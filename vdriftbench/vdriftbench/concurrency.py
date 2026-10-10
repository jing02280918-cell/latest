"""v3.14.1 样本级并行所需的共享状态保护。

v3.14 全量跑（314 条、目标 deepseek-v4-pro）实测 306 秒/样本，PAIR 基线在同一台机器、
同一个裁判栈上是 91 秒/样本。差距的来源不是"我们多问了几轮"——按调用计数，V-DriftBench
每样本约 4.8 次目标调用，PAIR 约 3 次，只差 1.6 倍，而墙钟差 3.4 倍。真正的原因是
**整条流水线全程单线程**：一次只有一个请求在飞，本地两块 vLLM（起草 Qwen2.5-32B-AWQ
@8001、裁判 Mistral-7B @8002）绝大部分时间在空转等下一条请求，GPU 利用率长期是 0。

vLLM 的连续批处理正好吃并发：几条请求同时在飞时，解码阶段合批，吞吐几乎线性上涨，
单条请求的延迟只小幅变差。所以把**样本之间**并行起来是这条流水线最大的一块提速，
而且它一个字都不改单个样本内部的流程——每个样本仍然严格按 R1→...→R5→安全复检 串行走。

唯一需要处理的是跨样本共享的三件可变状态：

  1. `ThompsonSamplingScheduler`：后验计数、影子候选池、转正事件
  2. `ResistArchetypeRegistry`：抵抗原型的增量聚类
  3. `EmbeddingClient`：一块本地 BGE-M3，`SentenceTransformer.encode` 不保证可重入

前两个用同一把可重入锁串起来（它们会互相调用：`registry.observe` 里要 embed，
`scheduler.update` 里可能触发转正、进而改 `principles.PRINCIPLES` 和
`bandit.DECISION_TABLE_CANDIDATES` 这两个模块级字典）。第三个单独一把锁，因为
embedding 调用远比 bandit 更新频繁，共用一把锁会把并行度压回 1。

**这会不会影响成功率？** 并行改变的只有一件事：一个样本做决策时看到的 bandit 后验，
可能少了若干条"和它同时在跑的样本"本该贡献的反馈（后验更新变得稍微陈旧）。
它不改任何 prompt、不改任何判分、不改轮次结构。而且 v3.14 的数据显示 bandit 早已
饱和——271 条里 196 条（72%）在第 1 轮（固定 P1，根本不经 bandit 选择）就已经判成功，
真正由 bandit 挑臂决定成败的样本是少数。并发度控制在个位数时，后验陈旧的程度相当于
"把样本顺序洗了个牌"，不构成方法上的改变。要完全复现旧行为，`--workers 1` 即可，
那条路径上连锁都不会被创建。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .archetypes import ArchetypeCluster, ResistArchetypeRegistry
from .bandit import ThompsonSamplingScheduler
from .embedding_client import EmbeddingClient


class LockedEmbeddingClient(EmbeddingClient):
    """串行化本地 embedding 调用。BGE-M3 跑在和两块 vLLM 同一张 4090 上，并发
    `encode` 既不安全也不会更快（它没有连续批处理），排队反而更稳。"""

    def __init__(self, inner: EmbeddingClient, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def embed(self, text: str) -> np.ndarray:
        with self._lock:
            return self._inner.embed(text)


class LockedScheduler:
    """`ThompsonSamplingScheduler` 的加锁代理。

    刻意用**显式转发**而不是 `__getattr__` 通配：漏掉一个方法应该在第一次调用时就
    `AttributeError` 炸出来，而不是悄悄绕过锁跑一整晚。`pipeline.run_single_sample`
    还直接读了 `_shadow_ids` 和调用了 `_find_shadow`，所以这两个私有成员也要转发。
    """

    def __init__(self, inner: ThompsonSamplingScheduler, lock: threading.RLock) -> None:
        self._inner = inner
        self._lock = lock

    @property
    def _shadow_ids(self) -> set:
        with self._lock:
            return set(self._inner._shadow_ids)

    def _find_shadow(self, candidate_id: str) -> Any:
        with self._lock:
            return self._inner._find_shadow(candidate_id)

    def select_action(self, *args, **kwargs):
        with self._lock:
            return self._inner.select_action(*args, **kwargs)

    def update(self, *args, **kwargs):
        with self._lock:
            return self._inner.update(*args, **kwargs)

    def snapshot(self, *args, **kwargs) -> dict:
        with self._lock:
            return self._inner.snapshot(*args, **kwargs)

    def propose_shadow_candidate(self, *args, **kwargs) -> str:
        with self._lock:
            return self._inner.propose_shadow_candidate(*args, **kwargs)

    def shadow_candidates_for_state(self, state: str):
        with self._lock:
            return self._inner.shadow_candidates_for_state(state)

    def save(self, path: str) -> None:
        with self._lock:
            self._inner.save(path)


class LockedArchetypeRegistry:
    """`ResistArchetypeRegistry` 的加锁代理。`observe` 会读改同一批簇心并调用
    embedding，两个线程同时进来会算出互相覆盖的簇。"""

    def __init__(self, inner: ResistArchetypeRegistry, lock: threading.RLock) -> None:
        self._inner = inner
        self._lock = lock

    def observe(self, category_macro: str, archetype_text: str, embed_client: EmbeddingClient) -> ArchetypeCluster:
        with self._lock:
            return self._inner.observe(category_macro, archetype_text, embed_client)

    def clusters_for(self, category_macro: str):
        with self._lock:
            return self._inner.clusters_for(category_macro)

    def snapshot(self) -> dict:
        with self._lock:
            return self._inner.snapshot()


def wrap_shared_state(
    scheduler: ThompsonSamplingScheduler,
    archetype_registry: ResistArchetypeRegistry,
    embed_client: EmbeddingClient,
) -> tuple[LockedScheduler, LockedArchetypeRegistry, LockedEmbeddingClient]:
    """一次性把三个共享对象换成加锁版本。`candidates_for_round` 里 bandit 与
    registry 会互相调用，所以前两者共用一把 `RLock`。"""

    bandit_lock = threading.RLock()
    return (
        LockedScheduler(scheduler, bandit_lock),
        LockedArchetypeRegistry(archetype_registry, bandit_lock),
        LockedEmbeddingClient(embed_client, threading.Lock()),
    )
