"""v3.14.1 样本级并行：加锁代理与 `run_dataset(workers=N)`。

并行是这次提速的主要来源，但它动的是**跨样本共享状态**，出错的方式很难在结果里看
出来（后验被覆盖、簇被算错、结果文件里两行 JSON 交错）。所以这里既测代理确实上了锁，
也测并行跑出来的东西和串行一致。
"""

import threading

from vdriftbench.archetypes import ResistArchetypeRegistry
from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.concurrency import (
    LockedArchetypeRegistry, LockedEmbeddingClient, LockedScheduler, wrap_shared_state,
)
from vdriftbench.config import Config
from vdriftbench.embedding_client import MockEmbeddingClient
from vdriftbench.llm_client import MockLLMClient
from vdriftbench.pipeline import run_dataset
from vdriftbench.schema import Sample


def _samples(n: int) -> list[Sample]:
    return [
        Sample(
            sample_id=f"T{i:04d}",
            prompt=f"示例prompt {i}",
            category="科学/伪科学",
            category_macro="科学与超现实类",
            target_claim=f"示例target_claim {i}",
            value_axis="事实准确性",
            risk_type="伪证据编造",
            expected_safe_behavior="指出该说法缺乏可靠证据支持",
        )
        for i in range(n)
    ]


def _clients():
    return (
        MockLLMClient(persona="target"),
        MockLLMClient(persona="judge"),
        MockLLMClient(persona="classifier"),
        MockEmbeddingClient(),
        MockLLMClient(persona="draft"),
    )


def test_wrap_shared_state_returns_locked_proxies_sharing_one_bandit_lock():
    scheduler, registry, embed = wrap_shared_state(
        ThompsonSamplingScheduler(seed=0),
        ResistArchetypeRegistry(),
        MockEmbeddingClient(),
    )
    assert isinstance(scheduler, LockedScheduler)
    assert isinstance(registry, LockedArchetypeRegistry)
    assert isinstance(embed, LockedEmbeddingClient)
    # bandit 与原型表会互相调用，必须是同一把可重入锁，否则 candidates_for_round
    # 那条路径上会自锁
    assert scheduler._lock is registry._lock
    assert embed._lock is not scheduler._lock


def test_locked_scheduler_serializes_concurrent_updates():
    inner = ThompsonSamplingScheduler(seed=0)
    locked = LockedScheduler(inner, threading.RLock())
    context = ("科学与超现实类", "S_engaged")
    candidates = [("P4", True), ("P4_rhetoric", True)]
    locked.select_action(context, candidates, pool_state="S_engaged")

    def worker():
        for _ in range(50):
            locked.update(context, "P4", 1, pool_state="S_engaged")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 4 线程 × 50 次全是 reward=1，一次都不能丢
    alpha, beta = inner.posterior[inner._context_key(context)]["P4"]
    assert (alpha, beta) == (200.0, 0.0)


def test_locked_scheduler_exposes_the_private_members_the_pipeline_reads():
    inner = ThompsonSamplingScheduler(seed=0)
    locked = LockedScheduler(inner, threading.RLock())
    cid = locked.propose_shadow_candidate("S_engaged", "先反问确认再顺势展开", round_idx=2)
    assert cid in locked._shadow_ids
    assert locked._find_shadow(cid) is not None
    assert locked._find_shadow("不存在的候选") is None


def test_locked_embedding_client_returns_the_same_vectors_as_the_inner_one():
    inner = MockEmbeddingClient()
    locked = LockedEmbeddingClient(inner, threading.Lock())
    assert (locked.embed("一段文本") == inner.embed("一段文本")).all()


def test_locked_registry_observe_still_clusters():
    inner = ResistArchetypeRegistry()
    locked = LockedArchetypeRegistry(inner, threading.RLock())
    embed = MockEmbeddingClient()
    first = locked.observe("科学与超现实类", "免责声明式回避", embed)
    again = locked.observe("科学与超现实类", "免责声明式回避", embed)
    assert first.cluster_id == again.cluster_id
    assert locked.clusters_for("科学与超现实类")


# --- run_dataset(workers=N) ---


def test_run_dataset_with_workers_processes_every_sample():
    target, judge, classifier, embed, draft = _clients()
    samples = _samples(8)
    seen: list[str] = []
    lock = threading.Lock()

    def on_done(i, sample, result):
        with lock:
            seen.append(sample.sample_id)

    results = run_dataset(
        samples, target, judge, classifier, embed,
        ThompsonSamplingScheduler(seed=0), Config(),
        draft_client=draft, skip_rwi_review=True,
        on_sample_done=on_done, workers=4,
    )
    assert len(results) == len(samples)
    assert sorted(seen) == sorted(s.sample_id for s in samples)


def test_run_dataset_workers_one_is_the_sequential_path():
    target, judge, classifier, embed, draft = _clients()
    samples = _samples(4)
    order: list[str] = []

    results = run_dataset(
        samples, target, judge, classifier, embed,
        ThompsonSamplingScheduler(seed=0), Config(),
        draft_client=draft, skip_rwi_review=True,
        on_sample_done=lambda i, s, r: order.append(s.sample_id),
        workers=1,
    )
    assert len(results) == 4
    assert order == [s.sample_id for s in samples]  # 串行时严格按提交顺序


def test_run_dataset_on_sample_done_is_never_called_concurrently():
    """调用方拿这个回调往同一个文件里 append + fsync，并发进来会写出坏行。"""

    target, judge, classifier, embed, draft = _clients()
    inside = 0
    overlaps = 0
    lock = threading.Lock()

    def on_done(i, sample, result):
        nonlocal inside, overlaps
        with lock:
            inside += 1
            if inside > 1:
                overlaps += 1
        # 停一下，真有并发调用就一定会被上面那段观察到
        threading.Event().wait(0.01)
        with lock:
            inside -= 1

    run_dataset(
        _samples(12), target, judge, classifier, embed,
        ThompsonSamplingScheduler(seed=0), Config(),
        draft_client=draft, skip_rwi_review=True,
        on_sample_done=on_done, workers=4,
    )
    assert overlaps == 0


def test_run_dataset_with_workers_skips_a_failing_sample_without_killing_the_run():
    target, judge, classifier, embed, draft = _clients()

    class _BoomTarget(MockLLMClient):
        def chat_with_logprobs(self, messages, *, temperature=0.7, top_logprobs=8):
            last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            if "示例prompt 2" in last or "示例target_claim 2" in last:
                raise RuntimeError("target endpoint blew up on this sample")
            return super().chat_with_logprobs(
                messages, temperature=temperature, top_logprobs=top_logprobs
            )

    samples = _samples(5)
    results = run_dataset(
        samples, _BoomTarget(persona="target"), judge, classifier, embed,
        ThompsonSamplingScheduler(seed=0), Config(),
        draft_client=draft, skip_rwi_review=True, workers=3,
    )
    # 失败的样本只是缺席（下一次 --resume 会重试它），其余照常跑完
    assert len(results) == 4
    assert "T0002" not in {r.sample.sample_id for r in results}
