"""只读诊断：把 v3.13 存下来的 R5 正文原封不动重新过一遍裁判。

用来切开"pilot 重跑 R5 时 61 条未成功样本有 55 条翻成成功"这件事的两个候选成因：
裁判侧打分方差，还是目标模型侧的回复变了（`deepseek-v4-flash` 是远端 API，两次
运行之间可能已更新）。

做法：对 v3.13 里 R5 未成功的那 61 条样本，取出**同一段已存储的 R5 回复**、同样的
对话历史、同一个裁判模型和同一份评分标准，只重新打一次分。完全不调用目标模型，
也不写回任何结果文件。

- 若同一段文字这次大量被判成功 -> 方差在裁判侧，v3.13 的 R5 数字本身不可复现；
- 若基本重现原来的低分 -> 裁判是稳的，差异来自目标模型侧。
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

# 必须先加载 .env：裁判的 VDB_JUDGE_BASE_URL(本地 vLLM 8002) 只写在 .env 里，不加载
# 的话 build_llm_client 拿不到 base_url，每一次调用都会失败并被 judge_response 兜底
# 成五维全零——那正好是这个脚本要诊断的那种噪声，会让整轮结果看起来"裁判全判零分"。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
for _k in list(os.environ):
    if "KEY" in _k or _k.endswith("_URL"):
        os.environ[_k] = os.environ[_k].strip()

from vdriftbench.judge import judge_response  # noqa: E402
from vdriftbench.llm_client import OpenAICompatibleClient  # noqa: E402
from vdriftbench.schema import JudgeScores  # noqa: E402

RESULTS = Path("/root/vdriftbench/data/v3.13_run/results.jsonl")
DIMS = ("VDS", "EFS", "NJS", "SCS", "IFR")


def sc(d: dict) -> JudgeScores:
    return JudgeScores(VDS=d["VDS"], EFS=d["EFS"], NJS=d["NJS"], SCS=d["SCS"], IFR=d["IFR"])


def main() -> int:
    rows = [json.loads(l) for l in RESULTS.open(encoding="utf-8") if l.strip()]

    tasks = []
    for r in rows:
        rounds = r.get("rounds", [])
        for i, x in enumerate(rounds):
            if x.get("principle_used") == "RAW_GENRE_PROMPT" and x.get("scores"):
                old = sc(x["scores"])
                if old.outcome() != "成功":  # 只看当初未成功的那批
                    hist = "\n".join(
                        f"[第{y['round_idx']}轮-{k}] {y.get(f, '')}"
                        for y in rounds[:i]
                        for k, f in (("提问", "prompt"), ("回复", "response"))
                    )
                    tasks.append((r["sample_id"], r.get("target_claim", ""),
                                  r.get("expected_safe_behavior", ""), hist,
                                  x.get("response", ""), old))
                break

    print(f"重判 {len(tasks)} 条【原封不动的 v3.13 R5 正文】")
    print("同一段文字 / 同一裁判模型 / 同一评分标准 / 不调用目标模型 / 不写回文件\n")

    # 不用 build_llm_client：它的默认 timeout 是 60s，而 R5 的裁判输入有上万字，
    # 本地 7B 裁判在并发下光 prefill 就可能超过 60s，请求会被客户端判超时、连续重试
    # 失败后被 judge_response 兜底成五维全零 —— 那正是这个脚本要诊断的噪声本身。
    # （这也是 v3.13 那 26 个兜底全零轮次的主因：显式给 300s 超时后，同样的输入
    # 一次就判出了真实分数，其中 3 条是 RS=7 的成功。）
    judge = OpenAICompatibleClient(
        model="judge-model",
        api_key=os.environ.get("VDB_JUDGE_API_KEY", "EMPTY"),
        base_url=os.environ.get("VDB_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1"),
        timeout=300.0,
        max_retries=3,
    )

    def go(t):
        sid, tc, esb, hist, resp, old = t
        new = judge_response(target_claim=tc, expected_safe_behavior=esb,
                             history_text=hist, latest_response=resp, judge_client=judge)
        return sid, old, new

    with ThreadPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(go, tasks))

    n = len(res)
    agree = sum(1 for _, o, w in res if o.outcome() == w.outcome())
    up = sum(1 for _, o, w in res if w.outcome() == "成功")
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  outcome 与原判一致:        {agree}/{n} = {100 * agree / n:.1f}%")
    print(f"  同一段文字这次被判成功:    {up}/{n} = {100 * up / n:.1f}%")
    print(f"  RiskScore 均值: 原 {statistics.mean(o.risk_score for _, o, _ in res):.2f} "
          f"-> 重判 {statistics.mean(w.risk_score for _, _, w in res):.2f}")
    print(f"  EFS 均值:       原 {statistics.mean(o.EFS for _, o, _ in res):.2f} "
          f"-> 重判 {statistics.mean(w.EFS for _, _, w in res):.2f}")
    print("  逐维一致率: " + "  ".join(
        f"{d}={100 * sum(1 for _, o, w in res if getattr(o, d) == getattr(w, d)) / n:.0f}%" for d in DIMS))
    print("\n  参考: pilot 里【重新生成回复】时 55/61 = 90.2% 判成成功")
    print("\n明细:")
    for sid, o, w in sorted(res):
        flag = "  <== 翻成成功" if w.outcome() == "成功" else ""
        print(f"  {sid}: 原 VDS={o.VDS} EFS={o.EFS} NJS={o.NJS} SCS={o.SCS} IFR={o.IFR} "
              f"RS={o.risk_score:.0f} {o.outcome():<4} -> 重判 VDS={w.VDS} EFS={w.EFS} "
              f"NJS={w.NJS} SCS={w.SCS} IFR={w.IFR} RS={w.risk_score:.0f} {w.outcome()}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
