"""验证 v3.14 裁判改动是否真的提升了自洽率。只读，不写回结果文件。

设计：在**随机分层**抽样的轮次上（不是按分数挑出来的低分尾巴，那会被向均值回归
污染），把同一段已存储的正文judge两遍，比较两遍的 outcome 是否一致。对两种配置各做
一次：

  旧配置 = 单次打分（v3.13 行为，judge_response）
  新配置 = 三次采样逐维取中位数（judge_response_stable, n_samples=3）

两者都跑在新的 300s 超时和"重试保持贪心"的阶梯上，所以这里量到的差异只来自
"单次 vs 中位数"。旧配置在 v3.13 那种 60s 超时下会更差。

对照基准：把 61 条未成功的 R5 正文原样重判时，旧配置的 outcome 自洽率是 21.3%
（那是低分尾巴上的最坏情况，不能直接和这里的随机抽样比）。
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
for _k in list(os.environ):
    if "KEY" in _k or _k.endswith("_URL"):
        os.environ[_k] = os.environ[_k].strip()

from vdriftbench.judge import judge_response, judge_response_stable  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402
from vdriftbench.schema import JudgeScores  # noqa: E402

RESULTS = Path("/root/vdriftbench/data/v3.13_run/results.jsonl")
N_PER_BAND = int(os.environ.get("N_PER_BAND", "10"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "3"))


def sc(d: dict) -> JudgeScores:
    return JudgeScores(VDS=d["VDS"], EFS=d["EFS"], NJS=d["NJS"], SCS=d["SCS"], IFR=d["IFR"])


def main() -> int:
    rows = [json.loads(l) for l in RESULTS.open(encoding="utf-8") if l.strip()]
    bands: dict[str, list] = {"成功": [], "部分成功": [], "失败": []}
    for r in rows:
        rounds = r.get("rounds", [])
        for i, x in enumerate(rounds):
            if not x.get("scores"):
                continue
            hist = "\n".join(
                f"[第{y['round_idx']}轮-{k}] {y.get(f, '')}"
                for y in rounds[:i]
                for k, f in (("提问", "prompt"), ("回复", "response"))
            )
            bands[sc(x["scores"]).outcome()].append(
                (r["sample_id"], x["round_idx"], r.get("target_claim", ""),
                 r.get("expected_safe_behavior", ""), hist, x.get("response", ""))
            )

    rnd = random.Random(20260906)
    tasks = []
    for band, items in bands.items():
        rnd.shuffle(items)
        tasks += [(band, *t) for t in items[:N_PER_BAND]]
    print(f"随机分层抽样 {len(tasks)} 个轮次: "
          + ", ".join(f"{b}={sum(1 for t in tasks if t[0] == b)}" for b in bands))
    print(f"每个轮次judge 4 遍: 旧配置(单次) x2 + 新配置(中位数 of {N_SAMPLES}) x2\n")

    judge = build_llm_client("judge-model", mock=False, persona="judge")

    def go(t):
        band, sid, ridx, tc, esb, hist, resp = t
        args = (tc, esb, hist, resp, judge)
        old_a = judge_response(*args)
        old_b = judge_response(*args)
        new_a = judge_response_stable(*args, n_samples=N_SAMPLES)
        new_b = judge_response_stable(*args, n_samples=N_SAMPLES)
        return band, sid, ridx, old_a, old_b, new_a, new_b

    with ThreadPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(go, tasks))

    def report(label, pairs):
        n = len(pairs)
        agree = sum(1 for a, b in pairs if a.outcome() == b.outcome())
        rs_same = sum(1 for a, b in pairs if a.risk_score == b.risk_score)
        dim_agree = {
            d: 100 * sum(1 for a, b in pairs if getattr(a, d) == getattr(b, d)) / n
            for d in ("VDS", "EFS", "NJS", "SCS", "IFR")
        }
        print(f"[{label}] n={n}")
        print(f"    outcome 两遍一致:   {agree}/{n} = {100 * agree / n:.1f}%")
        print(f"    RiskScore 完全相同: {rs_same}/{n} = {100 * rs_same / n:.1f}%")
        print("    逐维一致率: " + "  ".join(f"{d}={v:.0f}%" for d, v in dim_agree.items()))
        return 100 * agree / n

    print("=" * 74)
    old_pct = report("旧配置 单次打分", [(r[3], r[4]) for r in res])
    print()
    new_pct = report(f"新配置 中位数 of {N_SAMPLES}", [(r[5], r[6]) for r in res])
    print("=" * 74)
    print(f"\noutcome 自洽率: {old_pct:.1f}% -> {new_pct:.1f}%  ({new_pct - old_pct:+.1f}pt)")
    print("\n按原始分档:")
    for band in ("成功", "部分成功", "失败"):
        grp = [r for r in res if r[0] == band]
        if not grp:
            continue
        o = sum(1 for r in grp if r[3].outcome() == r[4].outcome())
        n_ = sum(1 for r in grp if r[5].outcome() == r[6].outcome())
        print(f"    {band}: 旧 {o}/{len(grp)} -> 新 {n_}/{len(grp)}")
    print("\n新配置下两遍判出的 outcome 分布:")
    print(f"    第一遍 {dict(Counter(r[5].outcome() for r in res))}")
    print(f"    第二遍 {dict(Counter(r[6].outcome() for r in res))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
