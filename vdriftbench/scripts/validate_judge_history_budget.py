#!/usr/bin/env python3
"""v3.14.1: 裁判历史预算缩到多少还不改判？

背景。裁判调用是整条流水线最贵的一步：一次典型的 R5 打分 prompt_tokens=13798，
输出只有 100 多个 token，墙钟时间几乎全在 prefill 上。这 13798 里约 45% 是对话历史
（`judge._MAX_HISTORY_TEXT_CHARS`，v3.14 是 6000 字），另一半是被判的正文。

v3.13 的判分边界写得很清楚，五个维度全部从**被判的那一段正文**里读（"评的是这段
回复实际提供了什么内容"）；历史只是让裁判看出这一轮在回应什么。所以历史是可压的，
正文不是。但"可压"要有数，不能拍脑袋——判分变了，成功率就变了，整条对比链就废了。

这个脚本把已跑完的结果原样回放：同一段正文、同一个裁判、同一份 rubric，只改历史
预算，看 outcome 判定和逐维分变不变。抽样按 outcome 分层（成功/失败各一半），并
只取历史确实超过基线预算的轮次——历史本来就没到 6000 字的轮次，改预算对它没有
任何影响，放进样本里只会把一致率注水。

注意：这个脚本会占用裁判 GPU（默认 40 条 × 4 个预算 = 160 次裁判调用）。跑之前先确认
没有别的实验在用 :8002，否则既会拖慢那个实验，测出来的 s/call 也没有参考价值。

用法：
    /hy-tmp/venv/bin/python scripts/validate_judge_history_budget.py \\
        --results data/v3.14_run/results.jsonl --n 40 --budgets 6000 3000 2000 1200

一致率达标（outcome 100%、逐维基本不动）后，把 `Config.judge_max_history_chars`
从 6000 改成选中的那个值即可，其余代码不用动。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.config import Config  # noqa: E402
from vdriftbench.enrich import load_enriched_jsonl  # noqa: E402
from vdriftbench.env_loader import load_dotenv_if_present  # noqa: E402
from vdriftbench.judge import DIMS, judge_response_stable  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402
from vdriftbench.schema import JudgeScores  # noqa: E402


def _outcome(sc: JudgeScores, config: Config) -> str:
    return sc.outcome(
        config.risk_success_threshold,
        config.risk_partial_threshold,
        config.enable_delivery_success_rule,
        config.outcome_mode,
    )


def _stored_outcome(scores: dict, config: Config) -> str:
    return _outcome(
        JudgeScores(
            scores["VDS"], scores["EFS"], scores["NJS"], scores["SCS"], scores["IFR"],
            non_delivery=bool(scores.get("non_delivery", False)),
        ),
        config,
    )


def _history_text(prior_rounds: list[dict]) -> str:
    lines = []
    for r in prior_rounds:
        lines.append(f"[第{r['round_idx']}轮-提问] {r['prompt']}")
        lines.append(f"[第{r['round_idx']}轮-回复] {r['response']}")
    return "\n".join(lines)


def collect_cases(results_path: str, dataset_path: str, config: Config, n: int, min_history: int, seed: int):
    samples = {s.sample_id: s for s in load_enriched_jsonl(dataset_path, verify_against_raw=False)}
    pool = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = samples.get(row["sample_id"])
            if sample is None:
                continue
            prior: list[dict] = []
            for rd in row["rounds"]:
                history_text = _history_text(prior)
                prior.append(rd)
                sc = rd.get("scores")
                # 未交付轮次在调用裁判**之前**就短路了，历史预算对它毫无影响。
                if sc is None or sc.get("non_delivery") or len(history_text) < min_history:
                    continue
                pool.append({
                    "sample_id": row["sample_id"],
                    "round_idx": rd["round_idx"],
                    "principle": rd["principle_used"],
                    "target_claim": sample.target_claim,
                    "expected_safe_behavior": sample.expected_safe_behavior,
                    "history_text": history_text,
                    "response": rd["response"],
                    "stored_outcome": _stored_outcome(sc, config),
                })
    rng = random.Random(seed)
    by_outcome: dict[str, list] = {}
    for case in pool:
        by_outcome.setdefault(case["stored_outcome"], []).append(case)
    picked = []
    per = max(1, n // max(1, len(by_outcome)))
    for outcome, cases in sorted(by_outcome.items()):
        rng.shuffle(cases)
        picked.extend(cases[:per])
    rng.shuffle(picked)
    return pool, picked[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="data/v3.14_run/results.jsonl")
    p.add_argument("--data", default="data/dataset_100.enriched.jsonl")
    p.add_argument("--judge-model", default="judge-model")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--min-history", type=int, default=6000,
                   help="只取历史长度 >= 此值的轮次（更短的轮次不受预算影响）")
    p.add_argument("--budgets", type=int, nargs="+", default=[6000, 3000, 2000, 1200])
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--out", default="data/v3.14_run/judge_history_budget.json")
    args = p.parse_args()

    # 没有这一句，`build_llm_client("judge-model")` 拿不到 VDB_JUDGE_BASE_URL，会退到
    # OpenAI 官方端点上干等 300 秒超时（这台机器没有外网），而且一声不吭。
    load_dotenv_if_present()

    config = Config()
    pool, cases = collect_cases(args.results, args.data, config, args.n, args.min_history, args.seed)
    print(f"候选轮次 {len(pool)} 条（历史>={args.min_history}字），分层抽样 {len(cases)} 条", flush=True)
    print("抽样 outcome 分布:", Counter(c["stored_outcome"] for c in cases), flush=True)
    print(f"将发出 {len(cases) * len(args.budgets)} 次裁判调用", flush=True)

    judge = build_llm_client(args.judge_model, mock=False, persona="judge")
    baseline = args.budgets[0]
    scored: dict[int, list[JudgeScores]] = {}
    timing: dict[int, float] = {}
    for budget in args.budgets:
        t0 = time.time()
        row = []
        for i, case in enumerate(cases, 1):
            sc = judge_response_stable(
                case["target_claim"], case["expected_safe_behavior"],
                case["history_text"], case["response"], judge,
                n_samples=config.judge_n_samples,
                per_dimension=config.judge_per_dimension,
                max_history_chars=budget,
            )
            row.append(sc)
            print(f"  budget={budget} [{i}/{len(cases)}] {case['sample_id']} r{case['round_idx']} "
                  f"-> {_outcome(sc, config)}", flush=True)
        scored[budget] = row
        timing[budget] = time.time() - t0

    report = {"n_cases": len(cases), "min_history": args.min_history, "budgets": {}}
    base_scores = scored[baseline]
    print("\n=== 相对 budget=%d 的一致率 ===" % baseline)
    for budget in args.budgets:
        row = scored[budget]
        same_outcome = sum(
            1 for a, b in zip(base_scores, row) if _outcome(a, config) == _outcome(b, config)
        )
        per_dim = {
            dim: sum(1 for a, b in zip(base_scores, row) if getattr(a, dim) == getattr(b, dim))
            for dim in DIMS
        }
        n_succ = sum(1 for sc in row if _outcome(sc, config) == "成功")
        report["budgets"][budget] = {
            "outcome_agreement": same_outcome / len(cases),
            "per_dim_agreement": {d: v / len(cases) for d, v in per_dim.items()},
            "n_success": n_succ,
            "seconds_total": timing[budget],
            "seconds_per_call": timing[budget] / len(cases),
        }
        print(
            f"budget={budget:5d}  outcome一致 {same_outcome}/{len(cases)} "
            f"({100*same_outcome/len(cases):.1f}%)  成功数 {n_succ}  "
            f"逐维 " + " ".join(f"{d}={100*v/len(cases):.0f}%" for d, v in per_dim.items())
            + f"  {timing[budget]/len(cases):.1f}s/call"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
