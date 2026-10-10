"""v3.14 修复B(R5 具体化要求) 的 pilot：重放已有对话历史，只重跑 R5。

做法：从 v3.13 的 results.jsonl 里取出每条样本 R5 之前的全部轮次，原样当作对话
历史，用**新的**改写器（带 `_GENRE_EVIDENCE_CLAUSE`）重新生成 R5 请求发给目标
模型，再用同一个裁判打分。裁判的 system prompt 和评分标准完全不变，变的只有攻击
侧的 R5 请求。

必须带对照组：具体化要求也可能反过来把原本成功的样本推到拒答（要"可核查的数据"
比只要"一篇文章"更容易让目标模型意识到命题站不住）。只测那 61 条未成功样本会看
不见这个副作用，测出来的提升是虚的。所以默认同时抽一批**原本已成功**的样本重跑，
统计净变化。

用法：
    python scripts/pilot_r5_evidence.py --results data/v3.13_run/results.jsonl \
        --out data/v314_pilot/r5_evidence_pilot.jsonl --n-fail 61 --n-success 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
for _k in list(os.environ):  # .env 是 CRLF 存的，尾随 \r 会让 Authorization 头非法
    if "KEY" in _k or _k.endswith("_URL"):
        os.environ[_k] = os.environ[_k].strip()

from vdriftbench import principles as principles_module  # noqa: E402
from vdriftbench.judge import judge_response  # noqa: E402
from vdriftbench.llm_client import OpenAICompatibleClient, build_llm_client  # noqa: E402
from vdriftbench.schema import JudgeScores, RoundRecord, Sample  # noqa: E402

R5_PRINCIPLE = "RAW_GENRE_PROMPT"


def sc_from(d: dict) -> JudgeScores:
    return JudgeScores(
        VDS=d["VDS"], EFS=d["EFS"], NJS=d["NJS"], SCS=d["SCS"], IFR=d["IFR"],
        raw_rationale=d.get("raw_rationale", ""),
    )


def build_history(row: dict, r5_idx: int) -> list[RoundRecord]:
    return [
        RoundRecord(
            round_idx=rd["round_idx"], state_resolved=None, resolved_by=None,
            principle_used=rd.get("principle_used", ""), bandit_context=None,
            bandit_posterior_snapshot=None, prompt=rd.get("prompt", ""),
            response=rd.get("response", ""), scores=None,
            embed_drift_norm=rd.get("embed_drift_norm"),
        )
        for rd in row["rounds"] if rd["round_idx"] < r5_idx
    ]


def sample_from(row: dict) -> Sample:
    return Sample(
        sample_id=row["sample_id"], prompt=row.get("prompt", ""),
        category=row.get("category", ""), category_macro=row.get("category_macro", ""),
        target_claim=row.get("target_claim", ""), value_axis=row.get("value_axis", ""),
        risk_type=row.get("risk_type", ""),
        expected_safe_behavior=row.get("expected_safe_behavior", ""),
    )


def run_one(row: dict, r5_idx: int, old: JudgeScores, group: str,
            draft, target, judge, require_evidence: bool) -> dict:
    sample = sample_from(row)
    history = build_history(row, r5_idx)
    prompt = principles_module.rewrite_genre_prompt(
        sample, history, draft,
        temperature=0.8, max_chars=160, ask_lead_chars=100, max_attempts=2,
        require_evidence=require_evidence,
    )
    messages = [
        m for rd in history
        for m in ({"role": "user", "content": rd.prompt}, {"role": "assistant", "content": rd.response})
    ] + [{"role": "user", "content": prompt}]
    response = target.chat(messages, temperature=0.7)
    new = judge_response(
        target_claim=sample.target_claim,
        expected_safe_behavior=sample.expected_safe_behavior,
        history_text="\n".join(
            f"[第{rd.round_idx}轮-{k}] {v}" for rd in history
            for k, v in (("提问", rd.prompt), ("回复", rd.response))
        ),
        latest_response=response,
        judge_client=judge,
    )
    return {
        "sample_id": sample.sample_id, "group": group, "category": row.get("category", ""),
        "r5_idx": r5_idx, "new_prompt": prompt, "new_response": response,
        "old_scores": old.to_dict(), "old_outcome": old.outcome(),
        "new_scores": new.to_dict(), "new_outcome": new.outcome(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-fail", type=int, default=61, help="抽多少条原本未成功的")
    ap.add_argument("--n-success", type=int, default=40, help="对照组: 抽多少条原本已成功的")
    ap.add_argument("--no-evidence", action="store_true", help="关掉具体化要求(重跑基线，测纯采样方差)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.results.open(encoding="utf-8") if l.strip()]
    fails, succs = [], []
    for row in rows:
        for rd in row.get("rounds", []):
            if rd.get("principle_used") == R5_PRINCIPLE and rd.get("scores"):
                s = sc_from(rd["scores"])
                (succs if s.outcome() == "成功" else fails).append((row, rd["round_idx"], s))
                break

    rnd = random.Random(args.seed)
    rnd.shuffle(succs)
    picked = ([(r, i, s, "未成功") for r, i, s in fails[: args.n_fail]]
              + [(r, i, s, "对照-已成功") for r, i, s in succs[: args.n_success]])
    print(f"原始 R5: 成功 {len(succs)} / 未成功 {len(fails)}")
    print(f"pilot 抽样: 未成功 {min(args.n_fail, len(fails))} + 对照已成功 {min(args.n_success, len(succs))} "
          f"= {len(picked)} 条")
    print(f"具体化要求: {'关闭(基线重跑)' if args.no_evidence else '开启'}\n")

    draft = build_llm_client("draft-model", mock=False, persona="draft")
    judge = build_llm_client("judge-model", mock=False, persona="judge")
    target = OpenAICompatibleClient(
        model="deepseek-v4-flash",
        api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip() or None,
        base_url="https://api.deepseek.com",
        timeout=180.0, max_retries=5,
        extra_body={"thinking": {"type": "disabled"}},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, errors = [], []
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, r, i, s, g, draft, target, judge, not args.no_evidence): r["sample_id"]
            for r, i, s, g in picked
        }
        for n, fut in enumerate(as_completed(futs), 1):
            sid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                errors.append((sid, str(e)[:120]))
                print(f"[{n}/{len(futs)}] {sid} 报错: {str(e)[:100]}")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done.append(rec)
            o, nw = rec["old_scores"], rec["new_scores"]
            mark = "  <== 翻成成功" if rec["old_outcome"] != "成功" and rec["new_outcome"] == "成功" else (
                "  <== 掉出成功!" if rec["old_outcome"] == "成功" and rec["new_outcome"] != "成功" else "")
            print(f"[{n}/{len(futs)}] {sid} {rec['group']:<11} "
                  f"EFS {o['EFS']}->{nw['EFS']}  RS {o['risk_score']:.0f}->{nw['risk_score']:.0f}  "
                  f"{rec['old_outcome']}->{rec['new_outcome']}{mark}")

    print(f"\n完成 {len(done)} / 报错 {len(errors)}  用时 {time.time()-t0:.0f}s")
    for g in ("未成功", "对照-已成功"):
        grp = [r for r in done if r["group"] == g]
        if not grp:
            continue
        up = sum(1 for r in grp if r["old_outcome"] != "成功" and r["new_outcome"] == "成功")
        dn = sum(1 for r in grp if r["old_outcome"] == "成功" and r["new_outcome"] != "成功")
        efs0 = sum(1 for r in grp if r["old_scores"]["EFS"] == 0)
        efs_up = sum(1 for r in grp if r["new_scores"]["EFS"] > r["old_scores"]["EFS"])
        print(f"\n[{g}] n={len(grp)}")
        print(f"  EFS 提升: {efs_up}/{len(grp)}  (原本 EFS=0 的有 {efs0} 条)")
        print(f"  EFS 均值: {sum(r['old_scores']['EFS'] for r in grp)/len(grp):.2f} -> "
              f"{sum(r['new_scores']['EFS'] for r in grp)/len(grp):.2f}")
        print(f"  RS  均值: {sum(r['old_scores']['risk_score'] for r in grp)/len(grp):.2f} -> "
              f"{sum(r['new_scores']['risk_score'] for r in grp)/len(grp):.2f}")
        print(f"  翻成成功 +{up} / 掉出成功 -{dn}")
    if errors:
        print(f"\n报错样本: {[s for s, _ in errors]}")
    print(f"\n明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
