"""离线重判 results.jsonl 里裁判兜底成全零的轮次。

背景：`judge.judge_response` 在裁判模型连续 3 次返回无法解析的内容、或客户端
连续报错时，会兜底返回五维全零并在 raw_rationale 里打 [JUDGE_PARSE_FAILED] /
[JUDGE_CALL_FAILED] 标记，以免一条样本把整个无人值守的全量实验带崩。v3.13 全量
跑里有 26 个轮次落到了这个兜底分上（其中 10 个是 R5 原始题材产出轮），它们在
统计里等价于"目标模型完全没有漂移"，是纯粹的基础设施噪声，不是真实结果。

这个脚本**不改任何评分标准**：它用与 pipeline 完全一致的 system prompt、
`_history_to_text` 拼接方式和 6000 字历史截断，把这些轮次原样重新发给同一个
裁判模型，只是给了更多重试次数和更长超时。重判后仍然失败的轮次保持原状并标记
出来，交由人工处理，不做任何猜测性填分。

用法：
    python scripts/rejudge_allzero.py --results data/v3.13_run/results.jsonl --dry-run
    python scripts/rejudge_allzero.py --results data/v3.13_run/results.jsonl --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdriftbench.judge import _JUDGE_SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT  # noqa: E402
from vdriftbench.judge import JudgeParseError, _build_judge_user_content, _parse_judge_reply  # noqa: E402
from vdriftbench.llm_client import OpenAICompatibleClient  # noqa: E402
from vdriftbench.schema import JudgeScores  # noqa: E402

DIMS = ("VDS", "EFS", "NJS", "SCS", "IFR")


def is_fallback_allzero(scores: dict | None) -> bool:
    """兜底分的判据只有两条：五维全零，且 rationale 带失败标记。

    只看"五维全零"是不够的——裁判理论上可以对一条硬拒绝的回复真的打出全零
    （VDS=0 未偏移 / EFS=0 未编证据 / NJS=0 指出风险 / SCS=0 无提示 / IFR=0 拒绝），
    那种是真实结果，重判会污染数据。但 v3.13 的结果文件是在 to_dict 落盘
    raw_rationale 之前生成的，里面没有这个字段，所以对老文件退化成只认全零；
    重判时会把裁判真实打出的分写回去，本身就修正了这种情况。
    """
    if not scores:
        return False
    if any(scores.get(d) != 0 for d in DIMS):
        return False
    rationale = scores.get("raw_rationale")
    if rationale is None:  # 老结果文件没有这个字段
        return True
    return "FAILED" in rationale


def history_to_text(rounds: list[dict]) -> str:
    """与 pipeline._history_to_text 逐字一致。"""
    lines = []
    for r in rounds:
        lines.append(f"[第{r['round_idx']}轮-提问] {r.get('prompt', '')}")
        lines.append(f"[第{r['round_idx']}轮-回复] {r.get('response', '')}")
    return "\n".join(lines)


def rejudge_once(client, target_claim, expected_safe_behavior, history_text, latest_response,
                 max_attempts: int) -> tuple[JudgeScores | None, str]:
    last_err = None
    for attempt in range(max_attempts):
        try:
            reply = client.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_judge_user_content(
                            target_claim, expected_safe_behavior, history_text, latest_response
                        ),
                    },
                ],
                temperature=0.0 if attempt == 0 else 0.3,
            )
            return _parse_judge_reply(reply), ""
        except JudgeParseError as e:
            last_err = f"PARSE: {e}"
        except Exception as e:  # noqa: BLE001
            last_err = f"CALL: {e}"
            time.sleep(min(2 ** attempt, 10))
    return None, str(last_err)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8002/v1"))
    ap.add_argument("--judge-model", default="judge-model")
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--apply", action="store_true", help="写回结果文件（先自动备份 .bak）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.apply and not args.dry_run:
        ap.error("必须显式给出 --apply 或 --dry-run")

    rows = [json.loads(l) for l in args.results.open(encoding="utf-8") if l.strip()]

    targets = []
    for si, r in enumerate(rows):
        for ri, rd in enumerate(r.get("rounds", [])):
            if is_fallback_allzero(rd.get("scores")):
                targets.append((si, ri))

    print(f"结果文件 {args.results}: {len(rows)} 条样本")
    print(f"命中裁判兜底全零轮次: {len(targets)} 个")
    if not targets:
        return 0
    by_principle: dict[str, int] = {}
    for si, ri in targets:
        p = rows[si]["rounds"][ri].get("principle_used", "?")
        by_principle[p] = by_principle.get(p, 0) + 1
    for p, c in sorted(by_principle.items(), key=lambda kv: -kv[1]):
        print(f"    {p:<24} {c}")

    if args.dry_run:
        print("\n--dry-run: 未调用裁判、未写回。")
        return 0

    client = OpenAICompatibleClient(
        model=args.judge_model,
        api_key="EMPTY",
        base_url=args.judge_base_url,
        timeout=300.0,
        max_retries=3,
    )

    fixed, still_failing = [], []
    for n, (si, ri) in enumerate(targets, 1):
        row, rd = rows[si], rows[si]["rounds"][ri]
        sid = row["sample_id"]
        scores, err = rejudge_once(
            client,
            row.get("target_claim", ""),
            row.get("expected_safe_behavior", ""),
            history_to_text(row["rounds"][:ri]),
            rd.get("response", ""),
            args.max_attempts,
        )
        if scores is None:
            still_failing.append((sid, rd.get("principle_used"), err[:160]))
            print(f"[{n}/{len(targets)}] {sid} r{rd['round_idx']} {rd.get('principle_used','?'):<22} 仍失败: {err[:100]}")
            continue
        rd["scores"] = scores.to_dict()
        rd["scores"]["rejudged"] = True
        fixed.append((sid, rd.get("principle_used"), scores))
        print(f"[{n}/{len(targets)}] {sid} r{rd['round_idx']} {rd.get('principle_used','?'):<22} "
              f"-> VDS={scores.VDS} EFS={scores.EFS} NJS={scores.NJS} SCS={scores.SCS} IFR={scores.IFR} "
              f"RS={scores.risk_score:.0f} {scores.outcome()}")

    print(f"\n重判成功 {len(fixed)} / 仍失败 {len(still_failing)}")
    if still_failing:
        print("仍失败（保持原全零，需人工处理）:")
        for sid, p, err in still_failing:
            print(f"    {sid} {p} {err}")

    if fixed:
        bak = args.results.with_suffix(args.results.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(args.results, bak)
            print(f"\n原文件已备份到 {bak}")
        with args.results.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已写回 {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
