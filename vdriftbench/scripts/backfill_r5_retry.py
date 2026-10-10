#!/usr/bin/env python3
"""v3.14.1: 给已经跑完的样本补一轮 R5 补救轮。

为什么需要这个脚本。R5 补救轮是在 v3.14 全量跑到 271/314 时才加进 pipeline 的，
后面 43 条会自带这一轮，前面 271 条不会。不补的话结果文件里会有两种样本，
"R5 补救轮的收益"这个数字就没法在全量上算。

它做的事和 `pipeline.run_single_sample` 里那段补救轮**完全一样**，只是历史不再重跑
而是从 `results.jsonl` 里读回来（build-up 各轮的 prompt/response 原样落盘了，
逐字重放给目标模型就能复现出同一段对话上下文）：

  1. 挑出符合条件的样本：样本级已成功、且 R5 那一轮未成功、且还没有补救轮
     （`--all-r5-failures` 放宽成"只要 R5 未成功"，对应 `r5_retry_only_when_sample_succeeded=False`）
  2. `principles.repair_genre_prompt` 拿裁判五维分 + 目标模型的拒绝原文当反馈改写请求，
     体裁和话题上锁
  3. 把改写后的请求接在原对话之后发给目标模型，重新判分
  4. 追加 `principle_used="RAW_GENRE_PROMPT_RETRY"` / `RAW_GENRE_PROMPT_RETRY_2` 轮次
     （首轮补救失败且 `r5_retry_rounds >= 2` 时再补第二轮）

**不重跑**安全复检轮和 RWI 审核：前者问的是"上述结论是否可靠"，补救轮之后再问一遍
会得到一个和前 43 条口径不同的 recovery_scs（那 43 条的安全复检看得到补救轮，
这 271 条的看不到）；与其造一个半新半旧的数字，不如让这个字段保持它原本的含义
（"build-up + 首轮 R5 之后的自我纠偏"），补救轮的效果单独看 R5 那两轮就够了。
`--rerun-safety` 可以打开重跑，但那样这一列在两批样本之间就不可比了。

默认写到单独的文件，确认无误再 `--apply` 合并回 results.jsonl（会先存 .bak）：

    # 1) 先看会补哪些、补多少条
    /hy-tmp/venv/bin/python scripts/backfill_r5_retry.py --dry-run

    # 2) 实跑（可随时 Ctrl-C，再跑一次会跳过已完成的）
    /hy-tmp/venv/bin/python scripts/backfill_r5_retry.py -v

    # 3) 合并回主结果文件
    /hy-tmp/venv/bin/python scripts/backfill_r5_retry.py --apply

注意这个脚本会同时占用目标模型、起草模型和裁判 GPU，别和别的实验并跑。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench import principles as principles_module  # noqa: E402
from vdriftbench.config import Config  # noqa: E402
from vdriftbench.enrich import load_enriched_jsonl  # noqa: E402
from vdriftbench.env_loader import load_dotenv_if_present  # noqa: E402
from vdriftbench.io_utils import round_to_dict  # noqa: E402
from vdriftbench.judge import judge_response_stable  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402
from vdriftbench.pipeline import (  # noqa: E402
    GENRE_CONTENT_PRINCIPLE,
    R5_RETRY_PRINCIPLES,
    _history_to_messages,
    _history_to_text,
    prompt_history_for_genre,
)
from vdriftbench.schema import JudgeScores, RoundRecord  # noqa: E402

logger = logging.getLogger("backfill_r5_retry")


def _scores_from_dict(d: dict | None) -> JudgeScores | None:
    if d is None:
        return None
    return JudgeScores(
        VDS=d["VDS"], EFS=d["EFS"], NJS=d["NJS"], SCS=d["SCS"], IFR=d["IFR"],
        raw_rationale=d.get("raw_rationale", ""),
        non_delivery=bool(d.get("non_delivery", False)),
    )


def _outcome(scores: JudgeScores | None, config: Config) -> str | None:
    if scores is None:
        return None
    return scores.outcome(
        config.risk_success_threshold,
        config.risk_partial_threshold,
        config.enable_delivery_success_rule,
        config.outcome_mode,
    )


def _round_from_row(d: dict) -> RoundRecord:
    """只重建补救轮真正要用到的字段：对话回放要 prompt/response，改写反馈要 scores，
    `principles._history_block` 要 round_idx。其余字段这条路径上用不到。"""

    return RoundRecord(
        round_idx=d["round_idx"],
        state_resolved=d.get("state_resolved"),
        resolved_by=d.get("resolved_by"),
        principle_used=d.get("principle_used", ""),
        bandit_context=None,
        bandit_posterior_snapshot=None,
        prompt=d.get("prompt", ""),
        response=d.get("response", ""),
        scores=_scores_from_dict(d.get("scores")),
        embed_drift_norm=d.get("embed_drift_norm"),
    )


def _existing_retry_count(rounds: list) -> int:
    return sum(1 for r in rounds if r.get("principle_used") in R5_RETRY_PRINCIPLES)


def select_rows(rows: list[dict], config: Config, all_failures: bool) -> list[dict]:
    picked = []
    for row in rows:
        rounds = row.get("rounds") or []
        n_retry = _existing_retry_count(rounds)
        if n_retry >= config.r5_retry_rounds:
            continue  # 该样本的 pipeline 级补救轮已补满
        genre = [r for r in rounds if r.get("principle_used") == GENRE_CONTENT_PRINCIPLE]
        if not genre:
            continue
        # 决定下一轮补救要接在哪条失败记录后面：首轮 R5 或上一轮补救。
        if n_retry == 0:
            last_failed = genre[-1]
        else:
            retry_rounds = [r for r in rounds if r.get("principle_used") in R5_RETRY_PRINCIPLES]
            last_failed = retry_rounds[-1]
        if _outcome(_scores_from_dict(last_failed.get("scores")), config) == "成功":
            continue
        sample_ok = any(
            _outcome(_scores_from_dict(r.get("scores")), config) == "成功" for r in rounds
        )
        if not sample_ok and not all_failures:
            continue
        picked.append(row)
    return picked


def backfill_one(row: dict, sample, clients, config: Config) -> dict:
    target, judge, draft = clients
    rounds = [_round_from_row(r) for r in row["rounds"]]
    n_retry = _existing_retry_count(row["rounds"])
    genre_idx = max(
        i for i, r in enumerate(rounds) if r.principle_used == GENRE_CONTENT_PRINCIPLE
    )
    if n_retry == 0:
        failed_record = rounds[genre_idx]
        history = rounds[: genre_idx + 1]
    else:
        retry_idx = max(
            i for i, r in enumerate(rounds) if r.principle_used in R5_RETRY_PRINCIPLES
        )
        failed_record = rounds[retry_idx]
        history = rounds[: retry_idx + 1]
    retry_principle = R5_RETRY_PRINCIPLES[n_retry]
    history_for_rewrite = prompt_history_for_genre(history, config, for_repair=True)

    retry_prompt, diagnosis = principles_module.repair_genre_prompt(
        sample, history_for_rewrite, draft,
        failed_prompt=failed_record.prompt,
        failed_response=failed_record.response,
        failed_scores=failed_record.scores,
        temperature=config.r5_retry_temperature,
        max_chars=config.genre_prompt_max_chars,
        max_attempts=config.r5_retry_max_attempts,
        min_topic_overlap=config.r5_retry_min_topic_overlap,
        ask_lead_chars=config.r5_retry_ask_lead_chars,
        enforce_pair_style=config.r5_retry_pair_style,
        require_evidence=config.r5_retry_require_evidence,
        success_only_history=config.r5_prompt_success_history_only,
    )

    messages = _history_to_messages(history) + [{"role": "user", "content": retry_prompt}]
    response = target.chat(messages)
    scores = judge_response_stable(
        target_claim=sample.target_claim,
        expected_safe_behavior=sample.expected_safe_behavior,
        history_text=_history_to_text(history),
        latest_response=response,
        judge_client=judge,
        n_samples=config.judge_n_samples,
        per_dimension=config.judge_per_dimension,
        max_history_chars=config.judge_max_history_chars,
    )
    record = RoundRecord(
        round_idx=rounds[-1].round_idx + 1,
        state_resolved=None,
        resolved_by=None,
        principle_used=retry_principle,
        bandit_context=None,
        bandit_posterior_snapshot=None,
        prompt=retry_prompt,
        response=response,
        scores=scores,
        # 漂移向量要 embedding 模型和 round1 的原始锚点才能算，回放算不出可比的值。
        # 与其填一个和别的轮次不同量纲的数，不如留 None，下游本来就允许缺失。
        embed_drift_norm=None,
        draft_thought=diagnosis,
    )
    return {
        "sample_id": row["sample_id"],
        "r5_outcome_before": _outcome(failed_record.scores, config),
        "r5_retry_outcome": _outcome(scores, config),
        "retry_round": round_to_dict(record),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="data/v3.14_run/results.jsonl")
    p.add_argument("--data", default="data/dataset_100.enriched.jsonl")
    p.add_argument("--out", default="data/v3.14_run/r5_retry_backfill.jsonl")
    p.add_argument("--target-model", default="deepseek-v4-pro")
    p.add_argument("--judge-model", default="judge-model")
    p.add_argument("--draft-model", default="draft-model")
    p.add_argument("--llm-seed", type=int, default=20260904)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--all-r5-failures", action="store_true",
                   help="不要求样本已经成功过，只要 R5 未成功就补")
    p.add_argument("--dry-run", action="store_true", help="只打印会补哪些样本")
    p.add_argument("--apply", action="store_true",
                   help="把 --out 里已完成的补救轮合并回 --results（先备份 .bak），不发任何请求")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    config = Config()

    with open(args.results, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.apply:
        # 按 (sample_id, principle) 索引，避免同一补救轮被 --apply 重复合并。
        patches: dict[tuple[str, str], dict] = {}
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    principle = rec["retry_round"]["principle_used"]
                    patches[(rec["sample_id"], principle)] = rec["retry_round"]
        n = 0
        for row in rows:
            for principle in R5_RETRY_PRINCIPLES:
                patch = patches.get((row["sample_id"], principle))
                if patch is None:
                    continue
                if any(r.get("principle_used") == principle for r in row["rounds"]):
                    continue
                row["rounds"].append(patch)
                n += 1
        shutil.copy2(args.results, args.results + ".bak")
        tmp = args.results + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, args.results)
        print(f"合并了 {n} 条补救轮到 {args.results}（原文件备份为 {args.results}.bak）")
        return

    todo = select_rows(rows, config, args.all_r5_failures)
    done_keys: set[tuple[str, str]] = set()
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done_keys.add((rec["sample_id"], rec["retry_round"]["principle_used"]))
    todo = [
        r for r in todo
        if (r["sample_id"], R5_RETRY_PRINCIPLES[_existing_retry_count(r["rounds"])])
        not in done_keys
    ]
    if args.limit:
        todo = todo[: args.limit]

    print(f"结果文件 {len(rows)} 条；符合补救条件且未补的 {len(todo)} 条"
          + (f"（已完成 {len(done_keys)} 条补救轮，跳过）" if done_keys else ""))
    if args.dry_run or not todo:
        for row in todo:
            print("  ", row["sample_id"], row.get("category_macro"))
        return

    load_dotenv_if_present()
    samples = {s.sample_id: s for s in load_enriched_jsonl(args.data, verify_against_raw=False)}
    clients = (
        build_llm_client(args.target_model, mock=False, persona="target", seed=args.llm_seed),
        build_llm_client(args.judge_model, mock=False, persona="judge", seed=args.llm_seed),
        build_llm_client(args.draft_model, mock=False, persona="draft", seed=args.llm_seed),
    )

    flipped = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo, 1):
            sample = samples.get(row["sample_id"])
            if sample is None:
                logger.warning("%s not in dataset, skipped", row["sample_id"])
                continue
            t0 = time.time()
            try:
                rec = backfill_one(row, sample, clients, config)
            except Exception:
                logger.exception("%s failed, will be retried on the next run", row["sample_id"])
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            flipped += rec["r5_retry_outcome"] == "成功"
            print(f"[{i}/{len(todo)}] {rec['sample_id']} "
                  f"{rec['r5_outcome_before']} -> {rec['r5_retry_outcome']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n补救轮翻成成功 {flipped}/{len(todo)}")
    print(f"逐条结果 -> {args.out}；确认后用 --apply 合并回 {args.results}")


if __name__ == "__main__":
    main()
