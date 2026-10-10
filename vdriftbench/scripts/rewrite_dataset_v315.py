#!/usr/bin/env python3
"""v3.15 dataset pass: rewrite PAIR-success / R5-fail gap prompts + expand 314→360.

Loads prompt text from data/dataset_v315_changes.json (curly-quote safe).

Usage:
    python scripts/rewrite_dataset_v315.py --dry-run
    python scripts/rewrite_dataset_v315.py
    python scripts/rewrite_dataset_v315.py --enrich
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_dataset import enrich_one_with_retry, is_unsafe_rewrite  # noqa: E402
from vdriftbench.category_map import CATEGORY_MACRO_MAP, to_macro  # noqa: E402
from vdriftbench.enrich import save_enriched_jsonl  # noqa: E402
from vdriftbench.env_loader import load_dotenv_if_present  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402
from vdriftbench.schema import Sample  # noqa: E402

MAX_PROMPT_CHARS = 120
TARGET_MEDIAN = (45, 85)
DATASET = Path("data/dataset_100.jsonl")
ENRICHED = Path("data/dataset_100.enriched.jsonl")
CHANGES = Path("data/dataset_v315_changes.json")
BACKUP_RAW = Path("data/dataset_100.v314_backup.jsonl")
BACKUP_ENR = Path("data/dataset_100.enriched.v314_backup.jsonl")
CHANGELOG = Path("data/dataset_optimization_changelog_v315.json")


def sid_of(idx: int) -> str:
    return f"S{idx:04d}"


def load_changes() -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]:
    raw = json.loads(CHANGES.read_text(encoding="utf-8"))
    gap = {k: tuple(v) for k, v in raw["gap_rewrites"].items()}
    new = [tuple(x) for x in raw["new_samples"]]
    return gap, new


def validate_prompt(sid: str, prompt: str, category: str) -> list[str]:
    errs = []
    if len(prompt) > MAX_PROMPT_CHARS:
        errs.append(f"{sid} 超长 {len(prompt)} > {MAX_PROMPT_CHARS}")
    if "\n" in prompt:
        errs.append(f"{sid} 含换行")
    why = is_unsafe_rewrite(prompt)
    if why:
        errs.append(f"{sid} 安全网: {why}")
    if category not in CATEGORY_MACRO_MAP:
        errs.append(f"{sid} 未知 category {category!r}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--enrich", action="store_true")
    ap.add_argument("--enrich-model", default="draft-model")
    args = ap.parse_args()

    gap_rewrites, new_samples = load_changes()
    if len(gap_rewrites) != 25:
        print(f"FATAL: expected 25 gap rewrites, got {len(gap_rewrites)}", file=sys.stderr)
        return 1
    if len(new_samples) != 46:
        print(f"FATAL: expected 46 new samples, got {len(new_samples)}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if len(rows) != 314:
        print(f"FATAL: expected 314 rows, got {len(rows)}", file=sys.stderr)
        return 1

    changes: list[dict] = []
    seen: set[str] = set()

    for idx, row in enumerate(rows):
        sid = sid_of(idx)
        if sid not in gap_rewrites:
            continue
        new_cat, new_prompt = gap_rewrites[sid]
        old_prompt, old_cat = row["prompt"], row["category"]
        row["prompt"], row["category"] = new_prompt, new_cat
        seen.add(sid)
        changes.append({
            "sample_id": sid,
            "action": "rewrite_gap25",
            "reason": "PAIR成功但v3.14 R5(含补救)失败；金样本结构+报告框重写",
            "old_category": old_cat,
            "new_category": new_cat,
            "old_prompt": old_prompt,
            "new_prompt": new_prompt,
            "old_len": len(old_prompt),
            "new_len": len(new_prompt),
        })

    if set(gap_rewrites) != seen:
        print(f"FATAL: missing gap ids {set(gap_rewrites) - seen}", file=sys.stderr)
        return 1

    for cat, prompt in new_samples:
        rows.append({"category": cat, "prompt": prompt})

    for idx in range(314, 360):
        sid = sid_of(idx)
        cat, prompt = rows[idx]["category"], rows[idx]["prompt"]
        changes.append({
            "sample_id": sid,
            "action": "add_new",
            "reason": "v3.15 扩充至360；现实争议轴",
            "old_category": None,
            "new_category": cat,
            "old_prompt": None,
            "new_prompt": prompt,
            "old_len": 0,
            "new_len": len(prompt),
        })

    errs: list[str] = []
    for c in changes:
        errs.extend(validate_prompt(c["sample_id"], c["new_prompt"], c["new_category"]))
    for idx, row in enumerate(rows):
        why = is_unsafe_rewrite(row["prompt"])
        if why:
            errs.append(f"{sid_of(idx)} (full scan) 安全网: {why}")

    new_lens = [c["new_len"] for c in changes if c["new_len"]]
    med = statistics.median(new_lens)
    macro = Counter(to_macro(r["category"]) for r in rows)

    print(f"总量: 314 -> {len(rows)}")
    print(f"改动: gap重写 {len(gap_rewrites)} + 新增 {len(new_samples)}")
    print(f"新/改 prompt 长度: 中位 {med:.0f} | 范围 {min(new_lens)}-{max(new_lens)}")
    print(f"大类分布: {dict(sorted(macro.items()))}")

    if errs:
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n[dry-run] 校验通过，未写入。")
        return 0

    shutil.copy2(DATASET, BACKUP_RAW)
    if ENRICHED.is_file():
        shutil.copy2(ENRICHED, BACKUP_ENR)

    with DATASET.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"prompt": row["prompt"], "category": row["category"]}, ensure_ascii=False) + "\n")

    changelog = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "from_n": 314,
        "to_n": 360,
        "gap_rewrites": len(gap_rewrites),
        "added": len(new_samples),
        "macro_distribution": dict(macro),
        "changes": changes,
    }
    CHANGELOG.write_text(json.dumps(changelog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {DATASET}（备份 {BACKUP_RAW}）")

    if not args.enrich:
        print("下一步: python scripts/rewrite_dataset_v315.py --enrich")
        return 0

    load_dotenv_if_present()
    client = build_llm_client(args.enrich_model, mock=False, persona="draft", seed=20260904)

    old_enriched: dict[str, Sample] = {}
    if ENRICHED.is_file():
        for idx, line in enumerate(ENRICHED.read_text().splitlines()):
            if line.strip():
                old_enriched[sid_of(idx)] = Sample(**json.loads(line))

    changed_sids = {c["sample_id"] for c in changes}
    samples: list[Sample] = []
    for idx, row in enumerate(rows):
        sid = sid_of(idx)
        if sid in changed_sids or sid not in old_enriched:
            fields = enrich_one_with_retry(row["prompt"], row["category"], client)
            samples.append(Sample(
                sample_id=sid,
                prompt=row["prompt"],
                category=row["category"],
                category_macro=to_macro(row["category"]),
                target_claim=fields["target_claim"],
                value_axis=fields["value_axis"],
                risk_type=fields["risk_type"],
                expected_safe_behavior=fields["expected_safe_behavior"],
            ))
            print(f"  enriched {sid}")
        else:
            old = old_enriched[sid]
            samples.append(Sample(
                sample_id=sid,
                prompt=old.prompt,
                category=old.category,
                category_macro=old.category_macro,
                target_claim=old.target_claim,
                value_axis=old.value_axis,
                risk_type=old.risk_type,
                expected_safe_behavior=old.expected_safe_behavior,
            ))

    save_enriched_jsonl(samples, str(ENRICHED))
    md5 = hashlib.md5(ENRICHED.read_bytes()).hexdigest()
    print(f"\n已写入 {ENRICHED}（{len(samples)} 条）")
    print(f"新 MD5: {md5}")
    print(f"跑实验: TOTAL=360 ENRICHED_MD5={md5} bash scripts/deploy/run_v314_full.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
