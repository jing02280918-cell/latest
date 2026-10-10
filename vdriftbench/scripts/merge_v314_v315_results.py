#!/usr/bin/env python3
"""Merge v3.14 full-run results with v3.15 delta (71 rewritten/new prompts).

Keeps v3.14 rows for unchanged sample_ids; replaces 25 rewrites + adds 46 new
from the delta run. Output is 360 rows sorted by sample_id.

Usage:
    python scripts/merge_v314_v315_results.py \\
        --v314 data/v3.14_run/results.jsonl \\
        --delta data/v3.15_delta_run/results.jsonl \\
        --out data/v3.15_run/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_CHANGELOG = Path("data/dataset_optimization_changelog_v315.json")


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v314", default="data/v3.14_run/results.jsonl")
    p.add_argument("--delta", default="data/v3.15_delta_run/results.jsonl")
    p.add_argument("--out", default="data/v3.15_run/results.jsonl")
    p.add_argument("--changelog", default=str(DEFAULT_CHANGELOG))
    p.add_argument("--enriched", default="data/dataset_100.enriched.jsonl",
                   help="refresh category/target_claim on merged rows from current enriched dataset")
    args = p.parse_args()

    changelog = json.loads(Path(args.changelog).read_text(encoding="utf-8"))
    rewrite_ids = {c["sample_id"] for c in changelog["changes"] if c["action"] == "rewrite_gap25"}
    delta_ids = {c["sample_id"] for c in changelog["changes"]}

    v314 = load_rows(Path(args.v314))
    delta = load_rows(Path(args.delta))

    missing_delta = sorted(delta_ids - set(delta))
    if missing_delta:
        raise SystemExit(f"delta run missing {len(missing_delta)} samples: {missing_delta[:10]}...")

    enriched = {}
    with open(args.enriched, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                enriched[d["sample_id"]] = d

    merged: dict[str, dict] = {}
    kept = replaced = added = 0
    for sid, row in v314.items():
        if sid in rewrite_ids:
            continue
        merged[sid] = row
        kept += 1

    for sid in sorted(delta_ids):
        if sid not in delta:
            continue
        merged[sid] = delta[sid]
        if sid in rewrite_ids:
            replaced += 1
        else:
            added += 1
        # sync metadata from v3.15 enriched dataset
        meta = enriched.get(sid)
        if meta:
            merged[sid]["category"] = meta["category"]
            merged[sid]["category_macro"] = meta["category_macro"]
            merged[sid]["target_claim"] = meta["target_claim"]

    ordered = [merged[sid] for sid in sorted(merged)]
    if len(ordered) != 360:
        raise SystemExit(f"expected 360 merged rows, got {len(ordered)} (kept={kept} replaced={replaced} added={added})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        shutil.copy2(out, str(out) + ".bak")
    with out.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"merged -> {out}")
    print(f"  kept from v3.14:     {kept}")
    print(f"  replaced (rewrite):  {replaced}")
    print(f"  added (new):         {added}")
    print(f"  total:               {len(ordered)}")


if __name__ == "__main__":
    main()
