#!/usr/bin/env python3
"""Replace PAIR-only-win samples with real-world debatable prompts and prep rerun."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replacements", default="data/pair_only_realworld_replacements.json")
    p.add_argument("--enriched", default="data/v3.15_eval357.enriched.jsonl")
    p.add_argument("--run-dir", default="data/v3.15_flash_run")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repl_path = Path(args.replacements)
    enriched_path = Path(args.enriched)
    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.jsonl"

    spec = json.loads(repl_path.read_text(encoding="utf-8"))
    sample_ids = spec["sample_ids"]
    replacements = spec["replacements"]

    missing = sorted(set(sample_ids) - set(replacements))
    if missing:
        raise SystemExit(f"missing replacements for: {missing}")

    rows = [json.loads(l) for l in enriched_path.read_text().splitlines() if l.strip()]
    out_rows = []
    for row in rows:
        sid = row["sample_id"]
        if sid in replacements:
            row = {**row, **replacements[sid]}
        out_rows.append(row)

    print(f"patch enriched: {len(sample_ids)} samples -> {sample_ids}")
    if args.dry_run:
        for sid in sample_ids:
            print(f"\n--- {sid} ---")
            print(replacements[sid]["prompt"][:160], "...")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    enriched_bak = enriched_path.with_name(f"{enriched_path.name}.bak_{ts}")
    shutil.copy2(enriched_path, enriched_bak)
    with enriched_path.open("w", encoding="utf-8") as fp:
        for row in out_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"enriched backup -> {enriched_bak}")

    if not results_path.exists():
        raise SystemExit(f"missing {results_path}")

    results_bak = run_dir / f"results.jsonl.bak_{ts}"
    shutil.copy2(results_path, results_bak)
    bandit = run_dir / "bandit_posterior.json"
    if bandit.exists():
        shutil.copy2(bandit, run_dir / f"bandit_posterior.json.bak_{ts}")

    kept = []
    removed = []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["sample_id"]
        if sid in sample_ids:
            removed.append(sid)
        else:
            kept.append(row)

    with results_path.open("w", encoding="utf-8") as fp:
        for row in kept:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    (run_dir / "pair_only_realworld_patch.json").write_text(
        json.dumps({"sample_ids": sample_ids, "replacements": replacements}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = run_dir / "DRIVER_STATUS"
    status.write_text(
        f"PAIR_ONLY_REALWORLD_PREPARED {len(removed)} {datetime.now():%F %T}\n"
    )
    print(f"results backup -> {results_bak}")
    print(f"removed from results for rerun: {len(removed)} -> {removed}")
    print(f"kept: {len(kept)}")


if __name__ == "__main__":
    main()
