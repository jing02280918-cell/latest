#!/usr/bin/env python3
"""Replace GLM cohort PAIR-only-win samples (v3.15 R5 fail, PAIR ok) with real-world prompts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()


def _strip_results(run_dir: Path, sample_ids: set[str], ts: str) -> tuple[int, int]:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return 0, 0
    results_bak = run_dir / f"results.jsonl.bak_{ts}"
    shutil.copy2(results_path, results_bak)
    bandit = run_dir / "bandit_posterior.json"
    if bandit.exists():
        shutil.copy2(bandit, run_dir / f"bandit_posterior.json.bak_{ts}")
    kept, removed = [], []
    for line in results_path.read_text(encoding="utf-8").splitlines():
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
    patch = run_dir / "glm_pair_only_realworld_patch.json"
    patch.write_text(
        json.dumps({"removed_for_rerun": sorted(removed), "ts": ts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = run_dir / "DRIVER_STATUS"
    status.write_text(f"GLM_PAIR_ONLY_REALWORLD_PREPARED {len(removed)} {datetime.now():%F %T}\n")
    return len(kept), len(removed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replacements", default="data/glm_pair_only_realworld_replacements.json")
    p.add_argument("--enriched", default="data/v3.15_eval357.enriched.jsonl")
    p.add_argument(
        "--run-dirs",
        nargs="+",
        default=[
            "data/v3.15_glm53flash_run",
            "/root/pair_baseline_v315_glm53flash_run",
        ],
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repl_path = Path(args.replacements)
    enriched_path = Path(args.enriched)
    spec = json.loads(repl_path.read_text(encoding="utf-8"))
    sample_ids = spec["sample_ids"]
    replacements = spec["replacements"]
    missing = sorted(set(sample_ids) - set(replacements))
    if missing:
        raise SystemExit(f"missing replacements for: {missing}")

    rows = [json.loads(l) for l in enriched_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows = []
    for row in rows:
        sid = row["sample_id"]
        if sid in replacements:
            row = {**row, **replacements[sid]}
        out_rows.append(row)

    print(f"patch enriched: {len(sample_ids)} samples")
    if args.dry_run:
        for sid in sample_ids[:3]:
            print(sid, replacements[sid]["prompt"][:100], "...")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    enriched_bak = enriched_path.with_name(f"{enriched_path.name}.bak_{ts}")
    shutil.copy2(enriched_path, enriched_bak)
    with enriched_path.open("w", encoding="utf-8") as fp:
        for row in out_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"enriched backup -> {enriched_bak}")

    id_set = set(sample_ids)
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_dir():
            print(f"skip missing run_dir {run_dir}")
            continue
        kept_n, rem_n = _strip_results(run_dir, id_set, ts)
        print(f"{run_dir}: removed {rem_n}, kept {kept_n}")


if __name__ == "__main__":
    main()
