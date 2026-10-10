#!/usr/bin/env python3
"""Remove specific sample_ids from results.jsonl so --resume will re-run them."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--run-dir", default="data/v3.15_flash_run")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.jsonl"
    remove_ids = set(args.ids)

    kept: list[dict] = []
    removed: list[str] = []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["sample_id"]
        if sid in remove_ids:
            removed.append(sid)
        else:
            kept.append(row)

    print(f"remove={len(removed)} keep={len(kept)}")
    print("removed:", " ".join(removed))
    if args.dry_run:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_dir / f"results.jsonl.bak_{ts}"
    shutil.copy2(results_path, backup)
    bandit = run_dir / "bandit_posterior.json"
    if bandit.exists():
        shutil.copy2(bandit, run_dir / f"bandit_posterior.json.bak_{ts}")

    with results_path.open("w", encoding="utf-8") as fp:
        for row in kept:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    (run_dir / "DRIVER_STATUS").write_text(
        f"RERUN_PREPARED {len(removed)} {datetime.now():%F %T}\n"
    )
    print(f"backup -> {backup}")


if __name__ == "__main__":
    main()
