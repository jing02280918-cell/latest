#!/usr/bin/env python3
"""Remove R5-failed rows from results.jsonl so --resume will re-run them."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.config import Config
from vdriftbench.pipeline import genre_rounds, _round_outcome
from vdriftbench.schema import JudgeScores, RoundRecord


def parse_rounds(d: dict) -> list[RoundRecord]:
    rounds = []
    for r in d.get("rounds", []):
        sc = r.get("scores")
        scores = (
            JudgeScores(
                VDS=sc["VDS"],
                EFS=sc["EFS"],
                NJS=sc["NJS"],
                SCS=sc["SCS"],
                IFR=sc["IFR"],
                raw_rationale=sc.get("raw_rationale", ""),
                non_delivery=sc.get("non_delivery", False),
            )
            if sc
            else None
        )
        rounds.append(
            RoundRecord(
                round_idx=r["round_idx"],
                state_resolved=r.get("state_resolved"),
                resolved_by=r.get("resolved_by"),
                principle_used=r["principle_used"],
                bandit_context=None,
                bandit_posterior_snapshot=None,
                prompt=r.get("prompt", ""),
                response=r.get("response", ""),
                scores=scores,
                embed_drift_norm=r.get("embed_drift_norm"),
            )
        )
    return rounds


def r5_retry_ok(rounds: list[RoundRecord], config: Config) -> bool:
    return any(_round_outcome(r, config) == "成功" for r in genre_rounds(rounds))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="data/v3.15_flash_run")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"missing {results_path}")

    config = Config()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_dir / f"results.jsonl.bak_{ts}"
    if not args.dry_run:
        shutil.copy2(results_path, backup)
        bandit = run_dir / "bandit_posterior.json"
        if bandit.exists():
            shutil.copy2(bandit, run_dir / f"bandit_posterior.json.bak_{ts}")

    kept: list[dict] = []
    remove_ids: list[str] = []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["sample_id"]
        if r5_retry_ok(parse_rounds(row), config):
            kept.append(row)
        else:
            remove_ids.append(sid)

    print(f"total={len(kept) + len(remove_ids)} keep={len(kept)} rerun={len(remove_ids)}")
    if remove_ids:
        print("rerun sample_ids:", " ".join(remove_ids))

    if args.dry_run:
        return

    with results_path.open("w", encoding="utf-8") as fp:
        for row in kept:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    status = run_dir / "DRIVER_STATUS"
    status.write_text(f"RERUN_PREPARED {len(remove_ids)} failures {datetime.now():%F %T}\n")
    print(f"backup -> {backup}")
    print(f"wrote {len(kept)} rows -> {results_path}")


if __name__ == "__main__":
    main()
