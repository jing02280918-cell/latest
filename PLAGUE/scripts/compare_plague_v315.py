#!/usr/bin/env python3
"""Compare PLAGUE v3.15 results against CoP and V-Drift baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "cop"))

from evaluate import evaluate_results  # noqa: E402
from vdb_config import (  # noqa: E402
    VDB_BASELINE_COP,
    VDB_BASELINE_VDRIFT,
    VDB_DATASET_PATH,
    VDB_OUTPUT_DIR,
)


def _load_jsonl(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def _vdrift_success(row: dict) -> bool:
    sys.path.insert(0, str(REPO_ROOT.parent / "vdriftbench"))
    from vdriftbench.config import Config
    from vdriftbench.schema import JudgeScores

    config = Config()
    for round_row in row.get("rounds", []):
        scores = round_row.get("scores") or {}
        js = JudgeScores(
            VDS=scores.get("VDS", 0),
            EFS=scores.get("EFS", 0),
            NJS=scores.get("NJS", 0),
            SCS=scores.get("SCS", 0),
            IFR=scores.get("IFR", 0),
            raw_rationale=scores.get("raw_rationale", ""),
            non_delivery=scores.get("non_delivery", False),
        )
        if js.outcome(mode=config.outcome_mode) == "成功":
            return True
    return False


def compare(
    *,
    plague_results_dir: str,
    cop_results_dir: str,
    dataset_path: str,
    vdrift_path: str,
) -> dict:
    plague = evaluate_results(plague_results_dir, dataset_path)
    cop = evaluate_results(cop_results_dir, dataset_path)
    plague_by_id = {d["sample_id"]: d for d in plague["details"]}
    cop_by_id = {d["sample_id"]: d for d in cop["details"]}
    vdrift_rows = _load_jsonl(vdrift_path)

    plague_ok = {sid for sid, d in plague_by_id.items() if d["cop_score"] == 10}
    cop_ok = {sid for sid, d in cop_by_id.items() if d["cop_score"] == 10}
    vdrift_ok = {sid for sid, row in vdrift_rows.items() if _vdrift_success(row)}
    overlap = sorted(set(plague_by_id) & set(cop_by_id) & set(vdrift_rows))

    both_plague_cop = plague_ok & cop_ok
    plague_only = sorted(plague_ok - cop_ok)
    cop_only = sorted(cop_ok - plague_ok)
    plague_miss_vdrift = sorted(vdrift_ok - plague_ok)
    plague_beat_vdrift = sorted(plague_ok - vdrift_ok)

    return {
        "plague": {
            "total": plague["total"],
            "successes": plague["successes"],
            "asr": plague["asr"],
            "results_dir": plague_results_dir,
        },
        "cop": {
            "total": cop["total"],
            "successes": cop["successes"],
            "asr": cop["asr"],
            "results_dir": cop_results_dir,
        },
        "vdrift_baseline": {
            "total": len(vdrift_rows),
            "successes": len(vdrift_ok),
            "asr": len(vdrift_ok) / len(vdrift_rows) if vdrift_rows else 0.0,
            "path": vdrift_path,
        },
        "overlap_n": len(overlap),
        "plague_vs_cop": {
            "both_success_n": len(both_plague_cop),
            "plague_only_success_n": len(plague_only),
            "plague_only_success_sample": plague_only[:20],
            "cop_only_success_n": len(cop_only),
            "cop_only_success_sample": cop_only[:20],
        },
        "plague_vs_vdrift": {
            "vdrift_success_plague_fail_n": len(plague_miss_vdrift),
            "vdrift_success_plague_fail_sample": plague_miss_vdrift[:20],
            "plague_success_vdrift_fail_n": len(plague_beat_vdrift),
            "plague_success_vdrift_fail_sample": plague_beat_vdrift[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(Path(VDB_OUTPUT_DIR) / "deepseek-v4-pro_vdb_draft-model"),
    )
    parser.add_argument(
        "--cop-results-dir",
        default=VDB_BASELINE_COP,
    )
    parser.add_argument("--dataset-path", default=VDB_DATASET_PATH)
    parser.add_argument("--vdrift-baseline", default=VDB_BASELINE_VDRIFT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary = compare(
        plague_results_dir=args.results_dir,
        cop_results_dir=args.cop_results_dir,
        dataset_path=args.dataset_path,
        vdrift_path=args.vdrift_baseline,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
