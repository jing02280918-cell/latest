#!/usr/bin/env python3
"""Resume the interrupted GLM v3.15 Core60 ablation without rerunning full."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


REMAINING_ARMS = [
    "v315_no_scheduler",
    "v315_no_score_feedback",
    "v315_no_dimension_escalation",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_source_hashes(project: Path, manifest: dict) -> None:
    mismatches = []
    for relative, expected in manifest["source_sha256"].items():
        path = project / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"source hash mismatch: {', '.join(mismatches)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project))

    import run_ablation as runner
    from vdriftbench.config import Config
    from vdriftbench.env_loader import load_dotenv_if_present
    from vdriftbench.enrich import load_enriched_jsonl

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest = read_json(source_dir / "manifest.json")
    if manifest.get("version") != "v3.15" or manifest.get("seeds") != [0]:
        raise RuntimeError("expected the v3.15 seed-0 source experiment")
    if manifest.get("presets") != ["full", *REMAINING_ARMS]:
        raise RuntimeError("unexpected source arm list")

    data_path = Path(manifest["data_path"])
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if data_hash != manifest["data_sha256"]:
        raise RuntimeError("dataset hash mismatch")
    validate_source_hashes(project, manifest)

    expected_ids = manifest["sample_ids_in_order"]
    samples = load_enriched_jsonl(data_path)
    if [sample.sample_id for sample in samples] != expected_ids:
        raise RuntimeError("dataset order no longer matches the source manifest")

    full_dir = source_dir / "seed_0" / "full"
    full_summary = read_json(full_dir / "summary.json")
    full_rows = read_jsonl(full_dir / "results.jsonl")
    full_ids = [row["sample_id"] for row in full_rows]
    if full_summary.get("status") != "complete" or full_ids != expected_ids:
        raise RuntimeError("the source full arm is not a complete ordered Core60 run")

    validation = {
        "status": "validated",
        "source_dir": str(source_dir),
        "data_sha256": data_hash,
        "n_samples": len(samples),
        "full_n_completed": full_summary["n_completed"],
        "remaining_arms": REMAINING_ARMS,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0
    if out_dir.exists():
        raise RuntimeError(f"output already exists: {out_dir}")

    load_dotenv_if_present()
    if Path("/hy-tmp/hf_cache").is_dir():
        os.environ.setdefault("HF_HOME", "/hy-tmp/hf_cache")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    out_dir.mkdir(parents=True, exist_ok=False)
    continuation_manifest = dict(manifest)
    continuation_manifest["continuation_of"] = str(source_dir)
    continuation_manifest["presets"] = REMAINING_ARMS
    continuation_manifest["source_full_dir"] = str(full_dir)
    continuation_manifest["resume_validation"] = validation
    continuation_manifest["glm_reasoning_effort"] = os.environ.get(
        "VDB_GLM_REASONING_EFFORT", ""
    )
    runner.write_json(out_dir / "manifest.json", continuation_manifest)

    base = Config(**manifest["base_config"])
    run_args = SimpleNamespace(mock=False, device=manifest["device"])
    rows_by_arm = {"full": full_rows}
    summary = {
        "version": "v3.15",
        "mock": False,
        "continuation_of": str(source_dir),
        "source_full_dir": str(full_dir),
        "runs": {"0": {"full": full_summary}},
        "paired": {},
        "status": "running",
    }
    runner.write_json(out_dir / "summary.json", summary)

    try:
        for name in REMAINING_ARMS:
            rows, report = runner.run_arm(
                samples, base, name, 0, out_dir, run_args
            )
            rows_by_arm[name] = rows
            summary["runs"]["0"][name] = report
            runner.write_json(out_dir / "summary.json", summary)

        summary["paired"]["0"] = {
            name: runner.paired_comparison(
                full_rows, rows_by_arm[name], expected_ids
            )
            for name in REMAINING_ARMS
        }
        all_names = ["full", *REMAINING_ARMS]
        summary["aggregate"] = runner.aggregate_reports(
            summary["runs"], all_names, [0]
        )
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "interrupted_or_failed"
        runner.write_json(out_dir / "summary.json", summary)
        raise

    runner.write_json(out_dir / "summary.json", summary)
    print(f"complete: {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
