#!/usr/bin/env python3
"""Build an ablation-style full/ arm from a main.py results.jsonl for --paired-full-dir.

Used when the eval357 v3.15 GLM full run already exists and only the ablation
arm (e.g. v315_no_target_history) needs to be executed (357 target calls).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from dataclasses import asdict

import run_ablation as runner
from vdriftbench.ablation import get_preset
from vdriftbench.config import Config
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.io_utils import load_results_jsonl, sample_result_from_dict, sample_result_to_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v3.15_eval357.enriched.jsonl")
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--out-dir", required=True, help="seed_0/full directory for pairing")
    parser.add_argument("--provenance", default="", help="note stored in manifest sidecar")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    source = Path(args.source_results).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite: {out_dir}")
    out_dir.mkdir(parents=True)

    samples = load_enriched_jsonl(data_path)
    expected_ids = [s.sample_id for s in samples]
    raw = load_results_jsonl(str(source))
    by_id = {row["sample_id"]: row for row in raw}
    if set(by_id) != set(expected_ids):
        missing = set(expected_ids) - set(by_id)
        extra = set(by_id) - set(expected_ids)
        raise SystemExit(f"id mismatch missing={len(missing)} extra={len(extra)}")

    base = get_preset("full").apply_to_config(Config())
    config = base.apply_overrides(seed=0, target_model="ZHIPU/GLM-5.3-Flash")

    rows = []
    start = time.monotonic()
    for sid in expected_ids:
        result = sample_result_from_dict(by_id[sid])
        rows.append(sample_result_to_dict(result, extra={
            "ablation_version": "v3.15",
            "ablation_preset": "full",
            "seed": 0,
            "ablation_metrics": runner.sample_metrics(result, config),
            "full_proxy_provenance": str(source),
        }))

    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = runner.summarize_arm(rows, expected_ids, {"target": 0}, time.monotonic() - start)
    runner.write_json(out_dir / "summary.json", report)
    runner.write_json(out_dir / "config.json", {
        "config": asdict(config),
        "ablation": asdict(get_preset("full")),
        "proxy": True,
        "source_results": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "provenance_note": args.provenance or "Imported from main.py run; not same-batch as ablation arm.",
    })
    runner.write_json(out_dir / "status.json", report)
    print(json.dumps({"status": report["status"], "n": len(rows), "out": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
