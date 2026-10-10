#!/usr/bin/env python3
"""Resume the interrupted GLM Core60 run in place without rerunning rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ARM_ORDER = [
    "full",
    "v315_no_scheduler",
    "v315_no_score_feedback",
    "v315_no_dimension_escalation",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_hashes(project: Path, manifest: dict) -> None:
    data_path = Path(manifest["data_path"])
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["data_sha256"]:
        raise RuntimeError("dataset hash mismatch")
    mismatches = []
    for relative, expected in manifest["source_sha256"].items():
        path = project / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"source hash mismatch: {', '.join(mismatches)}")


def resume_no_scheduler(runner, samples, base, out_dir: Path, run_args):
    name = "v315_no_scheduler"
    arm_dir = out_dir / "seed_0" / name
    results_path = arm_dir / "results.jsonl"
    existing = read_jsonl(results_path)
    expected_ids = [sample.sample_id for sample in samples]
    existing_ids = [row["sample_id"] for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("duplicate sample IDs in partial no_scheduler results")
    if not set(existing_ids).issubset(expected_ids):
        raise RuntimeError("partial no_scheduler results contain unexpected sample IDs")

    ablation = runner.get_preset(name)
    if not ablation.disable_bandit:
        raise RuntimeError("safe in-place continuation requires disable_bandit=True")
    config = ablation.apply_to_config(base).apply_overrides(seed=0)
    remaining = [sample for sample in samples if sample.sample_id not in set(existing_ids)]
    if not remaining:
        summary_path = arm_dir / "summary.json"
        if not summary_path.exists():
            raise RuntimeError("no samples remain but no completed arm summary exists")
        return existing, read_json(summary_path)

    roles = {
        "target": config.target_model,
        "judge": config.judge_model,
        "classifier": config.judge_model,
        "draft": config.draft_model,
    }
    clients = {
        role: runner.CountedClient(
            runner.build_llm_client(model, mock=False, persona=role, seed=0)
        )
        for role, model in roles.items()
    }
    embed = runner.build_embedding_client(
        config.embedding_model, mock=False, device=run_args.device
    )
    scheduler = runner.ThompsonSamplingScheduler(
        seed=0,
        kappa=config.bandit_kappa,
        disable_hierarchy=ablation.disable_hierarchy,
        shadow_pool_max_size=config.shadow_pool_max_size,
        shadow_promote_n_min=config.shadow_promote_n_min,
        shadow_promote_k_categories=config.shadow_promote_k_categories,
    )

    rows = list(existing)
    resume_start = time.monotonic()
    with results_path.open("a", encoding="utf-8") as handle:
        def save_sample(index, sample, result):
            row = runner.sample_result_to_dict(
                result,
                extra={
                    "ablation_version": "v3.15",
                    "ablation_preset": name,
                    "seed": 0,
                    "ablation_metrics": runner.sample_metrics(result, config),
                },
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            rows.append(row)
            scheduler.save(str(arm_dir / "bandit_posterior.json"))
            runner.write_json(
                arm_dir / "status.json",
                {
                    "status": "running",
                    "n_completed": len(rows),
                    "resumed_from": len(existing),
                },
            )
            print(
                f"[0/{name}] {len(rows)}/{len(samples)} {sample.sample_id} (resumed)",
                flush=True,
            )

        runner.run_dataset(
            remaining,
            clients["target"],
            clients["judge"],
            clients["classifier"],
            embed,
            scheduler,
            config,
            ablation,
            draft_client=clients["draft"],
            skip_rwi_review=True,
            on_sample_done=save_sample,
            workers=1,
        )

    resume_elapsed = time.monotonic() - resume_start
    # Target calls map one-to-one to recorded rounds. Other pre-resume role
    # counters were not checkpointed, so keep those totals explicitly unknown.
    target_calls = sum(len(row.get("rounds", [])) for row in rows)
    summary = runner.summarize_arm(
        rows,
        expected_ids,
        {
            "target": target_calls,
            "judge": None,
            "classifier": None,
            "draft": None,
        },
        resume_elapsed,
    )
    summary["elapsed_seconds"] = None
    summary["resume_elapsed_seconds"] = round(resume_elapsed, 3)
    summary["resumed_from_n_completed"] = len(existing)
    summary["resume_logical_calls_by_role"] = {
        role: client.calls for role, client in clients.items()
    }
    summary["accounting_note"] = (
        "Outcome metrics include all 60 rows. Only target logical calls can be "
        "reconstructed for the pre-resume rows; other role totals and total wall "
        "time were not checkpointed before interruption."
    )
    runner.write_json(arm_dir / "summary.json", summary)
    runner.write_json(arm_dir / "status.json", summary)
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project))
    import run_ablation as runner
    from vdriftbench.config import Config
    from vdriftbench.enrich import load_enriched_jsonl
    from vdriftbench.env_loader import load_dotenv_if_present

    out_dir = Path(args.out_dir).resolve()
    manifest = read_json(out_dir / "manifest.json")
    if manifest.get("version") != "v3.15" or manifest.get("seeds") != [0]:
        raise RuntimeError("expected a v3.15 seed-0 experiment")
    if manifest.get("presets") != ARM_ORDER:
        raise RuntimeError("unexpected arm order")
    validate_hashes(project, manifest)

    samples = load_enriched_jsonl(manifest["data_path"])
    expected_ids = manifest["sample_ids_in_order"]
    if [sample.sample_id for sample in samples] != expected_ids:
        raise RuntimeError("dataset order no longer matches manifest")

    full_dir = out_dir / "seed_0" / "full"
    full_rows = read_jsonl(full_dir / "results.jsonl")
    full_summary = read_json(full_dir / "summary.json")
    if (
        full_summary.get("status") != "complete"
        or [row["sample_id"] for row in full_rows] != expected_ids
    ):
        raise RuntimeError("full arm is not a complete ordered Core60 run")

    partial_path = out_dir / "seed_0" / "v315_no_scheduler" / "results.jsonl"
    partial_rows = read_jsonl(partial_path)
    validation = {
        "status": "validated",
        "out_dir": str(out_dir),
        "n_samples": len(samples),
        "full_n_completed": len(full_rows),
        "no_scheduler_n_completed": len(partial_rows),
        "no_scheduler_n_remaining": len(samples) - len(partial_rows),
        "not_started": ARM_ORDER[2:],
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    load_dotenv_if_present()
    if Path("/hy-tmp/hf_cache").is_dir():
        os.environ.setdefault("HF_HOME", "/hy-tmp/hf_cache")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    base = Config(**manifest["base_config"])
    run_args = SimpleNamespace(mock=False, device=manifest["device"])
    rows_by_arm = {"full": full_rows}
    summary = read_json(out_dir / "summary.json")
    summary.setdefault("runs", {}).setdefault("0", {})["full"] = full_summary
    summary["resume"] = validation | {
        "glm_reasoning_effort": os.environ.get("VDB_GLM_REASONING_EFFORT", "")
    }
    summary["status"] = "running"
    runner.write_json(out_dir / "summary.json", summary)

    try:
        scheduler_rows, scheduler_summary = resume_no_scheduler(
            runner, samples, base, out_dir, run_args
        )
        rows_by_arm["v315_no_scheduler"] = scheduler_rows
        summary["runs"]["0"]["v315_no_scheduler"] = scheduler_summary
        runner.write_json(out_dir / "summary.json", summary)

        for name in ARM_ORDER[2:]:
            rows, report = runner.run_arm(samples, base, name, 0, out_dir, run_args)
            rows_by_arm[name] = rows
            summary["runs"]["0"][name] = report
            runner.write_json(out_dir / "summary.json", summary)

        summary["paired"]["0"] = {
            name: runner.paired_comparison(full_rows, rows_by_arm[name], expected_ids)
            for name in ARM_ORDER[1:]
        }
        summary["aggregate"] = runner.aggregate_reports(
            summary["runs"], ARM_ORDER, [0]
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
