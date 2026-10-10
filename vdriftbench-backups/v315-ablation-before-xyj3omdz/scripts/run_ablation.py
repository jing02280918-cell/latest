"""Paired v3.15.1 ablations. See docs/ABLATION_v3.15.1.md.

--list lists the current arms; --dry-run validates and prints the plan without
building clients or creating outputs. Real runs always use a fresh full arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from vdriftbench.ablation import PRESETS, get_preset
from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.config import Config
from vdriftbench.embedding_client import build_embedding_client
from vdriftbench.enrich import load_enriched_jsonl
from vdriftbench.env_loader import load_dotenv_if_present
from vdriftbench.io_utils import sample_result_to_dict
from vdriftbench.llm_client import LLMClient, build_llm_client
from vdriftbench.pipeline import GENRE_PRINCIPLES, _round_outcome, genre_rounds, run_dataset
from vdriftbench.schema import OUTCOME_MODE_BINARY


class CountedClient(LLMClient):
    """Count logical role calls, not SDK/HTTP retries or billable token usage."""
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def chat(self, messages, *, temperature=0.7):
        self.calls += 1
        return self.inner.chat(messages, temperature=temperature)

    def chat_with_logprobs(self, messages, *, temperature=0.7, top_logprobs=8):
        self.calls += 1
        return self.inner.chat_with_logprobs(
            messages, temperature=temperature, top_logprobs=top_logprobs,
        )


def write_json(path, payload):
    """Atomic metadata update; incremental results use flush + fsync instead."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sample_metrics(result, config):
    content = genre_rounds(result.rounds)
    success = lambda r: _round_outcome(r, config) == "成功"
    scores = [r.scores.risk_score for r in result.rounds if r.scores is not None]
    return {
        "r5_first_success": bool(content and success(content[0])),
        "r5_with_retry_success": any(success(r) for r in content),
        "sample_success": any(success(r) for r in result.rounds),
        "best_risk_score": max(scores) if scores else None,
        "build_up_rounds": sum(r.principle_used not in GENRE_PRINCIPLES for r in result.rounds),
        "r5_retry_rounds": max(0, len(content) - 1),
        "hard_refusal_stopped": result.terminated_early,
    }


def summarize_arm(rows, expected_ids, calls, elapsed):
    ids = [r["sample_id"] for r in rows]
    complete = len(ids) == len(set(ids)) and set(ids) == set(expected_ids)
    metrics = [r["ablation_metrics"] for r in rows]
    n = len(rows)
    report = {
        "status": "complete" if complete else "incomplete",
        "n_expected": len(expected_ids), "n_completed": n,
        "missing_ids": sorted(set(expected_ids) - set(ids)),
        "logical_calls_by_role": calls, "elapsed_seconds": round(elapsed, 3),
        "resolved_by_distribution": dict(Counter(
            rd["resolved_by"] for row in rows for rd in row["rounds"] if rd["resolved_by"]
        )),
        "failure_score_rounds": sum(
            "[JUDGE_" in ((rd.get("scores") or {}).get("raw_rationale", ""))
            and "FAILED]" in ((rd.get("scores") or {}).get("raw_rationale", ""))
            for row in rows for rd in row["rounds"]
        ),
    }
    for metric in ("r5_first_success", "r5_with_retry_success", "sample_success"):
        count = sum(m[metric] for m in metrics)
        report[metric + "_count"] = count
        # An incomplete arm must never look like a completed high-ASR run.
        report[metric + "_pct"] = 100 * count / n if complete and n else None
    for key in ("best_risk_score", "build_up_rounds", "r5_retry_rounds"):
        values = [m[key] for m in metrics if m[key] is not None]
        report["mean_" + key] = statistics.mean(values) if complete and values else None
    report["hard_refusal_stopped_count"] = sum(m["hard_refusal_stopped"] for m in metrics)
    report["mean_target_logical_calls"] = calls.get("target", 0) / n if complete and n else None
    return report


def paired_comparison(full, arm, expected_ids):
    left = {r["sample_id"]: r["ablation_metrics"] for r in full}
    right = {r["sample_id"]: r["ablation_metrics"] for r in arm}
    expected = set(expected_ids)
    if (set(left) != expected or set(right) != expected
            or len(full) != len(left) or len(arm) != len(right)):
        return {"status": "incomplete", "n_common": len(set(left) & set(right))}
    report = {"status": "complete", "n_pairs": len(expected)}
    for metric in ("r5_first_success", "r5_with_retry_success", "sample_success"):
        full_only = sum(left[s][metric] and not right[s][metric] for s in expected)
        arm_only = sum(right[s][metric] and not left[s][metric] for s in expected)
        discordant = full_only + arm_only
        p = min(1.0, 2 * sum(math.comb(discordant, k)
                            for k in range(min(full_only, arm_only) + 1)) / (2 ** discordant))
        report[metric] = {
            "full_only": full_only, "ablation_only": arm_only,
            "delta_full_minus_ablation_pp": 100 * (full_only - arm_only) / len(expected),
            "mcnemar_exact_p_unadjusted": p,
        }
    return report


def build_plan(args):
    names = [v.strip() for v in args.presets.split(",") if v.strip()]
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if not names or len(set(names)) != len(names) or "full" not in names:
        raise ValueError("--presets must be unique and include full for paired comparisons")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("--seeds must contain unique nonnegative integers")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    presets = [get_preset(name) for name in names]
    base = Config.from_yaml(args.config) if args.config else Config()
    base = base.apply_overrides(
        target_model=args.target_model, judge_model=args.judge_model,
        draft_model=args.draft_model, embedding_model=args.embedding_model,
    )
    # Keep the advertised full-v3.15.1 mechanisms on; do not silently compare
    # a disabled component against another copy of itself through a YAML file.
    for field in ("r5_prompt_success_history_only", "r5_retry_enabled", "adaptive_rounds",
                  "early_termination_on_hard_refuse"):
        if not getattr(base, field):
            raise ValueError(f"full v3.15.1 requires {field}=True")
    if base.outcome_mode != OUTCOME_MODE_BINARY or base.r5_retry_rounds != 2:
        raise ValueError("full v3.15.1 requires 行为二元 scoring and r5_retry_rounds=2")
    if not (1 <= base.min_build_up_rounds < base.max_build_up_rounds <= 6):
        raise ValueError("require 1 <= min_build_up_rounds < max_build_up_rounds <= 6")
    samples = load_enriched_jsonl(args.data)
    if args.limit:
        samples = samples[:args.limit]
    ids = [s.sample_id for s in samples]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("dataset must be nonempty with unique sample_id values")
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "vdriftbench").glob("*.py")) + [Path(__file__).resolve()]
    manifest = {
        "version": "v3.15.1", "schema_version": 1,
        "mock": args.mock, "seeds": seeds, "presets": names,
        "sample_ids_in_order": ids, "n_samples": len(ids),
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "base_config": asdict(base), "device": args.device,
        "workers": 1, "rwi_review": False,
        "primary_metric": "r5_with_retry_success_pct",
        "arms": {p.name: {"description": p.description, "ablation": asdict(p),
                           "effective_config_without_seed": asdict(p.apply_to_config(base))}
                 for p in presets},
        "notes": [
            "Full is rerun under v3.15.1; inherited v3.15 results are not used as controls.",
            "Same ordered cohort and seeds, fresh scheduler for every arm/seed; workers=1.",
            "R5 is selected by principle token, not round_idx == 5.",
            "Logical role calls include recovery and failed calls, but not underlying HTTP retries/tokens.",
            "Remote endpoints may ignore seeds; paired p-values are exploratory and unadjusted.",
            "Legacy history restores history[:-1] for repairs, including earlier R5 attempts.",
            "RWI is disabled uniformly; recovery remains enabled. No existing result files are overwritten.",
        ],
    }
    return manifest, samples, base


def run_arm(samples, base, name, seed, out_dir, args):
    ablation = get_preset(name)
    config = ablation.apply_to_config(base).apply_overrides(seed=seed)
    arm_dir = out_dir / f"seed_{seed}" / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    write_json(arm_dir / "config.json", {"config": asdict(config), "ablation": asdict(ablation)})
    write_json(arm_dir / "status.json", {"status": "running"})
    roles = {"target": config.target_model, "judge": config.judge_model,
             "classifier": config.judge_model, "draft": config.draft_model}
    clients = {role: CountedClient(build_llm_client(model, mock=args.mock, persona=role, seed=seed))
               for role, model in roles.items()}
    embed = build_embedding_client(config.embedding_model, mock=args.mock, device=args.device)
    scheduler = ThompsonSamplingScheduler(
        seed=seed, kappa=config.bandit_kappa, disable_hierarchy=ablation.disable_hierarchy,
        shadow_pool_max_size=config.shadow_pool_max_size,
        shadow_promote_n_min=config.shadow_promote_n_min,
        shadow_promote_k_categories=config.shadow_promote_k_categories,
    )
    rows = []
    start = time.monotonic()
    with (arm_dir / "results.jsonl").open("x", encoding="utf-8") as handle:
        def save_sample(index, sample, result):
            row = sample_result_to_dict(result, extra={
                "ablation_version": "v3.15.1", "ablation_preset": name, "seed": seed,
                "ablation_metrics": sample_metrics(result, config),
            })
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            rows.append(row)
            scheduler.save(str(arm_dir / "bandit_posterior.json"))
            write_json(arm_dir / "status.json", {"status": "running", "n_completed": len(rows)})
            print(f"[{seed}/{name}] {len(rows)}/{len(samples)} {sample.sample_id}", flush=True)
        run_dataset(
            samples, clients["target"], clients["judge"], clients["classifier"], embed,
            scheduler, config, ablation, draft_client=clients["draft"],
            skip_rwi_review=True, on_sample_done=save_sample, workers=1,
        )
    summary = summarize_arm(rows, [s.sample_id for s in samples],
                            {role: c.calls for role, c in clients.items()}, time.monotonic() - start)
    write_json(arm_dir / "summary.json", summary)
    write_json(arm_dir / "status.json", summary)
    return rows, summary


def aggregate_reports(reports, names, seeds):
    output = {}
    for name in names:
        arms = [reports[str(seed)][name] for seed in seeds if name in reports.get(str(seed), {})]
        complete = len(arms) == len(seeds) and all(a["status"] == "complete" for a in arms)
        record = {"status": "complete" if complete else "incomplete", "n_seeds": len(arms)}
        for metric in ("r5_first_success_pct", "r5_with_retry_success_pct", "sample_success_pct",
                       "mean_best_risk_score", "mean_target_logical_calls"):
            values = [a[metric] for a in arms] if complete else []
            record[metric] = {
                "mean": statistics.mean(values) if values else None,
                "std_between_seeds": statistics.stdev(values) if len(values) > 1 else None,
            }
        output[name] = record
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="V-Drift v3.15.1 paired ablation experiments")
    parser.add_argument("--list", action="store_true", help="list active arms and exit")
    parser.add_argument("--dry-run", action="store_true", help="validate/print plan; no API calls or outputs")
    parser.add_argument("--data", default="data/v3.15_eval357.enriched.jsonl")
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--seeds", default="0,1,2", help="paired scheduler and LLM seeds")
    parser.add_argument("--config")
    parser.add_argument("--env-file")
    parser.add_argument("--mock", action="store_true")
    for role in ("target", "judge", "draft", "embedding"):
        parser.add_argument(f"--{role}-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default="data/ablation_v3151")
    args = parser.parse_args(argv)
    if args.list:
        for name, preset in PRESETS.items():
            print(f"{name}: {preset.description}")
        return 0
    try:
        manifest, samples, base = build_plan(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        parser.error(f"output already exists: {out_dir}; use a fresh --out-dir")
    if not args.mock:
        load_dotenv_if_present(args.env_file) if args.env_file else load_dotenv_if_present()
        if Path("/hy-tmp/hf_cache").is_dir():
            os.environ.setdefault("HF_HOME", "/hy-tmp/hf_cache")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "manifest.json", manifest)
    summary = {"version": "v3.15.1", "mock": args.mock, "runs": {}, "paired": {}}
    try:
        for index, seed in enumerate(manifest["seeds"]):
            names = manifest["presets"]
            # Rotate execution order across seeds to reduce a fixed arm/time association.
            offset = index % len(names)
            ordered = names[offset:] + names[:offset]
            rows_by_arm = {}
            summary["runs"][str(seed)] = {}
            for name in ordered:
                rows, report = run_arm(samples, base, name, seed, out_dir, args)
                rows_by_arm[name] = rows
                summary["runs"][str(seed)][name] = report
                write_json(out_dir / "summary.json", summary)
            summary["paired"][str(seed)] = {
                name: paired_comparison(rows_by_arm["full"], rows_by_arm[name], manifest["sample_ids_in_order"])
                for name in names if name != "full"
            }
        summary["aggregate"] = aggregate_reports(summary["runs"], manifest["presets"], manifest["seeds"])
        summary["status"] = ("complete" if all(r["status"] == "complete"
                             for r in summary["aggregate"].values()) else "incomplete")
    except BaseException:
        summary["status"] = "interrupted_or_failed"
        write_json(out_dir / "summary.json", summary)
        raise
    write_json(out_dir / "summary.json", summary)
    print(f"{summary['status']}: {out_dir / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
