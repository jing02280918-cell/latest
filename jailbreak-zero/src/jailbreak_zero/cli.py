from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from jailbreak_zero.artifacts import legacy_result_to_proposal_scores
from jailbreak_zero.config import apply_cli_overrides, config_hash, load_config
from jailbreak_zero.finetuning.preferences import rollout_scores_to_preference_pairs
from jailbreak_zero.metrics.summary import summarize_metrics
from jailbreak_zero.pipelines.example_based import run_example_based
from jailbreak_zero.pipelines.policy_based import run_policy_based
from jailbreak_zero.reproducibility import create_run_manifest
from jailbreak_zero.utils.jsonl import read_jsonl


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--allow-model-calls", action="store_true", default=False)
    parser.add_argument("--allow-api-calls", action="store_true", default=False)
    parser.add_argument("--allow-unsafe-artifact-writing", action="store_true", default=False)


def _load_with_overrides(args: argparse.Namespace):
    config = load_config(args.config)
    return apply_cli_overrides(config, seed=args.seed, output_dir=args.output_dir, limit=args.limit)


def cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    print(json.dumps({"valid": True, "config_hash": config_hash(config)}, indent=2))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    runner = run_example_based if config.evaluation_mode == "example_based" else run_policy_based
    plan = runner(config, dry_run=True, limit=args.limit)
    print(json.dumps(plan, indent=2))
    return 0


def cmd_run_example_based(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    start = time.time()
    result = run_example_based(
        config,
        dry_run=args.dry_run,
        limit=args.limit,
        allow_model_calls=args.allow_model_calls,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    manifest = create_run_manifest(
        run_id=config.experiment_name,
        config=config,
        seed=args.seed,
        command_line=sys.argv,
        output_paths={},
        start_time=start,
    )
    print(manifest.model_dump_json(indent=2))
    return 0


def cmd_run_policy_based(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    start = time.time()
    result = run_policy_based(
        config,
        dry_run=args.dry_run,
        limit=args.limit,
        allow_model_calls=args.allow_model_calls,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    manifest = create_run_manifest(
        run_id=config.experiment_name,
        config=config,
        seed=args.seed,
        command_line=sys.argv,
        output_paths={},
        start_time=start,
    )
    print(manifest.model_dump_json(indent=2))
    return 0


def cmd_score_proposals(args: argparse.Namespace) -> int:
    _load_with_overrides(args)
    path = Path(args.proposals)
    rows = list(read_jsonl(path))
    print(json.dumps({"proposal_records": len(rows), "source": str(path)}, indent=2))
    return 0


def _load_results(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return {"policy_results": list(read_jsonl(path))}
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "behavior_successes" in payload or "policy_results" in payload:
            return payload
        scores = legacy_result_to_proposal_scores(path)
        return {
            "behavior_successes": [
                {"successful": score.score >= score.max_score} for score in scores
            ]
        }
    raise ValueError(f"Unsupported results payload in {path}")


def cmd_compute_metrics(args: argparse.Namespace) -> int:
    _load_with_overrides(args)
    payload = _load_results(Path(args.results))
    summary = summarize_metrics(payload)
    print(summary.model_dump_json(indent=2))
    return 0


def cmd_build_dpo_dataset(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "pairs_per_policy": config.dpo.default_pairs_per_policy,
                    "negative_policy": config.dpo.negative_policy,
                    "max_negative_score": config.dpo.max_negative_score,
                },
                indent=2,
            )
        )
        return 0
    records = list(read_jsonl(args.rollout_scores))
    pairs = rollout_scores_to_preference_pairs(
        records,
        pairs_per_policy=config.dpo.default_pairs_per_policy,
        success_score=config.success_score,
        max_negative_score=config.dpo.max_negative_score,
        seed=args.seed or 0,
    )
    print(json.dumps({"pairs": len(pairs)}, indent=2))
    return 0


def cmd_summarize_results(args: argparse.Namespace) -> int:
    payload = _load_results(Path(args.results))
    print(summarize_metrics(payload).model_dump_json(indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jailbreak_zero")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    _add_common_flags(validate)
    validate.set_defaults(func=cmd_validate_config)

    dry_run = subparsers.add_parser("dry-run")
    _add_common_flags(dry_run)
    dry_run.set_defaults(func=cmd_dry_run)

    run_example = subparsers.add_parser("run-example-based")
    _add_common_flags(run_example)
    run_example.set_defaults(func=cmd_run_example_based)

    run_policy = subparsers.add_parser("run-policy-based")
    _add_common_flags(run_policy)
    run_policy.set_defaults(func=cmd_run_policy_based)

    score = subparsers.add_parser("score-proposals")
    _add_common_flags(score)
    score.add_argument("--proposals", required=True)
    score.set_defaults(func=cmd_score_proposals)

    metrics = subparsers.add_parser("compute-metrics")
    _add_common_flags(metrics)
    metrics.add_argument("--results", required=True)
    metrics.set_defaults(func=cmd_compute_metrics)

    dpo = subparsers.add_parser("build-dpo-dataset")
    _add_common_flags(dpo)
    dpo.add_argument("--rollout-scores", default="proposal_scores.jsonl")
    dpo.set_defaults(func=cmd_build_dpo_dataset)

    summarize = subparsers.add_parser("summarize-results")
    summarize.add_argument("results")
    summarize.set_defaults(func=cmd_summarize_results)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
