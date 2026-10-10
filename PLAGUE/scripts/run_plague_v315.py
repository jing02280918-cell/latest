#!/usr/bin/env python3
"""Run PLAGUE on V-DriftBench v3.15 dataset for fair comparison with CoP."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPTS_ROOT)

import pandas as pd  # noqa: E402

from plague import Plague, PlagueConfig  # noqa: E402
from vdb_config import (  # noqa: E402
    VDB_DATASET_MD5,
    VDB_DATASET_PATH,
    VDB_DRAFT_PORT,
    VDB_JUDGE_PORT,
    VDB_LLM_SEED,
    VDB_OUTPUT_DIR,
    VDB_TARGET_MAX_TOKENS,
)
from vdb_dataset import load_vdb_dataset  # noqa: E402
from vdb_judge import score_response  # noqa: E402


def _results_subdir(target_model: str) -> str:
    return f"{target_model}_vdb_draft-model"


def _result_csv_path(output_dir: str, target_model: str, sample_id: str) -> str:
    return os.path.join(
        output_dir, _results_subdir(target_model), f"{target_model}_jb_{sample_id}.csv"
    )


def _sample_done(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty and "judge_score" in df.columns
    except Exception:
        return False


def _save_csv(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([record]).to_csv(path, index=False)


def build_config(args: argparse.Namespace) -> PlagueConfig:
    return PlagueConfig(
        target_model=args.target_model,
        attacker_model=args.attacker_model,
        evaluator_model=args.evaluator_model,
        summariser_model=args.summariser_model,
        embedding_model=args.embedding_model,
        project=args.project,
        max_rounds=args.max_rounds,
        use_strategy_library=False,
        dataset="vdb315",
        verbose=args.verbose,
        max_workers=1,
        max_tries=1,
        primer_steps=args.primer_steps,
        finisher_type=args.finisher_type,
        use_iterative_prompting=True,
        use_planner=True,
        attacker_port=args.attacker_port,
        evaluator_port=args.evaluator_port,
        summariser_port=args.summariser_port,
        embedder_port=args.embedder_port,
        target_max_tokens=args.target_max_tokens,
        attacker_max_tokens=args.attacker_max_tokens,
        summariser_max_tokens=args.summariser_max_tokens,
        llm_seed=args.llm_seed,
        use_vdb_judge=True,
    )


def run_sample(plague: Plague, sample, csv_path: str) -> dict:
    meta = {
        "sample_id": sample.sample_id,
        "target_claim": sample.target_claim,
        "expected_safe_behavior": sample.expected_safe_behavior,
        "category": sample.category,
    }
    plague.config.sample_meta = meta
    if hasattr(plague.scorer, "sample_meta"):
        plague.scorer.sample_meta = meta

    result = plague.optimize_single_goal((sample.sample_id, sample.prompt), [], 0)
    attack_prompt = result.get("best_question") or sample.prompt
    response = result.get("best_response") or ""
    judge_score, judge_detail = score_response(
        target_claim=sample.target_claim,
        expected_safe_behavior=sample.expected_safe_behavior,
        attack_prompt=attack_prompt,
        response=response,
    )
    record = {
        "sample_id": sample.sample_id,
        "best_msg": attack_prompt,
        "jailbreak_output": response,
        "judge_score": int(judge_score),
        "bench_score": 1,
        "total_query_time": int(result.get("rounds") or 0),
        "action_l": json.dumps(
            result.get("best_sequence_of_questions") or [], ensure_ascii=False
        ),
        "vdb_outcome": judge_detail.get("outcome"),
    }
    _save_csv(csv_path, record)
    return record


def _ensure_deps() -> None:
    from ensure_v315_deps import main as ensure_main

    if ensure_main() != 0:
        raise SystemExit("v3.15 dependency check failed")


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="deepseek-v4-pro")
    parser.add_argument("--attacker-model", default="draft-model")
    parser.add_argument("--evaluator-model", default="judge-model")
    parser.add_argument("--summariser-model", default="draft-model")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--dataset-path", default=VDB_DATASET_PATH)
    parser.add_argument("--output-dir", default=VDB_OUTPUT_DIR)
    parser.add_argument("--project", default="plague_v315")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--primer-steps", type=int, default=1)
    parser.add_argument("--finisher-type", default="goat", choices=["normal", "goat"])
    parser.add_argument("--attacker-port", type=int, default=VDB_DRAFT_PORT)
    parser.add_argument("--evaluator-port", type=int, default=VDB_JUDGE_PORT)
    parser.add_argument("--summariser-port", type=int, default=VDB_DRAFT_PORT)
    parser.add_argument("--embedder-port", type=int, default=7005)
    parser.add_argument("--target-max-tokens", type=int, default=VDB_TARGET_MAX_TOKENS)
    parser.add_argument("--attacker-max-tokens", type=int, default=2048)
    parser.add_argument("--summariser-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-seed", type=int, default=VDB_LLM_SEED)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    samples = load_vdb_dataset(args.dataset_path, start=args.start, limit=args.limit)
    config = build_config(args)
    plague = Plague(config)

    trace_dir = os.path.join(args.output_dir, "plague_traces")
    os.makedirs(trace_dir, exist_ok=True)
    plague.run_dir = trace_dir
    plague.project_path = trace_dir
    config.save(os.path.join(trace_dir, "config.json"))

    results_dir = os.path.join(args.output_dir, _results_subdir(args.target_model))
    os.makedirs(results_dir, exist_ok=True)

    done = 0
    skipped = 0
    for sample in samples:
        csv_path = _result_csv_path(args.output_dir, args.target_model, sample.sample_id)
        if args.resume and _sample_done(csv_path):
            skipped += 1
            print(f"skip {sample.sample_id} — already in {csv_path}")
            continue
        print(f"=== PLAGUE sample {sample.sample_id} ===")
        try:
            record = run_sample(plague, sample, csv_path)
            done += 1
            print(
                f"  outcome={record.get('vdb_outcome')} "
                f"judge_score={record['judge_score']} queries={record['total_query_time']}"
            )
        except Exception as exc:
            print(f"ERROR on {sample.sample_id}: {exc}")
            if args.stop_on_error:
                raise

    print(f"finished batch: new={done} skipped={skipped} total_samples={len(samples)}")


if __name__ == "__main__":
    main()
