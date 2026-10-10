"""V-DriftBench v3.15 integration defaults for CoP baseline comparison."""

from __future__ import annotations

import os

# Dataset (must match v3.15 run byte-for-byte)
VDB_DATASET_PATH = os.environ.get(
    "VDB_DATASET_PATH", "/root/vdriftbench/data/dataset_100.enriched.jsonl"
)
VDB_DATASET_MD5 = "08d1bfa2296c6d907e78e0f18f243292"

# Shared LLM seed for draft/attack/policy (matches v3.15 V-Drift and PAIR runs)
VDB_LLM_SEED = int(os.environ.get("VDB_LLM_SEED", "20260904"))

# CoP v3.15: skip sending raw goal to target (saves 1 query; raw goal always filtered)
VDB_SKIP_BASELINE_PROBE = os.environ.get("VDB_SKIP_BASELINE_PROBE", "1") == "1"
VDB_HELPER_MAX_TOKENS = int(os.environ.get("VDB_HELPER_MAX_TOKENS", "1024"))
VDB_TARGET_MAX_TOKENS = int(os.environ.get("VDB_TARGET_MAX_TOKENS", "4096"))

# Max target-model queries per sample; exceeding without success counts as failure.
VDB_MAX_SAMPLE_QUERIES = int(os.environ.get("VDB_MAX_SAMPLE_QUERIES", "20"))

# Attack / helper model (local vLLM draft server)
VDB_DRAFT_MODEL = os.environ.get("VDB_DRAFT_MODEL", "draft-model")
VDB_DRAFT_BASE_URL = os.environ.get("VDB_DRAFT_BASE_URL", "http://127.0.0.1:8001/v1")
VDB_DRAFT_API_KEY = os.environ.get("VDB_DRAFT_API_KEY", "EMPTY")

# Target model (DeepSeek API)
VDB_TARGET_MODEL = os.environ.get("VDB_TARGET_MODEL", "deepseek-v4-pro")
VDB_TARGET_BASE_URL = os.environ.get("VDB_TARGET_BASE_URL", "https://api.deepseek.com")

# Judge model (local vLLM Mistral-7B)
VDB_JUDGE_MODEL = os.environ.get("VDB_JUDGE_MODEL", "judge-model")
VDB_JUDGE_BASE_URL = os.environ.get("VDB_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1")
VDB_JUDGE_API_KEY = os.environ.get("VDB_JUDGE_API_KEY", "EMPTY")

# Output directory for CoP results
VDB_OUTPUT_DIR = os.environ.get("VDB_OUTPUT_DIR", "/root/cop/outputs/v315")

# First-run failures pinned for v3.15 comparison (never rerun even with --rerun-failed)
VDB_FROZEN_SAMPLE_IDS = frozenset({"S0047", "S0054", "S0066", "S0080"})
VDB_FROZEN_ORIGINALS_DIR = os.environ.get(
    "VDB_FROZEN_ORIGINALS_DIR",
    os.path.join(VDB_OUTPUT_DIR, "frozen_originals"),
)

# v3.15 outcome mode (same as V-Drift and PAIR baselines)
VDB_OUTCOME_MODE = "行为二元"

# v3.15 baseline result paths for comparison
VDB_BASELINE_VDRIFT = os.environ.get(
    "VDB_BASELINE_VDRIFT", "/root/vdriftbench/data/v3.15_run/results.jsonl"
)
VDB_BASELINE_PAIR = os.environ.get(
    "VDB_BASELINE_PAIR", "/root/pair_baseline_v315_run/results.jsonl"
)
