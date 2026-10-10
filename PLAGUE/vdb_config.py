"""V-DriftBench v3.15 integration defaults for PLAGUE baseline comparison."""

from __future__ import annotations

import os

VDB_DATASET_PATH = os.environ.get(
    "VDB_DATASET_PATH", "/root/vdriftbench/data/dataset_100.enriched.jsonl"
)
VDB_DATASET_MD5 = "08d1bfa2296c6d907e78e0f18f243292"

VDB_LLM_SEED = int(os.environ.get("VDB_LLM_SEED", "20260904"))

VDB_DRAFT_MODEL = os.environ.get("VDB_DRAFT_MODEL", "draft-model")
VDB_DRAFT_BASE_URL = os.environ.get("VDB_DRAFT_BASE_URL", "http://127.0.0.1:8001/v1")
VDB_DRAFT_API_KEY = os.environ.get("VDB_DRAFT_API_KEY", "EMPTY")
VDB_DRAFT_PORT = int(os.environ.get("VDB_DRAFT_PORT", "8001"))

VDB_TARGET_MODEL = os.environ.get("VDB_TARGET_MODEL", "deepseek-v4-pro")
VDB_TARGET_BASE_URL = os.environ.get("VDB_TARGET_BASE_URL", "https://api.deepseek.com")
VDB_TARGET_MAX_TOKENS = int(os.environ.get("VDB_TARGET_MAX_TOKENS", "4096"))

VDB_JUDGE_MODEL = os.environ.get("VDB_JUDGE_MODEL", "judge-model")
VDB_JUDGE_BASE_URL = os.environ.get("VDB_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1")
VDB_JUDGE_API_KEY = os.environ.get("VDB_JUDGE_API_KEY", "EMPTY")
VDB_JUDGE_PORT = int(os.environ.get("VDB_JUDGE_PORT", "8002"))

VDB_OUTPUT_DIR = os.environ.get("VDB_OUTPUT_DIR", "/root/PLAGUE/outputs/v315")
VDB_OUTCOME_MODE = "行为二元"

VDB_BASELINE_VDRIFT = os.environ.get(
    "VDB_BASELINE_VDRIFT", "/root/vdriftbench/data/v3.15_run/results.jsonl"
)
VDB_BASELINE_COP = os.environ.get(
    "VDB_BASELINE_COP",
    "/root/cop/outputs/v315/deepseek-v4-pro_vdb_draft-model",
)
