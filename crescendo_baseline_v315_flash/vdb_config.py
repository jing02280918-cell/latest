"""V-DriftBench v3.15.1 alignment defaults for Crescendo baseline."""

from __future__ import annotations

import os


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


VDB_DATASET_PATH = _env(
    "VDB_DATASET_PATH", "/root/vdriftbench/data/v3.15_eval357.enriched.jsonl"
)
VDB_DATASET_MD5 = _env("VDB_DATASET_MD5", "e4748ddfa674710c97781bf974a8329f")

VDB_LLM_SEED = int(_env("VDB_LLM_SEED", "20260904"))

VDB_DRAFT_MODEL = _env("VDB_DRAFT_MODEL", "draft-model")
VDB_DRAFT_BASE_URL = _env("VDB_DRAFT_BASE_URL", "http://127.0.0.1:8001/v1")
VDB_DRAFT_API_KEY = _env("VDB_DRAFT_API_KEY", "EMPTY")

VDB_TARGET_MODEL = _env("VDB_TARGET_MODEL", "deepseek-v4-flash")
VDB_TARGET_BASE_URL = _env("VDB_TARGET_BASE_URL", "https://api.deepseek.com")
VDB_TARGET_API_KEY = _env("VDB_TARGET_API_KEY", "")
VDB_TARGET_MAX_TOKENS = int(_env("VDB_TARGET_MAX_TOKENS", "2048"))

VDB_JUDGE_MODEL = _env("VDB_JUDGE_MODEL", "judge-model")
VDB_JUDGE_BASE_URL = _env("VDB_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1")
VDB_JUDGE_API_KEY = _env("VDB_JUDGE_API_KEY", "EMPTY")

VDB_OUTPUT_DIR = os.environ.get(
    "VDB_OUTPUT_DIR", "/root/crescendo_baseline_v315_flash"
)
VDB_OUTCOME_MODE = "行为二元"

CRESCENDO_MAX_TURNS = int(os.environ.get("CRESCENDO_MAX_TURNS", "10"))
CRESCENDO_MAX_BACKTRACKS = int(os.environ.get("CRESCENDO_MAX_BACKTRACKS", "10"))
VDB_DRAFT_LAST_RESPONSE_CHARS = int(os.environ.get("VDB_DRAFT_LAST_RESPONSE_CHARS", "3000"))
VDB_DRAFT_MAX_COMPLETION_TOKENS = int(os.environ.get("VDB_DRAFT_MAX_COMPLETION_TOKENS", "384"))
