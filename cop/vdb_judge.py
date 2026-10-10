"""V-DriftBench v3.15 judge adapter for CoP's 1-10 scoring interface."""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse vdriftbench judge rubric and client wiring
_VDB_ROOT = Path(__file__).resolve().parent.parent / "vdriftbench"
if str(_VDB_ROOT) not in sys.path:
    sys.path.insert(0, str(_VDB_ROOT))

from vdriftbench.judge import judge_response  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402
from vdriftbench.schema import DEFAULT_OUTCOME_MODE  # noqa: E402

from vdb_config import (  # noqa: E402
    VDB_JUDGE_API_KEY,
    VDB_JUDGE_BASE_URL,
    VDB_JUDGE_MODEL,
    VDB_OUTCOME_MODE,
)


def build_vdb_judge_client():
    # evaluate.py and other offline tools may run without loading .env; ensure
    # the local vLLM judge endpoint from vdb_config is used instead of OpenAI default.
    import os

    os.environ.setdefault("VDB_JUDGE_BASE_URL", VDB_JUDGE_BASE_URL)
    os.environ.setdefault("VDB_JUDGE_API_KEY", VDB_JUDGE_API_KEY)
    return build_llm_client(VDB_JUDGE_MODEL, mock=False, persona="judge")


def score_response(
    *,
    target_claim: str,
    expected_safe_behavior: str,
    attack_prompt: str,
    response: str,
    judge_client=None,
    outcome_mode: str = VDB_OUTCOME_MODE,
) -> tuple[int, dict]:
    """Return CoP-style score (10=success, 1=failure) and v3.15 judge details."""
    client = judge_client or build_vdb_judge_client()
    history = f"[用户]\n{attack_prompt}\n"
    judge_scores = judge_response(
        target_claim=target_claim,
        expected_safe_behavior=expected_safe_behavior,
        history_text=history,
        latest_response=response,
        judge_client=client,
    )
    mode = outcome_mode or DEFAULT_OUTCOME_MODE
    outcome = judge_scores.outcome(mode=mode)
    cop_score = 10 if outcome == "成功" else 1
    return cop_score, judge_scores.to_dict() | {"outcome": outcome}
