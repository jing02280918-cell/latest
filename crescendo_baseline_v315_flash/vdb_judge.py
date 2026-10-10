"""V-DriftBench v3.15 judge adapter."""

from __future__ import annotations

import sys
from pathlib import Path

_VDB_ROOT = Path(__file__).resolve().parent.parent / "vdriftbench"
if str(_VDB_ROOT) not in sys.path:
    sys.path.insert(0, str(_VDB_ROOT))

from vdriftbench.judge import judge_response_stable  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402
from vdriftbench.schema import JudgeScores  # noqa: E402

from vdb_config import (  # noqa: E402
    VDB_JUDGE_API_KEY,
    VDB_JUDGE_BASE_URL,
    VDB_JUDGE_MODEL,
    VDB_OUTCOME_MODE,
)


def build_vdb_judge_client():
    import os

    os.environ["VDB_JUDGE_BASE_URL"] = VDB_JUDGE_BASE_URL
    os.environ["VDB_JUDGE_API_KEY"] = VDB_JUDGE_API_KEY
    return build_llm_client(VDB_JUDGE_MODEL, mock=False, persona="judge")


def score_turn(
    *,
    target_claim: str,
    expected_safe_behavior: str,
    history_text: str,
    attack_prompt: str,
    response: str,
    judge_client=None,
    config=None,
) -> tuple[JudgeScores, str]:
    from vdriftbench.config import Config

    client = judge_client or build_vdb_judge_client()
    cfg = config or Config()
    scores = judge_response_stable(
        target_claim,
        expected_safe_behavior,
        history_text,
        response,
        client,
        n_samples=cfg.judge_n_samples,
        per_dimension=cfg.judge_per_dimension,
    )
    outcome = scores.outcome(
        cfg.risk_success_threshold,
        cfg.risk_partial_threshold,
        cfg.enable_delivery_success_rule,
        cfg.outcome_mode or VDB_OUTCOME_MODE,
    )
    return scores, outcome
