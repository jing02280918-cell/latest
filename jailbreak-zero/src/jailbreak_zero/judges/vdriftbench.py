"""V-DriftBench v3.14 judge for jailbreak-zero."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from jailbreak_zero.schemas import Behavior, JudgeConfig, JudgeDecision, ResponseSample

_VDB_ROOT = Path(__file__).resolve().parents[4] / "vdriftbench"
if str(_VDB_ROOT) not in sys.path:
    sys.path.insert(0, str(_VDB_ROOT))

from vdriftbench.judge import judge_response  # noqa: E402
from vdriftbench.llm_client import build_llm_client  # noqa: E402


class VdriftbenchJudge:
    """Maps v3.14 five-dimensional rubric to jailbreak-zero JudgeDecision labels."""

    def __init__(
        self,
        config: JudgeConfig,
        behaviors: Sequence[Behavior],
    ) -> None:
        self.config = config
        self.model_id = config.model_id
        self.outcome_mode = config.outcome_mode
        self._behavior_map = {b.behavior_id: b for b in behaviors}
        self._judge_client = build_llm_client(config.model_id, mock=False, persona="judge")

    def judge(self, responses: Sequence[ResponseSample]) -> Sequence[JudgeDecision]:
        decisions: list[JudgeDecision] = []
        for response in responses:
            behavior_id = response.behavior_id or response.proposal_id
            behavior = self._behavior_map.get(behavior_id or "")
            target_claim = behavior.target_claim if behavior else ""
            expected_safe = behavior.expected_safe_behavior if behavior else ""
            attack_prompt = response.attack_prompt or (behavior.text if behavior else "")
            history = f"[用户]\n{attack_prompt}\n"
            judge_scores = judge_response(
                target_claim=target_claim or "",
                expected_safe_behavior=expected_safe or "",
                history_text=history,
                latest_response=response.text,
                judge_client=self._judge_client,
            )
            outcome = judge_scores.outcome(mode=self.outcome_mode)
            label = "unsafe" if outcome == "成功" else "safe"
            decisions.append(
                JudgeDecision(
                    judge_result_id=f"vdb-{response.response_id}",
                    response_id=response.response_id,
                    label=label,
                    score=judge_scores.risk_score,
                    raw_output=judge_scores.raw_rationale,
                    judge_model=self.model_id,
                    judge_template_version=self.config.template_version,
                    policy_or_behavior_id=behavior_id or "unknown",
                )
            )
        return decisions
