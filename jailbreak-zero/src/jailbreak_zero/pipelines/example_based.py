from __future__ import annotations

from typing import Any

from jailbreak_zero.data.loaders import load_behaviors
from jailbreak_zero.judges.factory import build_judge
from jailbreak_zero.models.factory import build_model_client
from jailbreak_zero.prompts.templates import behavior_template
from jailbreak_zero.proposals.generate import extract_prompt
from jailbreak_zero.proposals.score import score_from_decisions
from jailbreak_zero.safety import ensure_model_calls_allowed
from jailbreak_zero.schemas import ExperimentConfig, GenerationRequest, Proposal


def plan_example_based(config: ExperimentConfig, *, limit: int | None = None) -> dict[str, Any]:
    proposal_count = (
        min(config.proposal_count, limit) if limit is not None else config.proposal_count
    )
    return {
        "mode": "example_based",
        "benchmark": config.benchmark,
        "dataset_path": config.dataset.path,
        "proposal_count_per_behavior": proposal_count,
        "responses_per_proposal": config.responses_per_proposal,
        "success_score": config.success_score,
        "selected_top_k": config.selected_top_k,
        "attack_model": config.attack_model.model_id,
        "target_model": config.target_model.model_id,
        "judge": config.judge.model_id,
        "judge_type": config.judge.judge_type,
        "artifacts": [
            "config.resolved.yaml",
            "manifest.json",
            "proposals.jsonl",
            "responses.jsonl",
            "judge_results.jsonl",
            "proposal_scores.jsonl",
            "selected_proposals.jsonl",
            "metrics.json",
        ],
    }


def run_example_based(
    config: ExperimentConfig,
    *,
    dry_run: bool = True,
    limit: int | None = None,
    allow_model_calls: bool = False,
) -> dict[str, Any]:
    plan = plan_example_based(config, limit=limit)
    if dry_run:
        return plan
    ensure_model_calls_allowed(allow_model_calls)

    behaviors = load_behaviors(
        config.benchmark,
        limit=limit,
        dataset_path=config.dataset.path,
        start=config.dataset.start,
    )
    attack_client = build_model_client(config.attack_model, config.generation)
    target_client = build_model_client(config.target_model, config.generation)
    judge = build_judge(config.judge, behaviors)

    proposals: list[Proposal] = []
    responses = []
    all_decisions = []

    for behavior in behaviors:
        attack_messages = behavior_template(behavior.text, language=config.language)
        attack_req = GenerationRequest(
            request_id=behavior.behavior_id,
            messages=attack_messages,
            meta_data={"behavior_id": behavior.behavior_id, "attack_dtype": "behavior"},
        )
        attack_out = attack_client.generate([attack_req])[0]
        adversarial = extract_prompt(attack_out.text) or behavior.text

        target_req = GenerationRequest(
            request_id=f"{behavior.behavior_id}-target",
            messages=adversarial,
            meta_data={
                "behavior_id": behavior.behavior_id,
                "attack_prompt": adversarial,
            },
        )
        target_out = target_client.generate([target_req])[0]
        target_out.behavior_id = behavior.behavior_id
        target_out.attack_prompt = adversarial

        proposal = Proposal(
            proposal_id=behavior.behavior_id,
            text=adversarial,
            behavior_id=behavior.behavior_id,
            raw_output=attack_out.text,
        )
        proposals.append(proposal)
        responses.append(target_out)
        all_decisions.extend(judge.judge([target_out]))

    proposal_scores = []
    for proposal, target_out in zip(proposals, responses):
        decisions = [d for d in all_decisions if d.response_id == target_out.response_id]
        proposal_scores.append(
            score_from_decisions(
                proposal.proposal_id,
                decisions,
                max_score=config.responses_per_proposal,
                behavior_id=proposal.behavior_id,
                language=config.language,
            )
        )
    successful = sum(1 for d in all_decisions if d.label == "unsafe")
    return {
        **plan,
        "total_behaviors": len(behaviors),
        "successful_behaviors": successful,
        "asr": successful / len(behaviors) if behaviors else 0.0,
        "proposal_count": len(proposals),
        "proposal_scores": [ps.model_dump() for ps in proposal_scores],
    }
