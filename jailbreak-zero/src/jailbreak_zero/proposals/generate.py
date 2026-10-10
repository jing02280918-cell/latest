from __future__ import annotations

from typing import Any

from jailbreak_zero.schemas import Proposal

PUNCTUATION_TO_TRIM = {'"', "'", "(", ")", "*", "[", "]", "`", ":"}


def extract_prompt(response: str) -> str | None:
    if "Adversarial Prompt:" not in response:
        return None
    text = response.split("Adversarial Prompt:")[-1]
    if not text:
        return None
    text = text.strip()
    while text and text[0] in PUNCTUATION_TO_TRIM:
        text = text[1:].strip()
    while text and text[-1] in PUNCTUATION_TO_TRIM:
        text = text[:-1].strip()
    return text or None


def proposals_from_model_output(row: dict[str, Any]) -> list[Proposal]:
    meta = row.get("meta_data", {})
    candidates = row.get("candidate", [])
    proposals: list[Proposal] = []
    for idx, candidate in enumerate(candidates):
        text = extract_prompt(candidate)
        if text is None:
            continue
        request_id = str(meta.get("request_id", "unknown"))
        proposals.append(
            Proposal(
                proposal_id=f"{request_id}-{idx}",
                text=text,
                behavior_id=request_id if meta.get("attack_dtype") == "behavior" else None,
                policy_id=request_id if meta.get("attack_dtype") != "behavior" else None,
                raw_output=candidate,
            )
        )
    return proposals
