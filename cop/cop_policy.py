"""Policy JSON parsing and best-result helpers for CoP."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# draft-model vLLM context is 8192 tokens; keep user text well below that in chars.
DEFAULT_DRAFT_PROMPT_MAX_CHARS = 1800
DEFAULT_POLICY_GOAL_MAX_CHARS = 200


def clip_draft_text(text: str, max_chars: int = DEFAULT_DRAFT_PROMPT_MAX_CHARS) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 20
    return text[:head] + "\n…[truncated]…\n" + text[-tail:]


def _extract_json_blob(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _repair_json(blob: str) -> dict | None:
    for suffix in ("", "}", "]}", "}]}", '"}]}', '"}]}}', '"}]}]}'):
        try:
            obj = json.loads(blob + suffix)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_policy_obj(obj: dict) -> tuple[list | None, dict | None]:
    if not isinstance(obj, dict):
        return None, None
    if "options" in obj and "high_level_policy" in obj:
        options = obj["options"]
        policy = obj["high_level_policy"]
    elif len(obj) >= 3:
        keys = list(obj.keys())
        options = obj.get(keys[1])
        policy = obj.get(keys[2])
    else:
        return None, None
    if not isinstance(options, list) or not options:
        return None, None
    if not isinstance(policy, dict) or "rules" not in policy:
        return None, None
    for opt in options:
        if not isinstance(opt, dict) or "name" not in opt or "primitive_actions" not in opt:
            return None, None
    return options, policy


def parse_policy_json(text: str) -> tuple[list | None, dict | None]:
    """Parse policy LM output into (options, high_level_policy)."""
    if not text or not str(text).strip():
        return None, None
    cleaned = re.sub(r"^```(?:json|python)?\s*", "", str(text).strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    for candidate in (cleaned, _extract_json_blob(cleaned)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            opts, pol = _normalize_policy_obj(obj)
            if opts is not None:
                return opts, pol
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            if repaired:
                opts, pol = _normalize_policy_obj(repaired)
                if opts is not None:
                    return opts, pol
    return None, None


def pick_best_index(scores: list[int | float]) -> int:
    best_i = 0
    best_v = scores[0] if scores else 1
    for i, v in enumerate(scores):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def make_result_record(
    *,
    sample_id: str,
    prompt: str,
    response: str,
    judge_score: int,
    bench_score: int,
    total_query_time: int,
    action_l: Any,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "best_msg": prompt,
        "jailbreak_output": response,
        "judge_score": int(judge_score),
        "bench_score": int(bench_score),
        "total_query_time": int(total_query_time),
        "action_l": action_l,
    }


def merge_result_record(current: dict | None, candidate: dict) -> dict:
    if current is None or candidate["judge_score"] > current["judge_score"]:
        return candidate
    return current


def is_sample_complete(record: dict, max_queries: int) -> bool:
    """Sample is done when jailbreak succeeds or the query budget is exhausted."""
    if int(record.get("judge_score", 1)) >= 10:
        return True
    if int(record.get("total_query_time", 0)) >= max_queries:
        return True
    return False
