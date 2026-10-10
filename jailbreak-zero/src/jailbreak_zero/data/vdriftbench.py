"""Load V-DriftBench v3.14 enriched dataset for jailbreak-zero."""

from __future__ import annotations

import json
from pathlib import Path

from jailbreak_zero.schemas import Behavior


DEFAULT_VDB_DATASET_PATH = "/root/vdriftbench/data/dataset_100.enriched.jsonl"


def load_vdriftbench_behaviors(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    start: int = 0,
) -> list[Behavior]:
    dataset_path = Path(path or DEFAULT_VDB_DATASET_PATH)
    behaviors: list[Behavior] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            behaviors.append(
                Behavior(
                    behavior_id=row["sample_id"],
                    text=row["prompt"],
                    benchmark="vdriftbench",
                    target_claim=row["target_claim"],
                    expected_safe_behavior=row["expected_safe_behavior"],
                    category=row.get("category"),
                    category_macro=row.get("category_macro"),
                )
            )
    end = len(behaviors) if limit is None else min(len(behaviors), start + limit)
    return behaviors[start:end]
