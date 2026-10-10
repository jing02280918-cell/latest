"""Load V-DriftBench v3.15 enriched dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VDBSample:
    sample_id: str
    prompt: str
    category: str
    category_macro: str
    target_claim: str
    value_axis: str
    risk_type: str
    expected_safe_behavior: str


def load_vdb_dataset(path: str | Path, *, start: int = 0, limit: int | None = None) -> list[VDBSample]:
    samples: list[VDBSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            samples.append(
                VDBSample(
                    sample_id=row["sample_id"],
                    prompt=row["prompt"],
                    category=row["category"],
                    category_macro=row["category_macro"],
                    target_claim=row["target_claim"],
                    value_axis=row["value_axis"],
                    risk_type=row["risk_type"],
                    expected_safe_behavior=row["expected_safe_behavior"],
                )
            )
    end = len(samples) if limit is None else min(len(samples), start + limit)
    return samples[start:end]
