#!/usr/bin/env python3
"""Ensure v3.15 deps in the shared venv — install only packages that are missing.

Same policy as PAIR / CoP: reuse /hy-tmp/venv, never `pip install -r requirements.txt`.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

# (import_name, pip_spec) — minimal set for run_plague_v315.py + vdriftbench judge
V315_PACKAGES: list[tuple[str, str]] = [
    ("openai", "openai>=1.0.0"),
    ("pandas", "pandas>=2.0.0"),
    ("dotenv", "python-dotenv>=1.0.0"),
    ("numpy", "numpy>=1.24.0"),
    ("tqdm", "tqdm>=4.66.0"),
]


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _verify_plague_import() -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    from plague import Plague, PlagueConfig  # noqa: F401
    from vdb_judge import build_vdb_judge_client  # noqa: F401


def main() -> int:
    missing_specs = [spec for name, spec in V315_PACKAGES if not _can_import(name)]
    if missing_specs:
        print(f"installing missing v3.15 packages: {missing_specs}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_specs])
    else:
        print("v3.15 deps already present (no pip install needed)", flush=True)

    still_missing = [name for name, _ in V315_PACKAGES if not _can_import(name)]
    if still_missing:
        print(f"FATAL: imports still missing after install: {still_missing}", file=sys.stderr)
        return 1

    try:
        _verify_plague_import()
    except Exception as exc:
        print(f"FATAL: PLAGUE v3.15 import check failed: {exc}", file=sys.stderr)
        return 1

    print("PLAGUE v3.15 import chain ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
