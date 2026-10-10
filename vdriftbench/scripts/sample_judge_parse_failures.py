"""采样裁判真实的解析失败回复，用来据实放宽解析器（而不是猜格式）。

拿 v3.13 已存储的轮次当输入，反复调用裁判，把 `_parse_judge_reply` 抛
JudgeParseError 的原始回复原样打出来。只读，不写回任何结果文件。
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
for _k in list(os.environ):
    if "KEY" in _k or _k.endswith("_URL"):
        os.environ[_k] = os.environ[_k].strip()

from vdriftbench.judge import (  # noqa: E402
    _JUDGE_SYSTEM_PROMPT,
    _build_judge_user_content,
    _parse_judge_reply,
    JudgeParseError,
)
from vdriftbench.llm_client import build_llm_client  # noqa: E402

RESULTS = Path("/root/vdriftbench/data/v3.13_run/results.jsonl")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "80"))


def main() -> int:
    rows = [json.loads(l) for l in RESULTS.open(encoding="utf-8") if l.strip()]
    tasks = []
    for r in rows:
        rounds = r.get("rounds", [])
        for i, x in enumerate(rounds):
            if not x.get("scores"):
                continue
            hist = "\n".join(
                f"[第{y['round_idx']}轮-{k}] {y.get(f, '')}"
                for y in rounds[:i]
                for k, f in (("提问", "prompt"), ("回复", "response"))
            )
            tasks.append((r["sample_id"], x["round_idx"], r.get("target_claim", ""),
                          r.get("expected_safe_behavior", ""), hist, x.get("response", "")))
    step = max(1, len(tasks) // N_SAMPLES)
    tasks = tasks[::step][:N_SAMPLES]
    print(f"抽样 {len(tasks)} 个轮次(跨样本均匀取)，逐个调用裁判并记录解析失败的原始回复\n")

    judge = build_llm_client("judge-model", mock=False, persona="judge")

    def go(t):
        sid, ridx, tc, esb, hist, resp = t
        content = _build_judge_user_content(tc, esb, hist, resp)
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            reply = judge.chat(messages, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            return (sid, ridx, "CALL_FAILED", str(e)[:200])
        try:
            _parse_judge_reply(reply)
            return (sid, ridx, "OK", reply[:120])
        except JudgeParseError:
            return (sid, ridx, "PARSE_FAILED", reply)

    with ThreadPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(go, tasks))

    c = Counter(r[2] for r in res)
    n = len(res)
    print("=" * 74)
    print(f"attempt-0 (temperature=0, 新的300s超时) 结果: n={n}")
    for k, v in c.most_common():
        print(f"    {k}: {v} ({100 * v / n:.1f}%)")
    print("=" * 74)
    fails = [r for r in res if r[2] == "PARSE_FAILED"]
    if not fails:
        print("\n没有解析失败 —— 当前解析器已能覆盖这批回复。")
    for sid, ridx, _, reply in fails:
        print(f"\n--- {sid} r{ridx} 原始回复 ---")
        print(reply[:600])
    calls = [r for r in res if r[2] == "CALL_FAILED"]
    for sid, ridx, _, err in calls:
        print(f"\n--- {sid} r{ridx} 调用失败 --- {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
