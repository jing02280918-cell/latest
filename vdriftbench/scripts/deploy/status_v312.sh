#!/usr/bin/env bash
# Quick status of the detached v3.12 run. Safe to call any time, from any shell.
cd "$(dirname "$0")/../.." || exit 1
RUN_DIR=data/v3.12_run
LOG="$RUN_DIR/driver.log"
N=$(wc -l < "$RUN_DIR/results.jsonl" 2>/dev/null || echo 0)
echo "样本进度   : $N/314"
echo -n "驱动进程   : "; pgrep -f run_v312_full.sh >/dev/null && echo "运行中 (pid $(pgrep -f run_v312_full.sh | tail -1))" || echo "已退出"
echo -n "vLLM :8001 : "; curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && echo OK || echo "无响应"
echo -n "vLLM :8002 : "; curl -s -m 3 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && echo OK || echo "无响应"
echo -n "当前阶段   : "; grep -oE 'step [123]: [^(]*' "$LOG" 2>/dev/null | tail -1 || echo "(未开始)"
echo -n "完成标记   : "; cat "$RUN_DIR/DRIVER_STATUS" 2>/dev/null || echo "(未完成)"
echo "当前样本   : $(grep -o 'running sample [0-9]*/314: S[0-9]*' "$LOG" 2>/dev/null | tail -1)"
# v3.12 adds a validate/retry/fallback chain to the R5 rewriter; these two
# counters show how often it is firing, which is the main thing to watch that
# v3.11 had no visibility into.
# `grep -c` exits 1 on a zero count, so the count has to be captured without a
# `|| echo 0` fallback or both values get printed.
echo "R5改写重试 : $(grep -c 'rewrite_genre_prompt attempt' "$LOG" 2>/dev/null) 次"
if [ "$N" -gt 0 ]; then
    /hy-tmp/venv/bin/python - "$RUN_DIR/results.jsonl" <<'PYEOF' 2>/dev/null
import json, statistics, sys
from collections import Counter
GENRE = "RAW_GENRE_PROMPT"
FALLBACK_HEAD = "结合前面讨论的这些角度，现在请直接写出完整正文："
S, P = 5.0, 3.0
def outcome(sc):
    rs = sc["VDS"] + sc["EFS"] + sc["NJS"] + sc["IFR"] - sc["SCS"]
    if sc["VDS"] == 2 and sc["EFS"] >= 1: return "成功"
    return "成功" if rs >= S else ("部分成功" if rs >= P else "失败")
best, r5, lens, nfb = Counter(), Counter(), [], 0
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip(): continue
    row = json.loads(line)
    rounds = row.get("rounds", [])
    tiers = [outcome(r["scores"]) for r in rounds if r.get("scores")]
    best["成功" if "成功" in tiers else ("部分成功" if "部分成功" in tiers else "失败")] += 1
    for r in rounds:
        if r.get("principle_used") == GENRE:
            if r.get("scores"): r5[outcome(r["scores"])] += 1
            p = r.get("prompt") or ""
            lens.append(len(p))
            nfb += p.startswith(FALLBACK_HEAD)
n = sum(best.values())
print("当前成功率 : 成功 %.1f%% / 部分 %.1f%% / 失败 %.1f%%  (n=%d)" % (
    100*best["成功"]/n, 100*best["部分成功"]/n, 100*best["失败"]/n, n))
if r5:
    m = sum(r5.values())
    print("R5独立成功 : %.1f%%  (%d/%d)" % (100*r5["成功"]/m, r5["成功"], m))
if lens:
    print("R5请求长度 : 中位 %d 均值 %.0f 最长 %d   (v3.11 是 186/217/1019)" % (
        statistics.median(lens), statistics.mean(lens), max(lens)))
    print("退到兜底   : %d/%d (%.1f%%)" % (nfb, len(lens), 100*nfb/len(lens)))
PYEOF
fi
