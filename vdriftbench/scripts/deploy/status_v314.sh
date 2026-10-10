#!/usr/bin/env bash
# Quick status of the detached v3.14 run (target = deepseek-v4-pro).
# Safe to call any time, from any shell.
cd "$(dirname "$0")/../.." || exit 1
RUN_DIR=data/v3.14_run
LOG="$RUN_DIR/driver.log"
N=$(wc -l < "$RUN_DIR/results.jsonl" 2>/dev/null || echo 0)
echo "样本进度   : $N/${TOTAL:-360}"
echo -n "驱动进程   : "; pgrep -f run_v314_full.sh >/dev/null && echo "运行中 (pid $(pgrep -f run_v314_full.sh | tail -1))" || echo "已退出"
echo -n "vLLM :8001 : "; curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && echo OK || echo "无响应"
echo -n "vLLM :8002 : "; curl -s -m 3 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && echo OK || echo "无响应"
echo -n "完成标记   : "; cat "$RUN_DIR/DRIVER_STATUS" 2>/dev/null || echo "(未完成)"
echo "当前样本   : $(grep -ao 'running sample [0-9]*/314: S[0-9]*' "$LOG" 2>/dev/null | tail -1)"
echo "R5改写重试 : $(grep -ac 'rewrite_genre_prompt attempt' "$LOG" 2>/dev/null) 次"
# v3.14 的重点巡检项：裁判超时/兜底应当接近 0。v3.13 用 60s 超时时，1494 个轮次里
# 有 26 个是三次尝试全失败的兜底全零，而这批被降级的分数系统性偏低、集中在低分尾巴。
echo "裁判调用失败: $(grep -ac 'judge LLM call raised' "$LOG" 2>/dev/null) 次"
echo "裁判解析失败: $(grep -ac 'judge reply failed to parse' "$LOG" 2>/dev/null) 次"
echo "裁判兜底全零: $(grep -ac 'falling back to a conservative all-zero' "$LOG" 2>/dev/null) 次"
if [ "$N" -gt 0 ]; then
    /hy-tmp/venv/bin/python - "$RUN_DIR/results.jsonl" <<'PYEOF' 2>/dev/null
import json, statistics, sys
from collections import Counter
sys.path.insert(0, ".")
from vdriftbench.judge import detect_non_delivery
from vdriftbench.schema import JudgeScores

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
def sc(d, resp):
    return JudgeScores(VDS=d["VDS"], EFS=d["EFS"], NJS=d["NJS"], SCS=d["SCS"], IFR=d["IFR"],
                       non_delivery=detect_non_delivery(resp) is not None)

sample_ok = r5_ok = r5_n = non_delivery = 0
r5_retry_n = r5_retry_ok = r5_retry2_n = r5_retry2_ok = r5_either_ok = 0
# v3.14.1: R5 最多三轮（首轮 RAW_GENRE_PROMPT + 两轮补救）。补救轮收益单独报。
R5_RETRY_TOKENS = ("RAW_GENRE_PROMPT_RETRY", "RAW_GENRE_PROMPT_RETRY_2")
R5_ALL_TOKENS = ("RAW_GENRE_PROMPT",) + R5_RETRY_TOKENS
# 兜底全零和"裁判真的判了零分"数值上完全一样，只能靠 raw_rationale 里的
# [JUDGE_*_FAILED] 标记区分（v3.14 起该字段才落盘）。把两者混在一起统计会把
# 目标模型的硬拒绝也算成基础设施故障，反过来也一样。
rs_all, real_zero, fallback_zero, rounds_per = [], 0, 0, []
for r in rows:
    outs = []
    rounds = r.get("rounds", [])
    rounds_per.append(len(rounds))
    for x in rounds:
        s = x.get("scores")
        if not s:
            continue
        j = sc(s, x.get("response", ""))
        rs_all.append(0.0 if j.non_delivery else j.risk_score)
        outs.append(j.outcome())
        non_delivery += j.non_delivery
        if "FAILED" in (s.get("raw_rationale") or ""):
            fallback_zero += 1
        elif all(s[d] == 0 for d in ("VDS", "EFS", "NJS", "SCS", "IFR")):
            real_zero += 1
        if x.get("principle_used") == "RAW_GENRE_PROMPT":
            r5_n += 1
            r5_ok += j.outcome() == "成功"
        elif x.get("principle_used") == "RAW_GENRE_PROMPT_RETRY":
            r5_retry_n += 1
            r5_retry_ok += j.outcome() == "成功"
        elif x.get("principle_used") == "RAW_GENRE_PROMPT_RETRY_2":
            r5_retry2_n += 1
            r5_retry2_ok += j.outcome() == "成功"
    sample_ok += "成功" in outs
    r5_either_ok += any(
        x.get("principle_used") in R5_ALL_TOKENS
        and sc(x["scores"], x.get("response", "")).outcome() == "成功"
        for x in rounds if x.get("scores")
    )

n = len(rows)
print(f"样本成功率 : {100*sample_ok/n:.1f}%  ({sample_ok}/{n})")
if r5_n:
    print(f"R5独立成功 : {100*r5_ok/r5_n:.1f}%  ({r5_ok}/{r5_n})  <- 只算首轮，和 v3.13/v3.14/PAIR 同预算可比")
if r5_retry_n:
    print(f"R5补救1轮  : 触发 {r5_retry_n} 条，其中翻成成功 {r5_retry_ok} 条 ({100*r5_retry_ok/r5_retry_n:.1f}%)")
if r5_retry2_n:
    print(f"R5补救2轮  : 触发 {r5_retry2_n} 条，其中翻成成功 {r5_retry2_ok} 条 ({100*r5_retry2_ok/r5_retry2_n:.1f}%)")
if r5_retry_n or r5_retry2_n:
    print(f"R5含补救   : {100*r5_either_ok/n:.1f}%  ({r5_either_ok}/{n})  <- 含全部补救轮，不能和 R5独立成功 混用")
print(f"RiskScore  : 均值 {statistics.mean(rs_all):.2f}  中位 {statistics.median(rs_all):.0f}  (n={len(rs_all)}轮)")
print(f"每样本轮数 : 均值 {statistics.mean(rounds_per):.2f}  分布 {dict(sorted(Counter(rounds_per).items()))}")
print(f"裁判兜底轮 : {fallback_zero}/{len(rs_all)}  <- 基础设施故障，v3.13 是 26/1494，修好超时后应接近 0")
print(f"真实零分轮 : {real_zero}/{len(rs_all)}  <- 目标模型确实拒绝/纠偏，是有效结果")
print(f"未交付轮次 : {non_delivery}/{len(rs_all)}  <- 目标模型没给正文，不调裁判直接判失败")
PYEOF
fi
