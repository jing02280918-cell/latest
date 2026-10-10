#!/usr/bin/env bash
# v3.14 续跑 -> 完成后自动续跑 PAIR（与 run_pair_then_resume_v314.sh 顺序相反）
set -uo pipefail

V314_LOG=/root/vdriftbench/data/v3.14_run/driver.log
PAIR_LOG=/root/pair_baseline_v314_run/driver.log
V314_RESULTS=/root/vdriftbench/data/v3.14_run/results.jsonl
PAIR_RESULTS=/root/pair_baseline_v314_run/results.jsonl
TOTAL=314
PYTHON=/hy-tmp/venv/bin/python
WORKERS="${WORKERS:-3}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$V314_LOG"; }

log "=== v3.14 -> PAIR orchestrator starting (pid $$, WORKERS=$WORKERS) ==="

cd /root/vdriftbench
bash scripts/deploy/run_v314_full.sh >> "$V314_LOG" 2>&1
v314_rc=$?
v314_n=$(wc -l < "$V314_RESULTS" 2>/dev/null || echo 0)
log "v3.14 driver exited rc=$v314_rc, results=$v314_n/$TOTAL"

if [ "$v314_n" -lt "$TOTAL" ]; then
    log "WARNING: v3.14 did not reach $TOTAL; resuming PAIR anyway"
fi

log "resuming PAIR baseline..."
cd /root
$PYTHON run_pair_baseline.py --resume -v >> "$PAIR_LOG" 2>&1
pair_rc=$?
pair_n=$(wc -l < "$PAIR_RESULTS" 2>/dev/null || echo 0)
log "PAIR exited rc=$pair_rc, results=$pair_n/$TOTAL"
log "=== orchestrator done ==="
