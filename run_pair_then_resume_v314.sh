#!/usr/bin/env bash
# Run v3.14-aligned PAIR to completion, then resume the paused v3.14 full run.
set -uo pipefail

PAIR_LOG=/root/pair_baseline_v314_run/driver.log
V314_LOG=/root/vdriftbench/data/v3.14_run/driver.log
PYTHON=/hy-tmp/venv/bin/python

log() { echo "[$(date '+%F %T')] $*"; }

log "=== PAIR -> v3.14 resume orchestrator starting (pid $$) ==="

cd /root
log "starting PAIR baseline (v3.14-aligned)..."
$PYTHON run_pair_baseline.py --resume -v >> "$PAIR_LOG" 2>&1
pair_rc=$?
log "PAIR exited rc=$pair_rc"

pair_n=$(wc -l < /root/pair_baseline_v314_run/results.jsonl 2>/dev/null || echo 0)
log "PAIR results: $pair_n lines in pair_baseline_v314_run/results.jsonl"

if [ "$pair_rc" -ne 0 ] && [ "$pair_n" -lt 314 ]; then
    log "WARNING: PAIR did not finish cleanly; resuming v3.14 anyway"
fi

log "resuming v3.14 full run..."
cd /root/vdriftbench
nohup bash scripts/deploy/run_v314_full.sh >> "$V314_LOG" 2>&1 &
v314_pid=$!
log "v3.14 driver restarted (pid $v314_pid)"
log "=== orchestrator done ==="
