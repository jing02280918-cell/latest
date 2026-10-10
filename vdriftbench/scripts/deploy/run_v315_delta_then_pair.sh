#!/usr/bin/env bash
# Wait for V-Drift v3.15 delta (71 samples) to finish, then run PAIR delta + merge.
#
# Usage (background):
#   cd /root/vdriftbench
#   mkdir -p data/v3.15_delta_run /root/pair_baseline_v315_delta_run
#   setsid nohup bash scripts/deploy/run_v315_delta_then_pair.sh > data/v3.15_delta_run/orchestrator.log 2>&1 < /dev/null &

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

VDRIFT_STATUS="data/v3.15_delta_run/DRIVER_STATUS"
VDRIFT_RESULTS="data/v3.15_delta_run/results.jsonl"
VDRIFT_TOTAL=71
POLL_SEC="${POLL_SEC:-60}"
LOG="data/v3.15_delta_run/orchestrator.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== v3.15 orchestrator: wait V-Drift delta -> PAIR delta (pid $$) ==="

while true; do
    status=""
    [ -f "$VDRIFT_STATUS" ] && status=$(cat "$VDRIFT_STATUS" | awk '{print $1}')
    n=$([ -f "$VDRIFT_RESULTS" ] && wc -l < "$VDRIFT_RESULTS" || echo 0)

    if [[ "$status" == OK_merged_360* ]]; then
        log "V-Drift delta finished (status=$status, results=$n/$VDRIFT_TOTAL)"
        break
    fi
    if [ "$n" -ge "$VDRIFT_TOTAL" ] && [ -f "data/v3.15_run/results.jsonl" ]; then
        merged_n=$(wc -l < data/v3.15_run/results.jsonl)
        if [ "$merged_n" -ge 360 ]; then
            log "V-Drift delta + merge detected (delta=$n, merged=$merged_n)"
            break
        fi
    fi
    if [[ "$status" == FAILED* ]] || [[ "$status" == INCOMPLETE* ]]; then
        log "WARNING: V-Drift delta status=$status; starting PAIR delta anyway if partial results exist"
        [ "$n" -gt 0 ] && break
        log "FATAL: no V-Drift delta progress, aborting orchestrator"
        exit 1
    fi

    log "waiting V-Drift delta... $n/$VDRIFT_TOTAL (status=${status:-running})"
    sleep "$POLL_SEC"
done

log "starting PAIR v3.15 delta + merge..."
bash scripts/deploy/run_pair_v315_delta.sh >> "$LOG" 2>&1
pair_rc=$?
pair_status=""
[ -f /root/pair_baseline_v315_delta_run/DRIVER_STATUS ] && pair_status=$(cat /root/pair_baseline_v315_delta_run/DRIVER_STATUS)

log "orchestrator done: PAIR rc=$pair_rc status=${pair_status:-unknown}"
exit "$pair_rc"
