#!/usr/bin/env bash
# Wait for CoP v3.15 to finish, then start PLAGUE v3.15 full run.
#
# Usage (background):
#   setsid nohup bash /root/PLAGUE/scripts/deploy/run_plague_after_cop.sh \
#       > /root/PLAGUE/outputs/v315/wait_for_cop.log 2>&1 < /dev/null &

set -uo pipefail

COP_RUN_DIR="${COP_RUN_DIR:-/root/cop/outputs/v315}"
COP_RESULTS="${COP_RESULTS:-$COP_RUN_DIR/deepseek-v4-pro_vdb_draft-model}"
COP_TOTAL="${COP_TOTAL:-360}"
POLL_SEC="${POLL_SEC:-30}"
PLAGUE_DRIVER="/root/PLAGUE/scripts/deploy/run_plague_v315_full.sh"
WAIT_LOG="${WAIT_LOG:-/root/PLAGUE/outputs/v315/wait_for_cop.log}"

mkdir -p /root/PLAGUE/outputs/v315

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WAIT_LOG"; }

cop_done() {
    local n status
    n=$(ls "$COP_RESULTS"/*_jb_*.csv 2>/dev/null | wc -l)
    if [ "$n" -ge "$COP_TOTAL" ]; then
        return 0
    fi
    if [ -f "$COP_RUN_DIR/DRIVER_STATUS" ]; then
        status=$(cut -d' ' -f1 < "$COP_RUN_DIR/DRIVER_STATUS")
        [[ "$status" == OK_* ]]
        return
    fi
    return 1
}

log "=== waiting for CoP v3.15 ($COP_TOTAL samples) before PLAGUE ==="
log "CoP results dir: $COP_RESULTS"

while true; do
    n=$(ls "$COP_RESULTS"/*_jb_*.csv 2>/dev/null | wc -l)
    if cop_done; then
        log "CoP complete ($n/$COP_TOTAL) — starting PLAGUE"
        break
    fi
    log "CoP progress: $n/$COP_TOTAL — sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
done

# Brief pause so CoP main.py / evaluate can release vLLM connections
sleep 10

log "launching PLAGUE driver: $PLAGUE_DRIVER"
exec bash "$PLAGUE_DRIVER"
