#!/usr/bin/env bash
# Run V-Drift v3.15 GLM (resume → typically 35 rewritten eval357 samples), then PLAGUE GLM.
#
# Usage:
#   setsid nohup bash /root/vdriftbench/scripts/deploy/run_v315_glm53flash_then_plague.sh \
#       > /root/vdriftbench/data/v3.15_glm53flash_run/chain_then_plague.log 2>&1 < /dev/null &

set -uo pipefail

VDRIFT_DRIVER="/root/vdriftbench/scripts/deploy/run_v315_glm53flash_full.sh"
PLAGUE_DRIVER="/root/PLAGUE/scripts/deploy/run_plague_v315_glm53flash_full.sh"
VDRIFT_RUN_DIR="${VDRIFT_RUN_DIR:-/root/vdriftbench/data/v3.15_glm53flash_run}"
PLAGUE_RUN_DIR="${PLAGUE_RUN_DIR:-/root/PLAGUE/outputs/v315_glm53flash}"
CHAIN_LOG="${CHAIN_LOG:-$VDRIFT_RUN_DIR/chain_then_plague.log}"
VDRIFT_TOTAL="${VDRIFT_TOTAL:-357}"

mkdir -p "$VDRIFT_RUN_DIR" "$PLAGUE_RUN_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

log "=== chain: v3.15 GLM (resume) → PLAGUE GLM (resume) pid $$ ==="
log "v3.15 driver: $VDRIFT_DRIVER"
log "plague driver: $PLAGUE_DRIVER"

if ! bash "$VDRIFT_DRIVER"; then
    log "FATAL: v3.15 GLM driver failed (see $VDRIFT_RUN_DIR/driver.log)"
    echo "CHAIN_FAILED_VDRIFT $(date '+%F %T')" > "$VDRIFT_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

n=$(wc -l < "$VDRIFT_RUN_DIR/results.jsonl" | tr -d ' ')
if [ "$n" -lt "$VDRIFT_TOTAL" ]; then
    log "FATAL: v3.15 incomplete ($n/$VDRIFT_TOTAL) — not starting PLAGUE"
    echo "CHAIN_FAILED_VDRIFT_INCOMPLETE_${n}/${VDRIFT_TOTAL} $(date '+%F %T')" > "$VDRIFT_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

log "v3.15 GLM OK ($n/$VDRIFT_TOTAL) — pause 15s then start PLAGUE GLM"
sleep 15

echo "PLAGUE_RESUMED_AFTER_VDRIFT $(date '+%F %T')" > "$PLAGUE_RUN_DIR/PAUSE_NOTE"
if ! bash "$PLAGUE_DRIVER"; then
    log "FATAL: PLAGUE GLM driver failed (see $PLAGUE_RUN_DIR/driver.log)"
    echo "CHAIN_FAILED_PLAGUE $(date '+%F %T')" > "$VDRIFT_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

log "=== chain complete: v3.15 + PLAGUE GLM ==="
echo "CHAIN_OK $(date '+%F %T')" > "$VDRIFT_RUN_DIR/CHAIN_STATUS"
