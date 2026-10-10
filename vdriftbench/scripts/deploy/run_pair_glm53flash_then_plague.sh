#!/usr/bin/env bash
# PAIR GLM resume (35 rewritten eval357 samples) → PLAGUE GLM resume.
#
# Usage:
#   setsid nohup bash /root/vdriftbench/scripts/deploy/run_pair_glm53flash_then_plague.sh \
#       > /root/pair_baseline_v315_glm53flash_run/chain_then_plague.log 2>&1 < /dev/null &

set -uo pipefail

PAIR_DRIVER="/root/vdriftbench/scripts/deploy/run_pair_v315_glm53flash_full.sh"
PLAGUE_DRIVER="/root/PLAGUE/scripts/deploy/run_plague_v315_glm53flash_full.sh"
PAIR_RUN_DIR="${PAIR_RUN_DIR:-/root/pair_baseline_v315_glm53flash_run}"
PLAGUE_RUN_DIR="${PLAGUE_RUN_DIR:-/root/PLAGUE/outputs/v315_glm53flash}"
CHAIN_LOG="${CHAIN_LOG:-$PAIR_RUN_DIR/chain_then_plague.log}"
PAIR_TOTAL="${PAIR_TOTAL:-357}"
PLAGUE_TOTAL="${PLAGUE_TOTAL:-357}"

mkdir -p "$PAIR_RUN_DIR" "$PLAGUE_RUN_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

log "=== chain: PAIR GLM (resume) → PLAGUE GLM (resume) pid $$ ==="

if ! bash "$PAIR_DRIVER"; then
    log "FATAL: PAIR GLM driver failed (see $PAIR_RUN_DIR/driver.log)"
    echo "CHAIN_FAILED_PAIR $(date '+%F %T')" > "$PAIR_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

pair_n=$(wc -l < "$PAIR_RUN_DIR/results.jsonl" | tr -d ' ')
if [ "$pair_n" -lt "$PAIR_TOTAL" ]; then
    log "FATAL: PAIR incomplete ($pair_n/$PAIR_TOTAL) — not starting PLAGUE"
    echo "CHAIN_FAILED_PAIR_INCOMPLETE_${pair_n}/${PAIR_TOTAL} $(date '+%F %T')" > "$PAIR_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

log "PAIR GLM OK ($pair_n/$PAIR_TOTAL) — pause 15s then start PLAGUE GLM"
sleep 15

echo "PLAGUE_RESUMED_AFTER_PAIR $(date '+%F %T')" > "$PLAGUE_RUN_DIR/PAUSE_NOTE"
if ! bash "$PLAGUE_DRIVER"; then
    log "FATAL: PLAGUE GLM driver failed (see $PLAGUE_RUN_DIR/driver.log)"
    echo "CHAIN_FAILED_PLAGUE $(date '+%F %T')" > "$PAIR_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

plague_n=$(ls "$PLAGUE_RUN_DIR/glm-5.3-flash_vdb_draft-model"/*_jb_*.csv 2>/dev/null | wc -l)
if [ "$plague_n" -lt "$PLAGUE_TOTAL" ]; then
    log "FATAL: PLAGUE incomplete ($plague_n/$PLAGUE_TOTAL)"
    echo "CHAIN_FAILED_PLAGUE_INCOMPLETE_${plague_n}/${PLAGUE_TOTAL} $(date '+%F %T')" > "$PAIR_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

log "=== chain complete: PAIR + PLAGUE GLM ==="
echo "CHAIN_OK $(date '+%F %T')" > "$PAIR_RUN_DIR/CHAIN_STATUS"
