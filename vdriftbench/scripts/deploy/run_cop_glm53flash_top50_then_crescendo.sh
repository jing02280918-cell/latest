#!/usr/bin/env bash
# CoP GLM top50 → Crescendo GLM top50 (sequential chain).
#
# Usage:
#   setsid nohup bash /root/vdriftbench/scripts/deploy/run_cop_glm53flash_top50_then_crescendo.sh \
#       > /root/cop/outputs/v315_glm53flash_top50/chain.log 2>&1 < /dev/null &

set -uo pipefail

COP_DRIVER="/root/cop/scripts/deploy/run_cop_v315_glm53flash_top50.sh"
CRESC_DRIVER="/root/crescendo_baseline_v315_flash/scripts/deploy/run_crescendo_v315_glm53flash_top50.sh"
COP_RUN_DIR="${COP_RUN_DIR:-/root/cop/outputs/v315_glm53flash_top50}"
CRESC_RUN_DIR="${CRESC_RUN_DIR:-/root/crescendo_baseline_v315_glm53flash_top50}"
TOTAL="${TOTAL:-50}"
TARGET_MODEL="${TARGET_MODEL:-glm-5.3-flash}"
RESULTS_SUBDIR="${TARGET_MODEL}_vdb_draft-model"

mkdir -p "$COP_RUN_DIR" "$CRESC_RUN_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

log "=== chain: CoP GLM top50 → Crescendo GLM top50 (pid $$) ==="

if ! bash "$COP_DRIVER"; then
    log "FATAL: CoP GLM driver failed (see $COP_RUN_DIR/driver.log)"
    echo "CHAIN_FAILED_COP $(date '+%F %T')" > "$COP_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

cop_n=$(ls "$COP_RUN_DIR/$RESULTS_SUBDIR"/*_jb_*.csv 2>/dev/null | wc -l)
if [ "$cop_n" -lt "$TOTAL" ]; then
    log "FATAL: CoP incomplete ($cop_n/$TOTAL) — not starting Crescendo"
    echo "CHAIN_FAILED_COP_INCOMPLETE_${cop_n}/${TOTAL} $(date '+%F %T')" > "$COP_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

log "CoP GLM OK ($cop_n/$TOTAL) — pause 15s then start Crescendo GLM top50"
sleep 15

echo "CRESCENDO_STARTED_AFTER_COP $(date '+%F %T')" > "$CRESC_RUN_DIR/PAUSE_NOTE"
if ! bash "$CRESC_DRIVER"; then
    log "FATAL: Crescendo GLM driver failed (see $CRESC_RUN_DIR/driver.log)"
    echo "CHAIN_FAILED_CRESCENDO $(date '+%F %T')" > "$COP_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

cresc_n=$(wc -l < "$CRESC_RUN_DIR/results.jsonl" | tr -d ' ')
if [ "$cresc_n" -lt "$TOTAL" ]; then
    log "FATAL: Crescendo incomplete ($cresc_n/$TOTAL)"
    echo "CHAIN_FAILED_CRESCENDO_INCOMPLETE_${cresc_n}/${TOTAL} $(date '+%F %T')" > "$COP_RUN_DIR/CHAIN_STATUS"
    exit 1
fi

log "=== chain complete: CoP + Crescendo GLM top50 ==="
echo "CHAIN_OK $(date '+%F %T')" > "$COP_RUN_DIR/CHAIN_STATUS"
