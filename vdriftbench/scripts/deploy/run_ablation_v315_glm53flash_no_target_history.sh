#!/usr/bin/env bash
# Core60 ablation: v315_no_target_history vs existing GLM full arm.
#
# Background:
#   setsid -f bash /root/vdriftbench/scripts/deploy/run_ablation_v315_glm53flash_no_target_history.sh \
#       > /root/vdriftbench/data/ablation_v315_core60_glm53flash_no_target_history.log 2>&1 < /dev/null &
#
set -euo pipefail

ROOT="/root/vdriftbench"
cd "$ROOT"

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
DATA="${DATA:-data/v3.15_ablation_core60.enriched.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-ZHIPU/GLM-5.3-Flash}"
DRAFT_MODEL="${DRAFT_MODEL:-draft-model}"
JUDGE_MODEL="${JUDGE_MODEL:-judge-model}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
OUT_DIR="${OUT_DIR:-data/ablation_v315_core60_glm53flash_no_target_history}"
PAIRED_FULL="${PAIRED_FULL:-data/ablation_v315_core60_glm53flash/seed_0/full}"
LOG="${LOG:-data/ablation_v315_core60_glm53flash_no_target_history.log}"

export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== v3.15 no-target-history ablation (target=$TARGET_MODEL) pid $$ ==="
log "paired full: $PAIRED_FULL"
log "out: $OUT_DIR"

exec "$PYTHON" scripts/run_ablation.py \
  --env-file .env \
  --data "$DATA" \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --embedding-model "$EMBED_MODEL" \
  --presets v315_no_target_history \
  --seeds 0 \
  --ablation-only \
  --paired-full-dir "$PAIRED_FULL" \
  --out-dir "$OUT_DIR"
