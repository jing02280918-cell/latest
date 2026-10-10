#!/usr/bin/env bash
# eval357 paired ablation: full v3.15 vs v315_no_target_history (GLM-5.3-Flash).
# Same-batch full + ablation per docs/ABLATION_v3.15.md (357 unique sample_id).
#
# Background:
#   setsid -f bash /root/vdriftbench/scripts/deploy/run_ablation_v315_glm53flash_eval357_target_history.sh \
#       > /root/vdriftbench/data/ablation_v315_eval357_glm53flash_target_history.log 2>&1 < /dev/null &
#
set -euo pipefail

ROOT="/root/vdriftbench"
cd "$ROOT"

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
DATA="${DATA:-data/v3.15_eval357.enriched.jsonl}"
N_SAMPLES="${N_SAMPLES:-357}"
TARGET_MODEL="${TARGET_MODEL:-ZHIPU/GLM-5.3-Flash}"
DRAFT_MODEL="${DRAFT_MODEL:-draft-model}"
JUDGE_MODEL="${JUDGE_MODEL:-judge-model}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
OUT_DIR="${OUT_DIR:-data/ablation_v315_eval357_glm53flash_target_history}"
SEEDS="${SEEDS:-0}"

export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"

actual_n="$("$PYTHON" -c "import json; from pathlib import Path; p=Path('$DATA'); print(sum(1 for _ in p.open() if _.strip()))")"
if [[ "$actual_n" != "$N_SAMPLES" ]]; then
  echo "FATAL: $DATA has $actual_n lines, expected N_SAMPLES=$N_SAMPLES" >&2
  exit 1
fi

echo "[$(date '+%F %T')] === eval357 target-history ablation (target=$TARGET_MODEL, n=$N_SAMPLES) pid $$ ==="
echo "[$(date '+%F %T')] presets: full,v315_no_target_history seeds: $SEEDS"
echo "[$(date '+%F %T')] out: $OUT_DIR"

exec "$PYTHON" scripts/run_ablation.py \
  --env-file .env \
  --data "$DATA" \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --embedding-model "$EMBED_MODEL" \
  --presets full,v315_no_target_history \
  --seeds "$SEEDS" \
  --out-dir "$OUT_DIR"
