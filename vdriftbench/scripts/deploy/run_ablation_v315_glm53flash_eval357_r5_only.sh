#!/usr/bin/env bash
# eval357: v315_r5_only (357 samples), paired vs GLM full proxy.
#
#   setsid -f bash scripts/deploy/run_ablation_v315_glm53flash_eval357_r5_only.sh \
#       > data/ablation_v315_eval357_glm53flash_r5_only.log 2>&1 < /dev/null &
#
set -euo pipefail

ROOT="/root/vdriftbench"
cd "$ROOT"

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
DATA="${DATA:-data/v3.15_eval357.enriched.jsonl}"
SOURCE_FULL="${SOURCE_FULL:-data/v3.15_glm53flash_run/results.jsonl}"
PROXY_FULL="${PROXY_FULL:-data/ablation_v315_eval357_glm53flash_full_proxy/seed_0/full}"
TARGET_MODEL="${TARGET_MODEL:-ZHIPU/GLM-5.3-Flash}"
OUT_DIR="${OUT_DIR:-data/ablation_v315_eval357_glm53flash_r5_only}"
WORKERS="${WORKERS:-2}"

export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"

n_src="$(wc -l < "$SOURCE_FULL" | tr -d ' ')"
n_data="$("$PYTHON" -c "print(sum(1 for l in open('$DATA') if l.strip()))")"
[[ "$n_src" == "$n_data" ]] || { echo "FATAL: SOURCE_FULL=$n_src DATA=$n_data" >&2; exit 1; }

if [[ ! -f "$PROXY_FULL/results.jsonl" ]]; then
  echo "[$(date '+%F %T')] building full proxy from $SOURCE_FULL"
  "$PYTHON" scripts/export_ablation_full_proxy_from_run.py \
    --data "$DATA" --source-results "$SOURCE_FULL" --out-dir "$PROXY_FULL"
fi

RESUME_FLAG=()
RESULTS="$OUT_DIR/seed_0/v315_r5_only/results.jsonl"
if [[ -f "$RESULTS" ]]; then
  n_done="$(wc -l < "$RESULTS" | tr -d ' ')"
  if [[ "$n_done" -lt "$n_data" ]]; then
    RESUME_FLAG=(--resume)
    echo "[$(date '+%F %T')] resume R5-only from $n_done/$n_data"
  fi
fi

echo "[$(date '+%F %T')] === eval357 R5-only ablation (n=$n_data, workers=$WORKERS) pid $$ ==="

exec "$PYTHON" scripts/run_ablation.py \
  "${RESUME_FLAG[@]}" \
  --env-file .env \
  --data "$DATA" \
  --target-model "$TARGET_MODEL" \
  --draft-model draft-model \
  --judge-model judge-model \
  --embedding-model BAAI/bge-m3 \
  --presets v315_r5_only \
  --seeds 0 \
  --ablation-only \
  --paired-full-dir "$PROXY_FULL" \
  --workers "$WORKERS" \
  --out-dir "$OUT_DIR"
