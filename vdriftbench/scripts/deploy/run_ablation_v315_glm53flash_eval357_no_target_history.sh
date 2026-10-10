#!/usr/bin/env bash
# eval357: only v315_no_target_history (357 samples). Full baseline = existing GLM v3.15 run.
#
# Background:
#   setsid -f bash /root/vdriftbench/scripts/deploy/run_ablation_v315_glm53flash_eval357_no_target_history.sh \
#       > /root/vdriftbench/data/ablation_v315_eval357_glm53flash_no_target_history.log 2>&1 < /dev/null &
#
set -euo pipefail

ROOT="/root/vdriftbench"
cd "$ROOT"

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
DATA="${DATA:-data/v3.15_eval357.enriched.jsonl}"
SOURCE_FULL="${SOURCE_FULL:-data/v3.15_glm53flash_run/results.jsonl}"
PROXY_FULL="${PROXY_FULL:-data/ablation_v315_eval357_glm53flash_full_proxy/seed_0/full}"
TARGET_MODEL="${TARGET_MODEL:-ZHIPU/GLM-5.3-Flash}"
OUT_DIR="${OUT_DIR:-data/ablation_v315_eval357_glm53flash_no_target_history}"
WORKERS="${WORKERS:-2}"

export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"

n_src="$(wc -l < "$SOURCE_FULL" | tr -d ' ')"
n_data="$("$PYTHON" -c "print(sum(1 for l in open('$DATA') if l.strip()))")"
if [[ "$n_src" != "$n_data" ]]; then
  echo "FATAL: SOURCE_FULL has $n_src lines, DATA has $n_data" >&2
  exit 1
fi

if [[ ! -f "$PROXY_FULL/results.jsonl" ]]; then
  echo "[$(date '+%F %T')] building ablation full proxy from $SOURCE_FULL"
  "$PYTHON" scripts/export_ablation_full_proxy_from_run.py \
    --data "$DATA" \
    --source-results "$SOURCE_FULL" \
    --out-dir "$PROXY_FULL" \
    --provenance "v3.15_glm53flash_run eval357 with target multi-turn history"
fi

echo "[$(date '+%F %T')] === eval357 no-target-history ONLY (n=$n_data, workers=$WORKERS) pid $$ ==="
echo "[$(date '+%F %T')] paired full proxy: $PROXY_FULL"

exec "$PYTHON" scripts/run_ablation.py \
  --env-file .env \
  --data "$DATA" \
  --target-model "$TARGET_MODEL" \
  --draft-model draft-model \
  --judge-model judge-model \
  --embedding-model BAAI/bge-m3 \
  --presets v315_no_target_history \
  --seeds 0 \
  --ablation-only \
  --paired-full-dir "$PROXY_FULL" \
  --workers "$WORKERS" \
  --out-dir "$OUT_DIR"
