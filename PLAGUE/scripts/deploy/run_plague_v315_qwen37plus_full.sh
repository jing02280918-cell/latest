#!/usr/bin/env bash
# PLAGUE v3.15 on eval357 cohort, target = Qwen3.7-Plus (DashScope).
#
# Usage:
#   setsid nohup bash /root/PLAGUE/scripts/deploy/run_plague_v315_qwen37plus_full.sh \
#       > /root/PLAGUE/outputs/v315_qwen37plus/driver.log 2>&1 < /dev/null &

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-/root/PLAGUE/outputs/v315_qwen37plus}"
TARGET_MODEL="${TARGET_MODEL:-qwen3.7-plus}"
RESULTS_SUBDIR="${TARGET_MODEL}_vdb_draft-model"
RESULTS_DIR="$RUN_DIR/$RESULTS_SUBDIR"
DATASET="${DATASET:-/root/vdriftbench/data/v3.15_eval357.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-68cc8f3d75c55a1939aa967a11a9e448}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VDB_TARGET_BASE_URL="${VDB_TARGET_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR" "$RESULTS_DIR"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }
count_done() { ls "$RESULTS_DIR"/*_jb_*.csv 2>/dev/null | wc -l; }

log "=== PLAGUE v3.15 qwen37plus driver starting (pid $$) ==="
log "config: target=$TARGET_MODEL total=$TOTAL run_dir=$RUN_DIR"

if ! $PYTHON scripts/ensure_v315_deps.py; then
    finish "FAILED_DEPS"
    exit 1
fi

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        log "FATAL: no vLLM on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done

if ! grep -qE '^DASHSCOPE_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null; then
    log "FATAL: DASHSCOPE_API_KEY empty"
    finish "FAILED_PREFLIGHT_NO_DASHSCOPE_KEY"
    exit 1
fi

if ! $PYTHON -c "
import sys
sys.path.insert(0, 'src')
from plague.infer import query_dashscope
r = query_dashscope('回复OK', model='$TARGET_MODEL', max_tokens=5)
sys.exit(0 if (r or '').strip() else 1)
" >/dev/null 2>&1; then
    log "FATAL: target $TARGET_MODEL not reachable"
    finish "FAILED_PREFLIGHT_TARGET"
    exit 1
fi

actual_md5=$(md5sum "$DATASET" | cut -d' ' -f1)
if [ "$actual_md5" != "$ENRICHED_MD5" ]; then
    log "FATAL: dataset md5 mismatch expected=$ENRICHED_MD5 actual=$actual_md5"
    finish "FAILED_DATASET_FINGERPRINT"
    exit 1
fi
if ! $PYTHON -c "n=sum(1 for l in open('$DATASET') if l.strip()); assert n==$TOTAL"; then
    finish "FAILED_DATASET_COUNT"
    exit 1
fi
log "preflight ok: $TOTAL rows md5=$actual_md5"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(count_done)
    if [ "$n" -ge "$TOTAL" ]; then
        break
    fi
    log "run attempt $attempt/$MAX_ATTEMPTS ($n/$TOTAL)"
    $PYTHON scripts/run_plague_v315.py \
        --target-model "$TARGET_MODEL" \
        --attacker-model draft-model \
        --evaluator-model judge-model \
        --summariser-model draft-model \
        --dataset-path "$DATASET" \
        --output-dir "$RUN_DIR" \
        --llm-seed "$LLM_SEED" \
        --max-rounds 6 \
        --finisher-type goat \
        --resume
    rc=$?
    n=$(count_done)
    log "attempt $attempt exited rc=$rc, now $n/$TOTAL"
    if [ "$n" -ge "$TOTAL" ]; then
        break
    fi
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        sleep 20
    fi
done

n=$(count_done)
if [ "$n" -lt "$TOTAL" ]; then
    finish "INCOMPLETE_$n/$TOTAL"
    exit 1
fi

log "=== PLAGUE qwen37plus complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
