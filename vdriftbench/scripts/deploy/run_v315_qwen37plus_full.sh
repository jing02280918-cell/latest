#!/usr/bin/env bash
# v3.15 pipeline full run (357 eval cohort) with remote target = Qwen3.7-Plus.
#
# Prerequisites:
#   bash scripts/deploy/serve_draft.sh   # :8001
#   bash scripts/deploy/serve_judge.sh   # :8002
#   DASHSCOPE_API_KEY in .env
#
# Usage:
#   cd /root/vdriftbench
#   setsid nohup bash scripts/deploy/run_v315_qwen37plus_full.sh > data/v3.15_qwen37plus_run/driver.log 2>&1 < /dev/null &

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-data/v3.15_qwen37plus_run}"
RESULTS="$RUN_DIR/results.jsonl"
ENRICHED="${ENRICHED:-data/v3.15_eval357.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-68cc8f3d75c55a1939aa967a11a9e448}"
TARGET_MODEL="${TARGET_MODEL:-qwen3.7-plus}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
WORKERS="${WORKERS:-3}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"
DASHSCOPE_BASE="${VDB_TARGET_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VDB_TARGET_BASE_URL="${VDB_TARGET_BASE_URL:-$DASHSCOPE_BASE}"

mkdir -p "$RUN_DIR"
: > "$RESULTS"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }
count_done() { wc -l < "$RESULTS" 2>/dev/null | tr -d ' '; }

log "=== v3.15 full run (target=$TARGET_MODEL) starting (pid $$) ==="
log "config: run_dir=$RUN_DIR llm_seed=$LLM_SEED total=$TOTAL workers=$WORKERS"

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        log "FATAL: no vLLM server on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done
log "preflight: vLLM :8001 and :8002 both responding"

if ! grep -qE '^DASHSCOPE_API_KEY=.+' .env 2>/dev/null; then
    log "FATAL: DASHSCOPE_API_KEY is empty in .env"
    finish "FAILED_PREFLIGHT_NO_DASHSCOPE_KEY"
    exit 1
fi
log "preflight: DASHSCOPE_API_KEY present"

if ! $PYTHON -c "
import os, sys
from vdriftbench.env_loader import load_dotenv_if_present
load_dotenv_if_present('.env')
from vdriftbench.llm_client import build_llm_client
c = build_llm_client('$TARGET_MODEL', mock=False, persona='target')
r = c.chat([{'role': 'user', 'content': '回复OK'}])
sys.exit(0 if (r or '').strip() else 1)
" > /dev/null 2>&1; then
    log "FATAL: target model '$TARGET_MODEL' is not reachable via DashScope"
    finish "FAILED_PREFLIGHT_TARGET_MODEL"
    exit 1
fi
log "preflight: target model $TARGET_MODEL responds"

actual_md5=$(md5sum "$ENRICHED" | cut -d' ' -f1)
if [ "$actual_md5" != "$ENRICHED_MD5" ]; then
    log "FATAL: enriched dataset fingerprint mismatch (expected $ENRICHED_MD5, got $actual_md5)"
    finish "FAILED_DATASET_FINGERPRINT"
    exit 1
fi
if ! $PYTHON -c "
from vdriftbench.enrich import load_enriched_jsonl
n = len(load_enriched_jsonl('$ENRICHED', verify_against_raw=False))
assert n == $TOTAL, f'{n} rows, expected $TOTAL'
"; then
    finish "FAILED_ENRICHED_VERIFY"
    exit 1
fi
log "preflight: dataset ok ($TOTAL rows, md5 $actual_md5)"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(count_done)
    if [ "$n" -ge "$TOTAL" ]; then
        log "all $n/$TOTAL samples done"
        break
    fi

    log "run attempt $attempt/$MAX_ATTEMPTS (currently $n/$TOTAL)"
    $PYTHON main.py \
        --data "$ENRICHED" \
        --no-verify-enriched \
        --target-model "$TARGET_MODEL" \
        --judge-model judge-model \
        --classifier-model judge-model \
        --draft-model draft-model \
        --plausibility-model draft-model \
        --embedding-model BAAI/bge-m3 \
        --device cuda \
        --out "$RESULTS" \
        --bandit-state "$RUN_DIR/bandit_posterior.json" \
        --llm-seed "$LLM_SEED" \
        --workers "$WORKERS" \
        --resume \
        -v
    rc=$?
    n=$(count_done)
    log "attempt $attempt exited rc=$rc, now $n/$TOTAL"

    if [ "$n" -ge "$TOTAL" ]; then
        break
    fi
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        log "retrying in 20s"
        sleep 20
    fi
done

n=$(count_done)
if [ "$n" -lt "$TOTAL" ]; then
    finish "INCOMPLETE_$n/$TOTAL"
    exit 1
fi

log "=== v3.15 qwen37plus run complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
