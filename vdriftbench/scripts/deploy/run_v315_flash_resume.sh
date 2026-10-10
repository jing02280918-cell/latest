#!/usr/bin/env bash
# Continue / rerun v3.15 flash without wiping results.jsonl.
# Use after prepare_v315_flash_rerun_failures.py removed R5-failed rows.
#
# Usage:
#   setsid nohup bash scripts/deploy/run_v315_flash_resume.sh >> data/v3.15_flash_run/driver.log 2>&1 < /dev/null &

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-data/v3.15_flash_run}"
RESULTS="$RUN_DIR/results.jsonl"
ENRICHED="${ENRICHED:-data/v3.15_eval357.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-68cc8f3d75c55a1939aa967a11a9e448}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
WORKERS="${WORKERS:-3}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$RUN_DIR"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }
count_done() { wc -l < "$RESULTS" 2>/dev/null | tr -d ' '; }

log "=== v3.15 flash RESUME (target=$TARGET_MODEL, pair-style-r5, early-stop ON, r5-retry-all-failures) pid $$ ==="

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        log "FATAL: no vLLM server on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done

if ! grep -qE '^DEEPSEEK_API_KEY=.+' .env 2>/dev/null; then
    finish "FAILED_PREFLIGHT_NO_DEEPSEEK_KEY"
    exit 1
fi

actual_md5=$(md5sum "$ENRICHED" | cut -d' ' -f1)
if [ "$actual_md5" != "$ENRICHED_MD5" ]; then
    finish "FAILED_DATASET_FINGERPRINT"
    exit 1
fi

n_start=$(count_done)
log "resume from $n_start/$TOTAL rows in $RESULTS"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(count_done)
    if [ "$n" -ge "$TOTAL" ]; then
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
        --r5-retry-all-failures \
        -v
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

finish "OK_$n/$TOTAL"
log "=== v3.15 flash resume complete: $n/$TOTAL ==="
