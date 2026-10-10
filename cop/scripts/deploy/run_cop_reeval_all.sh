#!/usr/bin/env bash
# Re-evaluate all CoP v3.15 runs with the v3.15 judge (offline, reads CSV only).
#
# Usage:
#   cd /root/cop
#   setsid nohup bash scripts/deploy/run_cop_reeval_all.sh \
#       > outputs/reeval_all.log 2>&1 < /dev/null &

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
DATASET="${DATASET:-/root/vdriftbench/data/v3.15_eval357.enriched.jsonl}"
LOG_DIR="${LOG_DIR:-/root/cop/outputs}"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%F %T')] $*"; }

preflight() {
    for port in 8002; do
        if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            log "FATAL: no vLLM judge on :$port"
            exit 1
        fi
    done
    log "preflight: judge :8002 ok"
}

run_one() {
    local name="$1"
    local results_dir="$2"
    local out="$3"
    log "=== re-eval start: $name ==="
    log "results_dir=$results_dir"
    $PYTHON evaluate.py \
        --results-dir "$results_dir" \
        --dataset-path "$DATASET" \
        --out "$out"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        log "FAILED re-eval $name rc=$rc"
        return "$rc"
    fi
    $PYTHON -c "import json; s=json.load(open('$out')); print(json.dumps({'run':'$name','total':s['total'],'successes':s['successes'],'asr':round(s['asr']*100,2)}, ensure_ascii=False))"
    log "=== re-eval done: $name -> $out ==="
}

preflight

run_one "deepseek-v4-pro" \
    "$LOG_DIR/v315/deepseek-v4-pro_vdb_draft-model" \
    "$LOG_DIR/v315/eval_summary.json"

run_one "deepseek-v4-flash" \
    "$LOG_DIR/v315_flash/deepseek-v4-flash_vdb_draft-model" \
    "$LOG_DIR/v315_flash/eval_summary.json"

run_one "qwen3.7-plus" \
    "$LOG_DIR/v315_qwen37plus/qwen3.7-plus_vdb_draft-model" \
    "$LOG_DIR/v315_qwen37plus/eval_summary.json"

log "=== all CoP re-evals complete ==="
