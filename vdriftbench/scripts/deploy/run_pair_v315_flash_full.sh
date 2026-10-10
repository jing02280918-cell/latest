#!/usr/bin/env bash
# PAIR on v3.15 eval357 cohort, target = deepseek-v4-flash (DeepSeek API).
#
# Prerequisites:
#   bash scripts/deploy/serve_draft.sh   # :8001
#   bash scripts/deploy/serve_judge.sh   # :8002
#   DEEPSEEK_API_KEY in .env
#
# Usage:
#   setsid nohup bash scripts/deploy/run_pair_v315_flash_full.sh \
#       > /root/pair_baseline_v315_flash_run/driver.log 2>&1 < /dev/null &

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
PAIR_SCRIPT="/root/run_pair_baseline.py"
RUN_DIR="${RUN_DIR:-/root/pair_baseline_v315_flash_run}"
DATA="${DATA:-data/v3.15_eval357.enriched.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
RESULTS="$RUN_DIR/results.jsonl"
STATUS="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$RUN_DIR"
touch "$RESULTS"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$STATUS"; log "SENTINEL: $1"; }
count_done() { wc -l < "$RESULTS" 2>/dev/null | tr -d ' '; }

log "=== PAIR v3.15 eval357 (target=$TARGET_MODEL) starting (pid $$) ==="

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
        log "FATAL: no vLLM on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done

if ! grep -qE '^DEEPSEEK_API_KEY=.+' .env 2>/dev/null; then
    log "FATAL: DEEPSEEK_API_KEY empty in .env"
    finish "FAILED_PREFLIGHT_NO_DEEPSEEK_KEY"
    exit 1
fi

if ! $PYTHON -c "
import sys
sys.path.insert(0, '.')
from vdriftbench.env_loader import load_dotenv_if_present
load_dotenv_if_present('.env')
from vdriftbench.llm_client import build_llm_client
c = build_llm_client('$TARGET_MODEL', mock=False, persona='target')
r = c.chat([{'role': 'user', 'content': '回复OK'}])
sys.exit(0 if (r or '').strip() else 1)
" >/dev/null 2>&1; then
    log "FATAL: target $TARGET_MODEL not reachable"
    finish "FAILED_PREFLIGHT_TARGET"
    exit 1
fi

if ! $PYTHON -c "
from vdriftbench.enrich import load_enriched_jsonl
n = len(load_enriched_jsonl('$DATA', verify_against_raw=False))
assert n == $TOTAL, n
"; then
    log "FATAL: dataset $DATA != $TOTAL rows"
    finish "FAILED_DATASET"
    exit 1
fi
log "preflight ok: dataset $TOTAL rows, target responds"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(count_done)
    if [ "$n" -ge "$TOTAL" ]; then
        break
    fi
    log "PAIR attempt $attempt/$MAX_ATTEMPTS ($n/$TOTAL)"
    $PYTHON "$PAIR_SCRIPT" \
        --data "$DATA" \
        --out-dir "$RUN_DIR" \
        --target-model "$TARGET_MODEL" \
        --attack-model draft-model \
        --judge-model judge-model \
        --plausibility-model draft-model \
        --llm-seed "$LLM_SEED" \
        --n-iterations 20 \
        --resume \
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
log "=== PAIR flash complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
