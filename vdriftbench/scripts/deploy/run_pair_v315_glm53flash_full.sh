#!/usr/bin/env bash
# PAIR on v3.15 eval357 cohort, target = ZHIPU/GLM-5.3-Flash (Bailian / DashScope).
#
# Prerequisites:
#   bash /root/vdriftbench/scripts/deploy/serve_draft.sh   # :8001
#   bash /root/vdriftbench/scripts/deploy/serve_judge.sh   # :8002
#   DASHSCOPE_API_KEY in /root/vdriftbench/.env
#   GLM (智谱) product activated on Model Studio for this API key
#
# Usage:
#   setsid nohup bash /root/vdriftbench/scripts/deploy/run_pair_v315_glm53flash_full.sh \
#       > /root/pair_baseline_v315_glm53flash_run/driver.log 2>&1 < /dev/null &

set -uo pipefail

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
PAIR_SCRIPT="/root/run_pair_baseline.py"
RUN_DIR="${RUN_DIR:-/root/pair_baseline_v315_glm53flash_run}"
DATA="${DATA:-/root/vdriftbench/data/v3.15_eval357.enriched.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-ZHIPU/GLM-5.3-Flash}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
RESULTS="$RUN_DIR/results.jsonl"
STATUS="$RUN_DIR/DRIVER_STATUS"

export VDB_TARGET_BASE_URL="${VDB_TARGET_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"

mkdir -p "$RUN_DIR"
touch "$RESULTS"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$STATUS"; log "SENTINEL: $1"; }
count_done() { wc -l < "$RESULTS" 2>/dev/null | tr -d ' '; }

log "=== PAIR v3.15 eval357 (target=$TARGET_MODEL, DashScope) starting (pid $$) ==="

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
        log "FATAL: no vLLM on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done

if ! grep -qE '^DASHSCOPE_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null; then
    log "FATAL: DASHSCOPE_API_KEY empty in .env"
    finish "FAILED_PREFLIGHT_NO_DASHSCOPE_KEY"
    exit 1
fi

if [ "${SKIP_PAIR_TARGET_PREFLIGHT:-0}" != "1" ]; then
    if ! $PYTHON -c "
import sys
sys.path.insert(0, '/root/vdriftbench')
from vdriftbench.env_loader import load_dotenv_if_present
load_dotenv_if_present('/root/vdriftbench/.env')
from vdriftbench.llm_client import build_llm_client
c = build_llm_client('$TARGET_MODEL', mock=False, persona='target')
r = c.chat([{'role': 'user', 'content': '回复OK'}])
sys.exit(0 if (r or '').strip() else 1)
" 2>"$RUN_DIR/preflight_target.err"; then
        log "FATAL: target $TARGET_MODEL not reachable (see $RUN_DIR/preflight_target.err)"
        finish "FAILED_PREFLIGHT_TARGET"
        exit 1
    fi
else
    log "WARN: SKIP_PAIR_TARGET_PREFLIGHT=1 — skipping target API check"
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
log "=== PAIR GLM-5.3-Flash complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
