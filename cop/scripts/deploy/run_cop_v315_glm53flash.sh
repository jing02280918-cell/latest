#!/usr/bin/env bash
# CoP v3.15.1 eval357 — first 50 samples, target = GLM-5.3-Flash (DashScope).
#
# Usage:
#   setsid nohup bash /root/cop/scripts/deploy/run_cop_v315_glm53flash/measured.sh \
#       > /root/cop/outputs/v315_glm53flash/measured/driver.log 2>&1 < /dev/null &

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-/root/cop/outputs/v315_glm53flash/measured}"
TARGET_MODEL="${TARGET_MODEL:-glm-5.3-flash}"
RESULTS_SUBDIR="${TARGET_MODEL}_vdb_draft-model"
RESULTS_DIR="$RUN_DIR/$RESULTS_SUBDIR"
DATASET="${DATASET:-/root/vdriftbench/data/v3.15_eval357.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-e4748ddfa674710c97781bf974a8329f}"
ATTACK_MODEL="${ATTACK_MODEL:-draft-model}"
JUDGE_MODEL="${JUDGE_MODEL:-vdb-judge}"
LLM_SEED="${LLM_SEED:-20260904}"
START="${START:-0}"
LIMIT="${LIMIT:-50}"
TOTAL="${TOTAL:-50}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
MAX_SAMPLE_QUERIES="${MAX_SAMPLE_QUERIES:-20}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export VDB_TARGET_MODEL="$TARGET_MODEL"
export VDB_TARGET_BASE_URL="${VDB_TARGET_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"
export VDB_TARGET_TIMEOUT_S="${VDB_TARGET_TIMEOUT_S:-300}"
export VDB_MAX_SAMPLE_QUERIES="$MAX_SAMPLE_QUERIES"
export VDB_OUTPUT_DIR="$RUN_DIR"

mkdir -p "$RUN_DIR" "$RESULTS_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }
count_done() { ls "$RESULTS_DIR"/*_jb_*.csv 2>/dev/null | wc -l; }

log "=== CoP v3.15 GLM top50 driver starting (pid $$) ==="
log "config: target=$TARGET_MODEL start=$START limit=$LIMIT total=$TOTAL run_dir=$RUN_DIR"

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        log "FATAL: no vLLM server on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done

if ! grep -qE '^DASHSCOPE_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null; then
    log "FATAL: DASHSCOPE_API_KEY empty in /root/vdriftbench/.env"
    finish "FAILED_PREFLIGHT_NO_DASHSCOPE_KEY"
    exit 1
fi

if ! $PYTHON -c "
import sys
sys.path.insert(0, '/root/vdriftbench')
from vdriftbench.env_loader import load_dotenv_if_present
load_dotenv_if_present('/root/vdriftbench/.env')
from vdriftbench.llm_client import build_llm_client
c = build_llm_client('ZHIPU/GLM-5.3-Flash', mock=False, persona='target')
r = c.chat([{'role': 'user', 'content': '回复OK'}])
sys.exit(0 if (r or '').strip() else 1)
" >"$RUN_DIR/preflight_target.log" 2>&1; then
    log "FATAL: GLM-5.3-Flash not reachable (see $RUN_DIR/preflight_target.log)"
    finish "FAILED_PREFLIGHT_TARGET_MODEL"
    exit 1
fi
log "preflight: GLM-5.3-Flash responds"

if [ ! -f "$DATASET" ]; then
    log "FATAL: missing dataset $DATASET"
    finish "FAILED_NO_DATASET"
    exit 1
fi
actual_md5=$(md5sum "$DATASET" | cut -d' ' -f1)
if [ "$actual_md5" != "$ENRICHED_MD5" ]; then
    log "FATAL: enriched dataset fingerprint mismatch (expected $ENRICHED_MD5, got $actual_md5)"
    finish "FAILED_DATASET_FINGERPRINT"
    exit 1
fi
log "preflight: dataset ok (slice start=$START limit=$LIMIT from eval357)"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(count_done)
    if [ "$n" -ge "$TOTAL" ]; then
        log "all $n/$TOTAL samples done"
        break
    fi

    log "run attempt $attempt/$MAX_ATTEMPTS (currently $n/$TOTAL)"
    $PYTHON main.py \
        --dataset-path "$DATASET" \
        --output-dir "$RUN_DIR" \
        --target-model "$TARGET_MODEL" \
        --attack-model "$ATTACK_MODEL" \
        --helper-model "$ATTACK_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --llm-seed "$LLM_SEED" \
        --helper-max-n-tokens "${HELPER_MAX_TOKENS:-1024}" \
        --target-max-n-tokens "${TARGET_MAX_TOKENS:-4096}" \
        --max-sample-queries "$MAX_SAMPLE_QUERIES" \
        --start "$START" \
        --limit "$LIMIT" \
        --skip-baseline-probe \
        --resume
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
    log "run did not reach $TOTAL after $MAX_ATTEMPTS attempts (stopped at $n)"
    finish "INCOMPLETE_$n/$TOTAL"
    exit 1
fi

log "evaluating CoP GLM with v3.15 judge"
$PYTHON evaluate.py \
    --results-dir "$RESULTS_DIR" \
    --dataset-path "$DATASET" \
    --out "$RUN_DIR/eval_summary.json"

log "=== CoP GLM complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
