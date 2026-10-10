#!/usr/bin/env bash
# PLAGUE v3.15 on eval357 cohort (real-world 35 patch), target = GLM-5.3-Flash (DashScope / ZHIPU).
#
# Prerequisites:
#   bash /root/vdriftbench/scripts/deploy/serve_draft.sh   # :8001
#   bash /root/vdriftbench/scripts/deploy/serve_judge.sh   # :8002
#   DASHSCOPE_API_KEY in /root/vdriftbench/.env
#
# Usage:
#   cd /root/PLAGUE
#   setsid nohup bash scripts/deploy/run_plague_v315_glm53flash_full.sh \
#       > outputs/v315_glm53flash/driver.log 2>&1 < /dev/null &
#
# Monitor:
#   tail -f /root/PLAGUE/outputs/v315_glm53flash/driver.log
#   ls outputs/v315_glm53flash/glm-5.3-flash_vdb_draft-model/*_jb_*.csv | wc -l

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-/root/PLAGUE/outputs/v315_glm53flash}"
TARGET_MODEL="${TARGET_MODEL:-glm-5.3-flash}"
RESULTS_SUBDIR="${TARGET_MODEL}_vdb_draft-model"
RESULTS_DIR="$RUN_DIR/$RESULTS_SUBDIR"
DATASET="${DATASET:-/root/vdriftbench/data/v3.15_eval357.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-e4748ddfa674710c97781bf974a8329f}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-2048}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VDB_TARGET_BASE_URL="${VDB_TARGET_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VDB_GLM_REASONING_EFFORT="${VDB_GLM_REASONING_EFFORT:-low}"
export VDB_TARGET_TIMEOUT_S="${VDB_TARGET_TIMEOUT_S:-300}"
export VDB_OUTPUT_DIR="$RUN_DIR"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR" "$RESULTS_DIR"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }
count_done() { ls "$RESULTS_DIR"/*_jb_*.csv 2>/dev/null | wc -l; }

log "=== PLAGUE v3.15 GLM-5.3-Flash driver starting (pid $$) ==="
log "config: target=$TARGET_MODEL total=$TOTAL target_max_tokens=$TARGET_MAX_TOKENS run_dir=$RUN_DIR"

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
log "preflight: vLLM :8001 and :8002 both responding"

if ! grep -qE '^DASHSCOPE_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null; then
    log "FATAL: DASHSCOPE_API_KEY empty"
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
    finish "FAILED_PREFLIGHT_TARGET"
    exit 1
fi
log "preflight: GLM-5.3-Flash responds"

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
        --target-max-tokens "$TARGET_MAX_TOKENS" \
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

log "evaluating PLAGUE GLM results with v3.15 judge"
PYTHONPATH="/root/cop:$RUN_DIR/.." $PYTHON /root/cop/evaluate.py \
    --results-dir "$RESULTS_DIR" \
    --dataset-path "$DATASET" \
    --out "$RUN_DIR/eval_summary.json"

log "=== PLAGUE v3.15 GLM-5.3-Flash complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
