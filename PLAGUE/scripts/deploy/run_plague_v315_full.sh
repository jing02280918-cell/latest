#!/usr/bin/env bash
# PLAGUE v3.15 full-run driver: 360-sample attack aligned with CoP / V-Drift v3.15 setup.
#
# Prerequisites:
#   bash /root/vdriftbench/scripts/deploy/serve_draft.sh   # :8001 draft-model
#   bash /root/vdriftbench/scripts/deploy/serve_judge.sh   # :8002 judge-model
#   DEEPSEEK_API_KEY in /root/vdriftbench/.env
#   Uses /hy-tmp/venv — same env as CoP v3.15 / PAIR.
#   Do NOT pip install -r requirements.txt; see requirements-v315.txt.
#   Missing deps are installed on demand by scripts/ensure_v315_deps.py.
#
# Usage (background):
#   cd /root/PLAGUE
#   setsid nohup bash scripts/deploy/run_plague_v315_full.sh > outputs/v315/driver.log 2>&1 < /dev/null &
#
# Monitor:
#   tail -f /root/PLAGUE/outputs/v315/driver.log
#   ls outputs/v315/deepseek-v4-pro_vdb_draft-model/*_jb_*.csv | wc -l

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-/root/PLAGUE/outputs/v315}"
RESULTS_SUBDIR="deepseek-v4-pro_vdb_draft-model"
RESULTS_DIR="$RUN_DIR/$RESULTS_SUBDIR"
DATASET="${DATASET:-/root/vdriftbench/data/dataset_100.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-08d1bfa2296c6d907e78e0f18f243292}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-pro}"
ATTACK_MODEL="${ATTACK_MODEL:-draft-model}"
JUDGE_MODEL="${JUDGE_MODEL:-judge-model}"
LLM_SEED="${LLM_SEED:-20260904}"
TOTAL="${TOTAL:-360}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR" "$RESULTS_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }
count_done() { ls "$RESULTS_DIR"/*_jb_*.csv 2>/dev/null | wc -l; }

log "=== PLAGUE v3.15 full-run driver starting (pid $$) ==="
log "config: target=$TARGET_MODEL attack=$ATTACK_MODEL judge=$JUDGE_MODEL seed=$LLM_SEED total=$TOTAL"

log "checking v3.15 python deps (install only if missing, PAIR-style)"
if ! $PYTHON scripts/ensure_v315_deps.py; then
    finish "FAILED_DEPS"
    exit 1
fi

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        log "FATAL: no vLLM server on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done
log "preflight: vLLM :8001 and :8002 both responding"

if ! grep -qE '^DEEPSEEK_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null; then
    log "FATAL: DEEPSEEK_API_KEY is empty in /root/vdriftbench/.env"
    finish "FAILED_PREFLIGHT_NO_API_KEY"
    exit 1
fi
log "preflight: DEEPSEEK_API_KEY present"

if [ ! -f "$DATASET" ]; then
    log "FATAL: missing dataset $DATASET"
    finish "FAILED_NO_DATASET"
    exit 1
fi
actual_md5=$(md5sum "$DATASET" | cut -d' ' -f1)
if [ "$actual_md5" != "$ENRICHED_MD5" ]; then
    log "FATAL: enriched dataset fingerprint mismatch"
    log "  expected $ENRICHED_MD5"
    log "  actual   $actual_md5"
    finish "FAILED_DATASET_FINGERPRINT"
    exit 1
fi
if ! $PYTHON -c "
import json
n = sum(1 for line in open('$DATASET') if line.strip())
assert n == $TOTAL, f'{n} rows, expected $TOTAL'
"; then
    log "FATAL: dataset row count != $TOTAL"
    finish "FAILED_DATASET_COUNT"
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
    $PYTHON scripts/run_plague_v315.py \
        --target-model "$TARGET_MODEL" \
        --attacker-model "$ATTACK_MODEL" \
        --evaluator-model "$JUDGE_MODEL" \
        --summariser-model "$ATTACK_MODEL" \
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

log "evaluating PLAGUE results with v3.15 judge"
PYTHONPATH="/root/cop:$RUN_DIR/.." $PYTHON /root/cop/evaluate.py \
    --results-dir "$RESULTS_DIR" \
    --dataset-path "$DATASET" \
    --out "$RUN_DIR/eval_summary.json"

log "comparing PLAGUE vs CoP vs v3.15 baselines"
$PYTHON scripts/compare_plague_v315.py \
    --results-dir "$RESULTS_DIR" \
    --dataset-path "$DATASET" \
    --out "$RUN_DIR/compare_v315.json"

log "=== PLAGUE v3.15 run complete: $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
