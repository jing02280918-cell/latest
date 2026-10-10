#!/usr/bin/env bash
# v3.15 PAIR delta: only 71 rewritten + new prompts, then merge with v3.14 PAIR.
#
# Usage (background):
#   cd /root/vdriftbench
#   setsid nohup bash scripts/deploy/run_pair_v315_delta.sh > /root/pair_baseline_v315_delta_run/driver.log 2>&1 < /dev/null &
#
# Monitor:
#   tail -f /root/pair_baseline_v315_delta_run/driver.log
#   wc -l /root/pair_baseline_v315_delta_run/results.jsonl

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
PAIR_SCRIPT="/root/run_pair_baseline.py"
LLM_SEED="${LLM_SEED:-20260904}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"

RUN_DIR="/root/pair_baseline_v315_delta_run"
MERGED_DIR="/root/pair_baseline_v315_run"
DELTA_ENRICHED="data/v3.15_delta_run/delta71.enriched.jsonl"
DELTA_IDS="data/v3.15_delta_run/delta71_ids.txt"
RESULTS="$RUN_DIR/results.jsonl"
LOG="$RUN_DIR/driver.log"
STATUS="$RUN_DIR/DRIVER_STATUS"

mkdir -p "$RUN_DIR" "$MERGED_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

finish() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; }

log "=== PAIR v3.15 delta driver starting (pid $$) ==="

# preflight vLLM + API
for port in 8001 8002; do
    curl -s -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null || {
        log "FATAL: vLLM :$port not responding"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    }
done
grep -qE '^DEEPSEEK_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null || {
    log "FATAL: DEEPSEEK_API_KEY missing"
    finish "FAILED_PREFLIGHT_NO_API_KEY"
    exit 1
}

if [ ! -f "$DELTA_ENRICHED" ]; then
    log "FATAL: missing $DELTA_ENRICHED (run v3.15 delta driver first to build subset)"
    finish "FAILED_NO_DELTA_SUBSET"
    exit 1
fi

TOTAL=$(wc -l < "$DELTA_IDS")
log "PAIR delta samples: $TOTAL"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$([ -f "$RESULTS" ] && wc -l < "$RESULTS" || echo 0)
    if [ "$n" -ge "$TOTAL" ]; then
        log "PAIR delta complete: $n/$TOTAL"
        break
    fi
    log "PAIR attempt $attempt/$MAX_ATTEMPTS ($n/$TOTAL done)"
    $PYTHON "$PAIR_SCRIPT" \
        --data "$DELTA_ENRICHED" \
        --out-dir "$RUN_DIR" \
        --target-model deepseek-v4-pro \
        --attack-model draft-model \
        --judge-model judge-model \
        --plausibility-model draft-model \
        --llm-seed "$LLM_SEED" \
        --resume \
        -v >> "$LOG" 2>&1 || true
    n=$([ -f "$RESULTS" ] && wc -l < "$RESULTS" || echo 0)
    log "after PAIR attempt $attempt: $n/$TOTAL"
    [ "$n" -ge "$TOTAL" ] && break
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep 20
done

n=$([ -f "$RESULTS" ] && wc -l < "$RESULTS" || echo 0)
if [ "$n" -lt "$TOTAL" ]; then
    log "INCOMPLETE PAIR delta: $n/$TOTAL"
    finish "INCOMPLETE_$n/$TOTAL"
    exit 1
fi

log "merging v3.14 PAIR (289 unchanged) + delta (71) -> $MERGED_DIR/results.jsonl"
$PYTHON scripts/merge_pair_v314_v315_results.py \
    --v314 /root/pair_baseline_v314_run/results.jsonl \
    --delta "$RESULTS" \
    --out "$MERGED_DIR/results.jsonl"

finish "OK_merged_360"
log "=== PAIR v3.15 merge complete: 360 rows in $MERGED_DIR/results.jsonl ==="
