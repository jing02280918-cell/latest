#!/usr/bin/env bash
# v3.15 delta run: only the 71 rewritten + new prompts, then merge with v3.14.
#
# Usage (background):
#   cd /root/vdriftbench
#   setsid nohup bash scripts/deploy/run_v315_delta.sh > data/v3.15_delta_run/driver.log 2>&1 < /dev/null &
#
# Monitor:
#   tail -f data/v3.15_delta_run/driver.log
#   wc -l data/v3.15_delta_run/results.jsonl

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-pro}"
LLM_SEED="${LLM_SEED:-20260904}"
WORKERS="${WORKERS:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"

RUN_DIR="data/v3.15_delta_run"
MERGED_DIR="data/v3.15_run"
ENRICHED="data/dataset_100.enriched.jsonl"
ENRICHED_MD5="${ENRICHED_MD5:-08d1bfa2296c6d907e78e0f18f243292}"
DELTA_ENRICHED="$RUN_DIR/delta71.enriched.jsonl"
DELTA_IDS="$RUN_DIR/delta71_ids.txt"
RESULTS="$RUN_DIR/results.jsonl"
BANDIT="$RUN_DIR/bandit_posterior.json"
LOG="$RUN_DIR/driver.log"
STATUS="$RUN_DIR/DRIVER_STATUS"

mkdir -p "$RUN_DIR" "$MERGED_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

finish() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"; }

log "=== v3.15 delta driver starting (pid $$, WORKERS=$WORKERS) ==="

# preflight vLLM + API
for port in 8001 8002; do
    curl -s -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null || {
        log "FATAL: vLLM :$port not responding"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    }
done
grep -qE '^DEEPSEEK_API_KEY=.+' .env 2>/dev/null || {
    log "FATAL: DEEPSEEK_API_KEY missing"
    finish "FAILED_PREFLIGHT_NO_API_KEY"
    exit 1
}

actual_md5=$(md5sum "$ENRICHED" | cut -d' ' -f1)
if [ "$actual_md5" != "$ENRICHED_MD5" ]; then
    log "FATAL: enriched md5 mismatch (expected $ENRICHED_MD5, got $actual_md5)"
    finish "FAILED_DATASET_FINGERPRINT"
    exit 1
fi

# build 71-sample subset + id list from changelog
$PYTHON <<'PYEOF'
import json
from pathlib import Path
from vdriftbench.enrich import load_enriched_jsonl

cl = json.loads(Path("data/dataset_optimization_changelog_v315.json").read_text())
ids = sorted({c["sample_id"] for c in cl["changes"]})
all_samples = {s.sample_id: s for s in load_enriched_jsonl("data/dataset_100.enriched.jsonl", verify_against_raw=False)}
missing = [i for i in ids if i not in all_samples]
if missing:
    raise SystemExit(f"missing enriched samples: {missing}")
subset = [all_samples[i] for i in ids]
Path("data/v3.15_delta_run/delta71_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
with Path("data/v3.15_delta_run/delta71.enriched.jsonl").open("w", encoding="utf-8") as f:
    for s in subset:
        f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
print(f"delta subset: {len(subset)} samples")
PYEOF

TOTAL=$(wc -l < "$DELTA_IDS")
log "delta samples: $TOTAL"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
    if [ "$n" -ge "$TOTAL" ]; then
        log "delta run complete: $n/$TOTAL"
        break
    fi
    log "run attempt $attempt/$MAX_ATTEMPTS ($n/$TOTAL done)"
    $PYTHON main.py \
        --data "$DELTA_ENRICHED" \
        --no-verify-enriched \
        --target-model "$TARGET_MODEL" \
        --judge-model judge-model \
        --classifier-model judge-model \
        --draft-model draft-model \
        --plausibility-model draft-model \
        --embedding-model BAAI/bge-m3 \
        --device cuda \
        --out "$RESULTS" \
        --bandit-state "$BANDIT" \
        --llm-seed "$LLM_SEED" \
        --workers "$WORKERS" \
        --resume \
        -v || true
    n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
    log "after attempt $attempt: $n/$TOTAL"
    [ "$n" -ge "$TOTAL" ] && break
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep 20
done

n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
if [ "$n" -lt "$TOTAL" ]; then
    log "INCOMPLETE delta run: $n/$TOTAL"
    finish "INCOMPLETE_$n/$TOTAL"
    exit 1
fi

log "merging v3.14 (289 unchanged) + delta (71) -> $MERGED_DIR/results.jsonl"
$PYTHON scripts/merge_v314_v315_results.py \
    --v314 data/v3.14_run/results.jsonl \
    --delta "$RESULTS" \
    --out "$MERGED_DIR/results.jsonl"

# copy bandit from v3.14 as base reference (delta bandit is partial)
cp -f data/v3.14_run/bandit_posterior.json "$MERGED_DIR/bandit_posterior_v314_base.json" 2>/dev/null || true
cp -f "$BANDIT" "$MERGED_DIR/bandit_posterior_delta71.json" 2>/dev/null || true

finish "OK_merged_360"
log "=== v3.15 merge complete: 360 rows in $MERGED_DIR/results.jsonl ==="
