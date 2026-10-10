#!/usr/bin/env bash
# Crescendo on v3.15.1 eval357 cohort, target = deepseek-v4-flash.
# Attacker: draft-model (:8001)  Judge: judge-model (:8002) via v3.15 rubric.
#
# Prerequisites:
#   bash /root/vdriftbench/scripts/deploy/serve_draft.sh
#   bash /root/vdriftbench/scripts/deploy/serve_judge.sh
#   DEEPSEEK_API_KEY in /root/vdriftbench/.env
#
# Usage:
#   setsid nohup bash /root/crescendo_baseline_v315_flash/scripts/deploy/run_crescendo_v315_flash_full.sh \
#       > /root/crescendo_baseline_v315_flash/driver.log 2>&1 < /dev/null &

set -uo pipefail

ROOT="/root/crescendo_baseline_v315_flash"
PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT}"
DATASET="${DATASET:-/root/vdriftbench/data/v3.15_eval357.enriched.jsonl}"
ENRICHED_MD5="${ENRICHED_MD5:-e4748ddfa674710c97781bf974a8329f}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"
TOTAL="${TOTAL:-357}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
MAX_TURNS="${CRESCENDO_MAX_TURNS:-10}"
MAX_BACKTRACKS="${CRESCENDO_MAX_BACKTRACKS:-10}"
RESULTS="$RUN_DIR/results.jsonl"
STATUS="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VDB_TARGET_MODEL="$TARGET_MODEL"
export VDB_LLM_SEED="$LLM_SEED"
export VDB_OUTPUT_DIR="$RUN_DIR"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR"
touch "$RESULTS"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$STATUS"; log "SENTINEL: $1"; }
count_done() { wc -l < "$RESULTS" 2>/dev/null | tr -d ' '; }

log "=== Crescendo v3.15.1 driver starting (pid $$) ==="
log "config: target=$TARGET_MODEL attacker=draft-model judge=judge-model turns=$MAX_TURNS early_stop=genre_delivery draft_ctx=1 run_dir=$RUN_DIR"

for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
        log "FATAL: no vLLM on :$port"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done

if ! grep -qE '^DEEPSEEK_API_KEY=.+' /root/vdriftbench/.env 2>/dev/null; then
    log "FATAL: DEEPSEEK_API_KEY empty"
    finish "FAILED_PREFLIGHT_NO_DEEPSEEK_KEY"
    exit 1
fi

if ! $PYTHON -c "import pyrit" 2>/dev/null; then
    log "installing PyRIT editable package"
    $PYTHON -m pip install -e /root/PyRIT -q || { finish "FAILED_PYRIT_INSTALL"; exit 1; }
fi

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
    $PYTHON "$ROOT/run_crescendo_v315.py" \
        --dataset-path "$DATASET" \
        --out-dir "$RUN_DIR" \
        --max-turns "$MAX_TURNS" \
        --max-backtracks "$MAX_BACKTRACKS" \
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

log "summarizing Crescendo jsonl results"
$PYTHON - <<'PY'
import json, statistics, sys
from pathlib import Path
p = Path("$RESULTS")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
succ = sum(1 for r in rows if r.get("best_outcome") == "成功")
risks = [r.get("best_risk_score", 0) for r in rows]
summary = {
    "total": len(rows),
    "successes": succ,
    "asr": succ / len(rows) if rows else 0,
    "mean_risk": round(statistics.mean(risks), 2) if risks else 0,
}
out = Path("$RUN_DIR/eval_summary.json")
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY

log "=== Crescendo v3.15.1 complete ($TARGET_MODEL): $n/$TOTAL ==="
finish "OK_$n/$TOTAL"
