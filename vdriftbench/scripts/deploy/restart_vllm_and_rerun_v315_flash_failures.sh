#!/usr/bin/env bash
# Restart vLLM and resume v3.15 flash (re-run R5-failed + remaining samples).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

LOG="/hy-tmp/restart_v315_flash_rerun.log"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== prepare: drop R5-failed rows from results.jsonl ==="
/hy-tmp/venv/bin/python scripts/prepare_v315_flash_rerun_failures.py

log "=== killing stale vLLM / EngineCore ==="
pkill -9 -f "run_v315_flash" 2>/dev/null || true
pkill -9 -f "main.py" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 3
# EngineCore can outlive the parent vLLM process; kill by PID if GPU still occupied.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read -r pid; do
            [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
        done
fi
sleep 5

log "=== starting judge (:8002) ==="
: > logs/serve_judge.log
nohup bash scripts/deploy/serve_judge.sh >> logs/serve_judge.log 2>&1 &
for i in $(seq 1 120); do
    curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q judge-model && break
    sleep 5
done
if ! curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q judge-model; then
    echo "FAILED_JUDGE $(date '+%F %T')" > data/v3.15_flash_run/DRIVER_STATUS
    exit 1
fi

log "=== starting draft (:8001) ==="
: > logs/serve_draft.log
nohup bash scripts/deploy/serve_draft.sh >> logs/serve_draft.log 2>&1 &
for i in $(seq 1 120); do
    curl -s -m 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q draft-model \
    && curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q judge-model && break
    sleep 5
done
if ! curl -s -m 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q draft-model; then
    echo "FAILED_VLLM $(date '+%F %T')" > data/v3.15_flash_run/DRIVER_STATUS
    exit 1
fi

log "=== launching resume driver (target=$TARGET_MODEL) ==="
export TARGET_MODEL
exec bash scripts/deploy/run_v315_flash_resume.sh >> "$LOG" 2>&1
