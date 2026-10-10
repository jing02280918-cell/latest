#!/usr/bin/env bash
# Kill stale vLLM, restart judge->draft, then launch v3.15 flash (deepseek-v4-flash).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

LOG="/hy-tmp/restart_v315_flash.log"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== killing stale vLLM / EngineCore ==="
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 5
# Orphan EngineCore can hold VRAM without showing in nvidia-smi process list.
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    if ls -l "/proc/$pid/fd" 2>/dev/null | grep -q nvidia; then
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        case "$cmd" in *VLLM::*|*vllm*) kill -9 "$pid" 2>/dev/null || true ;; esac
    fi
done
sleep 3
log "GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

log "=== starting judge (:8002) ==="
: > logs/serve_judge.log
nohup bash scripts/deploy/serve_judge.sh >> logs/serve_judge.log 2>&1 &
for i in $(seq 1 120); do
    if curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q '"id":"judge-model"'; then
        log "judge ready after ${i}*5s"
        break
    fi
    if [ $((i % 12)) -eq 0 ]; then
        log "waiting judge (${i}*5s) GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
        tail -1 logs/serve_judge.log >> "$LOG" 2>/dev/null || true
    fi
    sleep 5
done
if ! curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q judge-model; then
    log "FATAL: judge failed to start"
    tail -20 logs/serve_judge.log >> "$LOG"
    echo "FAILED_JUDGE $(date '+%F %T')" > data/v3.15_flash_run/DRIVER_STATUS
    exit 1
fi

log "=== starting draft (:8001) ==="
: > logs/serve_draft.log
nohup bash scripts/deploy/serve_draft.sh >> logs/serve_draft.log 2>&1 &
for i in $(seq 1 120); do
    ok=1
    curl -s -m 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q draft-model || ok=0
    curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q judge-model || ok=0
    if [ "$ok" = 1 ]; then
        log "draft+judge ready after ${i}*5s"
        break
    fi
    if [ $((i % 12)) -eq 0 ]; then
        log "waiting draft (${i}*5s) GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
        tail -1 logs/serve_draft.log >> "$LOG" 2>/dev/null || true
    fi
    sleep 5
done
if ! curl -s -m 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q draft-model \
    || ! curl -s -m 5 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q judge-model; then
    log "FATAL: draft/judge pair failed"
    tail -20 logs/serve_draft.log >> "$LOG"
    echo "FAILED_VLLM $(date '+%F %T')" > data/v3.15_flash_run/DRIVER_STATUS
    exit 1
fi

log "=== launching v3.15 flash (target=$TARGET_MODEL) ==="
export TARGET_MODEL
exec bash scripts/deploy/run_v315_flash_full.sh >> "$LOG" 2>&1
