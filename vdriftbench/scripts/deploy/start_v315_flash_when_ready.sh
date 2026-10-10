#!/usr/bin/env bash
# Wait for model weights, then start vLLM + v3.15 full run (target=deepseek-v4-flash).
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

LOG="${LOG:-/hy-tmp/start_v315_flash.log}"
DOWNLOAD_LOG="${DOWNLOAD_LOG:-/hy-tmp/model_download.log}"
HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-flash}"

mkdir -p data/v3.15_flash_run logs
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

models_ready() {
    local draft="$HF_HOME/models--ibrahimkettaneh--Qwen2.5-32B-Instruct-abliterated-pass2-AWQ"
    local judge="$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3"
    local embed="$HF_HOME/models--BAAI--bge-m3"
    [ -d "$draft/snapshots" ] && [ -d "$judge/snapshots" ] && [ -d "$embed/snapshots" ]
}

log "=== waiting for model download to finish ==="
while pgrep -f "scripts/deploy/download_models.py" >/dev/null 2>&1; do
    du -sh "$HF_HOME" 2>/dev/null | awk -v t="$HF_HOME" '{print strftime("%F %T"), "cache:", $1, t}' >> "$LOG"
    sleep 60
done

if grep -q "All requested models downloaded." "$DOWNLOAD_LOG" 2>/dev/null; then
    log "download script reported success"
elif models_ready; then
    log "all three model snapshots present"
else
    log "FATAL: download finished but models incomplete"
    echo "FAILED_MODELS $(date '+%F %T')" > data/v3.15_flash_run/DRIVER_STATUS
    exit 1
fi

log "=== starting vLLM draft (:8001) and judge (:8002) ==="
if ! curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    nohup bash scripts/deploy/serve_draft.sh > logs/serve_draft.log 2>&1 &
fi
if ! curl -s -m 3 http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
    nohup bash scripts/deploy/serve_judge.sh > logs/serve_judge.log 2>&1 &
fi

vllm_has_model() {
    local port="$1" name="$2"
    curl -s -m 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"id\":\"$name\""
}

for i in $(seq 1 180); do
    ok=1
    vllm_has_model 8001 draft-model || ok=0
    vllm_has_model 8002 judge-model || ok=0
    if [ "$ok" = 1 ]; then
        log "vLLM ready after ${i}*10s"
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        log "still waiting for vLLM (${i}0s)..."
        tail -2 logs/serve_draft.log 2>/dev/null >> "$LOG" || true
        tail -2 logs/serve_judge.log 2>/dev/null >> "$LOG" || true
    fi
    sleep 10
done

if ! vllm_has_model 8001 draft-model || ! vllm_has_model 8002 judge-model; then
    log "FATAL: vLLM not ready after 30min"
    echo "FAILED_VLLM $(date '+%F %T')" > data/v3.15_flash_run/DRIVER_STATUS
    exit 1
fi

log "=== launching v3.15 full run (target=$TARGET_MODEL) ==="
export HF_HOME HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TARGET_MODEL
exec bash scripts/deploy/run_v315_flash_full.sh >> "$LOG" 2>&1
