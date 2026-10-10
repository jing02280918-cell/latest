#!/usr/bin/env bash
# Starts the local draft (O-T-R free-generation) model with vLLM, per
# HANDOFF.md 4.1 / section 6 step 0a-2. Occupies ~0.45 of the single GPU's
# memory (~21GB on a 48GB card).
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$HF_HOME/hub"
for _repo in "$HF_HOME"/models--*; do
    [ -d "$_repo" ] || continue
    ln -sfn "$_repo" "$HF_HOME/hub/$(basename "$_repo")"
done

DRAFT_MODEL="${DRAFT_MODEL:-ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ}"
DRAFT_LOCAL="${DRAFT_LOCAL:-$HF_HOME/models--ibrahimkettaneh--Qwen2.5-32B-Instruct-abliterated-pass2-AWQ/snapshots/725e146001172f54f842e6ff179cc5cd9dbb7a6c}"
if [ -f "$DRAFT_LOCAL/config.json" ]; then
    DRAFT_MODEL="$DRAFT_LOCAL"
fi
PORT="${DRAFT_PORT:-8001}"
GPU_UTIL="${DRAFT_GPU_UTIL:-0.48}"
# Model's native max_seq_len (32768) needs ~8GiB of KV cache (32B model, more
# layers/heads than the judge), which doesn't fit alongside the judge
# model's share of the GPU; 8192 covers a long multi-turn draft conversation
# and needs only ~2GiB.
MAX_MODEL_LEN="${DRAFT_MAX_MODEL_LEN:-8192}"

# --enforce-eager skips torch.compile / CUDA-graph capture (very slow /
# unreliable to warm up on first run in this environment) — fine for the
# draft/free-generation workload here.
#
# v3.14.1: 同 serve_judge.sh，做成开关但默认不变（`ENFORCE_EAGER=0` 打开 CUDA graph）。
# 起草模型每轮要生成几百 token 的 O-T-R-F 话术，是这两块模型里 decode 占比更高的一块，
# 所以真要试 CUDA graph，先从这一块试收益更大。
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
eager_flag=()
[ "$ENFORCE_EAGER" = "1" ] && eager_flag=(--enforce-eager)

source /hy-tmp/venv/bin/activate

exec vllm serve "$DRAFT_MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --quantization awq \
    --served-model-name draft-model \
    "${eager_flag[@]}" \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --download-dir "$HF_HOME"
