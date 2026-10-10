#!/usr/bin/env bash
# Starts the local judge (5-dim scoring / Layer2 classifier / 9.2 RWI review)
# model with vLLM, per HANDOFF.md 4.1 / section 6 step 0a-2. Occupies ~0.35 of
# the single GPU's memory (~16GB on a 48GB card).
#
# NOTE: HANDOFF.md originally selected meta-llama/Llama-3.1-8B-Instruct for
# this role, but that repo is HF-gated and no approved token was available at
# deployment time. This script defaults to mistralai/Mistral-7B-Instruct-v0.3
# instead (ungated, similar size, different model family from the Qwen-based
# draft model and from common remote target models). See
# scripts/deploy/download_models.py docstring for details, or override
# JUDGE_MODEL below once you have an HF_TOKEN with Llama-3.1 access.
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
# download_models.py sets this; vLLM must too or weight load falls back to XET
# (cas-server.xethub.hf.co → 401 from this network).
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Weights from download_models.py land in $HF_HOME/models--*; huggingface_hub / vLLM
# resolve via $HF_HOME/hub/models--*. Link them so startup uses local shards only.
mkdir -p "$HF_HOME/hub"
for _repo in "$HF_HOME"/models--*; do
    [ -d "$_repo" ] || continue
    ln -sfn "$_repo" "$HF_HOME/hub/$(basename "$_repo")"
done

JUDGE_MODEL="${JUDGE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
JUDGE_LOCAL="${JUDGE_LOCAL:-$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71}"
if [ -f "$JUDGE_LOCAL/config.json" ] && [ -f "$JUDGE_LOCAL/model.safetensors.index.json" ]; then
    JUDGE_MODEL="$JUDGE_LOCAL"
fi
PORT="${JUDGE_PORT:-8002}"
GPU_UTIL="${JUDGE_GPU_UTIL:-0.40}"
# Model's native max_seq_len (32768) needs ~4GiB of KV cache, which doesn't
# fit alongside the draft model's share of the GPU; 16384 is plenty for
# scoring a multi-turn conversation transcript and needs only ~2GiB.
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"

# v3.14.1: 裁判是整条流水线最贵的一步（一次 R5 打分 prompt_tokens 可到 13798，输出
# 只有 100 多 token，墙钟几乎全在 prefill 上），而 `main.py --workers N` 现在会同时
# 打进来 N 条请求。下面两个参数是配合并行用的：
#
#   --enable-prefix-caching：同一个样本连续几轮的裁判输入共享一长段前缀（系统提示 +
#     target_claim + expected_safe_behavior + 前几轮历史），命中缓存就不用重复 prefill。
#     vLLM v1 引擎默认已开，这里显式写出来是为了别在换版本时被静默关掉。
#   --max-num-batched-tokens：默认 8192 会把一条 13798 token 的裁判请求切成两段
#     chunked prefill。放到 16384 让它一趟走完，和 --max-model-len 对齐。
MAX_BATCHED_TOKENS="${JUDGE_MAX_BATCHED_TOKENS:-16384}"

# --enforce-eager skips torch.compile / CUDA-graph capture (which was very
# slow / unreliable to warm up on first run in this environment) — fine for
# judge scoring workloads where per-request latency isn't the bottleneck.
#
# v3.14.1: 保留为默认，但做成开关。CUDA graph 主要加速 decode，裁判的 decode 只有
# 100~320 个 token，收益是次要的；而这台机器上两块模型共占 0.88 的显存，graph capture
# 额外要一块显存，起不来就会把整轮续跑卡在 preflight。想试的话：
#   ENFORCE_EAGER=0 bash scripts/deploy/serve_judge.sh
# 起来之后用 scripts/validate_judge_history_budget.py 打印的 s/call 对比前后。
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
eager_flag=()
[ "$ENFORCE_EAGER" = "1" ] && eager_flag=(--enforce-eager)

source /hy-tmp/venv/bin/activate

exec vllm serve "$JUDGE_MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name judge-model \
    "${eager_flag[@]}" \
    --enable-prefix-caching \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --download-dir "$HF_HOME"
