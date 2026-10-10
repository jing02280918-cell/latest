#!/usr/bin/env bash
# Fast env bootstrap when default Aliyun pip mirror stalls on large wheels.
set -euo pipefail

MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
VLLM_VER="${VLLM_VER:-0.28.0}"
VLLM_WHEEL="/hy-tmp/wheels/vllm-${VLLM_VER}-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_URL="${VLLM_URL:-https://pypi.tuna.tsinghua.edu.cn/packages/87/d7/97f6ecc2ae883e601e08d7cef87cb54ececeefcfe6b5e12d5d92f8d06d6b/vllm-${VLLM_VER}-cp38-abi3-manylinux_2_28_x86_64.whl}"
VENV="/hy-tmp/venv"
LOG="/hy-tmp/fast_setup.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

mkdir -p /hy-tmp/wheels "$(dirname "$LOG")"

if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv with system-site-packages"
    python3 -m venv --system-site-packages --without-pip "$VENV"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV/bin/python" /tmp/get-pip.py -q
fi

"$VENV/bin/python" -c "import torch; print('torch', torch.__version__)" | tee -a "$LOG"

if [ ! -f "$VLLM_WHEEL" ]; then
    log "downloading vllm ${VLLM_VER} wheel via aria2 (16 conn)"
    aria2c -x 16 -s 16 -k 1M -c -o "$(basename "$VLLM_WHEEL")" -d /hy-tmp/wheels "$VLLM_URL"
fi

log "installing python deps (small wheels only, mirror=$MIRROR)"
"$VENV/bin/pip" install -U pip -i "$MIRROR"
"$VENV/bin/pip" install pytest sentence-transformers huggingface_hub -i "$MIRROR" \
    --retries 10 --timeout 300 --prefer-binary

log "installing vllm from local wheel"
"$VENV/bin/pip" install "$VLLM_WHEEL" -i "$MIRROR" --retries 10 --timeout 300

"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$LOG"
log "env ready"
