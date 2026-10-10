#!/usr/bin/env bash
# Quick status of the detached v3.11 run. Safe to call any time, from any shell.
cd "$(dirname "$0")/../.." || exit 1
RUN_DIR=data/v3.11_run
N=$(wc -l < "$RUN_DIR/results.jsonl" 2>/dev/null || echo 0)
echo "样本进度   : $N/314"
echo -n "驱动进程   : "; pgrep -f run_v311_full.sh >/dev/null && echo "运行中 (pid $(pgrep -f run_v311_full.sh | tail -1))" || echo "已退出"
echo -n "vLLM :8001 : "; curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && echo OK || echo "无响应"
echo -n "vLLM :8002 : "; curl -s -m 3 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && echo OK || echo "无响应"
echo -n "完成标记   : "; cat "$RUN_DIR/DRIVER_STATUS" 2>/dev/null || echo "(未完成)"
echo "当前样本   : $(grep -o 'running sample [0-9]*/314: S[0-9]*' "$RUN_DIR/driver.log" 2>/dev/null | tail -1)"
