#!/usr/bin/env bash
# Crescendo on v3.15.1 eval357 cohort, target = deepseek-v4-pro.
# Same codebase as flash; genre early-stop (VDS>=2 & IFR>=2) in vdb_scorer.
#
# Usage:
#   setsid nohup bash /root/crescendo_baseline_v315_flash/scripts/deploy/run_crescendo_v315_pro_full.sh \
#       > /root/crescendo_baseline_v315_pro_genre/driver.log 2>&1 < /dev/null &

set -uo pipefail

export RUN_DIR="${RUN_DIR:-/root/crescendo_baseline_v315_pro_genre}"
export TARGET_MODEL="${TARGET_MODEL:-deepseek-v4-pro}"

exec bash "$(dirname "$0")/run_crescendo_v315_flash_full.sh"
