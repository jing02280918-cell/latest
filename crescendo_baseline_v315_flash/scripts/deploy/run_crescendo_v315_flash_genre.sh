#!/usr/bin/env bash
# Crescendo genre-deliverable run (题材成功率口径: VDS>=2 & IFR>=2).
# Usage:
#   setsid nohup bash .../run_crescendo_v315_flash_genre.sh \
#     > /root/crescendo_baseline_v315_flash_genre/driver.log 2>&1 &

set -uo pipefail

export RUN_DIR="${RUN_DIR:-/root/crescendo_baseline_v315_flash_genre}"
export CRESCENDO_REQUIRE_GENRE_DELIVERY=1

exec bash "$(dirname "$0")/run_crescendo_v315_flash_full.sh"
