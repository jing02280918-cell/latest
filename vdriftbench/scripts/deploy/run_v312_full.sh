#!/usr/bin/env bash
# v3.12 full-run driver: enrich -> 314-sample run (resume-retried) -> annotate.
#
# What's different from v3.11 (scripts/deploy/run_v311_full.sh):
#   1. Dataset: 89 of 314 rows were re-curated — 历史类 removed entirely (29 rows
#      re-assigned to the other seven macro categories) and 60 weak non-history
#      rows rewritten onto live contested axes. See
#      data/dataset_optimization_changelog_v312.json.
#   2. R5 rewriter fix: `rewrite_genre_prompt` now validates its own output
#      (production ask must land inside the first `genre_ask_lead_chars`
#      characters, total length <= `genre_prompt_max_chars`), retries once at
#      temperature 0 naming the defect, then falls back to a deterministic
#      ask-first template. v3.11 shipped a buried ask for 249/314 samples.
#   3. --llm-seed: forwards `seed` to every OpenAI-compatible endpoint so the
#      draft model's attack utterances are reproducible. Local vLLM honours it
#      (verified); deepseek-v4-flash ignores it (also verified), so the target's
#      own sampling variance remains.
#
# `--draft-temperature` is deliberately left at its 0.9 default: the seed
# already makes generation reproducible, and lowering the temperature would
# change the *distribution* of attack utterances too, confounding v3.12-vs-v3.11
# with a third intervention.
#
# Designed to be launched detached so it survives the launching terminal, the
# SSH connection and the editor all going away:
#
#   cd /root/vdriftbench
#   setsid nohup bash scripts/deploy/run_v312_full.sh > data/v3.12_run/driver.log 2>&1 < /dev/null &
#
# `setsid` puts it in a brand-new session with no controlling terminal, so no
# SIGHUP can reach it; stdin from /dev/null keeps it from ever blocking on a
# read. Progress goes to the log above, and a sentinel file is written on exit
# so completion can be checked without parsing logs.
#
# The two vLLM servers (scripts/deploy/serve_draft.sh on :8001,
# serve_judge.sh on :8002) must already be up — this script checks and refuses
# to start otherwise, rather than burning the enrich spend and then failing.
#
# Safe to re-run: enrichment is skipped when the enriched file already matches
# the raw dataset, and the run itself uses --resume, so an interrupted job
# picks up where it left off instead of redoing completed samples.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"
RUN_DIR="${RUN_DIR:-data/v3.12_run}"
RESULTS="$RUN_DIR/results.jsonl"
RAW="data/dataset_100.jsonl"
ENRICHED="data/dataset_100.enriched.jsonl"
# Distinct from v3.11's backup name on purpose: reusing it would hit the
# "backup already exists" guard below and silently overwrite the enriched file
# that v3.11's results.jsonl must be read against.
ENRICHED_BACKUP="data/dataset_100.enriched.v311_pre_v312.jsonl"
TOTAL="${TOTAL:-314}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
LLM_SEED="${LLM_SEED:-20260904}"
SENTINEL="$RUN_DIR/DRIVER_STATUS"

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$RUN_DIR"
log() { echo "[$(date '+%F %T')] $*"; }
finish() { echo "$1 $(date '+%F %T')" > "$SENTINEL"; log "SENTINEL: $1"; }

log "=== v3.12 full-run driver starting (pid $$) ==="
log "config: llm_seed=$LLM_SEED total=$TOTAL run_dir=$RUN_DIR"

# ---- preflight -------------------------------------------------------------
for port in 8001 8002; do
    if ! curl -s -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        log "FATAL: no vLLM server on :$port — start serve_draft.sh / serve_judge.sh first"
        finish "FAILED_PREFLIGHT_VLLM_$port"
        exit 1
    fi
done
log "preflight: vLLM :8001 and :8002 both responding"

if ! grep -qE '^DEEPSEEK_API_KEY=.+' .env 2>/dev/null; then
    log "FATAL: DEEPSEEK_API_KEY is empty in .env"
    finish "FAILED_PREFLIGHT_NO_API_KEY"
    exit 1
fi
log "preflight: DEEPSEEK_API_KEY present"

# ---- step 1: enrichment ----------------------------------------------------
# The enriched file carries its own copy of `prompt`, so it goes stale whenever
# the raw dataset is curated. load_enriched_jsonl() raises on a mismatch, which
# doubles as the "is enrichment needed?" test.
if $PYTHON -c "
from vdriftbench.enrich import load_enriched_jsonl
load_enriched_jsonl('$ENRICHED')
" > /dev/null 2>&1; then
    log "step 1: enriched file already matches $RAW — skipping enrichment"
else
    # v3.11's results are only interpretable alongside the enriched file that
    # produced them (it holds that run's target_claim/value_axis), and
    # enrichment overwrites it in place, so preserve it once.
    if [ -f "$ENRICHED" ] && [ ! -f "$ENRICHED_BACKUP" ]; then
        cp "$ENRICHED" "$ENRICHED_BACKUP"
        log "step 1: backed up v3.11 enriched file -> $ENRICHED_BACKUP"
    fi

    log "step 1: enriching $TOTAL rows on deepseek-v4-flash (one call per row)"
    if ! $PYTHON main.py --enrich --data "$RAW" \
            --enriched-out "$ENRICHED" --enrich-model deepseek-v4-flash; then
        log "FATAL: enrichment failed"
        finish "FAILED_ENRICH"
        exit 1
    fi

    # Enrichment writes as it consumes a generator, so a mid-way API failure can
    # leave a short file. Re-verify rather than trusting the exit code alone.
    if ! $PYTHON -c "
from vdriftbench.enrich import load_enriched_jsonl
n = len(load_enriched_jsonl('$ENRICHED'))
assert n == $TOTAL, f'{n} rows, expected $TOTAL'
"; then
        log "FATAL: enriched file is incomplete or still mismatched after enrichment"
        finish "FAILED_ENRICH_VERIFY"
        exit 1
    fi
    log "step 1: enrichment complete and verified against $RAW"
fi

# ---- step 2: the run -------------------------------------------------------
# pipeline.py already swallows per-sample errors, so this loop is only for
# process-level deaths (endpoint drop, OOM, transient network). --resume makes
# each attempt skip samples already in results.jsonl.
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
    if [ "$n" -ge "$TOTAL" ]; then
        log "step 2: all $n/$TOTAL samples done"
        break
    fi

    log "step 2: run attempt $attempt/$MAX_ATTEMPTS (currently $n/$TOTAL)"
    $PYTHON main.py \
        --data "$ENRICHED" \
        --target-model deepseek-v4-flash \
        --judge-model judge-model \
        --classifier-model judge-model \
        --draft-model draft-model \
        --plausibility-model draft-model \
        --embedding-model BAAI/bge-m3 \
        --device cuda \
        --out "$RESULTS" \
        --bandit-state "$RUN_DIR/bandit_posterior.json" \
        --llm-seed "$LLM_SEED" \
        --resume \
        -v
    rc=$?
    n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
    log "step 2: attempt $attempt exited rc=$rc, now $n/$TOTAL"

    if [ "$n" -ge "$TOTAL" ]; then
        break
    fi
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        log "step 2: retrying in 60s"
        sleep 60
    fi
done

n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
if [ "$n" -lt "$TOTAL" ]; then
    log "run did not reach $TOTAL after $MAX_ATTEMPTS attempts (stopped at $n)"
    finish "INCOMPLETE_$n/$TOTAL"
    exit 1
fi

# ---- step 3: annotation ----------------------------------------------------
log "step 3: annotating results"
if ! $PYTHON scripts/annotate_results.py \
        --results "$RESULTS" \
        --enriched "$ENRICHED" \
        --out-annotated "$RUN_DIR/dataset_100.annotated.jsonl" \
        --out-failed "$RUN_DIR/failed_samples.jsonl"; then
    log "annotation failed, but the run itself is complete at $n/$TOTAL"
    finish "RUN_OK_ANNOTATE_FAILED_$n"
    exit 1
fi

log "=== done: $n/$TOTAL samples, results in $RUN_DIR ==="
finish "SUCCESS_$n/$TOTAL"
