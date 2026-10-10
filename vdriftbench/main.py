"""CLI entry point for the V-DriftBench v3.1 method (no experiment harness).

Examples
--------
Offline structural smoke test, no network / API key / model download needed:
    python main.py --mock --limit 5

Enrich the raw dataset once (adds target_claim / value_axis / risk_type /
expected_safe_behavior via an LLM call per row):
    python main.py --enrich --data data/dataset_100.jsonl \
        --enriched-out data/dataset_100.enriched.jsonl --enrich-model deepseek-v4-flash

Real run against an OpenAI-compatible endpoint (set VDB_API_KEY / VDB_BASE_URL,
or pass --api-key/--base-url):
    python main.py --data data/dataset_100.enriched.jsonl --limit 20 \
        --target-model deepseek-v4-flash --judge-model judge-model \
        --draft-model draft-model --plausibility-model draft-model \
        --embedding-model BAAI/bge-m3 --device cuda -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from vdriftbench.ablation import get_preset
from vdriftbench.archetypes import ResistArchetypeRegistry
from vdriftbench.bandit import ThompsonSamplingScheduler
from vdriftbench.config import Config
from vdriftbench.embedding_client import build_embedding_client
from vdriftbench.enrich import enrich_dataset, load_raw_jsonl, save_enriched_jsonl, load_enriched_jsonl
from vdriftbench.env_loader import load_dotenv_if_present
from vdriftbench.io_utils import load_results_jsonl, load_sample_results_jsonl, sample_result_to_dict
from vdriftbench.llm_client import build_llm_client
from vdriftbench.pipeline import run_dataset, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="V-DriftBench v3.1 method runner")
    parser.add_argument(
        "--env-file", default=None,
        help="path to a .env file with API keys/base URLs (default: ./.env or repo root .env if present)",
    )
    parser.add_argument("--data", default="data/dataset_100.jsonl", help="raw or enriched jsonl path")
    parser.add_argument("--config", default=None, help="YAML file overriding Config defaults (see configs/default.yaml)")
    parser.add_argument("--enriched-out", default="data/dataset_100.enriched.jsonl")
    parser.add_argument("--enrich", action="store_true", help="run the data-enrichment step and exit")
    parser.add_argument(
        "--no-verify-enriched", action="store_true",
        help="skip the enriched-vs-raw prompt consistency check (only to replay a historical enriched snapshot)",
    )
    parser.add_argument("--enrich-model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N samples")
    parser.add_argument("--mock", action="store_true", help="use offline mock clients (no network/API key)")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--classifier-model", default=None, help="defaults to judge-model")
    parser.add_argument("--draft-model", default=None, help="v3: 起草LLM(O-T-R自由生成)，默认与--judge-model相同")
    parser.add_argument("--plausibility-model", default=None, help="v3 9.2节: 现实可信度/说服力审核模型，默认与--draft-model相同")
    parser.add_argument("--skip-rwi-review", action="store_true", help="v3: 跳过9.2节现实可信度/说服力审核")
    parser.add_argument("--ablation", default="full", help="method ablation preset, see vdriftbench/ablation.py PRESETS")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default="cuda", help="device for the embedding model (cuda/cpu)")
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip sample_ids already present in --out (read from a previous, possibly crashed, run) instead of overwriting it",
    )
    parser.add_argument("--bandit-state", default="bandit_posterior.json")
    parser.add_argument("--seed", type=int, default=0, help="Thompson Sampling 臂选择的随机种子（不影响LLM采样）")
    parser.add_argument(
        "--llm-seed", type=int, default=None,
        help="v3.12: 透传给所有OpenAI兼容端点的 `seed` 参数，让起草LLM的话术可复现。"
             "本地vLLM支持；远端API（如deepseek）可能忽略，故只能消掉起草侧方差。默认不发送（沿用v3.11行为）",
    )
    parser.add_argument(
        "--draft-temperature", type=float, default=None,
        help="v3.12: 起草LLM自由生成话术的温度，默认0.9。配合 --llm-seed 降低跑间方差",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="v3.14.1: 同时跑几个样本。1 = 旧的严格串行行为。>1 时 bandit/原型表/embedding "
             "自动换成 concurrency.py 里的加锁代理；每个样本内部仍严格按轮次串行。"
             "本地两块 vLLM 靠连续批处理吃并发，这是整条流水线最大的一块提速",
    )
    parser.add_argument(
        "--no-r5-retry", action="store_true",
        help="v3.14.1: 关掉 R5 补救轮（默认开启，仅在样本已成功而 R5 这一轮未成功时补一轮）",
    )
    parser.add_argument(
        "--r5-retry-all-failures", action="store_true",
        help="v3.14.1: 只要 R5 未成功就补一轮，不再要求样本已经成功过（默认只补后者）",
    )
    parser.add_argument(
        "--no-early-termination", action="store_true",
        help="v3.15: 不在 S_hard_refuse 时提前结束 build-up，继续跑满 build-up 预算再进 R5",
    )
    parser.add_argument(
        "--no-r5-success-history-only", action="store_true",
        help="v3.15：R5/补救 prompt 参照全部 build-up（默认已关闭成功历史筛选）",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    loaded_keys = load_dotenv_if_present(args.env_file) if args.env_file else load_dotenv_if_present()
    if loaded_keys:
        logging.info("loaded %d var(s) from .env: %s", len(loaded_keys), ", ".join(loaded_keys))

    # Safety net matching scripts/deploy/env.sh's defaults: if that script
    # wasn't sourced first, an unset HF_HOME/HF_ENDPOINT makes
    # sentence-transformers/huggingface_hub try to reach the real (blocked)
    # huggingface.co instead of the local cache dir / mirror, and hang for
    # minutes on a dead TCP connection instead of failing fast.
    if os.path.isdir("/hy-tmp/hf_cache"):
        os.environ.setdefault("HF_HOME", "/hy-tmp/hf_cache")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    config = Config.from_yaml(args.config) if args.config else Config()
    config = config.apply_overrides(
        target_model=args.target_model,
        judge_model=args.judge_model,
        draft_model=args.draft_model,
        embedding_model=args.embedding_model,
        seed=args.llm_seed,
        draft_temperature=args.draft_temperature,
    )
    config = config.apply_overrides(
        r5_retry_enabled=False if args.no_r5_retry else None,
        r5_retry_only_when_sample_succeeded=False if args.r5_retry_all_failures else None,
        early_termination_on_hard_refuse=False if args.no_early_termination else None,
        r5_prompt_success_history_only=False if args.no_r5_success_history_only else None,
    )
    ablation = get_preset(args.ablation)
    config = ablation.apply_to_config(config)

    if args.enrich:
        raw_rows = load_raw_jsonl(args.data)
        enrich_client = build_llm_client(args.enrich_model, mock=args.mock, persona="enrich")
        samples = enrich_dataset(raw_rows, enrich_client)
        save_enriched_jsonl(samples, args.enriched_out)
        print(f"enriched {len(raw_rows)} rows -> {args.enriched_out}")
        return

    samples = load_enriched_jsonl(args.data, verify_against_raw=not args.no_verify_enriched)
    if args.limit:
        samples = samples[: args.limit]

    already_done_ids: set[str] = set()
    if args.resume and os.path.isfile(args.out):
        already_done_ids = {row["sample_id"] for row in load_results_jsonl(args.out)}
        n_before = len(samples)
        samples = [s for s in samples if s.sample_id not in already_done_ids]
        print(f"--resume: {len(already_done_ids)} sample(s) already in {args.out}, {n_before - len(samples)} skipped, {len(samples)} remaining")

    target_client = build_llm_client(config.target_model, mock=args.mock, persona="target", seed=config.seed)
    judge_client = build_llm_client(config.judge_model, mock=args.mock, persona="judge", seed=config.seed)
    classifier_model = args.classifier_model or config.judge_model
    classifier_client = build_llm_client(classifier_model, mock=args.mock, persona="classifier", seed=config.seed)
    draft_model = args.draft_model or config.draft_model
    draft_client = build_llm_client(draft_model, mock=args.mock, persona="draft", seed=config.seed)
    plausibility_model = args.plausibility_model or draft_model
    plausibility_client = (
        None if args.skip_rwi_review
        else build_llm_client(plausibility_model, mock=args.mock, persona="plausibility", seed=config.seed)
    )
    embed_client = build_embedding_client(config.embedding_model, mock=args.mock, device=args.device)

    scheduler = ThompsonSamplingScheduler.load_or_create(
        args.bandit_state,
        seed=args.seed,
        kappa=config.bandit_kappa,
        disable_hierarchy=ablation.disable_hierarchy,
        shadow_pool_max_size=config.shadow_pool_max_size,
        shadow_promote_n_min=config.shadow_promote_n_min,
        shadow_promote_k_categories=config.shadow_promote_k_categories,
    )
    archetype_registry = ResistArchetypeRegistry(
        tau_archetype=config.tau_archetype, n_archetype_min=config.n_archetype_min
    )

    # Write + fsync after every sample (rather than buffering the whole run
    # in memory and writing once at the end): a real run against a paid API
    # can take many hours, and without this a crash near the end would lose
    # every completed sample instead of just needing a `--resume` restart.
    out_mode = "a" if args.resume else "w"
    out_file = open(args.out, out_mode, encoding="utf-8")
    # `--workers > 1` 时样本不再按提交顺序完成，进度只能靠自己数完成了几条
    # （`i` 是提交序号，用它当进度会来回跳）。run_dataset 保证这个回调是串行调用的。
    n_completed = 0

    def _on_sample_done(i, sample, result) -> None:
        nonlocal n_completed
        out_file.write(json.dumps(sample_result_to_dict(result), ensure_ascii=False) + "\n")
        out_file.flush()
        os.fsync(out_file.fileno())
        scheduler.save(args.bandit_state)
        n_completed += 1
        n_done = len(already_done_ids) + n_completed
        print(
            f"[{n_done}/{len(samples) + len(already_done_ids)}] {sample.sample_id} "
            f"done ({time.time() - t_start:.0f}s elapsed) -> {args.out}"
        )

    t_start = time.time()
    try:
        results = run_dataset(
            samples,
            target_client,
            judge_client,
            classifier_client,
            embed_client,
            scheduler,
            config,
            ablation,
            draft_client=draft_client,
            archetype_registry=archetype_registry,
            plausibility_client=plausibility_client,
            skip_rwi_review=args.skip_rwi_review,
            on_sample_done=_on_sample_done,
            workers=args.workers,
        )
    finally:
        out_file.close()
    if samples:
        elapsed = time.time() - t_start
        print(
            f"ran {len(samples)} sample(s) with workers={args.workers} in {elapsed / 60:.1f} min "
            f"({elapsed / len(samples):.0f}s/sample wall-clock)"
        )

    if args.resume and already_done_ids:
        results = load_sample_results_jsonl(args.out)  # reload everything for an accurate summary, not just the newly-run tail
        print(f"summary below covers all {len(results)} sample(s) in {args.out} (including {len(already_done_ids)} from before --resume)")
    summary = summarize(results, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"per-round records -> {args.out}")
    print(f"bandit posterior -> {args.bandit_state}")


if __name__ == "__main__":
    main()
