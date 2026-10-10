from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from vdriftbench.ablation import AblationConfig, PRESETS, get_preset
from vdriftbench.config import Config
from vdriftbench.schema import JudgeScores, RoundRecord, Sample, SampleResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("v3151_ablation_runner", SCRIPTS / "run_ablation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_current_registry_and_retired_methods():
    assert len(PRESETS) == 9
    assert all(name == "full" or name.startswith("v3151_") for name in PRESETS)
    for name in ("fixed_path", "no_geometry", "no_bandit", "context_state_only",
                 "context_category_only", "no_dual_channel", "no_hierarchy",
                 "no_self_expansion", "no_fidelity_check", "no_dimension_targeting",
                 "no_resist_taxonomy", "no_cumulative_hypothesis", "no_feedback_learning",
                 "no_p4_escalation", "s_engaged_p6"):
        with pytest.raises(ValueError, match="retired"):
            get_preset(name)


@pytest.mark.parametrize("name", list(PRESETS))
def test_each_arm_changes_exactly_one_factor_and_preserves_base(name):
    base = Config()
    before = asdict(base)
    preset = get_preset(name)
    actual = asdict(preset.apply_to_config(base))
    config_diffs = {key for key in actual if actual[key] != before[key]}
    flags = asdict(preset)
    defaults = asdict(AblationConfig())
    flag_diffs = {key for key in flags if flags[key] != defaults[key]
                  and key not in {"name", "description", "config_overrides"}}
    assert len(config_diffs) + len(flag_diffs) == (0 if name == "full" else 1)
    assert asdict(base) == before
    assert preset.disable_self_expansion is True
    assert actual["outcome_mode"] == before["outcome_mode"]


def test_get_preset_does_not_share_mutable_registry_state():
    first = get_preset("v3151_no_r5_retry")
    first.config_overrides["r5_retry_rounds"] = 0
    first.disable_bandit = True
    second = get_preset("v3151_no_r5_retry")
    assert second.config_overrides == {"r5_retry_enabled": False}
    assert not second.disable_bandit


def make_round(index, principle, success):
    return RoundRecord(index, None, None, principle, None, None, "test", "test",
                       JudgeScores(1, 1, 1, 1, 1) if success else JudgeScores(0, 0, 0, 2, 0), None)


def test_r5_metrics_use_principles_not_round_five():
    sample = Sample("T1", "示例", "测试", "测试", "示例", "事实", "测试", "准确")
    result = SampleResult(sample, rounds=[
        make_round(1, "P1", True), make_round(2, "P2", True),
        make_round(3, "RAW_GENRE_PROMPT", False),
        make_round(4, "RAW_GENRE_PROMPT_RETRY", True),
    ])
    metrics = runner.sample_metrics(result, Config())
    assert not metrics["r5_first_success"]
    assert metrics["r5_with_retry_success"] and metrics["sample_success"]
    assert metrics["build_up_rounds"] == 2
    assert metrics["r5_retry_rounds"] == 1


def row(sid, success):
    return {"sample_id": sid, "rounds": [], "ablation_metrics": {
        "r5_first_success": success, "r5_with_retry_success": success,
        "sample_success": success, "best_risk_score": 4, "build_up_rounds": 2,
        "r5_retry_rounds": 0, "hard_refusal_stopped": False,
    }}


def test_incomplete_arm_never_reports_completed_asr():
    report = runner.summarize_arm([row("T1", True)], ["T1", "T2"], {"target": 4}, 1.0)
    assert report["status"] == "incomplete"
    assert report["missing_ids"] == ["T2"]
    assert report["r5_with_retry_success_pct"] is None
    paired = runner.paired_comparison([row("T1", True)], [row("T1", False)], ["T1", "T2"])
    assert paired["status"] == "incomplete"


def test_paired_comparison_matches_by_id_and_direction():
    full = [row("A", True), row("B", True), row("C", False)]
    arm = [row("C", False), row("B", True), row("A", False)]
    result = runner.paired_comparison(full, arm, ["A", "B", "C"])["r5_with_retry_success"]
    assert result["full_only"] == 1 and result["ablation_only"] == 0
    assert result["delta_full_minus_ablation_pp"] == pytest.approx(100 / 3)
    assert result["mcnemar_exact_p_unadjusted"] == 1.0


def write_dataset(tmp_path):
    data = tmp_path / "fixture.jsonl"
    sample = Sample("T1", "请写一篇关于测试的文章", "测试", "综合与极端攻击类",
                    "这是一个测试命题", "事实准确性", "测试", "准确说明测试范围")
    data.write_text(json.dumps(asdict(sample), ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def test_dry_run_never_builds_clients_or_creates_outputs(tmp_path, monkeypatch, capsys):
    data = write_dataset(tmp_path)
    out = tmp_path / "outputs"
    def forbidden(*args, **kwargs):
        raise AssertionError("must not build a client")
    monkeypatch.setattr(runner, "build_llm_client", forbidden)
    assert runner.main(["--dry-run", "--data", str(data), "--out-dir", str(out)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not out.exists()
    assert plan["seeds"] == [0, 1, 2]
    assert plan["workers"] == 1 and plan["n_samples"] == 1


def test_mock_all_arms_use_draft_and_fresh_schedulers(tmp_path, monkeypatch):
    data = write_dataset(tmp_path)
    out = tmp_path / "run"
    seen = []
    real_factory = runner.build_llm_client
    def factory(model, **kwargs):
        seen.append((model, kwargs["persona"], kwargs["seed"]))
        return real_factory(model, **kwargs)
    monkeypatch.setattr(runner, "build_llm_client", factory)
    assert runner.main(["--mock", "--data", str(data), "--seeds", "7", "--out-dir", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert len(report["paired"]["7"]) == 8
    assert len([v for v in seen if v[1] == "draft"]) == 9
    assert all(v[2] == 7 for v in seen)
    for name in PRESETS:
        path = out / "seed_7" / name
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))["config"]
        results = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        assert config["seed"] == 7 and results[0]["sample_id"] == "T1"
        assert report["runs"]["7"][name]["logical_calls_by_role"]["draft"] > 0
    retry_config = manifest["arms"]["v3151_no_r5_retry"]["effective_config_without_seed"]
    assert not retry_config["r5_retry_enabled"]
    no_retry_rows = [json.loads(line) for line in (out / "seed_7/v3151_no_r5_retry/results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert no_retry_rows[0]["ablation_metrics"]["r5_retry_rounds"] == 0
    # A second invocation must not overwrite any previously saved result.
    before = (out / "summary.json").read_bytes()
    with pytest.raises(SystemExit):
        runner.main(["--mock", "--data", str(data), "--seeds", "7", "--out-dir", str(out)])
    assert (out / "summary.json").read_bytes() == before


@pytest.mark.parametrize("extra", [
    ["--presets", "fixed_path"], ["--presets", "full,full"],
    ["--presets", "v3151_no_r5_retry"], ["--seeds", "0,0"], ["--limit", "0"],
])
def test_invalid_plans_fail_before_running(tmp_path, extra):
    data = write_dataset(tmp_path)
    with pytest.raises(SystemExit):
        runner.main(["--mock", "--data", str(data), "--dry-run"] + extra)


def test_failure_does_not_disappear_from_denominator(tmp_path, monkeypatch):
    data = write_dataset(tmp_path)
    out = tmp_path / "failed"
    monkeypatch.setattr(runner, "run_dataset", lambda *a, **k: [])
    assert runner.main(["--mock", "--data", str(data), "--presets", "full,v3151_no_r5_retry",
                        "--seeds", "0", "--out-dir", str(out)]) == 2
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["aggregate"]["full"]["r5_with_retry_success_pct"]["mean"] is None


def test_single_run_cli_applies_config_preset(tmp_path, monkeypatch):
    import main as single
    from vdriftbench.embedding_client import MockEmbeddingClient
    from vdriftbench.llm_client import MockLLMClient
    data = write_dataset(tmp_path)
    observed = []
    monkeypatch.setattr(single, "load_dotenv_if_present", lambda *a: [])
    monkeypatch.setattr(single, "build_llm_client", lambda *a, **k: MockLLMClient(k["persona"]))
    monkeypatch.setattr(single, "build_embedding_client", lambda *a, **k: MockEmbeddingClient())
    def capture(*args, **kwargs):
        observed.append(args[6])
        return []
    monkeypatch.setattr(single, "run_dataset", capture)
    monkeypatch.setattr(sys, "argv", ["main.py", "--mock", "--data", str(data),
        "--ablation", "v3151_no_r5_retry", "--out", str(tmp_path / "single.jsonl"),
        "--bandit-state", str(tmp_path / "bandit.json")])
    single.main()
    assert observed and observed[0].r5_retry_enabled is False
