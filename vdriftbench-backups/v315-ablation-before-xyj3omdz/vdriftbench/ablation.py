"""Active v3.15.1 ablations; earlier named experimental methods are retired.

Each registered arm changes one mechanism relative to the current full method.
Low-level fields remain for library callers and historical unit tests, but the
old v3.0-v3.10 presets are deliberately absent from PRESETS / the CLI.
Config overrides must be applied before clients and the scheduler are built.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from .config import Config

ContextMode = Literal["category_state", "state_only", "category_only"]

# Low-level compatibility only; no fixed_path experiment is registered.
FIXED_PATH = {2: "P2", 3: "P3", 4: "P4", 5: "P4_continue", 6: "P4_continue"}


@dataclass
class AblationConfig:
    name: str = "full"
    fixed_path: bool = False
    disable_geometry: bool = False
    disable_bandit: bool = False
    context_mode: ContextMode = "category_state"
    disable_dual_channel: bool = False
    disable_hierarchy: bool = False
    disable_self_expansion: bool = True
    disable_fidelity_check: bool = False
    disable_dimension_targeting: bool = False
    disable_resist_taxonomy: bool = False
    disable_cumulative_hypothesis: bool = False
    disable_feedback_learning: bool = False
    disable_p4_escalation: bool = False
    s_engaged_override: str | None = None
    config_overrides: dict = field(default_factory=dict)
    description: str = "完整 v3.15.1，作为同批重跑的基线"

    def build_context(self, category_macro: str, state: str) -> tuple:
        if self.context_mode == "state_only":
            return ("*", state)
        if self.context_mode == "category_only":
            return (category_macro, "*")
        return (category_macro, state)

    def apply_to_config(self, config: Config) -> Config:
        return config.apply_overrides(**self.config_overrides)


PRESETS: dict[str, AblationConfig] = {
    "full": AblationConfig(),
    "v3151_legacy_r5_history": AblationConfig(
        name="v3151_legacy_r5_history",
        config_overrides={"r5_prompt_success_history_only": False},
        description="回退旧 R5 历史策略：首轮使用全部 build-up；补救使用 history[:-1]",
    ),
    "v3151_no_r5_retry": AblationConfig(
        name="v3151_no_r5_retry", config_overrides={"r5_retry_enabled": False},
        description="关闭 R5 补救，保留首次题材产出轮",
    ),
    "v3151_no_readiness_stop": AblationConfig(
        name="v3151_no_readiness_stop", config_overrides={"adaptive_rounds": False},
        description="关闭就绪早停，仍保留硬拒绝早停及原最大轮数",
    ),
    "v3151_no_hard_refusal_stop": AblationConfig(
        name="v3151_no_hard_refusal_stop",
        config_overrides={"early_termination_on_hard_refuse": False},
        description="关闭硬拒绝早停，仍保留就绪早停",
    ),
    "v3151_no_fastpath": AblationConfig(
        name="v3151_no_fastpath", disable_geometry=True,
        description="关闭 Layer 1 几何/首 token 快速判定，保留关键词规则和 LLM 分类",
    ),
    "v3151_no_scheduler": AblationConfig(
        name="v3151_no_scheduler", disable_bandit=True,
        description="固定选状态默认动作，关闭 Thompson 采样和后验更新；保留维度升级",
    ),
    "v3151_no_score_feedback": AblationConfig(
        name="v3151_no_score_feedback", disable_dimension_targeting=True,
        description="去除给起草模型的评分提示，仍保留裁判、奖励、就绪判断和维度升级",
    ),
    "v3151_no_dimension_escalation": AblationConfig(
        name="v3151_no_dimension_escalation", disable_p4_escalation=True,
        description="关闭维度定向策略升级，保留起草模型评分反馈和常规调度",
    ),
}


def get_preset(name: str) -> AblationConfig:
    try:
        return deepcopy(PRESETS[name])
    except KeyError as exc:
        raise ValueError(
            f"unknown or retired ablation preset {name!r}; v3.15.1 choices: {sorted(PRESETS)}"
        ) from exc
