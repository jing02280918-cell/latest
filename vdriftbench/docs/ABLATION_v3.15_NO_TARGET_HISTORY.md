# V-Drift v3.15：目标模型无多轮历史消融（GLM-5.3-Flash）

## 主实验：eval357 无历史消融（357 条新跑）

**样本数 357** 指：只对 **无历史臂** 做 357 次完整 pipeline 调用。  
对照组 **full（有多轮目标历史）** 使用已有 `data/v3.15_glm53flash_run/results.jsonl`（同 eval357 cohort），经 `export_ablation_full_proxy_from_run.py` 转成消融配对格式，**不再重跑 357 条 full**。

（若严格同批双臂重跑，才是 357×2=714 次；本实验不做。）

| 项 | 值 |
|---|---|
| 数据 | `data/v3.15_eval357.enriched.jsonl`（357） |
| 新跑 | `v315_no_target_history` × 357 |
| 对照 | `v3.15_glm53flash_run` → `data/ablation_v315_eval357_glm53flash_full_proxy/seed_0/full` |
| 目标 | `ZHIPU/GLM-5.3-Flash` |
| 输出 | `data/ablation_v315_eval357_glm53flash_no_target_history/` |
| 并行 | 默认 `--workers 2`（与 `v3.15_glm53flash_run` 一致；Bandit 后验更新顺序与 workers=1 略有不同，见 `concurrency.py`） |

```bash
cd /root/vdriftbench

setsid -f bash scripts/deploy/run_ablation_v315_glm53flash_eval357_no_target_history.sh \
  > data/ablation_v315_eval357_glm53flash_no_target_history.log 2>&1 < /dev/null
```

进度：

```bash
tail -f /root/vdriftbench/data/ablation_v315_eval357_glm53flash_no_target_history.log
wc -l data/ablation_v315_eval357_glm53flash_no_target_history/seed_0/v315_no_target_history/results.jsonl
```

---

## 探索性：Core60（60 条，已 supersede）

## 实验目的

完整 v3.15 在 build-up 与 R5 阶段都会把**先前各轮的 user/assistant 消息**一并送入目标模型 API，使后续 prompt 依赖 in-context 铺垫。本消融只关闭这一机制：

- **对照 `full`**：目标模型每轮看到完整对话历史（同批 GLM Core60 已完成的 `full` 臂）。
- **`v315_no_target_history`**：目标模型每轮**仅收到当前一条 user 消息**（与 PAIR baseline 对目标侧的无历史设定一致）。

起草 LLM、裁判、Bandit、维度升级、R5 补救等仍使用完整轮次记录与评分反馈，因此差异**只归因于目标侧是否保留多轮上下文**，用于回答「多轮交互是否真正发挥作用」。

## 固定配置

与 [`ABLATION_v3.15_CORE60.md`](ABLATION_v3.15_CORE60.md) 一致，仅目标模型换为 GLM：

| 项 | 值 |
|---|---|
| 数据 | `data/v3.15_ablation_core60.enriched.jsonl`（60 条） |
| 目标 | `ZHIPU/GLM-5.3-Flash`（`VDB_GLM_REASONING_EFFORT=low`） |
| 起草 / 裁判 | `draft-model` / `judge-model`（本地 vLLM） |
| Seed | `0` |
| 配对基线 | `data/ablation_v315_core60_glm53flash/seed_0/full`（同 cohort 同 seed 重跑的 full） |

## 运行

```bash
cd /root/vdriftbench

setsid -f bash scripts/deploy/run_ablation_v315_glm53flash_no_target_history.sh \
  > data/ablation_v315_core60_glm53flash_no_target_history.log 2>&1 < /dev/null
```

进度：

```bash
tail -f /root/vdriftbench/data/ablation_v315_core60_glm53flash_no_target_history.log
wc -l /root/vdriftbench/data/ablation_v315_core60_glm53flash_no_target_history/seed_0/v315_no_target_history/results.jsonl
```

## 主指标

与 Core60 主消融相同：R5 含补救 ASR（`r5_with_retry_success_pct`），并与已有 `full` 臂做样本级 McNemar 配对（脚本在 `--ablation-only` 模式下自动引用 `paired_full_dir`）。

若完整方法显著高于无历史组，则支持「多轮 in-context 铺垫对 GLM 上的 v3.15 成功率有必要贡献」；若差异很小，则 build-up 的收益可能主要来自**当轮 prompt 文案**而非目标侧记忆。
