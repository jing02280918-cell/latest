# V-Drift v3.15 Core60 主消融实验

本文记录时间受限条件下的 v3.15 主消融。它是 `ABLATION_v3.15.md` 所定义完整九组方案的
精简验证：固定一个代表性目标模型，在分层抽取的 60 个样本上比较完整方法与三个核心机制消融。

## 1. 实验目的

本实验回答三个核心问题：

- 跨样本 Thompson Sampling 调度是否优于按状态使用默认动作；
- 将上一轮五维评分反馈给起草模型是否有效；
- 针对 NJS/EFS 低分触发的维度定向升级是否有效。

这三个组件会改变后续动作或提示内容，无法仅靠历史结果可靠地做反事实回放，因此采用同批真实模型调用。
R5 retry、早停和快速状态判定等机制优先用历史结果或效率统计补充，不纳入本次主消融。

## 2. 固定版本与模型

- 方法版本：V-Drift v3.15；所有组强制 `r5_prompt_success_history_only=false`。
- 目标模型：`deepseek-v4-flash`，关闭思考模式，使用 `.env` 中的 DeepSeek API。
- 起草模型：`draft-model`，服务器本地 Qwen2.5-32B-Instruct Abliterated AWQ。
- 裁判与 Layer-2 状态分类：`judge-model`，服务器本地 Mistral-7B-Instruct-v0.3。
- 向量模型：`BAAI/bge-m3`，设备为 CUDA。
- RWI：全部组统一关闭；安全恢复逻辑保持开启。
- 评估 seed：`0`；每个实验组创建独立的新调度器，样本按相同顺序串行执行。

除被消融组件外，数据、样本顺序、模型、提示规则、评分口径和公共配置均保持一致。

## 3. Core60 分层样本

母集为 `data/v3.15_eval357.enriched.jsonl` 的 357 条样本。Core60 按 `category_macro`
比例分配，每个大类至少保留一条；类内用 `SHA-256(selection_seed + sample_id)` 排序后取样，
最后恢复为母集中的原始顺序。选择 seed 为 `31560`，与模型评估 seed 相互独立。

各大类数量如下：

- 政治与社会类：15
- 科技与未来类：11
- 价值观与情感操控类：10
- 综合与极端攻击类：9
- 文化与宗教类：7
- 科学与超现实类：4
- 经济类：4

数据与选择记录：

- 子集：`data/v3.15_ablation_core60.enriched.jsonl`
- 选择清单：`data/v3.15_ablation_core60.selection.json`
- 母集 SHA-256：`1694d22d470d870b60aa00c8af0510eba5ff45e88b986ae0df9a600ea27b8b03`
- 子集 SHA-256：`4f7caf73fd9bdd60b39ba605775742e0cea1d0f4793e007a8207db8cf5de4c7c`
- 选择清单 SHA-256：`482b0e9f3996c192101ccd360e7947340ee77c2bad8c6a95d379b25f7006e4c8`

## 4. 实验组

- `full`：完整 v3.15，同批基线。
- `v315_no_scheduler`：关闭 Thompson Sampling 和后验更新，按状态使用默认动作；保留评分反馈和维度升级。
- `v315_no_score_feedback`：不向 build-up 起草模型提供上一轮五维评分；裁判分数仍参与成功判断、
  Bandit 奖励、就绪判断和维度升级。
- `v315_no_dimension_escalation`：关闭 NJS/EFS 低分触发的强制策略升级；保留普通调度和评分反馈。

总运行量为 `4 组 × 60 条 × 1 seed = 240` 个样本-实验组。`full` 在同一批次重新运行，
历史 v3.15 结果不直接充当对照。

## 5. 实际运行命令

```bash
cd /root/vdriftbench

setsid -f /hy-tmp/venv/bin/python scripts/run_ablation.py \
  --env-file .env \
  --data data/v3.15_ablation_core60.enriched.jsonl \
  --target-model deepseek-v4-flash \
  --draft-model draft-model \
  --judge-model judge-model \
  --embedding-model BAAI/bge-m3 \
  --presets full,v315_no_scheduler,v315_no_score_feedback,v315_no_dimension_escalation \
  --seeds 0 \
  --out-dir data/ablation_v315_core60_deepseek \
  > data/ablation_v315_core60_deepseek.log 2>&1 < /dev/null
```

本次任务于 2026-10-05 23:41:41 CST 启动，启动 PID 为 `2208352`。PID 仅用于本次进程定位，
进程结束后不再具有复现实验的意义。2026-10-05 23:48 CST 的状态快照为：进程运行中，
`full` 已完成 2/60，尚未观察到错误。

运行产物：

- 根目录：`data/ablation_v315_core60_deepseek/`
- 日志：`data/ablation_v315_core60_deepseek.log`
- 实验清单：`data/ablation_v315_core60_deepseek/manifest.json`
- 最终汇总：`data/ablation_v315_core60_deepseek/summary.json`
- 分组结果：`data/ablation_v315_core60_deepseek/seed_0/<组名>/results.jsonl`
- 分组状态：`data/ablation_v315_core60_deepseek/seed_0/<组名>/status.json`

查看实时进度：

```bash
tail -f /root/vdriftbench/data/ablation_v315_core60_deepseek.log
```

## 6. 指标与统计

主指标为 R5 含补救 ASR，即 `r5_with_retry_success_pct`。同时报告：

- R5 首轮 ASR；
- 样本任意轮成功率；
- 样本内最大 RiskScore 均值；
- 平均 build-up 轮数与 R5 补救轮数；
- 目标模型平均逻辑调用数；
- 硬拒绝早停数量、状态判定层分布和裁判失败标记数。

每个消融组与 `full` 按相同 `sample_id` 配对，报告 full-only、ablation-only、百分点差和双侧精确
McNemar p 值。由于本次同时检验三个机制，正文应同时报告效应量；若将显著性作为正式结论，
应对三个主比较做 Holm 校正。

## 7. 结果填写模板

运行完成后，从 `summary.json` 提取下列内容：

- `full`：R5 含补救 ASR、样本成功率、平均目标调用数；
- 每个消融组：同一组指标及其相对 `full` 的百分点变化；
- 配对统计：full-only、ablation-only、未校正 p 值及 Holm 校正结果；
- 效率：各组总用时、平均轮数和调用数。

结论应按“关闭组件后指标下降、接近不变或上升”陈述，不应只根据单个 p 值判断组件是否有效。

## 8. 有效性边界

- 本实验只有一个模型 seed，不能估计 seed 间方差；远程目标模型也可能忽略 seed。
- Core60 是覆盖七大类的固定分层子集，但仍不是完整 357 条总体，置信区间会更宽。
- 这是对 DeepSeek v4 Flash 的主消融，结论不能自动推广到 GLM 或 Qwen。
- 各组按固定顺序执行，API 服务随时间变化可能带来时间效应；样本顺序与配置在各组保持一致，
  且比较使用样本级配对，以降低但不能完全消除这一影响。
- 当前运行器不支持严格等价的组内断点续跑；若某组不完整，不得用已完成部分计算完整 ASR。

因此，本实验适合作为时间受限条件下的主要机制证据，并应与历史版本演进分析共同呈现；
不得把历史跨版本差异直接解释为单变量因果消融。
