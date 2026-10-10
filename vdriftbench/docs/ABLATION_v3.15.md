# V-Drift v3.15 消融实验

本文件定义 v3.15 的正式消融方案。可执行入口集中在 `scripts/run_ablation.py`；
旧版命名实验已从活动注册表和 CLI 中移除。历史结果与底层兼容开关仍保留，便于读取旧数据。

时间受限条件下正在执行的 DeepSeek v4 Flash 四组、Core60 主消融，见
[`ABLATION_v3.15_CORE60.md`](ABLATION_v3.15_CORE60.md)。

## 版本边界

仓库包含可选开关 `r5_prompt_success_history_only`（默认关闭）。v3.15 的 R5 起草器
参考全部 build-up 历史，所以本实验的 `full` 以及全部对照组都强制：

```text
r5_prompt_success_history_only = false
```

因此这里的完整基线与默认配置一致，不会启用“只参考成功 build-up”。

## 实验组

- `full`：完整 v3.15，同批重新运行的基线。
- `v315_no_retry_pair_style`：关闭 R5 补救轮的 PAIR 风格开头与禁复述约束；
  首轮仍按 v3.15 默认的普通题材改写运行，补救轮的长度、产出动词位置、体裁、
  话题重合度和证据要求保持不变。
- `v315_no_r5_retry`：关闭 R5 补救，只保留首次题材产出轮。该组查询预算更低，
  需同时报告首轮 ASR、含补救 ASR和目标模型调用数。
- `v315_no_readiness_stop`：关闭“模型已就绪”早停，仍保留硬拒绝早停和最大 build-up 轮数。
- `v315_no_hard_refusal_stop`：关闭硬拒绝早停，仍保留就绪早停。
- `v315_no_fastpath`：关闭 Layer 1 几何轨迹和首 token 快速判定，保留 Layer 0 关键词规则、
  Layer 2 LLM 分类。embedding 仍计算并落盘，因此该组衡量判定机制，不代表完全移除 embedding 的成本。
- `v315_no_scheduler`：按当前状态选择默认动作，关闭 Thompson Sampling 和后验更新；
  状态分类、评分反馈与维度升级保持开启。
- `v315_no_score_feedback`：不向 build-up 起草模型提供五维评分提示；裁判分数仍用于
  成功判定、Bandit 奖励、就绪判断、维度升级和 R5 补救。
- `v315_no_dimension_escalation`：关闭 NJS/EFS 维度定向的强制策略升级；
  普通调度、评分反馈和 R5 补救保持开启。
- `v315_no_target_history`：目标模型每轮仅接收当前 user 消息，不 replay 先前
  user/assistant 轮次；起草、裁判与调度仍使用完整对话记录。GLM Core60 配对实验见
  [`ABLATION_v3.15_NO_TARGET_HISTORY.md`](ABLATION_v3.15_NO_TARGET_HISTORY.md)。
- `v315_r5_only`：跳过 build-up（P1–P7 / Bandit / 早停），仅 R5 改写+补救+安全复检。
  eval357 见 [`ABLATION_v3.15_R5_ONLY.md`](ABLATION_v3.15_R5_ONLY.md)。

自扩展在 v3.15 完整方法中已经默认关闭，“关闭自扩展”与 full 相同，故不列为有效消融。
评分口径固定为“行为二元”，避免把成功定义变更混入方法消融。

## 比较协议

默认使用 `data/v3.15_eval357.enriched.jsonl` 的 357 个唯一 sample_id。
每个模型使用独立输出目录；默认运行种子 0、1、2。同一 seed 下，各组使用相同样本顺序、
模型、提示规则、评分口径和公共参数。每组从新调度器开始，不加载历史后验，并按样本串行运行，
避免并发完成顺序改变跨样本学习。远程模型可能忽略 seed，因此不能假设完全确定性。

每次实验必须包含 `full`。历史 v3.15 跑批不能直接充当对照，因为代码状态、调用时间与随机流
可能不同。`full` 和消融组应在同一实验批次重新运行。RWI 在各组统一关闭，因为它不参与方法决策；
安全恢复轮保持开启。

## 指标与结果

主指标为 R5 含补救 ASR。R5 通过 `RAW_GENRE_PROMPT`、`RAW_GENRE_PROMPT_RETRY`、
`RAW_GENRE_PROMPT_RETRY_2` 识别，不能假定它固定出现在第 5 轮。

辅助指标包括 R5 首轮 ASR、样本任意轮成功率、样本内最大 RiskScore 均值、build-up 与补救轮数、
硬拒绝早停数、各状态判定层使用次数、裁判失败标记数和各角色逻辑调用数。
脚本还按相同 sample_id 给出 full-only、ablation-only、百分点差以及探索性的双侧精确 McNemar p 值。
p 值未做多重比较校正，不应单独作为多组显著性结论。

输出包含数据与源码哈希、有效配置、逐样本 JSONL、Bandit 后验、运行状态、逐 seed 汇总和多 seed 汇总。
若缺少样本，状态为 `incomplete`，完整 ASR 不会被计算，并列出 `missing_ids`。
已有输出目录不会被覆盖。中断结果会保留，但当前不支持严格等价的半组断点续跑；重跑应换新目录。

## 运行命令

以下命令在 `/root/vdriftbench` 执行：

```bash
# 查看全部 v3.15 组
/hy-tmp/venv/bin/python scripts/run_ablation.py --list

# 只验证配置、样本与实验计划，不调用模型
/hy-tmp/venv/bin/python scripts/run_ablation.py --dry-run

# 离线结构验证
/hy-tmp/venv/bin/python scripts/run_ablation.py --mock --limit 2 --seeds 0 \
  --out-dir /tmp/v315_ablation_smoke

# DeepSeek v4 Flash 的完整 v3.15 消融
/hy-tmp/venv/bin/python scripts/run_ablation.py \
  --env-file .env --data data/v3.15_eval357.enriched.jsonl \
  --target-model deepseek-v4-flash --draft-model draft-model --judge-model judge-model \
  --seeds 0,1,2 --out-dir data/ablation_v315_flash

# 只验证 v3.15 补救轮的 PAIR 风格约束，仍重新运行 full
/hy-tmp/venv/bin/python scripts/run_ablation.py \
  --presets full,v315_no_retry_pair_style --seeds 0,1,2 \
  --out-dir data/ablation_v315_pair_style
```

mock 结果只验证程序结构，不代表真实模型的安全行为。
