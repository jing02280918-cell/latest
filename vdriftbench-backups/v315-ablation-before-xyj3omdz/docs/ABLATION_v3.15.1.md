# V-Drift v3.15.1 消融实验

本方案替换此前 v3.0–v3.10 的命名消融实验。代码入口为 `scripts/run_ablation.py`，
唯一预设注册表为 `vdriftbench/ablation.py`。旧注册项已经删除；历史数据、历史方法文档
以及仍被库调用方使用的底层兼容字段保留。以下是实验设计，不是已取得的真实实验结论。

## 实验组

- `full`：当前 v3.15.1，作为本次实验中新跑的完整基线。
- `v3151_legacy_r5_history`：将 `r5_prompt_success_history_only` 设为 False，
  回退到旧生成输入策略。R5 首轮参考所有 build-up；补救轮参考 `history[:-1]`，
  因而第二次补救会看到较早的 R5 失败轮。这是“整个历史选择策略”的消融，
  不能解释成仅加入失败 build-up 的纯因素实验。目标模型上下文保持原逻辑。
- `v3151_no_r5_retry`：将 `r5_retry_enabled` 设为 False，只保留首个题材轮。
  用于测量补救的增量与额外调用成本。比较首轮 ASR 和含补救 ASR，并同时报告调用数；
  该组预算更少，不能据此宣称相同查询预算下的优势。
- `v3151_no_readiness_stop`：将 `adaptive_rounds` 设为 False，不因“已就绪”提前进入 R5。
  保留硬拒绝早停、状态判断和最大 build-up 轮数，因此不保证每个样本都跑满六轮。
- `v3151_no_hard_refusal_stop`：将 `early_termination_on_hard_refuse` 设为 False，
  硬拒绝不再直接结束 build-up；就绪早停保持开启。
- `v3151_no_fastpath`：开启 `disable_geometry`，关闭 Layer 1 的几何轨迹与首 token
  快速判定，保留 Layer 0 关键词规则、Layer 2 LLM 分类；几何指标仍计算并记录。
  它测量的是状态快速判定的贡献，不能解释为完全删除 embedding 的算力节省。
- `v3151_no_scheduler`：开启 `disable_bandit`，选当前状态的默认动作，禁用采样及后验更新。
  状态分类、起草反馈、维度定向升级仍开启。S_engaged 本身只有一个候选，效果主要来自
  仍存在多候选的状态/轮次；这是调度系统整体对照，不单独识别跨样本学习的效应。
- `v3151_no_score_feedback`：开启 `disable_dimension_targeting`，去掉传给起草模型的
  五维评分提示。裁判分数仍用于成功判定、Bandit 奖励、就绪判断、维度升级及 R5 补救。
  对应结论只限于 build-up 的显式评分提示。
- `v3151_no_dimension_escalation`：开启 `disable_p4_escalation`，去掉维度定向的强制策略升级。
  普通调度、起草评分提示和 R5 补救仍保留。

此前的“关闭自扩展”与当前完整模型相同，因为自扩展在 v3.15.1 已默认关闭，故不再列为实验组。
评分口径固定为“行为二元”，不把修改成功定义混入方法消融。

## 比较协议

默认使用 `data/v3.15_eval357.enriched.jsonl` 中的 357 个唯一 sample_id；
`--data` 允许改用已划分的 held-out 数据。数据集文件哈希与选中样本的顺序写入 manifest。
数据集曾参考旧方法结果修改，因此在同一 cohort 上的结果只能支撑该 cohort 内的机制比较，
泛化结论需要独立测试集；本脚本不自动重写、删除或重划数据。

每个模型单独设置输出目录。默认种子 0、1、2，同一 seed 下各组共享样本顺序、
目标/起草/裁判/分类模型、提示规则、评分口径和轮数上限。各组各 seed 使用新的调度器，
不加载历史后验；按样本严格串行（workers=1），避免并发完成顺序影响跨样本学习。
每个 seed 轮换组的执行顺序以减轻固定时间顺序的影响，但不能消除服务端随时间变化。
种子同时传入调度器和各模型客户端；远程端点可能忽略种子，不能保证完全复现。

显式创建独立 draft 客户端；classifier 默认复用 judge 的模型名及其自身角色路由。
RWI 在所有组统一关闭，因为它不参与方法决策；安全恢复轮保持执行。
自定义 YAML 可统一改变公共配置，但不得把完整基线中的四个被研究机制关掉，或改变
行为二元口径与两次补救的基线定义。`main.py --ablation` 也会应用预设的 Config 覆盖。

Pro/Qwen 现有 v3.15.1 结果由 v3.15 继承，不作为本次对照。运行时必须包含 `full`，
完整方法与消融组在同一套代码、数据和运行配置下重新执行。

## 指标与落盘

主指标是 R5 含补救 ASR。R5 根据 `RAW_GENRE_PROMPT` / `RAW_GENRE_PROMPT_RETRY`
/ `RAW_GENRE_PROMPT_RETRY_2` 识别，绝不按 `round_idx == 5` 识别。
辅助指标包括 R5 首轮 ASR、样本任意轮成功率、样本内最大 RiskScore 的均值、
build-up/补救轮数、硬拒绝早停数、各判定层使用次数、裁判失败标记数及各角色逻辑调用数。
目标逻辑调用数包括安全恢复与失败尝试，不包含客户端内部 HTTP 重试，亦非 token 或计费统计。

结果按 `seed_N/预设名/` 隔离；逐样本写入 `results.jsonl` 并 flush/fsync，
同时保存调度后验、有效配置和状态。根目录有代码哈希、数据哈希与完整计划 manifest，
以及逐组汇总、多种子均值/样本标准差和按同 ID 配对比较。配对差异以“full 减消融组”的
百分点表示；同时给出 full-only / ablation-only 数与探索性的双侧精确 McNemar p 值。
p 值未做多重比较校正，不应直接作为多组显著性结论；跨种子标准差也不是置信区间。

若缺样本，状态为 incomplete，ASR 和跨 seed 汇总不填入可误读为完整结果的数值，
并列出 missing_ids；程序退出码为 2。完成的配对要求双方都覆盖完整 cohort。
程序中断会保留已写入记录。当前入口不支持从半组断点续跑：随机流和其他在线状态并未
完整检查点化，不能仅恢复 Bandit 后验就声称等价续跑。重新执行应使用新输出目录，
可用 `--presets full,待重跑组` 和 `--seeds` 缩小范围。已有输出目录一律拒绝覆盖。

## 运行

以下命令从 `/root/vdriftbench` 执行。列出方案和检查计划均不会发起模型调用。

```bash
/hy-tmp/venv/bin/python scripts/run_ablation.py --list
/hy-tmp/venv/bin/python scripts/run_ablation.py --dry-run

# 离线验证：新输出目录，9 组，每组 2 个样本、1 个种子
/hy-tmp/venv/bin/python scripts/run_ablation.py --mock --limit 2 --seeds 0 \
  --out-dir /tmp/v3151_ablation_smoke

# 真实单模型完整实验：9 组 × 357 样本 × 3 种子 = 9639 个样本运行
# 需对应目标模型的角色路由/API 配置，显式使用独立起草模型和本地裁判。
/hy-tmp/venv/bin/python scripts/run_ablation.py \
  --env-file .env --data data/v3.15_eval357.enriched.jsonl \
  --target-model deepseek-v4-flash --draft-model draft-model --judge-model judge-model \
  --seeds 0,1,2 --out-dir data/ablation_v3151_flash

# 只研究 v3.15.1 历史选择改动；full 仍须重跑
/hy-tmp/venv/bin/python scripts/run_ablation.py \
  --presets full,v3151_legacy_r5_history --seeds 0,1,2 \
  --out-dir data/ablation_v3151_history
```

真实模型实验需单独执行上述命令；实现验证中的 mock 结果不代表模型安全行为。
