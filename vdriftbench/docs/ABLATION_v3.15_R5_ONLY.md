# V-Drift v3.15：仅 R5 消融（eval357 / GLM-5.3-Flash）

## 目的

隔离 **多轮 build-up 策略（P1–P7、Bandit、状态早停等）** 的贡献：直接进入 v3.15 的 **R5 题材改写 + 补救 + 安全复检**，与完整 v3.15（含 build-up）对照。

| 臂 | 行为 |
|---|---|
| **full**（对照） | 完整 v3.15（`v3.15_glm53flash_run` 代理，357 条，目标侧有多轮历史） |
| **`v315_r5_only`** | **无 build-up**；`rewrite_genre_prompt` 在无历史下改写 `sample.prompt`；保留 R5 补救与安全轮；目标默认 **带** R5 轮内多轮 API 历史（仅 R5 段） |

与 `v315_no_target_history` 的区别：本臂 **不跑 build-up**，不是「全 pipeline 但目标无记忆」。

## 运行

```bash
cd /root/vdriftbench

setsid -f bash scripts/deploy/run_ablation_v315_glm53flash_eval357_r5_only.sh \
  > data/ablation_v315_eval357_glm53flash_r5_only.log 2>&1 < /dev/null
```

进度：

```bash
tail -f data/ablation_v315_eval357_glm53flash_r5_only.log
wc -l data/ablation_v315_eval357_glm53flash_r5_only/seed_0/v315_r5_only/results.jsonl
```

主指标：R5 含补救 ASR；完成后见 `summary.json` 中相对 full 代理的配对 McNemar。
