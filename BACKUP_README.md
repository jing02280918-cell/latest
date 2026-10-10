# 服务器 `/root` 备份说明

本仓库是运行环境 **`/root` 目录的快照**（用于恢复脚本、实验结果与项目代码），**不是**可启动的整机镜像。

## 未纳入备份的内容

| 类别 | 典型路径 | 说明 |
| --- | --- | --- |
| 模型权重 | `/hy-tmp/hf_cache` | 见根目录 **`DOWNLOADS.md`** |
| Python 虚拟环境 | `/hy-tmp/venv` 等 | 用各项目 `requirements.txt` / `pyproject.toml` 重建 |
| IDE / 远程会话缓存 | `.cache/`, `.cursor-server/`, `.vscode-server/` | 可再生成 |
| 密钥与凭据 | `.ssh/`, `**/.env` | 仅保留 `*.example` |
| Git 元数据 | 各子目录内原 `.git/` | 避免嵌套仓库；上游地址见 **`DOWNLOADS.md`** |
| 大体积 bundle | `vdriftbench-backup.bundle` | 与当前树重复 |
| 超大日志 | 如 `crescendo_baseline_v315_qwen37plus/driver.log`（>100MB） | GitHub 单文件限制，仅留服务器本地 |

## 主要目录

| 路径 | 说明 |
| --- | --- |
| `vdriftbench/` | V-DriftBench v3.15 主项目 |
| `PyRIT/`, `PLAGUE/`, `JailbreakingLLMs/`, `cop/` | 对照/参考仓库（源码快照） |
| `pair_baseline_*`, `crescendo_baseline_*` | 基线实验输出与脚本 |
| `run_pair_baseline.py` 等 | `/root` 级运行脚本 |

恢复时：克隆本仓库到目标机 `root` 或任意目录，按 **`DOWNLOADS.md`** 拉取模型与依赖，再复制 `.env`（勿提交 Git）。
