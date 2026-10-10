# 外链与下载清单（服务器级，唯一入口）

大文件与缓存**不入库**；按本节与 `vdriftbench/MODELS.md` 在目标机器上下载即可。

---

## 1. V-DriftBench 本地模型（Hugging Face，约 45GB）

完整表格、镜像与 `download_models.py` 用法见：

**[`vdriftbench/MODELS.md`](vdriftbench/MODELS.md)**

默认缓存目录：`/hy-tmp/hf_cache`（不在本仓库内）。

---

## 2. 第三方 Git 仓库（本备份不含 `.git`，需自行 clone）

| 目录 | 上游地址 |
| --- | --- |
| `PyRIT/` | https://github.com/microsoft/PyRIT.git |
| `PLAGUE/` | https://github.com/madhavaggar/PLAGUE.git |
| `JailbreakingLLMs/` | https://github.com/patrickrchao/JailbreakingLLMs.git |
| `cop/` | https://github.com/IBM/cop.git |
| `vdriftbench/`（上游参考） | https://github.com/jing02280918-cell/vdriftbench.git |

本备份以**文件快照**为准；与上游 diff 时可用上表重新 clone 对比。

---

## 3. 运行时环境（不在 `/root` 备份内）

| 资源 | 路径 / 获取方式 |
| --- | --- |
| 实验用 venv | `/hy-tmp/venv` — 在目标机 `python -m venv` 后 `pip install -r vdriftbench/requirements.txt` 等 |
| vLLM 服务 | 见 `vdriftbench/scripts/deploy/serve_draft.sh`、`serve_judge.sh` 与 [vLLM 安装文档](https://github.com/vllm-project/vllm) |
| DeepSeek / DashScope API | 控制台申请 Key，本地配置 `vdriftbench/.env`（模板：`vdriftbench/.env.example`） |

---

## 4. 权重文件后缀（一律勿提交 Git）

`*.safetensors`, `*.pt`, `*.pth`, `*.bin`, `*.gguf`, `*.ckpt`, `models--*` 缓存目录。
