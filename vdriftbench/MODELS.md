# 外链与下载清单（唯一说明文件）

本仓库**不上传**任何需从网络拉取的大文件（Hugging Face 权重、本地缓存目录等），只在此文件记录下载地址与复现方式。  
实验数据集与结果 JSONL 已在 `data/` 中随库提交；**模型权重默认落在 `/hy-tmp/hf_cache`（约 45GB），勿提交 Git**。

> 本环境常无法直连 `huggingface.co`，下载与加载时建议设置 `HF_ENDPOINT=https://hf-mirror.com`（见下文）。

---

## 1. Hugging Face 本地权重（必下，不入库）

| 角色 | 模型 ID | 约大小 | 部署 | HF 原始 | 国内镜像 (hf-mirror) |
| --- | --- | ---: | --- | --- | --- |
| 起草 draft | `ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ` | ~19 GB | 端口 8001，`scripts/deploy/serve_draft.sh` | [huggingface.co/…/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ](https://huggingface.co/ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ) | [hf-mirror.com/…/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ](https://hf-mirror.com/ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ) |
| 裁判 judge | `mistralai/Mistral-7B-Instruct-v0.3` | ~15 GB | 端口 8002，`scripts/deploy/serve_judge.sh` | [huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | [hf-mirror.com/mistralai/Mistral-7B-Instruct-v0.3](https://hf-mirror.com/mistralai/Mistral-7B-Instruct-v0.3) |
| 合理性 plausibility | 与 draft 相同 | — | 不单独部署 | 同上 | 同上 |
| Embedding | `BAAI/bge-m3` | ~2 GB | 进程内 `sentence-transformers` | [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | [hf-mirror.com/BAAI/bge-m3](https://hf-mirror.com/BAAI/bge-m3) |

### 一键下载（推荐）

```bash
cd /path/to/vdriftbench
export HF_HOME=/hy-tmp/hf_cache          # 或任意大磁盘路径
export HF_ENDPOINT=https://hf-mirror.com # 可选；脚本内也有默认
python scripts/deploy/download_models.py --all
```

仅下载子集：

```bash
python scripts/deploy/download_models.py --only embedding,judge
python scripts/deploy/download_models.py --only draft
```

缓存目录可用 `--cache-dir` 覆盖；需 gated 仓库时在环境变量中设置 `HF_TOKEN` 后重跑（见下「备选模型」）。

### 角色说明（简）

- **Draft**：原设计 `huihui-ai/Qwen2.5-32B-Instruct-abliterated` 为 bf16 全量（~65GB），单卡 48GB 放不下；实际使用上表 AWQ 量化版。
- **Judge**：原设计 `meta-llama/Llama-3.1-8B-Instruct` 为 gated，无 HF 授权 token 时改用 Mistral-7B，保持「裁判 ≠ 目标模型家族」。
- **vLLM 对外名称**：draft → `draft-model`，judge → `judge-model`（与 CLI `--draft-model` / `--judge-model` 一致）。

---

## 2. 被测目标模型（远程 API，无权重文件）

| 模型名（示例） | 用途 | API 文档 / 控制台 | 默认 Base URL | 环境变量 |
| --- | --- | --- | --- | --- |
| `deepseek-v4-flash` | 主实验 target | [platform.deepseek.com](https://platform.deepseek.com/) | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `deepseek-v4-pro` | 部分 v3.15 消融 | 同上 | 同上 | `DEEPSEEK_API_KEY` |
| `ZHIPU/GLM-5.3-Flash` | GLM 系列 target | [DashScope 模型服务](https://help.aliyun.com/zh/model-studio/getting-started/models) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| `qwen-plus` 等 `qwen*` | 通义 target（可选） | 同上 | 同上 | `DASHSCOPE_API_KEY` |

密钥与端点模板见仓库根目录 `.env.example`（`.env` 本身**不入库**）。

---

## 3. 备选 / 未默认使用的 Hugging Face 仓库

| 说明 | 模型 ID | HF 原始 | 镜像 |
| --- | --- | --- | --- |
| Draft 原 bf16 全量（过大） | `huihui-ai/Qwen2.5-32B-Instruct-abliterated` | [链接](https://huggingface.co/huihui-ai/Qwen2.5-32B-Instruct-abliterated) | [镜像](https://hf-mirror.com/huihui-ai/Qwen2.5-32B-Instruct-abliterated) |
| Judge 原选型（gated，需 `HF_TOKEN`） | `meta-llama/Llama-3.1-8B-Instruct` | [链接](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [镜像](https://hf-mirror.com/meta-llama/Llama-3.1-8B-Instruct) |

改用 Judge 全量 Llama 时示例：

```bash
HF_TOKEN=hf_xxx python scripts/deploy/download_models.py --only judge \
  --judge-repo meta-llama/Llama-3.1-8B-Instruct
```

---

## 4. Python 依赖（PyPI，非 Git 大文件）

无单独权重包；在干净环境中安装：

```bash
pip install -e ".[llm,embedding,calibration,dev]"
# 或
pip install -r requirements.txt
```

本地 vLLM 服务需自行在运行环境中安装 [vLLM](https://github.com/vllm-project/vllm)（版本与 CUDA 匹配，见 `scripts/deploy/serve_*.sh`）。

---

## 5. 明确不要提交 Git 的内容

| 路径 / 模式 | 说明 |
| --- | --- |
| `/hy-tmp/hf_cache` 及任意 `models--*` 缓存 | Hugging Face 已下载权重 |
| `*.safetensors`, `*.pt`, `*.pth`, `*.bin`, `*.gguf`, `*.ckpt` | 权重文件 |
| `.env` | API Key |
| `models/`（若本地另存权重） | 本地模型目录 |

`.gitignore` 已忽略上述权重后缀；**下载地址以本文件为准，勿再维护 `docs/MODELS.md` 等副本。**
