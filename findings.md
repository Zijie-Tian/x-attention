# Findings: RULER Benchmark 非 Docker 环境兼容性

## Discovery 1: huggingface_hub 版本冲突（致命）

**Date:** 2026-01-13
**Source:** 运行 `call_api.py` 导入测试

### 错误信息
```
ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'
```

### 原因分析
- `ModelFilter` 类在 huggingface_hub 0.24+ 被废弃并移除
- NeMo 1.23.0 的 `nemo/core/classes/common.py:31` 仍依赖此类
- 当前环境: huggingface_hub **0.36.0**
- Docker 镜像: huggingface_hub **0.23.4**

### 影响
- 无法导入 `call_api.py`
- RULER 预测阶段完全无法运行

---

## Discovery 2: transformers 版本差异

**Date:** 2026-01-13
**Source:** 版本对比

### 版本对比
| 环境 | transformers 版本 |
|------|-------------------|
| RULER requirements.txt | 4.45.2 |
| Docker 镜像 | 4.45.2 |
| conda minference | 4.51.0 |

### 影响
- API 可能存在细微差异
- tokenizer 行为可能不同
- 风险级别：中等

---

## Discovery 3: transformer_engine 缺失

**Date:** 2026-01-13
**Source:** pip show 检查

### 现状
- Docker 镜像预装 transformer_engine 2.11.0
- 当前 conda 环境未安装

### 影响
- 某些 NeMo 功能可能受限
- 风险级别：低（非核心依赖）

---

## Discovery 4: 已正常工作的依赖

**Date:** 2026-01-13
**Source:** 导入测试

### 正常工作列表
| 包名 | 版本 | 状态 |
|------|------|------|
| PyTorch | 2.9.1+cu128 | ✅ |
| CUDA | 12.8 | ✅ |
| flash_attn | 2.8.3 | ✅ |
| flashinfer | 0.5.3 | ✅ |
| vllm | 0.12.1.dev1 | ✅ |
| nemo_toolkit | 1.23.0 | ✅ (被 huggingface_hub 阻塞) |
| nltk punkt_tab | - | ✅ |
| youtokentome | 1.0.6 | ✅ |
| Cython | 3.2.2 | ✅ |
| NanoVLLMModel wrapper | - | ✅ (可导入) |

---

## Discovery 5: Docker 镜像优势

**Date:** 2026-01-13
**Source:** docker inspect / docker run

### Docker 镜像 tzj/ruler:v0.3 特性
- 基于 nvcr.io/nvidia/pytorch:24.08-py3
- CUDA 12.6，NCCL 2.22.3
- 预编译的 transformer_engine
- 精确锁定的依赖版本
- 环境隔离，不影响主机

---

## 解决方案评估

### 方案 A: 降级包版本
```bash
pip install huggingface_hub==0.23.4 transformers==4.45.2
pip install transformer_engine[pytorch]
```
- **优点:** 快速，不需要新环境
- **缺点:** 可能破坏其他项目依赖

### 方案 B: 创建专用 conda 环境
```bash
conda create -n ruler python=3.10
conda activate ruler
pip install -r eval/RULER/requirements.txt
```
- **优点:** 隔离，不影响现有环境
- **缺点:** 需要管理多个环境

### 方案 C: 继续使用 Docker（推荐）
- **优点:** 完全隔离，版本精确匹配，已验证可用
- **缺点:** 需要 Docker，略有性能开销

---

## Discovery 6: NeMo 2.6.1 已修复 ModelFilter 问题

**Date:** 2026-01-13
**Source:** 网络调研 + pip dry-run 测试

### 关键发现

1. **ModelFilter 问题已修复**
   - GitHub Issue [#10272](https://github.com/NVIDIA-NeMo/NeMo/issues/10272) 已于 2024-09-07 关闭
   - NeMo 开发者确认: "It's been removed already from HF utils in NeMo"
   - NeMo 2.x 系列不再依赖已废弃的 `ModelFilter` 类

2. **huggingface_hub 兼容性**
   - NeMo 2.6.1 要求: `huggingface_hub>=0.24`
   - 当前环境 0.36.0 **完全兼容** ✅

3. **transformers 版本**
   - NeMo 2.6.1 核心包**不硬性依赖** transformers 特定版本
   - 但 NLP/LLM 功能可能有额外要求
   - 已知问题: transformers 4.41.0 移除了 `ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST`
   - 推荐版本: 4.40.0 或 4.51+ (绕过 4.41.0)

### NeMo 版本对比

| 特性 | NeMo 1.23.0 (当前) | NeMo 2.6.1 (最新) |
|------|-------------------|-------------------|
| huggingface_hub 要求 | `>=0.20.3` (实际需要 <0.24) | `>=0.24` ✅ |
| ModelFilter 依赖 | ❌ 有 (已废弃) | ✅ 已移除 |
| transformers 要求 | 4.45.2 (严格) | 宽松 (4.51+ 可用) |
| Python 要求 | >=3.8 | >=3.10 |
| 架构 | NeMo 1.x | NeMo 2.x (重构) |

### NeMo 2.x 重大变化 (2025年12月公告)

> "As of December 2025, this NeMo repo will pivot to focus on speech models collections only.
> NeMo 2.0, with its support for LLMs and VLMs will be deprecated by 25.11,
> and replaced by NeMo Megatron-Bridge and NeMo AutoModel."

**影响:**
- NeMo 2.x 主仓库将专注于语音模型
- LLM/VLM 功能迁移到 NeMo Megatron-Bridge 和 NeMo AutoModel
- 对于 RULER benchmark (主要使用 ASR/NLP 功能)，NeMo 2.6.1 仍可用

---

## Discovery 7: 升级到 NeMo 2.6.1 的风险评估

**Date:** 2026-01-13
**Source:** 依赖分析

### 潜在风险

1. **API 不兼容**
   - NeMo 1.x → 2.x 是重大版本升级
   - RULER 的 `call_api.py` 可能需要修改导入路径
   - 模块组织结构可能已变化

2. **RULER 代码兼容性未知**
   - RULER 原本针对 NeMo 1.23.0 设计
   - 需要测试 `nemo.collections.asr.parts.utils.manifest_utils` 是否存在

3. **其他依赖连锁反应**
   - 升级 NeMo 可能触发其他包版本变化
   - 需要在隔离环境中测试

### 建议测试步骤

```bash
# 1. 创建隔离测试环境
conda create -n nemo26-test python=3.10 -y
conda activate nemo26-test

# 2. 安装 NeMo 2.6.1
pip install nemo-toolkit[asr]==2.6.1

# 3. 测试关键导入
python -c "from nemo.collections.asr.parts.utils.manifest_utils import read_manifest; print('OK')"

# 4. 测试 RULER call_api.py 导入
cd /path/to/x-attention
python -c "import sys; sys.path.insert(0, 'eval/RULER/scripts/pred'); from call_api import *"
```

---

## 更新后的解决方案评估

### 方案 D: 升级到 NeMo 2.6.1 (新增)

```bash
# 在隔离环境中测试
conda create -n ruler-nemo26 python=3.10 -y
conda activate ruler-nemo26
pip install nemo-toolkit[asr]==2.6.1
pip install flash-attn flashinfer  # 如果需要
```

**优点:**
- 解决 huggingface_hub 版本冲突
- 与当前 transformers 4.51.0 兼容
- 不需要降级任何包

**缺点:**
- RULER 代码可能需要适配 NeMo 2.x API
- 需要验证 ASR/NLP 模块路径是否变化
- 重大版本升级风险

**推荐度:** ⭐⭐⭐ (中等 - 需要测试验证)

---

---

## Discovery 8: 固定版本库的 transformers 约束冲突

**Date:** 2026-01-13
**Source:** pip show + pkg_resources 分析

### 固定版本库的 transformers 要求

| 库 | 版本 | transformers 要求 |
|----|------|------------------|
| torch | 2.9.1 | 无 |
| vllm | 0.12.1.dev1 | `>=4.56.0, <5` |
| flashinfer | 0.5.3 | 无 |
| flash_attn | 2.8.3 | 无 |
| sglang | 0.5.6 | `==4.57.1` (严格!) |

### NeMo 2.6.1 的 transformers 要求

| Extra | transformers 要求 |
|-------|------------------|
| base (无 extra) | **无要求** ✅ |
| [asr] | `~=4.53.0` (4.53.x) |
| [nlp] | `~=4.53.0` (4.53.x) |

### 核心矛盾

```
sglang:    transformers == 4.57.1  (严格)
vllm:      transformers >= 4.56.0
NeMo ASR:  transformers ~= 4.53.0  (4.53.x only)

结论: 不存在同时满足三者的 transformers 版本!
```

### 当前环境实际状态

```bash
$ pip check | grep transform
sglang 0.5.6 has requirement transformers==4.57.1, but you have transformers 4.51.0.
vllm 0.12.1.dev1 has requirement transformers>=4.56.0, but you have transformers 4.51.0.
```

当前环境 transformers 4.51.0 实际上与 vllm/sglang 都不兼容，但仍能运行（可能是 --no-deps 安装或功能未触发冲突）。

---

## Discovery 9: RULER 对 NeMo 的实际依赖极简

**Date:** 2026-01-13
**Source:** grep 分析 RULER 代码

### RULER 使用的 NeMo 功能

```python
# 仅使用两个模块:
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
```

### manifest_utils 功能分析

这是简单的 JSONL 工具函数:
- `read_manifest`: 读取 JSONL 文件，返回字典列表
- `write_manifest`: 将字典列表写入 JSONL 文件

**完全可以用 10 行 Python 代码替代!**

### 推荐方案: 本地实现替代

```python
# eval/RULER/scripts/utils/manifest_utils.py
import json

def read_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_manifest(manifest_path, data, ensure_ascii=True):
    with open(manifest_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + '\n')
```

---

## Sources

- [NeMo Issue #10272: stop using ModelFilter](https://github.com/NVIDIA-NeMo/NeMo/issues/10272)
- [NeMo Issue #9793: ModelFilter import error](https://github.com/NVIDIA-NeMo/NeMo/issues/9793)
- [NeMo Issue #9272: transformers version breaking NLP modules](https://github.com/NVIDIA-NeMo/NeMo/issues/9272)
- [NeMo PyPI](https://pypi.org/project/nemo-toolkit/)
- [huggingface_hub Issue #2710](https://github.com/huggingface/huggingface_hub/issues/2710)
- [vLLM Releases](https://github.com/vllm-project/vllm/releases)
