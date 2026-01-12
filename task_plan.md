# Task Plan: 脱离 Docker 运行 RULER (使用 conda 环境)

## Goal
在不修改 torch、flashinfer、flashattn 版本的前提下，新建 conda 环境，脱离 Docker 运行 RULER benchmark，并尽可能提高 transformers 版本。

## Status: `planning`

---

## 原始 RULER requirements.txt 分析

```
nemo-toolkit[all]           # NeMo 全量安装 → 改为 base
tritonclient[all]           # Triton 推理客户端
transformer_engine[pytorch] # NVIDIA Transformer Engine
flask                       # Web 框架
flask_restful              # REST API
html2text                  # HTML 转文本
google-generativeai        # Gemini API
sshtunnel_requests         # SSH 隧道
wonderwords                # 随机词生成
openai                     # OpenAI API
tiktoken                   # Tokenizer
tenacity                   # 重试逻辑
accelerate                 # HuggingFace Accelerate
huggingface_hub==0.23.4    # → 升级到 0.36.0+
transformers==4.45.2       # → 升级到 4.57.3
torch==2.4.0               # → 升级到 2.9.1 (固定)
torchaudio==2.4.0          # → 升级到 2.9.1
torchdiffeq==0.2.5         # 微分方程求解
torchmetrics==1.6.1        # → 升级到 1.8.2
torchsde==0.2.6            # 随机微分方程
torchvision==0.19.0        # → 升级到 0.24.1
```

---

## 完整包需求规划

### 1. 核心深度学习框架 (固定版本)

| 包名 | 原版本 | 目标版本 | 安装方式 | 说明 |
|------|--------|---------|----------|------|
| torch | 2.4.0 | **2.9.1** | pip | 固定，CUDA 12.8 |
| torchaudio | 2.4.0 | **2.9.1** | pip | 匹配 torch 版本 |
| torchvision | 0.19.0 | **0.24.1** | pip | 匹配 torch 版本 |
| flash-attn | - | **源码编译** | submodule | 从 fork 仓库编译 |
| flashinfer | - | **源码编译** | submodule | 从 fork 仓库编译 |

### 1.1 Flash-Attn 和 FlashInfer 源码编译配置

**重要**: flash-attn 和 flashinfer 需要从用户 fork 的仓库源码编译安装。

| 项目 | 仓库地址 | 分支 | 安装路径 |
|------|---------|------|----------|
| flash-attention | `git@github.com:Zijie-Tian/flash-attention.git` | `tzj/minference` | `/home/tzj/Code/COMPASS/3rdparty/flash-attention` |
| flashinfer | `git@github.com:Zijie-Tian/flashinfer.git` | `tzj/minference` | `/home/tzj/Code/COMPASS/3rdparty/flashinfer` |

**安装顺序**:
1. 先在 `/home/tzj/Code/COMPASS` 下创建 `3rdparty` 目录
2. 添加两个仓库作为 git submodule
3. 切换到 `tzj/minference` 分支
4. 源码编译安装

### 2. HuggingFace 生态 (升级版本)

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| transformers | 4.45.2 | **4.57.3** | 升级到最新 |
| huggingface_hub | 0.23.4 | **0.36.0+** | 自动升级 |
| accelerate | - | **1.12.0** | 最新版 |
| tokenizers | - | **0.21.0+** | transformers 依赖 |
| safetensors | - | **0.5.0+** | transformers 依赖 |

### 3. NVIDIA 工具链 (可选/按需)

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| nemo-toolkit | [all] | **2.6.1 (base)** | 仅安装 base，无 extra |
| transformer_engine | [pytorch] | **2.11.0** | 可选，需要时安装 |
| tritonclient | [all] | **2.63.0** | 可选，Triton 服务时需要 |

### 4. Tokenizer 相关

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| tiktoken | - | **0.12.0** | OpenAI tokenizer |
| sentencepiece | - | **0.2.1** | SP tokenizer |

### 5. API 客户端

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| openai | - | **2.6.1** | OpenAI API |
| google-generativeai | - | **0.8.5** | Gemini API |

### 6. Web 框架 (API 服务用)

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| flask | - | **2.2.5** | Web 框架 |
| flask_restful | - | **0.3.10** | REST API |

### 7. 工具库

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| wonderwords | - | **3.0.1** | 随机词生成 (RULER 数据生成) |
| tenacity | - | **9.1.2** | 重试逻辑 |
| html2text | - | **2025.4.15** | HTML 转文本 |
| nltk | - | **3.9.1** | 自然语言工具包 |

### 8. PyTorch 扩展 (可选)

| 包名 | 原版本 | 目标版本 | 说明 |
|------|--------|---------|------|
| torchmetrics | 1.6.1 | **1.8.2** | 评估指标 |
| torchdiffeq | 0.2.5 | **0.2.5** | 保持原版本 |
| torchsde | 0.2.6 | **0.2.6** | 保持原版本 |

### 9. 本地实现 (替代 NeMo 依赖)

| 文件 | 功能 | 说明 |
|------|------|------|
| `utils/manifest_utils.py` | JSONL 读写 | 替代 nemo.collections.asr |

---

## 新建 conda 环境的完整命令

### 方案 A: 最小安装 (仅 RULER 核心功能)

```bash
# ============================================
# Phase 0: 设置 3rdparty submodules
# ============================================

# 0.1 创建 3rdparty 目录
mkdir -p /home/tzj/Code/COMPASS/3rdparty
cd /home/tzj/Code/COMPASS

# 0.2 添加 flash-attention submodule
git submodule add -b tzj/minference \
    git@github.com:Zijie-Tian/flash-attention.git \
    3rdparty/flash-attention

# 0.3 添加 flashinfer submodule
git submodule add -b tzj/minference \
    git@github.com:Zijie-Tian/flashinfer.git \
    3rdparty/flashinfer

# 0.4 初始化 submodules (如果已添加)
git submodule update --init --recursive

# ============================================
# Phase 1: 创建 conda 环境并安装基础依赖
# ============================================

# 1. 创建环境
conda create -n ruler python=3.10 -y
conda activate ruler

# 2. 安装 PyTorch 生态 (固定版本)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================
# Phase 2: 源码编译 Flash-Attn 和 FlashInfer
# ============================================

# 3. 编译安装 Flash Attention
cd /home/tzj/Code/COMPASS/3rdparty/flash-attention
pip install ninja packaging
pip install -e . --no-build-isolation

# 4. 编译安装 FlashInfer
cd /home/tzj/Code/COMPASS/3rdparty/flashinfer
pip install -e . --no-build-isolation

# ============================================
# Phase 3: 安装其他依赖
# ============================================

# 5. 安装 HuggingFace 生态 (升级版本)
pip install transformers==4.57.3 accelerate==1.12.0

# 6. 安装 Tokenizers
pip install tiktoken==0.12.0 sentencepiece==0.2.1

# 7. 安装 RULER 数据生成依赖
pip install wonderwords==3.0.1 nltk==3.9.1

# 8. 安装 API 客户端 (按需)
pip install openai==2.6.1 tenacity==9.1.2

# 9. 安装 NeMo base (可选，如果需要其他 NeMo 功能)
# pip install nemo-toolkit==2.6.1

# 10. 下载 NLTK 数据
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
```

### 方案 B: 完整安装 (包含所有可选功能)

```bash
# ============================================
# Phase 0: 设置 3rdparty submodules (同方案 A)
# ============================================

mkdir -p /home/tzj/Code/COMPASS/3rdparty
cd /home/tzj/Code/COMPASS

git submodule add -b tzj/minference \
    git@github.com:Zijie-Tian/flash-attention.git \
    3rdparty/flash-attention

git submodule add -b tzj/minference \
    git@github.com:Zijie-Tian/flashinfer.git \
    3rdparty/flashinfer

git submodule update --init --recursive

# ============================================
# Phase 1: 创建 conda 环境
# ============================================

conda create -n ruler-full python=3.10 -y
conda activate ruler-full

# 安装 PyTorch 生态
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================
# Phase 2: 源码编译 Flash-Attn 和 FlashInfer
# ============================================

# 编译安装 Flash Attention
cd /home/tzj/Code/COMPASS/3rdparty/flash-attention
pip install ninja packaging
pip install -e . --no-build-isolation

# 编译安装 FlashInfer
cd /home/tzj/Code/COMPASS/3rdparty/flashinfer
pip install -e . --no-build-isolation

# ============================================
# Phase 3: 安装其他依赖
# ============================================

# 安装 HuggingFace 生态
pip install transformers==4.57.3 accelerate==1.12.0

# 安装 NeMo base
pip install nemo-toolkit==2.6.1

# 安装 Transformer Engine (可选，GPU 优化)
pip install transformer_engine==2.11.0

# 安装 Triton Client (可选，Triton 服务)
pip install tritonclient[all]==2.63.0

# 安装 Tokenizers
pip install tiktoken==0.12.0 sentencepiece==0.2.1

# 安装 RULER 依赖
pip install wonderwords==3.0.1 nltk==3.9.1 tenacity==9.1.2

# 安装 API 客户端
pip install openai==2.6.1 google-generativeai==0.8.5

# 安装 Web 框架 (vLLM/sglang 服务用)
pip install flask==2.2.5 flask_restful==0.3.10

# 安装 PyTorch 扩展
pip install torchmetrics==1.8.2 torchdiffeq==0.2.5 torchsde==0.2.6

# 安装 html2text
pip install html2text==2025.4.15

# 下载 NLTK 数据
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
```

---

## 新版 requirements.txt (建议)

创建 `eval/RULER/requirements-ruler.txt`:

```txt
# ===========================================
# RULER Benchmark Requirements (Upgraded)
# ===========================================
# PyTorch 2.9.1 + CUDA 12.8 + transformers 4.57.3
# ===========================================

# --- Core PyTorch (固定版本) ---
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1

# --- Flash Attention & FlashInfer (源码编译) ---
# 需要从 fork 仓库源码编译安装，不通过 pip:
#   flash-attention: git@github.com:Zijie-Tian/flash-attention.git (tzj/minference)
#   flashinfer: git@github.com:Zijie-Tian/flashinfer.git (tzj/minference)
# 安装路径: /home/tzj/Code/COMPASS/3rdparty/

# --- HuggingFace 生态 (升级版本) ---
transformers==4.57.3
accelerate==1.12.0
# huggingface_hub>=0.36.0  # transformers 依赖自动安装

# --- Tokenizers ---
tiktoken==0.12.0
sentencepiece==0.2.1

# --- RULER 数据生成 ---
wonderwords==3.0.1
nltk>=3.9.0

# --- API 客户端 ---
openai>=2.0.0
tenacity>=9.0.0

# --- 可选: NeMo base (不带 ASR/NLP extra) ---
# nemo-toolkit==2.6.1

# --- 可选: NVIDIA 工具 ---
# transformer_engine==2.11.0
# tritonclient[all]==2.63.0

# --- 可选: Web 框架 (API 服务) ---
# flask>=2.2.0
# flask_restful>=0.3.10

# --- 可选: PyTorch 扩展 ---
# torchmetrics>=1.8.0
# torchdiffeq==0.2.5
# torchsde==0.2.6

# --- 其他 ---
# google-generativeai>=0.8.0  # Gemini API
# html2text>=2024.0.0
```

---

## 执行计划

### Phase 0: 设置 3rdparty Submodules (新增)
- **Status:** `pending`
- **Location:** `/home/tzj/Code/COMPASS/3rdparty/`
- **Submodules:**
  - `flash-attention` from `git@github.com:Zijie-Tian/flash-attention.git` @ `tzj/minference`
  - `flashinfer` from `git@github.com:Zijie-Tian/flashinfer.git` @ `tzj/minference`
- **Effort:** 中
- **Notes:** 需要先确认 COMPASS 仓库存在，然后添加 submodules

### Phase 1: 创建本地 manifest_utils
- **Status:** `pending`
- **File:** `eval/RULER/scripts/utils/manifest_utils.py`
- **Effort:** 低

### Phase 2: 修改 RULER 导入路径
- **Status:** `pending`
- **Files:** 7 个 Python 文件
- **Effort:** 中

### Phase 3: 处理 SentencePieceTokenizer
- **Status:** `pending`
- **File:** `eval/RULER/scripts/data/tokenizer.py`
- **Effort:** 低

### Phase 4: 创建 conda 环境并源码编译
- **Status:** `pending`
- **Commands:** 见上方 "方案 A: 最小安装"
- **Sub-steps:**
  1. 创建 conda 环境
  2. 安装 PyTorch 生态
  3. 源码编译 flash-attention (从 submodule)
  4. 源码编译 flashinfer (从 submodule)
  5. 安装其他 pip 依赖
- **Effort:** 高 (编译耗时)

### Phase 5: 创建新版 requirements.txt
- **Status:** `pending`
- **File:** `eval/RULER/requirements-ruler.txt`
- **Effort:** 低

### Phase 6: 验证 RULER 运行
- **Status:** `pending`
- **Test:** 运行 RULER benchmark
- **Effort:** 低

---

## 版本对比总结

| 包名 | 原版本 | 新版本 | 安装方式 | 变化 |
|------|--------|--------|----------|------|
| torch | 2.4.0 | 2.9.1 | pip | ⬆️ +0.5.1 |
| torchvision | 0.19.0 | 0.24.1 | pip | ⬆️ +0.5.1 |
| torchaudio | 2.4.0 | 2.9.1 | pip | ⬆️ +0.5.1 |
| **transformers** | **4.45.2** | **4.57.3** | pip | ⬆️ **+0.12.1** |
| huggingface_hub | 0.23.4 | 0.36.0+ | pip (auto) | ⬆️ +0.13+ |
| nemo-toolkit | [all] | base only | pip | 🔄 简化 |
| flash-attn | pip wheel | **源码编译** | submodule | 🔧 Fork 仓库 |
| flashinfer | pip wheel | **源码编译** | submodule | 🔧 Fork 仓库 |

---

## Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| 移除 sglang/vllm 依赖 | 解除 transformers 版本限制 | 2026-01-13 |
| 使用本地 manifest_utils | 避免 NeMo ASR extra 依赖 | 2026-01-13 |
| transformers 4.57.3 | 用户要求最新版本 | 2026-01-13 |
| NeMo base only | 避免 transformers ~=4.53.0 约束 | 2026-01-13 |
| flash-attn 源码编译 | 用户要求从 fork 仓库编译 | 2026-01-13 |
| flashinfer 源码编译 | 用户要求从 fork 仓库编译 | 2026-01-13 |
| 使用 git submodule | 用户要求在 COMPASS/3rdparty 下管理 | 2026-01-13 |

---

## Next Actions

1. [ ] 用户确认此计划
2. [ ] 确认 `/home/tzj/Code/COMPASS` 仓库存在
3. [ ] 创建 `3rdparty` 目录并添加 submodules
4. [ ] 创建 `eval/RULER/scripts/utils/__init__.py`
5. [ ] 创建 `eval/RULER/scripts/utils/manifest_utils.py`
6. [ ] 修改 7 个 RULER 文件的导入
7. [ ] 处理 `tokenizer.py` 的 SentencePieceTokenizer
8. [ ] 创建 conda 环境
9. [ ] 源码编译 flash-attention
10. [ ] 源码编译 flashinfer
11. [ ] 安装其他 pip 依赖
12. [ ] 创建 `eval/RULER/requirements-ruler.txt`
13. [ ] 测试 RULER 运行

---

## 3rdparty Submodule 详细说明

### 目录结构

```
/home/tzj/Code/COMPASS/
├── .gitmodules           # submodule 配置文件
├── 3rdparty/
│   ├── flash-attention/  # Zijie-Tian/flash-attention @ tzj/minference
│   │   ├── csrc/
│   │   ├── flash_attn/
│   │   ├── setup.py
│   │   └── ...
│   └── flashinfer/       # Zijie-Tian/flashinfer @ tzj/minference
│       ├── csrc/
│       ├── flashinfer/
│       ├── setup.py
│       └── ...
└── ...
```

### .gitmodules 内容预览

```ini
[submodule "3rdparty/flash-attention"]
    path = 3rdparty/flash-attention
    url = git@github.com:Zijie-Tian/flash-attention.git
    branch = tzj/minference

[submodule "3rdparty/flashinfer"]
    path = 3rdparty/flashinfer
    url = git@github.com:Zijie-Tian/flashinfer.git
    branch = tzj/minference
```

### 编译注意事项

1. **环境变量** (可选，加速编译):
   ```bash
   export MAX_JOBS=32  # 并行编译任务数
   ```

2. **依赖**: 编译前确保安装:
   ```bash
   pip install ninja packaging wheel setuptools
   ```

3. **CUDA 版本**: 确保系统 CUDA 版本与 PyTorch CUDA 版本匹配 (12.8)

4. **编译时间**:
   - flash-attention: 约 10-30 分钟
   - flashinfer: 约 10-30 分钟

---

## Sources

- [PyTorch Versions](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions)
- [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)
- [Torchaudio Installation](https://docs.pytorch.org/audio/stable/installation.html)
- [NeMo PyPI](https://pypi.org/project/nemo-toolkit/)
- [Transformers PyPI](https://pypi.org/project/transformers/)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
