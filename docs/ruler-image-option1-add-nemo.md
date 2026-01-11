# 方案1：基于 tzj/xattn:v0.6 安装 NeMo

## 问题背景

RULER 基准测试需要同时满足两个依赖：

1. **NeMo 框架**：RULER 的数据生成、预测和评估脚本依赖 `nemo.collections.asr.parts.utils.manifest_utils`
2. **block_sparse_attn**：x-attention 的 AvgPool 等稀疏注意力实现依赖此库

当前镜像状态：

| 镜像 | PyTorch | CUDA | NeMo | block_sparse_attn | flash-attn | flashinfer |
|------|---------|------|------|-------------------|------------|------------|
| `tzj/ruler:v0.3` | 2.5.0 | 12.6 | ✅ | ❌ | ✅ | ✅ |
| `tzj/xattn:v0.6` | 2.9.1 | 12.8 | ❌ | ✅ | ✅ | ✅ |

## 需要解决的问题

### 1. NeMo 安装复杂度

NeMo 是一个大型深度学习框架，包含多个子模块（ASR、NLP、TTS 等）。RULER 只需要其中的 manifest 工具函数：

```python
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
```

完整安装 NeMo 会带来：
- 大量依赖包（megatron-core, apex, transformer-engine 等）
- 潜在的版本冲突
- 镜像体积显著增大

### 2. 版本兼容性

`tzj/xattn:v0.6` 基于 PyTorch 2.9.1 + CUDA 12.8，需要确认 NeMo 与此环境兼容。

NeMo 版本矩阵（参考）：
- NeMo 2.0+ 需要 PyTorch 2.2+
- NeMo 1.x 需要 PyTorch 1.13-2.1

### 3. huggingface_hub 版本冲突

测试中发现 NeMo 依赖旧版 `huggingface_hub`（需要 `ModelFilter` 类），但新版已移除此类。需要：
- 固定 huggingface_hub 版本，或
- 使用 `PYTHONNOUSERSITE=1` 隔离主机包

## 初步思路

### 方案 A：最小化安装（推荐）

只安装 RULER 实际需要的 NeMo 组件，而非完整框架。

```dockerfile
# 基于 tzj/xattn:v0.6
FROM tzj/xattn:v0.6

# 安装 NeMo 核心依赖（不含完整框架）
RUN pip install \
    nemo_toolkit[asr]==2.0.0 \
    huggingface_hub==0.21.4 \
    --no-deps

# 或者只安装 manifest 工具所需的最小依赖
RUN pip install \
    soundfile \
    librosa \
    editdistance
```

**优点**：
- 安装快速
- 镜像体积增加较少
- 减少版本冲突风险

**缺点**：
- 可能缺少某些间接依赖
- 需要测试确认功能完整

### 方案 B：完整安装 NeMo

```dockerfile
FROM tzj/xattn:v0.6

# 安装完整 NeMo 框架
RUN pip install nemo_toolkit[all]==2.0.0

# 固定 huggingface_hub 版本避免冲突
RUN pip install huggingface_hub==0.21.4
```

**优点**：
- 功能完整
- 官方支持的安装方式

**缺点**：
- 安装耗时长（30+ 分钟）
- 镜像体积大幅增加（+10GB）
- 可能引入版本冲突

### 方案 C：提取 manifest 工具

将 NeMo 的 manifest 工具提取为独立模块，避免完整依赖。

```python
# eval/RULER/scripts/utils/manifest_utils.py
# 从 NeMo 提取的最小化 manifest 读写功能

import json

def read_manifest(path):
    """Read a NeMo manifest file (jsonl format)."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def write_manifest(path, data):
    """Write a NeMo manifest file (jsonl format)."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
```

**优点**：
- 零外部依赖
- 不影响镜像
- 代码完全可控

**缺点**：
- 需要修改 RULER 脚本的导入路径
- 可能缺少 NeMo manifest 的高级功能

## 实施步骤

### 步骤 1：测试 NeMo 安装

```bash
# 在容器中测试安装
docker run --rm -it tzj/xattn:v0.6 bash

# 尝试最小化安装
pip install nemo_toolkit[asr]==2.0.0 huggingface_hub==0.21.4

# 测试导入
python -c "from nemo.collections.asr.parts.utils.manifest_utils import read_manifest; print('OK')"
```

### 步骤 2：验证兼容性

```bash
# 确认 block_sparse_attn 仍可用
python -c "from block_sparse_attn import block_sparse_attn_func; print('OK')"

# 确认 flash-attn 仍可用
python -c "from flash_attn import flash_attn_func; print('OK')"
```

### 步骤 3：构建新镜像

```bash
# 创建 Dockerfile
cat > Dockerfile.ruler-xattn << 'EOF'
FROM tzj/xattn:v0.6

# 安装 NeMo 和相关依赖
RUN pip install \
    nemo_toolkit[asr]==2.0.0 \
    huggingface_hub==0.21.4 \
    wonderwords \
    nltk

# 下载 NLTK 数据
RUN python -c "import nltk; nltk.download('punkt_tab')"
EOF

# 构建
docker build -t tzj/ruler-xattn:v0.1 -f Dockerfile.ruler-xattn .
```

### 步骤 4：运行 RULER 测试

```bash
# 更新 run_ruler_docker.sh 使用新镜像
IMAGE_NAME="tzj/ruler-xattn:v0.1"

# 运行测试
./scripts/run_ruler_docker.sh
```

## 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| NeMo 与 PyTorch 2.9.1 不兼容 | 中 | 高 | 测试安装，回退到方案 C |
| huggingface_hub 版本冲突 | 高 | 中 | 固定版本或使用 PYTHONNOUSERSITE |
| 镜像体积过大 | 低 | 低 | 使用方案 A 最小化安装 |

## 参考资料

- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [NeMo 安装指南](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
- [PyTorch NGC 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
