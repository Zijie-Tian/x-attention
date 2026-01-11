# 方案2：基于 tzj/ruler:v0.3 安装 block_sparse_attn

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

### 1. block_sparse_attn 编译要求

[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) 是 MIT-HAN-LAB 开发的高效块稀疏注意力 CUDA 内核。安装需要：

- CUDA 开发工具链（nvcc）
- PyTorch C++ 扩展编译环境
- 足够的编译时内存（建议 32GB+）
- 编译时间较长（10-30 分钟）

### 2. CUDA 版本兼容性

`tzj/ruler:v0.3` 基于 PyTorch NGC 24.08（CUDA 12.6），需要确认：
- block_sparse_attn 是否支持 CUDA 12.6
- 编译器版本是否匹配

### 3. PyTorch 版本差异

| 镜像 | PyTorch 版本 |
|------|--------------|
| `tzj/ruler:v0.3` | 2.5.0 |
| `tzj/xattn:v0.6` | 2.9.1 |

PyTorch C++ ABI 在不同版本间可能有差异，block_sparse_attn 需要重新编译以匹配 PyTorch 2.5.0。

### 4. 主机包污染问题

测试中发现 `~/.local` 下的主机包（如新版 `huggingface_hub`）会覆盖容器内的版本，导致 NeMo 导入失败：

```
ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'
```

需要使用 `PYTHONNOUSERSITE=1` 环境变量隔离。

## 初步思路

### 方案 A：容器内源码编译（推荐）

在运行的容器中编译 block_sparse_attn，然后 commit 为新镜像。

```bash
# 1. 启动容器
docker run -d --gpus all --name bsa-build \
    -e MAX_JOBS=32 \
    --shm-size=32g \
    tzj/ruler:v0.3 sleep infinity

# 2. 克隆源码
docker exec bsa-build git clone \
    https://github.com/mit-han-lab/Block-Sparse-Attention.git \
    /workspace/Block-Sparse-Attention

# 3. 编译安装
docker exec -w /workspace/Block-Sparse-Attention bsa-build \
    pip install -e .

# 4. 验证
docker exec bsa-build python -c \
    "from block_sparse_attn import block_sparse_attn_func; print('OK')"

# 5. 提交新镜像
docker commit bsa-build tzj/ruler-bsa:v0.1

# 6. 清理
docker rm -f bsa-build
```

**优点**：
- 保留完整的 NeMo 环境
- 不影响现有 flash-attn 和 flashinfer
- 编译环境与运行环境完全一致

**缺点**：
- 编译耗时长（10-30 分钟）
- 需要足够的磁盘空间和内存

### 方案 B：Dockerfile 多阶段构建

使用 Dockerfile 自动化构建过程。

```dockerfile
# Dockerfile.ruler-bsa
FROM tzj/ruler:v0.3

# 设置编译环境
ENV MAX_JOBS=32
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# 克隆并编译 block_sparse_attn
RUN git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git \
        /workspace/Block-Sparse-Attention && \
    cd /workspace/Block-Sparse-Attention && \
    pip install -e . && \
    # 清理编译缓存
    rm -rf build *.egg-info

# 验证安装
RUN python -c "from block_sparse_attn import block_sparse_attn_func; print('block_sparse_attn OK')"
```

```bash
# 构建（需要 GPU）
docker build --gpus all -t tzj/ruler-bsa:v0.1 -f Dockerfile.ruler-bsa .
```

**优点**：
- 可重复构建
- 便于版本管理

**缺点**：
- docker build 默认不支持 GPU，需要配置 nvidia-container-runtime
- 某些构建系统可能需要额外配置

### 方案 C：从 tzj/xattn:v0.6 复制编译产物

将已编译的 block_sparse_attn 从 xattn 镜像复制到 ruler 镜像。

```bash
# 1. 从 xattn 镜像提取编译产物
docker run --rm tzj/xattn:v0.6 \
    tar czf - /usr/local/lib/python3.10/dist-packages/block_sparse_attn* \
    > bsa-package.tar.gz

# 2. 注入到 ruler 镜像
docker run -d --name ruler-bsa tzj/ruler:v0.3 sleep infinity
docker cp bsa-package.tar.gz ruler-bsa:/tmp/
docker exec ruler-bsa tar xzf /tmp/bsa-package.tar.gz -C /
docker commit ruler-bsa tzj/ruler-bsa:v0.1
docker rm -f ruler-bsa
```

**注意**：此方案可能因 PyTorch 版本差异（2.5.0 vs 2.9.1）导致 ABI 不兼容，需要测试验证。

**优点**：
- 跳过编译过程，速度快

**缺点**：
- 高风险：PyTorch C++ ABI 可能不兼容
- 依赖路径可能不匹配

## 实施步骤（使用方案 A）

### 步骤 1：准备构建环境

```bash
# 确保有足够的磁盘空间
df -h

# 启动构建容器
docker run -d --gpus all --name bsa-build \
    -e MAX_JOBS=32 \
    --shm-size=32g \
    --ulimit memlock=-1 \
    tzj/ruler:v0.3 sleep infinity
```

### 步骤 2：克隆源码

```bash
docker exec bsa-build git clone \
    https://github.com/mit-han-lab/Block-Sparse-Attention.git \
    /workspace/Block-Sparse-Attention
```

### 步骤 3：编译安装

```bash
# 查看 CUDA 版本
docker exec bsa-build nvcc --version

# 编译（耗时 10-30 分钟）
docker exec -w /workspace/Block-Sparse-Attention -e MAX_JOBS=32 bsa-build \
    pip install -e .

# 查看编译日志（可选）
docker logs bsa-build
```

### 步骤 4：验证安装

```bash
# 验证 block_sparse_attn
docker exec bsa-build python -c "
from block_sparse_attn import block_sparse_attn_func
import torch
print('block_sparse_attn:', block_sparse_attn_func)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
"

# 验证 NeMo 仍可用
docker exec -e PYTHONNOUSERSITE=1 bsa-build python -c "
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
print('NeMo OK')
"

# 验证 flash-attn 仍可用
docker exec bsa-build python -c "
from flash_attn import flash_attn_func
print('flash-attn OK')
"
```

### 步骤 5：提交新镜像

```bash
# 提交为新镜像
docker commit bsa-build tzj/ruler-bsa:v0.1

# 添加描述性标签
docker tag tzj/ruler-bsa:v0.1 tzj/ruler-bsa:$(date +%Y%m%d)

# 清理构建容器
docker rm -f bsa-build
```

### 步骤 6：更新 run_ruler_docker.sh

```bash
# 修改镜像名称
sed -i 's/tzj\/ruler:v0.3/tzj\/ruler-bsa:v0.1/' scripts/run_ruler_docker.sh

# 或者手动编辑
vim scripts/run_ruler_docker.sh
# 将 IMAGE_NAME 改为 tzj/ruler-bsa:v0.1
```

### 步骤 7：运行 RULER 测试

```bash
./scripts/run_ruler_docker.sh
```

## 故障排除

### 编译失败：CUDA 架构不匹配

```bash
# 指定目标 GPU 架构
docker exec -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    -w /workspace/Block-Sparse-Attention bsa-build \
    pip install -e .
```

### 编译失败：内存不足

```bash
# 减少并行编译任务数
docker exec -e MAX_JOBS=8 \
    -w /workspace/Block-Sparse-Attention bsa-build \
    pip install -e .
```

### 运行时错误：符号未定义

可能是 PyTorch 版本不匹配，需要清理后重新编译：

```bash
docker exec -w /workspace/Block-Sparse-Attention bsa-build bash -c "
    rm -rf build *.egg-info
    pip uninstall -y block-sparse-attn
    pip install -e .
"
```

### NeMo 导入失败：ModelFilter 错误

确保使用 `PYTHONNOUSERSITE=1`：

```bash
# 在 run_ruler_docker.sh 中添加
DOCKER_CMD="$DOCKER_CMD -e PYTHONNOUSERSITE=1"
```

## 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 编译失败 | 中 | 高 | 检查 CUDA 版本，调整编译参数 |
| ABI 不兼容 | 低 | 高 | 源码编译而非复制二进制 |
| 编译时间过长 | 中 | 低 | 增加 MAX_JOBS，使用更快的机器 |
| 与现有包冲突 | 低 | 中 | 使用 pip install -e . 可编辑安装 |

## 预期结果

成功构建后，新镜像 `tzj/ruler-bsa:v0.1` 将包含：

- ✅ NeMo 框架（用于 RULER）
- ✅ block_sparse_attn（用于 AvgPool 等）
- ✅ flash-attn（用于高效注意力）
- ✅ flashinfer（用于 KV cache）
- ✅ 其他 RULER 依赖（nltk, wonderwords 等）

## 参考资料

- [Block-Sparse-Attention GitHub](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [PyTorch C++ 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)
