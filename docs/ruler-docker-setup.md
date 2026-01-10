# RULER Docker Image Setup Guide

## Overview

This document provides a **complete, step-by-step guide** for building Docker images to run the RULER benchmark with x-attention project. The build process creates three image versions, each adding additional CUDA extensions.

---

## Image Version Summary

| Tag | Base | Size | Key Components | Status |
|-----|------|------|----------------|--------|
| `tzj/ruler:v0.1` | `nvcr.io/nvidia/pytorch:24.08-py3` | ~28.7GB | NeMo 1.23.0, PyTorch 2.5.0, flash-attn 2.4.2 (broken) | Base |
| `tzj/ruler:v0.2` | `tzj/ruler:v0.1` | ~30.7GB | + flash-attn 2.8.3 (source compiled) | + Flash Attention |
| `tzj/ruler:v0.3` | `tzj/ruler:v0.2` | ~31.3GB | + flashinfer 0.5.3 (tzj/minference branch) | **Recommended** |

---

## Base Image Information

### NVIDIA PyTorch Container

```
nvcr.io/nvidia/pytorch:24.08-py3
```

| Component | Version |
|-----------|---------|
| PyTorch | 2.5.0a0+872d972 |
| CUDA | 12.6 |
| cuDNN | 9.x |
| Python | 3.10 |
| OS | Ubuntu 22.04 |

**Note**: The base image includes a pre-compiled flash-attn 2.4.2, but it has **ABI incompatibility** with PyTorch 2.5.0 and will cause `undefined symbol` errors. This is why we need to recompile from source.

---

## Step 1: Build v0.1 (Base RULER Environment)

### 1.1 Prerequisites

Ensure you have:
- Docker with NVIDIA Container Toolkit installed
- Sufficient disk space (~30GB for the image)
- Network access to pull base image and install packages

### 1.2 File Structure

```
eval/RULER/
├── Dockerfile           # Docker build file
├── build_docker.sh      # Build script
├── requirements.txt     # Python dependencies
└── scripts/
    ├── run.sh           # RULER entry point
    ├── config_models.sh # Model configurations
    └── config_tasks.sh  # Task configurations
```

### 1.3 Dockerfile Content

**File**: `eval/RULER/Dockerfile`

```dockerfile
# RULER Benchmark Execution Environment
# Base: NVIDIA PyTorch 24.08 (PyTorch 2.5.0, CUDA 12.6)
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set working directory
WORKDIR /workspace/RULER

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements first for better caching
COPY requirements.txt .

# Install build dependencies
# IMPORTANT: Cython must be installed BEFORE youtokentome
RUN pip install --no-cache-dir -q Cython

# Install youtokentome with no-build-isolation (needs Cython)
# This is required by NeMo toolkit
RUN pip install --no-cache-dir youtokentome --no-build-isolation

# Install all requirements including nemo-toolkit
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for text processing
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy RULER scripts
COPY . .

# Set default command
CMD ["/bin/bash"]
```

### 1.4 Requirements File

**File**: `eval/RULER/requirements.txt`

```
nemo-toolkit[all]
tritonclient[all]
transformer_engine[pytorch]
flask
flask_restful
html2text
google-generativeai
sshtunnel_requests
wonderwords
openai
tiktoken
tenacity
accelerate
huggingface_hub==0.23.4
transformers==4.45.2
torch==2.4.0
torchaudio==2.4.0
torchdiffeq==0.2.5
torchmetrics==1.6.1
torchsde==0.2.6
torchvision==0.19.0
```

**Note**: The torch version in requirements.txt (2.4.0) will be overridden by the base image's version (2.5.0). This is expected and NeMo is compatible with both versions.

### 1.5 Build Command

```bash
cd /home/zijie/Code/x-attention/eval/RULER
./build_docker.sh
```

Or manually:

```bash
docker build -t tzj/ruler:v0.1 -f Dockerfile .
```

### 1.6 Key Installed Packages in v0.1

| Package | Version | Purpose |
|---------|---------|---------|
| nemo-toolkit | 1.23.0 | NVIDIA NeMo for evaluation |
| transformers | 4.45.2 | HuggingFace Transformers |
| accelerate | latest | Model parallelism |
| numpy | 1.26.4 | Compatible with NeMo's tensorstore |
| transformer_engine | (from base) | NVIDIA Transformer Engine |
| flash-attn | 2.4.2 | **BROKEN** - needs recompilation |

### 1.7 Build Time

- **Estimated time**: 10-15 minutes
- **Main bottleneck**: NeMo toolkit installation

---

## Step 2: Build v0.2 (Add Flash Attention)

### 2.1 Problem Statement

The pre-installed flash-attn 2.4.2 in NVIDIA's container has **ABI incompatibility** with PyTorch 2.5.0:

```
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_l
```

**Solution**: Recompile flash-attention from source with matching PyTorch version.

### 2.2 Source Repository

| Repository | URL | Branch |
|------------|-----|--------|
| flash-attention | `https://github.com/Zijie-Tian/flash-attention.git` | main |

This is a fork with additional features for x-attention project.

### 2.3 Build Commands (Step by Step)

#### Step 2.3.1: Start Build Container

```bash
docker run -d --gpus all --name ruler-build --shm-size=16g tzj/ruler:v0.1 sleep infinity
```

#### Step 2.3.2: Uninstall Broken flash-attn

```bash
docker exec ruler-build pip uninstall flash-attn -y
```

#### Step 2.3.3: Clone flash-attention Repository

```bash
docker exec ruler-build git clone https://github.com/Zijie-Tian/flash-attention.git /workspace/flash-attention
```

#### Step 2.3.4: Build flash-attention from Source

**CRITICAL**: Use `FLASH_ATTN_CUDA_ARCHS` environment variable, **NOT** `TORCH_CUDA_ARCH_LIST`.

```bash
docker exec -w /workspace/flash-attention \
    -e MAX_JOBS=32 \
    -e FLASH_ATTN_CUDA_ARCHS="80;90" \
    ruler-build pip install -e . --no-build-isolation
```

**Environment Variables Explained**:
| Variable | Value | Purpose |
|----------|-------|---------|
| `MAX_JOBS` | 32 | Parallel compilation jobs (adjust based on CPU cores) |
| `FLASH_ATTN_CUDA_ARCHS` | "80;90" | Target CUDA architectures (Ampere=80, Hopper=90) |

**Why not include newer architectures?**
- `compute_100` (Blackwell) is not supported by CUDA 12.6
- `compute_110`, `compute_120` cause `nvcc fatal: Unsupported gpu architecture` error

#### Step 2.3.5: Verify Installation

```bash
docker exec ruler-build python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')"
```

Expected output:
```
flash-attn version: 2.8.3
```

#### Step 2.3.6: Commit as v0.2

```bash
docker commit ruler-build tzj/ruler:v0.2
```

#### Step 2.3.7: Cleanup

```bash
docker rm -f ruler-build
```

### 2.4 Build Time

- **Estimated time**: 15-20 minutes
- **Main bottleneck**: CUDA kernel compilation

### 2.5 Verification

```bash
docker run --rm --gpus all tzj/ruler:v0.2 python -c "
from flash_attn import flash_attn_func
import torch
q = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 128, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v, causal=True)
print(f'Flash attention output shape: {out.shape}')
print('Flash attention works correctly!')
"
```

---

## Step 3: Build v0.3 (Add FlashInfer)

### 3.1 Purpose

FlashInfer is required by x-attention for efficient attention decoding:
- `flashinfer.single_decode_with_kv_cache()` - Single token decoding
- `flashinfer.single_prefill_with_kv_cache()` - Full attention prefill

### 3.2 Source Repository

| Repository | URL | Branch |
|------------|-----|--------|
| flashinfer | `https://github.com/Zijie-Tian/flashinfer.git` | **tzj/minference** |

**Important**: Must use the `tzj/minference` branch, not main.

### 3.3 Build Commands (Step by Step)

#### Step 3.3.1: Start Build Container

```bash
docker run -d --gpus all --name ruler-build --shm-size=16g tzj/ruler:v0.2 sleep infinity
```

#### Step 3.3.2: Clone FlashInfer Repository

```bash
docker exec ruler-build git clone https://github.com/Zijie-Tian/flashinfer.git /workspace/flashinfer
```

#### Step 3.3.3: Checkout tzj/minference Branch

```bash
docker exec -w /workspace/flashinfer ruler-build git checkout tzj/minference
```

#### Step 3.3.4: Initialize Submodules

FlashInfer has submodule dependencies (cutlass, etc.):

```bash
docker exec -w /workspace/flashinfer ruler-build git submodule update --init --recursive
```

#### Step 3.3.5: Build FlashInfer from Source

**IMPORTANT**: Use build isolation (do NOT use `--no-build-isolation`).

```bash
docker exec -w /workspace/flashinfer \
    -e MAX_JOBS=32 \
    -e FLASHINFER_CUDA_ARCHITECTURES="80;90" \
    ruler-build pip install -e .
```

**Environment Variables Explained**:
| Variable | Value | Purpose |
|----------|-------|---------|
| `MAX_JOBS` | 32 | Parallel compilation jobs |
| `FLASHINFER_CUDA_ARCHITECTURES` | "80;90" | Target CUDA architectures |

#### Step 3.3.6: Verify Installation

```bash
docker exec ruler-build python -c "import flashinfer; print(f'flashinfer version: {flashinfer.__version__}')"
```

Expected output:
```
flashinfer version: 0.5.3
```

#### Step 3.3.7: Verify PyTorch Unchanged

```bash
docker exec ruler-build python -c "import torch; print(f'torch version: {torch.__version__}')"
```

Expected output:
```
torch version: 2.5.0a0+872d972
```

#### Step 3.3.8: Commit as v0.3

```bash
docker commit ruler-build tzj/ruler:v0.3
```

#### Step 3.3.9: Cleanup

```bash
docker rm -f ruler-build
```

### 3.4 Build Time

- **Estimated time**: 20-30 minutes
- **Main bottleneck**: CUDA kernel compilation

### 3.5 Verification

```bash
docker run --rm --gpus all tzj/ruler:v0.3 python -c "
import flashinfer
import torch
q = torch.randn(8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(8, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(8, 128, 64, device='cuda', dtype=torch.float16)
out = flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout='HND')
print(f'FlashInfer decode output shape: {out.shape}')
print('FlashInfer works correctly!')
"
```

---

## Complete Build Script (All-in-One)

For convenience, here's a complete script to build all three versions:

```bash
#!/bin/bash
# Complete RULER Docker Image Build Script
# Usage: ./build_all_ruler_images.sh

set -e

echo "=========================================="
echo "Building RULER Docker Images"
echo "=========================================="

# Step 1: Build v0.1
echo ""
echo ">>> Step 1: Building v0.1 (Base RULER Environment)"
cd /home/zijie/Code/x-attention/eval/RULER
docker build -t tzj/ruler:v0.1 -f Dockerfile .

# Step 2: Build v0.2
echo ""
echo ">>> Step 2: Building v0.2 (Add Flash Attention)"
docker run -d --gpus all --name ruler-build --shm-size=16g tzj/ruler:v0.1 sleep infinity

docker exec ruler-build pip uninstall flash-attn -y
docker exec ruler-build git clone https://github.com/Zijie-Tian/flash-attention.git /workspace/flash-attention
docker exec -w /workspace/flash-attention \
    -e MAX_JOBS=32 \
    -e FLASH_ATTN_CUDA_ARCHS="80;90" \
    ruler-build pip install -e . --no-build-isolation

docker exec ruler-build python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
docker commit ruler-build tzj/ruler:v0.2
docker rm -f ruler-build

# Step 3: Build v0.3
echo ""
echo ">>> Step 3: Building v0.3 (Add FlashInfer)"
docker run -d --gpus all --name ruler-build --shm-size=16g tzj/ruler:v0.2 sleep infinity

docker exec ruler-build git clone https://github.com/Zijie-Tian/flashinfer.git /workspace/flashinfer
docker exec -w /workspace/flashinfer ruler-build git checkout tzj/minference
docker exec -w /workspace/flashinfer ruler-build git submodule update --init --recursive
docker exec -w /workspace/flashinfer \
    -e MAX_JOBS=32 \
    -e FLASHINFER_CUDA_ARCHITECTURES="80;90" \
    ruler-build pip install -e .

docker exec ruler-build python -c "import flashinfer; print(f'flashinfer: {flashinfer.__version__}')"
docker commit ruler-build tzj/ruler:v0.3
docker rm -f ruler-build

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
docker images tzj/ruler
```

---

## Known Issues and Solutions

### Issue 1: Cython not found during youtokentome build

**Error**:
```
ModuleNotFoundError: No module named 'Cython'
```

**Cause**: youtokentome requires Cython at build time, but pip's build isolation creates a clean environment without Cython.

**Solution**:
1. Install Cython first: `pip install Cython`
2. Install youtokentome with `--no-build-isolation`

### Issue 2: flash-attn ABI incompatibility

**Error**:
```
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_l
```

**Cause**: Pre-compiled flash-attn binary was built with different PyTorch version.

**Solution**: Recompile flash-attn from source (see Step 2).

### Issue 3: Unsupported GPU architecture compute_120

**Error**:
```
nvcc fatal: Unsupported gpu architecture 'compute_120'
```

**Cause**: flash-attention's setup.py defaults to including future architectures (Blackwell), but CUDA 12.6 doesn't support them yet.

**Solution**: Set `FLASH_ATTN_CUDA_ARCHS="80;90"` before building.

### Issue 4: ValueError: not enough values to unpack (expected 3, got 2)

**Error**:
```python
hidden_states, self_attn_weights, present_key_value = self.self_attn(
ValueError: not enough values to unpack (expected 3, got 2)
```

**Cause**: x-attention's monkey-patched forward function returns 2 values instead of 3.

**Solution**: Fixed in `xattn/src/load_llama.py` - return `(attn_output, None, past_key_value)`.

### Issue 5: CUDA illegal memory access

**Error**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**Cause**: Hard-coded `.to("cuda")` in multi-GPU environment causes device mismatch.

**Solution**: Fixed in `xattn/src/load_llama.py` - use `.to(query_states.device)` instead.

### Issue 6: Docker GPU specification error

**Error**:
```
docker: Error response from daemon: cannot set both Count and DeviceIDs on device request
```

**Cause**: Using `--gpus device=2,3,4,5` format incorrectly.

**Solution**: Use `--gpus all` with `-e CUDA_VISIBLE_DEVICES=2,3,4,5`.

---

## Dependencies Status

### CUDA Extensions Required by x-attention

| Package | Version | Repository | Branch | Status |
|---------|---------|------------|--------|--------|
| flash-attn | 2.8.3 | `Zijie-Tian/flash-attention` | main | ✅ Installed in v0.2 |
| flashinfer | 0.5.3 | `Zijie-Tian/flashinfer` | tzj/minference | ✅ Installed in v0.3 |
| block_sparse_attn | - | MIT-HAN-LAB/Block-Sparse-Attention | - | ❌ Not installed |

### Import Status by Metric

| Metric | Required Imports | v0.3 Status |
|--------|-----------------|-------------|
| `full` | flashinfer | ✅ Works |
| `xattn` | flashinfer, block_sparse_attn | ❌ Missing block_sparse_attn |
| `minfer` | flashinfer, block_sparse_attn | ❌ Missing block_sparse_attn |
| `compass` | flashinfer, block_sparse_attn | ❌ Missing block_sparse_attn |
| `avgpool` | flashinfer, block_sparse_attn | ❌ Missing block_sparse_attn |

---

## Usage Guide

### Run RULER Benchmark (Recommended Method)

```bash
# From x-attention project root
./scripts/run_ruler_tasks.sh
```

This script runs the benchmark with configurable metrics. Edit the script to change:
- `RUN_FULL=true/false` - Full attention baseline
- `RUN_XATTN=true/false` - XAttn sparse attention
- `RUN_AVGPOOL=true/false` - AvgPool sparse attention

### Launch Interactive Container

```bash
./scripts/launch_ruler_docker.sh
```

### Manual Docker Run

```bash
docker run --gpus all -it --rm \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES=2,3,4,5 \
    -v /home/zijie/Code/x-attention:/workspace/x-attention \
    -v /home/zijie/models:/data/models:ro \
    -w /workspace/x-attention \
    tzj/ruler:v0.3 \
    /bin/bash
```

### Inside Container

```bash
# Install x-attention package
pip install -e . -q

# Run RULER benchmark
cd eval/RULER/scripts
./run.sh llama3.1-8b-chat synthetic --metric full
```

### Cleanup After Testing

```bash
./scripts/cleanup_docker.sh
```

---

## Script Reference

| Script | Purpose | Location |
|--------|---------|----------|
| `build_docker.sh` | Build v0.1 base image | `eval/RULER/` |
| `launch_ruler_docker.sh` | Launch interactive container | `scripts/` |
| `run_ruler_docker.sh` | Run benchmark in container | `scripts/` |
| `run_ruler_tasks.sh` | Orchestrate multiple test configurations | `scripts/` |
| `cleanup_docker.sh` | Clean up containers | `scripts/` |

---

## Environment Variables Reference

### Build-time Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MAX_JOBS` | Parallel compilation jobs | `32` |
| `FLASH_ATTN_CUDA_ARCHS` | CUDA architectures for flash-attn | `"80;90"` |
| `FLASHINFER_CUDA_ARCHITECTURES` | CUDA architectures for flashinfer | `"80;90"` |

### Runtime Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | Visible GPUs | `2,3,4,5` |
| `TORCH_EXTENSIONS_DIR` | PyTorch extensions cache | `/workspace/.cache/torch_extensions` |
| `PYTHONUSERBASE` | User pip install location | `/workspace/x-attention/.pip_cache` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-01-10 | Initial build with NeMo 1.23.0 (flash-attn broken) |
| v0.2 | 2026-01-10 | Added source-compiled flash-attn 2.8.3 |
| v0.3 | 2026-01-10 | Added flashinfer 0.5.3 (tzj/minference branch) |

---

## Troubleshooting Checklist

1. **Docker permission denied**: Run `sudo usermod -aG docker $USER && newgrp docker`
2. **Image not found**: Run `docker images tzj/ruler` to check available images
3. **CUDA out of memory**: Increase `--shm-size` or reduce `MAX_JOBS`
4. **Network issues during build**: Check proxy settings, use `--network=host`
5. **GPU not visible**: Ensure NVIDIA Container Toolkit is installed
6. **Slow builds**: Increase `MAX_JOBS` (default: 32)
