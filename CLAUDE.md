# x-attention Development Guide

## Docker Environment

### Container Image: `tzj/sparseattn`

Pre-built Docker image with sparse attention kernels for long-context LLM inference.

| Tag | Size | Contents |
|-----|------|----------|
| v0.1 | 27.6GB | PyTorch 2.9.1, CUDA 12.8, flash-attention, block-sparse-attn |
| v0.2 | 32.9GB | + vllm 0.9.0 |

### Pre-installed Libraries

Located in `/worksparse/` (symlinked to `/workspace/`):

| Library | Version | Path |
|---------|---------|------|
| flash-attention | 2.8.3 | `/workspace/flash-attention` |
| Block-Sparse-Attention | 0.0.1 | `/workspace/Block-Sparse-Attention` |
| vllm | 0.9.0+cu128 | `/workspace/vllm` |

All libraries are installed in editable mode (`pip install -e .`).

### Base Image

```
pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
```

### Quick Start

```bash
# Pull image (if not built locally)
docker pull tzj/sparseattn:v0.2

# Run interactive container
docker run --gpus all -it --rm \
    --shm-size=16g \
    -v $(pwd):/workspace/x-attention \
    -w /workspace/x-attention \
    tzj/sparseattn:v0.2 \
    /bin/bash
```

### Docker Scripts

#### Interactive Shell: `scripts/launch_docker.sh`
Launch an interactive Docker container for development:

```bash
./scripts/launch_docker.sh
```

#### Run Any Command: `scripts/run_docker.sh`
Execute any command in a Docker container without creating dedicated scripts:

```bash
# Basic usage
./scripts/run_docker.sh <command> [args...]

# Examples
./scripts/run_docker.sh python tools/visualize_rowmax_sparsity.py --seq_len 4096
./scripts/run_docker.sh python eval/efficiency/attention_speedup.py
./scripts/run_docker.sh bash  # Interactive shell
```

**Features**:
- Automatically runs `pip install -e . -q` before executing your command
- Uses the same configuration as `launch_docker.sh`
- Supports any Python script or shell command
- Replaces the need for individual `run_*.sh` scripts

Both scripts automatically:
- Use host network (`--network=host`)
- Pass proxy environment variables
- Sync user UID/GID for file permissions
- Mount home directory and project
- Set shared memory to 16GB

### Directory Structure Inside Container

```
/workspace/
├── flash-attention/        # Dao-AILab flash-attention (symlink)
├── Block-Sparse-Attention/ # MIT-HAN-LAB block sparse attn (symlink)
├── vllm/                   # vLLM inference engine (symlink)
├── MInference/             # (mounted via launch script)
└── x-attention/            # (mount your project here)

/worksparse/                # Actual location of pre-built libraries
├── flash-attention/
├── Block-Sparse-Attention/
└── vllm/

/data/models/               # Model directory (configurable in launch script)
```

### Environment Variables

The container respects these environment variables:

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `MAX_JOBS` | Parallel compilation jobs (default: 32) |
| `HTTP_PROXY` / `HTTPS_PROXY` | Proxy settings (auto-passed) |

### Building the Image

From MInference directory:

```bash
# Build base image (flash-attn + block-sparse-attn)
docker build -t tzj/sparseattn:v0.1 .

# Add vllm (run in container, then commit)
docker run -d --gpus all --name vllm-build tzj/sparseattn:v0.1 sleep infinity
docker cp /path/to/vllm vllm-build:/worksparse/vllm
docker exec -w /worksparse/vllm vllm-build python use_existing_torch.py
docker exec -w /worksparse/vllm -e SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 vllm-build \
    uv pip install --system -r requirements/build.txt
docker exec -w /worksparse/vllm -e SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 -e MAX_JOBS=32 vllm-build \
    uv pip install --system --no-build-isolation -e .
docker commit vllm-build tzj/sparseattn:v0.2
docker rm -f vllm-build
```

### Dockerfile Location

```
/home/zijie/Code/MInference/Dockerfile
```

### Usage Examples

```python
# Flash Attention
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)

# Block Sparse Attention
from block_sparse_attn import block_sparse_attn_func
out = block_sparse_attn_func(q, k, v, block_mask)

# vLLM
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

### Troubleshooting

**Docker permission denied:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**Image not found:**
```bash
# Check available images
docker images tzj/sparseattn
```

**CUDA out of memory:**
- Increase `--shm-size` (default: 16g)
- Reduce `MAX_JOBS` for compilation

### Related Repositories

| Repo | Description |
|------|-------------|
| MInference | Sparse attention for long-context LLM |
| flash-attention | Dao-AILab FlashAttention-2 |
| Block-Sparse-Attention | MIT-HAN-LAB block sparse kernels |
| vLLM | High-throughput LLM serving |

---

## Sparsity Visualization Tools

### Overview

The project includes tools for analyzing and comparing different sparsity detection methods:
- **XAttn**: `xattn_estimate`-based sparse detection (stride-based approach)
- **SpargeAttention**: Rowmax-based filtering method ([arXiv:2502.18137](https://arxiv.org/pdf/2502.18137))

### Figures Directory Structure

```
figures/
├── comparison/          # XAttn vs SpargeAttention comparison visualizations
├── xattn/              # XAttn sparsity pattern visualizations
├── rowmax/             # SpargeAttention rowmax-based visualizations
├── archive/            # Archived/experimental figures
└── *.pdf, *.jpg        # Paper figures and diagrams (tracked in git)
```

**Note**: All subdirectories under `figures/` are gitignored. Only PDF/JPG files in the root are tracked.

### Visualization Scripts

#### Recommended: Use `run_docker.sh`
The generic `run_docker.sh` script can run any visualization command:

```bash
# Compare XAttn vs SpargeAttention (all heads)
./scripts/run_docker.sh python tools/visualize_rowmax_sparsity.py \
    --seq_len 4096 --block_size 128 --stride 16 \
    --xattn_threshold 0.9 --rowmax_threshold 3.0

# Compare specific head
./scripts/run_docker.sh python tools/visualize_rowmax_sparsity.py \
    --seq_len 4096 --stride 16 --head_idx 0

# XAttn sparsity patterns
./scripts/run_docker.sh python tools/visualize_sparsity.py \
    --compare --stride 16

# Time breakdown analysis
./scripts/run_docker.sh python eval/efficiency/xattn_breakdown.py
```


### File Naming Convention

#### Comparison Files
```
figures/comparison/head{N}_seq{len}_bs{block_size}_s{stride}_xth{xattn_th}_rth{rowmax_th}.png
```

Example: `head0_seq4096_bs128_s16_xth0.9_rth3.0.png`
- `N`: Head index (0-31)
- `len`: Sequence length (e.g., 4096)
- `block_size`: Block size for tiling (e.g., 128)
- `stride`: Stride for xattn_estimate (e.g., 16)
- `xattn_th`: XAttn threshold (e.g., 0.9)
- `rowmax_th`: SpargeAttention rowmax threshold (e.g., 3.0)

#### XAttn Files
- `sparsity_head{N}_seq{len}_stride{S}.png` - Per-head sparsity pattern
- `sparsity_summary_seq{len}_stride{S}.png` - Summary across all heads
- `density_trend_stride{S}.png` - Density vs sequence length
- `sparsity_comparison_stride{S}.png` - Multi-length comparison

#### Rowmax Files
- `rowmax_heatmap_seq{len}_bs{size}.png` - Block rowmax values heatmap
- `rowmax_sparsity_seq{len}_bs{size}_th{threshold}.png` - Sparsity pattern

### Usage Examples

#### Generate Comparison Visualizations
```bash
./scripts/run_comparison.sh
```

This generates 32 comparison figures (one per attention head) showing:
1. XAttn sparsity pattern
2. SpargeAttention sparsity pattern
3. Side-by-side comparison with agreement analysis

#### Modify Parameters
Edit `scripts/run_comparison.sh` to change:
```bash
--seq_len 4096              # Sequence length
--block_size 128            # Block size
--stride 16                 # XAttn stride
--xattn_threshold 0.9       # XAttn threshold
--rowmax_threshold 3.0      # SpargeAttention threshold
--head_idx 0                # Visualize specific head only (omit for all)
```

### Current Configuration

Default parameters used in visualizations:
- **Sequence Length**: 4096
- **Block Size**: 128
- **Stride**: 16
- **XAttn Threshold**: 0.9
- **Rowmax Threshold**: 3.0

### Results Summary

#### Comparison Statistics (threshold=3.0)
- **Agreement**: 79.0% - 97.7% across 32 heads
- **XAttn Sparsity**: 67.8% - 87.4%
- **SpargeAttention Sparsity**: 53.5% - 88.7%

**Key Observations**:
- Head 12, 13: Highest agreement (93-94%)
- Head 19: Best agreement (97.7%), XAttn more conservative
- Head 1, 4, 8: Lower agreement (~80%), SpargeAttention keeps more blocks

### Tools Overview

#### Main Visualization Tool
`tools/visualize_rowmax_sparsity.py` - Comprehensive comparison tool
- Loads cached Q/K tensors from `output/` directory
- Computes both XAttn and SpargeAttention sparsity masks
- Generates per-head comparison visualizations
- Prints detailed statistics

#### Debug Tool
`tools/debug_xattn_shape.py` - Understand xattn_estimate behavior
- Analyzes padding and block calculations
- Shows effective vs padded regions
- Useful for understanding xattn_estimate internals

### Important Notes

1. **xattn_estimate Padding**: Due to chunk_size padding (default 16384), the output mask includes padded blocks. The tools automatically extract only the valid region.

2. **Block Size Matching**: The comparison tool ensures both methods use the same effective block size for fair comparison.

3. **Cached Data**: Visualizations use pre-computed Q/K tensors from `output/` directory. Run the main benchmark first to generate these files.
