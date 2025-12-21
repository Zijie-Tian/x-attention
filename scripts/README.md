# Scripts Directory

This directory contains utility scripts for running x-attention in Docker.

## Main Scripts

### `run_docker.sh` - Universal Docker Command Runner ⭐

Execute any command in a Docker container without creating dedicated wrapper scripts.

**Usage**:
```bash
./scripts/run_docker.sh <command> [args...]
```

**Features**:
- Automatically runs `pip install -e . -q` before your command
- Same configuration as `launch_docker.sh` (GPU, volumes, proxy, etc.)
- Supports any Python script or shell command
- No need to create individual run scripts

**Examples**:
```bash
# Run visualization
./scripts/run_docker.sh python tools/visualize_rowmax_sparsity.py --seq_len 4096

# Run benchmark
./scripts/run_docker.sh python eval/efficiency/attention_speedup.py

# Debug script
./scripts/run_docker.sh python tools/debug_xattn_shape.py

# Interactive shell
./scripts/run_docker.sh bash

# Check Python version
./scripts/run_docker.sh python --version
```

**Configuration Variables** (edit in script):
```bash
IMAGE_NAME="tzj/xattn:v0.3"    # Docker image to use
GPUS="all"                      # GPU access
MODEL_DIR="/home/zijie/models"  # Model directory to mount
SHM_SIZE="16g"                  # Shared memory size
AUTO_INSTALL=true               # Auto pip install -e .
```

### `launch_docker.sh` - Interactive Development Shell

Launch an interactive Docker container for development.

**Usage**:
```bash
./scripts/launch_docker.sh
```

Once inside, you need to manually:
```bash
pip install -e .
python your_script.py
```

## Other Scripts

The directory also contains evaluation scripts for various benchmarks:

- `format_longbench_result.py` - Format LongBench evaluation results
- `longbench.sh` - LongBench evaluation wrapper
- `run_demo.sh` - Run interactive demo
- `run_hunyuan.sh` - HunyuanVideo benchmark
- `run_longbench.sh` - Batch LongBench evaluation
- `run_ruler.sh` - RULER benchmark
- `run_vllms.sh` - VLMEvalKit setup and evaluation

These scripts run directly without Docker wrappers. For Docker execution, use `run_docker.sh`.

## Migration Guide

Instead of creating new `run_*.sh` scripts, just use `run_docker.sh`:

**Before**:
```bash
# Need to create scripts/run_my_tool.sh:
#!/bin/bash
docker run ... tzj/xattn:v0.3 bash -c "pip install -e . && python my_tool.py"
```

**After**:
```bash
# Just use run_docker.sh directly:
./scripts/run_docker.sh python my_tool.py --arg1 value1
```

## Tips

1. **Pass complex arguments**: Quote them properly
   ```bash
   ./scripts/run_docker.sh python script.py --config '{"key": "value"}'
   ```

2. **Disable auto-install**: Edit `run_docker.sh` and set `AUTO_INSTALL=false`

3. **Use different image**: Edit `IMAGE_NAME` in the script

4. **Debug**: Add `set -x` at the top of `run_docker.sh` to see exact docker command
