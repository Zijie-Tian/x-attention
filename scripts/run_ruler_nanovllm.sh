#!/bin/bash
# Docker runner script for RULER benchmark with NanoVLLM backend
# Usage: ./scripts/run_ruler_nanovllm.sh
#
# This script runs RULER benchmark using nano-vllm inference engine
# with CPU offload support for long-context inference.

set -e

#############################################
# Configuration
#############################################

# Docker settings
IMAGE_NAME="${IMAGE_NAME:-tzj/ruler:v0.3}"
VISIBLE_GPUS="${VISIBLE_GPUS:-2,3}"
MODEL_DIR="${MODEL_DIR:-/home/zijie/models}"
SHM_SIZE="${SHM_SIZE:-16g}"

# RULER benchmark settings
# Note: Use llama3.1-8b-nanovllm instead of qwen3-4b-nanovllm because
# Qwen3 requires transformers>=4.51.0 but ruler:v0.3 has 4.45.2
MODEL_NAME="${MODEL_NAME:-llama3.1-8b-nanovllm}"
BENCHMARK="${BENCHMARK:-synthetic}"

# NanoVLLM specific settings (exported as env vars for call_api.py)
# CPU offload supports single sequence, which is compatible with RULER (batch_size=1)
NANOVLLM_MAX_MODEL_LEN="${NANOVLLM_MAX_MODEL_LEN:-32768}"   # 32K for long-context benchmark
NANOVLLM_CPU_OFFLOAD="${NANOVLLM_CPU_OFFLOAD:-true}"        # Enable CPU offload for memory efficiency
NANOVLLM_NUM_GPU_BLOCKS="${NANOVLLM_NUM_GPU_BLOCKS:-4}"     # GPU blocks for offload
NANOVLLM_BLOCK_SIZE="${NANOVLLM_BLOCK_SIZE:-1024}"          # KV cache block size
NANOVLLM_GPU_UTIL="${NANOVLLM_GPU_UTIL:-0.9}"               # GPU memory utilization
NANOVLLM_ENFORCE_EAGER="${NANOVLLM_ENFORCE_EAGER:-true}"    # Disable CUDA graphs

#############################################
# Script logic
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

HOST_UID=$(id -u)
HOST_GID=$(id -g)

# Build docker command
DOCKER_CMD="docker run --rm"
DOCKER_CMD="$DOCKER_CMD --gpus all"
DOCKER_CMD="$DOCKER_CMD -e CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS"
DOCKER_CMD="$DOCKER_CMD --shm-size=$SHM_SIZE"
DOCKER_CMD="$DOCKER_CMD --ipc=host"
DOCKER_CMD="$DOCKER_CMD --ulimit memlock=-1"
DOCKER_CMD="$DOCKER_CMD --ulimit stack=67108864"
DOCKER_CMD="$DOCKER_CMD --network=host"
DOCKER_CMD="$DOCKER_CMD --user $HOST_UID:$HOST_GID"
DOCKER_CMD="$DOCKER_CMD -e HOME=$HOME"

# Mount volumes
DOCKER_CMD="$DOCKER_CMD -v $HOME:$HOME"
DOCKER_CMD="$DOCKER_CMD -v $PROJECT_DIR:/workspace/x-attention"
DOCKER_CMD="$DOCKER_CMD -v /home/zijie/Code/nano-vllm:/workspace/nano-vllm"
DOCKER_CMD="$DOCKER_CMD -v $MODEL_DIR:/data/models"
DOCKER_CMD="$DOCKER_CMD -v /etc/passwd:/etc/passwd:ro"
DOCKER_CMD="$DOCKER_CMD -v /etc/group:/etc/group:ro"

# Environment variables - Proxy settings
[[ -n "${http_proxy:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e http_proxy=$http_proxy"
[[ -n "${https_proxy:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e https_proxy=$https_proxy"
[[ -n "${HTTP_PROXY:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e HTTP_PROXY=$HTTP_PROXY"
[[ -n "${HTTPS_PROXY:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e HTTPS_PROXY=$HTTPS_PROXY"
[[ -n "${no_proxy:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e no_proxy=$no_proxy"
[[ -n "${NO_PROXY:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e NO_PROXY=$NO_PROXY"

# Environment variables - Python and cache paths
DOCKER_CMD="$DOCKER_CMD -e PYTHONNOUSERSITE=1"
DOCKER_CMD="$DOCKER_CMD -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions"
DOCKER_CMD="$DOCKER_CMD -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor"
DOCKER_CMD="$DOCKER_CMD -e MPLCONFIGDIR=/workspace/.cache/matplotlib"
# Use PYTHONPATH to add packages to Python path
DOCKER_CMD="$DOCKER_CMD -e PYTHONPATH=/workspace/x-attention:/workspace/nano-vllm"

# NOTE: Do NOT set TORCHDYNAMO_DISABLE=1 - it causes incorrect output on PyTorch 2.4
# The @torch.compile decorators in rotary_embedding.py need to work correctly.
# DOCKER_CMD="$DOCKER_CMD -e TORCHDYNAMO_DISABLE=1"
# DOCKER_CMD="$DOCKER_CMD -e TORCH_COMPILE_DISABLE=1"

# NanoVLLM specific environment variables
DOCKER_CMD="$DOCKER_CMD -e NANOVLLM_MAX_MODEL_LEN=$NANOVLLM_MAX_MODEL_LEN"
DOCKER_CMD="$DOCKER_CMD -e NANOVLLM_CPU_OFFLOAD=$NANOVLLM_CPU_OFFLOAD"
DOCKER_CMD="$DOCKER_CMD -e NANOVLLM_NUM_GPU_BLOCKS=$NANOVLLM_NUM_GPU_BLOCKS"
DOCKER_CMD="$DOCKER_CMD -e NANOVLLM_BLOCK_SIZE=$NANOVLLM_BLOCK_SIZE"
DOCKER_CMD="$DOCKER_CMD -e NANOVLLM_GPU_UTIL=$NANOVLLM_GPU_UTIL"
DOCKER_CMD="$DOCKER_CMD -e NANOVLLM_ENFORCE_EAGER=$NANOVLLM_ENFORCE_EAGER"

# Working directory
DOCKER_CMD="$DOCKER_CMD -w /workspace/x-attention"

# Print configuration
echo "========================================"
echo "RULER Benchmark with NanoVLLM (Docker)"
echo "========================================"
echo "Image:           $IMAGE_NAME"
echo "GPUs:            $VISIBLE_GPUS"
echo "Model Dir:       $MODEL_DIR -> /data/models"
echo "Model:           $MODEL_NAME"
echo "Benchmark:       $BENCHMARK"
echo ""
echo "NanoVLLM Settings:"
echo "  Max Model Len: $NANOVLLM_MAX_MODEL_LEN"
echo "  CPU Offload:   $NANOVLLM_CPU_OFFLOAD"
echo "  GPU Blocks:    $NANOVLLM_NUM_GPU_BLOCKS"
echo "  Block Size:    $NANOVLLM_BLOCK_SIZE"
echo "  GPU Util:      $NANOVLLM_GPU_UTIL"
echo "  Enforce Eager: $NANOVLLM_ENFORCE_EAGER"
echo "========================================"
echo ""

#############################################
# Build container command
#############################################

# Note: NanoVLLM doesn't use --metric flag (always full attention)
# The metric parameter in run.sh is ignored for nanovllm framework
RUN_ARGS="$MODEL_NAME $BENCHMARK"

# Download NLTK punkt_tab data if needed, then run RULER
CONTAINER_CMD="python3 -c \"import nltk; nltk.download('punkt_tab', quiet=True)\" && cd eval/RULER/scripts && ./run.sh $RUN_ARGS"

echo "Executing: $DOCKER_CMD $IMAGE_NAME bash -c \"$CONTAINER_CMD\""
$DOCKER_CMD $IMAGE_NAME bash -c "$CONTAINER_CMD"
