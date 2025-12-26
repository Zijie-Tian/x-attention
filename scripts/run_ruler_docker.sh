#!/bin/bash
# Docker runner script for RULER benchmark
# Usage: ./scripts/run_ruler_docker.sh
#
# To modify parameters, edit the Configuration section below

set -e

#############################################
# Configuration
#############################################

# Docker settings
IMAGE_NAME="tzj/xattn:v0.5"  # v0.5 has pre-compiled CUDA extensions
GPUS="device=0"              # Use only GPU 0 (A100)
MODEL_DIR="/home/zijie/models"
SHM_SIZE="16g"

# RULER benchmark settings
# Note: Setup is not needed for v0.4, data is already prepared in the image
MODEL_NAME="llama3.1-8b-chat"
BENCHMARK="synthetic"
METRIC="avgpool"               # Metric to use (e.g., xattn, avgpool, compass)
STRIDE="16"                  # Stride value for xattn (e.g., 16, 8, 4)
THRESHOLD=""                 # Threshold value (leave empty for default)
AVGPOOL_TOPK="64"              # Top-k blocks per row for avgpool (leave empty for default: 64)

#############################################
# Script logic
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

HOST_UID=$(id -u)
HOST_GID=$(id -g)

# Build docker command
DOCKER_CMD="docker run --rm"
DOCKER_CMD="$DOCKER_CMD --gpus $GPUS"
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

# Environment variables - Use pre-compiled cache from v0.5 image
DOCKER_CMD="$DOCKER_CMD -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions"
DOCKER_CMD="$DOCKER_CMD -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor"
DOCKER_CMD="$DOCKER_CMD -e MPLCONFIGDIR=/workspace/.cache/matplotlib"

# Working directory
DOCKER_CMD="$DOCKER_CMD -w /workspace/x-attention"

# Print configuration
echo "========================================"
echo "RULER Benchmark (Docker)"
echo "========================================"
echo "Image:        $IMAGE_NAME"
echo "GPUs:         $GPUS"
echo "Model Dir:    $MODEL_DIR -> /data/models"
echo "========================================"

#############################################
# Build container command
#############################################

# Build run command with configured parameters
RUN_ARGS="$MODEL_NAME $BENCHMARK --metric $METRIC"

if [ -n "$STRIDE" ]; then
    RUN_ARGS="$RUN_ARGS --stride $STRIDE"
fi

if [ -n "$THRESHOLD" ]; then
    RUN_ARGS="$RUN_ARGS --threshold $THRESHOLD"
fi

if [ -n "$AVGPOOL_TOPK" ]; then
    RUN_ARGS="$RUN_ARGS --avgpool_topk $AVGPOOL_TOPK"
fi

# Update MODEL_DIR in run.sh to point to /data/models
# Download NLTK punkt_tab data before running RULER
CONTAINER_CMD="pip install -e . -q && python3 -m nltk.downloader punkt_tab && cd eval/RULER && cd scripts && ./run.sh $RUN_ARGS"

echo "Mode:         Benchmark"
echo "Model:        $MODEL_NAME"
echo "Benchmark:    $BENCHMARK"
echo "Metric:       $METRIC"
[ -n "$STRIDE" ] && echo "Stride:       $STRIDE"
[ -n "$THRESHOLD" ] && echo "Threshold:    $THRESHOLD"
[ -n "$AVGPOOL_TOPK" ] && echo "AvgPool TopK: $AVGPOOL_TOPK"
echo "========================================"
echo ""

# Execute docker command
echo "Executing: $DOCKER_CMD $IMAGE_NAME bash -c \"$CONTAINER_CMD\""
$DOCKER_CMD $IMAGE_NAME bash -c "$CONTAINER_CMD"
