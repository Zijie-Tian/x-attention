#!/bin/bash
# Generic Docker command runner for x-attention
# Usage: ./scripts/run_docker.sh <command> [args...]
# Example: ./scripts/run_docker.sh python tools/visualize_rowmax_sparsity.py --seq_len 4096

set -e

#############################################
# Configuration (same as launch_docker.sh)
#############################################

IMAGE_NAME="tzj/xattn:v0.4"
GPUS="all"
VISIBLE_GPUS="0"  # Limit container to see only these GPUs (e.g., "0", "0,1", or "" for all)
MODEL_DIR="/home/zijie/models"
SHM_SIZE="16g"
AUTO_INSTALL=true  # Automatically run pip install -e . before command

#############################################
# Script logic
#############################################

# Cleanup any existing containers using the same image
EXISTING_CONTAINERS=$(docker ps -q --filter ancestor="$IMAGE_NAME" 2>/dev/null || true)
if [ -n "$EXISTING_CONTAINERS" ]; then
    echo "Stopping existing containers using $IMAGE_NAME..."
    docker stop $EXISTING_CONTAINERS 2>/dev/null || true
fi

# Check if command is provided
if [ $# -eq 0 ]; then
    echo "Error: No command provided"
    echo ""
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 python tools/visualize_rowmax_sparsity.py --seq_len 4096"
    echo "  $0 python eval/efficiency/attention_speedup.py"
    echo "  $0 bash  # Interactive shell"
    echo ""
    exit 1
fi

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

# GPU visibility - limit which GPUs the container can see
[[ -n "${VISIBLE_GPUS:-}" ]] && DOCKER_CMD="$DOCKER_CMD -e CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS"

# Working directory
DOCKER_CMD="$DOCKER_CMD -w /workspace/x-attention"

# Detect if interactive mode is needed (for bash, sh, etc.)
INTERACTIVE=false
if [[ "$1" == "bash" ]] || [[ "$1" == "sh" ]] || [[ "$1" == "/bin/bash" ]] || [[ "$1" == "/bin/sh" ]]; then
    INTERACTIVE=true
fi

# Add interactive flag if needed
if [ "$INTERACTIVE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -it"
fi

# Build the command to execute in container
if [ "$AUTO_INSTALL" = true ] && [ "$INTERACTIVE" = false ]; then
    # Auto install xattn package before running command
    CONTAINER_CMD="pip install -e . -q && $*"
else
    CONTAINER_CMD="$*"
fi

# Execute with docker group permissions
sg docker -c "$DOCKER_CMD $IMAGE_NAME bash -c '$CONTAINER_CMD'"
