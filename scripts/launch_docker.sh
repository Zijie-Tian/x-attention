#!/bin/bash
# Docker launcher script for x-attention
# Usage: ./scripts/launch_docker.sh

set -e

#############################################
# Configuration
#############################################

IMAGE_NAME="tzj/xattn:v0.4"
GPUS="all"
MODEL_DIR="/home/zijie/models"
CONTAINER_NAME="xattn"
SHM_SIZE="16g"

#############################################
# Script logic
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

HOST_UID=$(id -u)
HOST_GID=$(id -g)

# Remove existing container if exists
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Build docker command
DOCKER_CMD="docker run --rm"
DOCKER_CMD="$DOCKER_CMD --gpus $GPUS"
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
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

# Working directory
DOCKER_CMD="$DOCKER_CMD -w /workspace/x-attention"

# Print configuration
echo "========================================"
echo "x-attention Docker Launcher"
echo "========================================"
echo "Image:        $IMAGE_NAME"
echo "GPUs:         $GPUS"
echo "Model Dir:    $MODEL_DIR"
echo "Project Dir:  $PROJECT_DIR"
echo "========================================"
echo ""
echo "After entering the container, run:"
echo "  pip install -e .  # Install xattn package"
echo "  python eval/efficiency/attention_speedup.py"
echo ""

# Execute with docker group permissions
sg docker -c "$DOCKER_CMD -it $IMAGE_NAME /bin/bash"
