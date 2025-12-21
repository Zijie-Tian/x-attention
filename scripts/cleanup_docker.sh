#!/bin/bash
# Cleanup script for x-attention Docker containers
# Stops and removes all containers using tzj/xattn images

set -e

#############################################
# Configuration
#############################################

IMAGE_PREFIX="tzj/xattn"  # Prefix for x-attention images (matches tzj/xattn:v0.3, v0.4, etc.)
FORCE_REMOVE="true"       # Force remove containers (true/false)

#############################################
# Script logic
#############################################

echo "========================================"
echo "X-Attention Docker Cleanup"
echo "========================================"
echo "Image prefix: $IMAGE_PREFIX"
echo "Force remove: $FORCE_REMOVE"
echo "========================================"
echo ""

# Find all containers using x-attn images (grep for image prefix)
CONTAINERS=$(sg docker -c "docker ps -a --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}'" | grep "$IMAGE_PREFIX" || true)

if [ -z "$CONTAINERS" ]; then
    echo "No containers found using $IMAGE_PREFIX images."
    exit 0
fi

echo "Found containers:"
echo "$CONTAINERS"
echo ""

# Extract container IDs
CONTAINER_IDS=$(echo "$CONTAINERS" | awk '{print $1}' | tr '\n' ' ')

# Stop running containers
echo "Stopping containers..."
for CONTAINER_ID in $CONTAINER_IDS; do
    STATUS=$(sg docker -c "docker ps -a --filter id=$CONTAINER_ID --format '{{.Status}}'")
    if [[ $STATUS == Up* ]]; then
        echo "  Stopping $CONTAINER_ID..."
        sg docker -c "docker stop $CONTAINER_ID" > /dev/null
    else
        echo "  Container $CONTAINER_ID already stopped"
    fi
done

echo ""

# Remove containers
echo "Removing containers..."
for CONTAINER_ID in $CONTAINER_IDS; do
    if [ "$FORCE_REMOVE" = "true" ]; then
        sg docker -c "docker rm -f $CONTAINER_ID" > /dev/null 2>&1 || true
        echo "  Removed $CONTAINER_ID (forced)"
    else
        sg docker -c "docker rm $CONTAINER_ID" > /dev/null 2>&1 || true
        echo "  Removed $CONTAINER_ID"
    fi
done

echo ""
echo "========================================"
echo "Cleanup complete!"
echo "========================================"
