#!/bin/bash
# Cleanup script for x-attention Docker containers
# Stops and removes all containers using tzj/xattn and tzj/ruler images

set -e

#############################################
# Configuration
#############################################

IMAGE_PREFIXES=("tzj/xattn" "tzj/ruler")  # Prefixes for x-attention images
FORCE_REMOVE="true"                        # Force remove containers (true/false)

#############################################
# Script logic
#############################################

echo "========================================"
echo "X-Attention Docker Cleanup"
echo "========================================"
echo "Image prefixes: ${IMAGE_PREFIXES[*]}"
echo "Force remove: $FORCE_REMOVE"
echo "========================================"
echo ""

# Build grep pattern for all prefixes
GREP_PATTERN=$(IFS="|"; echo "${IMAGE_PREFIXES[*]}")

# Find all containers using x-attn or ruler images
CONTAINERS=$(sg docker -c "docker ps -a --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}'" | grep -E "$GREP_PATTERN" || true)

if [ -z "$CONTAINERS" ]; then
    echo "No containers found using these image prefixes."
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
