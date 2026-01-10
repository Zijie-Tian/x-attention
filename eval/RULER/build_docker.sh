#!/bin/bash
#############################################
# RULER Docker Image Build Script
#############################################

#############################################
# Configuration
#############################################

IMAGE_NAME="tzj/ruler"
IMAGE_TAG="v0.1"
DOCKERFILE_PATH="."

#############################################
# Build
#############################################

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Context: ${DOCKERFILE_PATH}"
echo ""

docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${DOCKERFILE_PATH}/Dockerfile" \
    "${DOCKERFILE_PATH}"

BUILD_STATUS=$?

if [ ${BUILD_STATUS} -eq 0 ]; then
    echo ""
    echo "Build successful: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To run the container:"
    echo "  docker run --gpus all -it --rm \\"
    echo "      --shm-size=16g \\"
    echo "      -v /data/models:/data/models:ro \\"
    echo "      ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash"
else
    echo ""
    echo "Build failed with exit code: ${BUILD_STATUS}"
    exit ${BUILD_STATUS}
fi
