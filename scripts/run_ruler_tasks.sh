#!/bin/bash
# Test different sparse attention configurations

set -e

#############################################
# Configuration
#############################################

RUN_FULL=false
RUN_XATTN=false
RUN_AVGPOOL=false
RUN_NANOVLLM=true

#############################################
# Full Attention Tests (Baseline - HuggingFace)
#############################################

if [ "$RUN_FULL" = true ]; then
    echo "=== Testing Full Attention (baseline) ==="

    METRIC="full" ./scripts/run_ruler_docker.sh

    echo "=== Full Attention tests completed ==="
fi

#############################################
# XAttn Tests
#############################################

if [ "$RUN_XATTN" = true ]; then
    echo "=== Testing XAttn stride configurations ==="

    # METRIC="xattn" STRIDE="4" ./scripts/run_ruler_docker.sh
    METRIC="xattn" STRIDE="8" ./scripts/run_ruler_docker.sh
    # METRIC="xattn" STRIDE="16" ./scripts/run_ruler_docker.sh

    echo "=== XAttn tests completed ==="
fi

#############################################
# AvgPool Tests
#############################################

if [ "$RUN_AVGPOOL" = true ]; then
    echo "=== Testing AvgPool top-k configurations ==="

    METRIC="avgpool" AVGPOOL_TOPP="" AVGPOOL_TOPK="32" ./scripts/run_ruler_docker.sh
    METRIC="avgpool" AVGPOOL_TOPP="" AVGPOOL_TOPK="64" ./scripts/run_ruler_docker.sh
    METRIC="avgpool" AVGPOOL_TOPP="" AVGPOOL_TOPK="128" ./scripts/run_ruler_docker.sh

    echo "=== Testing AvgPool top-p configurations ==="

    METRIC="avgpool" AVGPOOL_TOPK="" AVGPOOL_TOPP="0.9" ./scripts/run_ruler_docker.sh
    METRIC="avgpool" AVGPOOL_TOPK="" AVGPOOL_TOPP="0.95" ./scripts/run_ruler_docker.sh
    METRIC="avgpool" AVGPOOL_TOPK="" AVGPOOL_TOPP="0.99" ./scripts/run_ruler_docker.sh

    echo "=== AvgPool tests completed ==="
fi

#############################################
# NanoVLLM Tests (Full Attention via nano-vllm)
#############################################

if [ "$RUN_NANOVLLM" = true ]; then
    echo "=== Testing NanoVLLM Full Attention ==="

    # Default: Llama 3.1-8B with GPU-only mode (no CPU offload)
    # MODEL_NAME="llama3.1-8b-nanovllm" ./scripts/run_ruler_nanovllm.sh

    # With CPU offload enabled (single sequence only, for long context)
    # NANOVLLM_CPU_OFFLOAD="true" \
    # NANOVLLM_NUM_GPU_BLOCKS="2" \
    # NANOVLLM_MAX_MODEL_LEN="131072" \
    # MODEL_NAME="llama3.1-8b-nanovllm" ./scripts/run_ruler_nanovllm.sh

    # # GPU-only mode with different max lengths
    # NANOVLLM_MAX_MODEL_LEN="8192" \
    # MODEL_NAME="llama3.1-8b-nanovllm" ./scripts/run_ruler_nanovllm.sh

    # Uncomment for longer sequences (requires more GPU memory)
    # NANOVLLM_MAX_MODEL_LEN="16384" \
    # MODEL_NAME="llama3.1-8b-nanovllm" ./scripts/run_ruler_nanovllm.sh

    # 32K with CPU offload - no OOM with layerwise KV cache offload
    NANOVLLM_CPU_OFFLOAD="true" \
    NANOVLLM_NUM_GPU_BLOCKS="4" \
    NANOVLLM_MAX_MODEL_LEN="131072" \
    MODEL_NAME="llama3.1-8b-nanovllm" ./scripts/run_ruler_nanovllm.sh

    echo "=== NanoVLLM tests completed ==="
fi

echo "=== All tests completed ==="
