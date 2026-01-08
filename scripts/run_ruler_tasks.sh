#!/bin/bash
# Test different sparse attention configurations

set -e

#############################################
# Configuration
#############################################

RUN_FULL=true
RUN_XATTN=true
RUN_AVGPOOL=false

#############################################
# Full Attention Tests (Baseline)
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

echo "=== All tests completed ==="
