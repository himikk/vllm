#!/bin/bash
# Build vllm-multiquant container
#
# Usage:
#   ./build.sh                     # DGX Spark (SM121a + SM120a)
#   ./build.sh --rtx               # RTX PRO 6000 (SM120a only)
#   ./build.sh --jobs 8            # Limit parallel jobs
#   ./build.sh --no-cache          # Force full rebuild

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="vllm-multiquant"
BASE="nvcr.io/nvidia/vllm:26.02-py3"
ARCHS="12.0a;12.1a"
JOBS=16
EXTRA="--no-cache"  # Default: fresh git clone (ccache handles C++ speed)

while [[ $# -gt 0 ]]; do
    case $1 in
        --rtx)              ARCHS="12.0a"; shift ;;
        --jobs)             JOBS="$2"; shift 2 ;;
        --use-layer-cache)  EXTRA=""; shift ;;
        --tag)              IMAGE="$2"; shift 2 ;;
        *)                  echo "Unknown: $1"; exit 1 ;;
    esac
done

CACHE_BASE="/data/sources"
PIP_CACHE="$CACHE_BASE/pip-cache"
CCACHE_DIR="$CACHE_BASE/ccache"
TORCH_EXT="$CACHE_BASE/torch-extensions"
CUTLASS_DIR="$CACHE_BASE/cutlass"

mkdir -p "$PIP_CACHE" "$CCACHE_DIR" "$TORCH_EXT"

# Pre-clone CUTLASS locally (saves ~30s per build)
if [[ ! -d "$CUTLASS_DIR/.git" ]]; then
    echo "Cloning CUTLASS 4.4.1 locally..."
    git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi

echo "================================================"
echo "  Building ${IMAGE}"
echo "  Base:   ${BASE}"
echo "  Archs:  ${ARCHS}"
echo "  Jobs:   ${JOBS}"
echo "  Caches: ${CACHE_BASE}/"
echo "================================================"

# Copy FlashInfer wheels if available
if [ ! -f build/wheels/flashinfer_python*.whl ]; then
    MARLIN_WHEELS="$HOME/vllm-marlin-sm12x/build/wheels"
    if [ -d "$MARLIN_WHEELS" ]; then
        echo "Copying FlashInfer wheels from vllm-marlin-sm12x..."
        mkdir -p build/wheels
        cp "$MARLIN_WHEELS"/flashinfer*.whl build/wheels/
    else
        echo "No local FlashInfer wheels — will install from PyPI (slower)"
    fi
fi

podman build \
    -f Dockerfile.multiquant \
    -t "$IMAGE" \
    --build-arg BASE_IMAGE="$BASE" \
    --build-arg CUDA_ARCHS="$ARCHS" \
    --build-arg MAX_JOBS="$JOBS" \
    -v "$PIP_CACHE:/pip-cache:rw" \
    -v "$CCACHE_DIR:/ccache:rw" \
    -v "$TORCH_EXT:/root/.cache/torch_extensions:rw" \
    -v "$CUTLASS_DIR:/opt/cutlass-local:ro" \
    $EXTRA \
    .

echo ""
EXPECTED=$(git rev-parse --short HEAD)
ACTUAL=$(podman run --rm "$IMAGE" cat /opt/vllm-commit 2>/dev/null || echo "???")
echo "================================================"
echo "  Done: ${IMAGE}"
echo "  Size:   $(podman image inspect "$IMAGE" --format '{{.Size}}' | numfmt --to=iec)"
echo "  Commit: ${ACTUAL} (expected: ${EXPECTED})"
echo "  ccache: $(du -sh "$CCACHE_DIR" 2>/dev/null | cut -f1)"
echo "  pip:    $(du -sh "$PIP_CACHE" 2>/dev/null | cut -f1)"
echo "================================================"
if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "  WARNING: Image commit mismatch! Push before build?"
fi
