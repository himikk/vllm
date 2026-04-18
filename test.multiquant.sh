#!/bin/bash
# Run MultiQuant unit tests inside the container
set -euo pipefail

IMAGE="${1:-localhost/vllm-multiquant}"
echo "MultiQuant Tests — Image: $IMAGE"
echo "================================"

podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  "$IMAGE" \
  python3 -m pytest /opt/tests/multiquant/ -v --tb=short --noconftest 2>&1

echo ""
echo "Done."
