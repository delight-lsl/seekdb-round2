#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-minivectordb:latest}"
DATA_DIR="${2:-$(pwd)/data}"

mkdir -p "$DATA_DIR"
chmod 777 "$DATA_DIR" 2>/dev/null || true

exec docker run --rm -i \
  --network=none \
  --cpus=2 \
  --memory=2g \
  --memory-swap=2g \
  --pids-limit=128 \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  -v "$DATA_DIR:/data" \
  "$IMAGE_NAME"
