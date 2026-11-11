#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${1:-minivectordb:latest}"
EXPORT_TAR="${2:-minivectordb.tar}"

echo "[build] Ensuring Cargo.lock is up-to-date..."
cargo check --quiet

if docker image inspect rust:1.93.1-slim-bookworm >/dev/null 2>&1 && docker image inspect ubuntu:24.04 >/dev/null 2>&1; then
    echo "[build] Building from source with standard multi-stage Dockerfile (rust:1.93.1-slim-bookworm -> ubuntu:24.04)..."
    docker build -t "$IMAGE_NAME" -f Dockerfile .
else
    echo "[build] Base images not cached locally; building static x86_64-unknown-linux-musl binary with Rust $(rustc --version)..."
    cargo build --release --target x86_64-unknown-linux-musl --locked
    echo "[build] Packaging minimal multi-stage Docker image via Dockerfile.musl..."
    docker build -t "$IMAGE_NAME" -f Dockerfile.musl .
fi

echo "[build] Exporting Docker image to $EXPORT_TAR..."
docker save -o "$EXPORT_TAR" "$IMAGE_NAME"
ls -lh "$EXPORT_TAR"
docker images "$IMAGE_NAME"
echo "[build] Done: $IMAGE_NAME"
