# Multi-stage Dockerfile for MiniVectorDB
# Stage 1: Build from source with locked Rust 1.93.1
FROM rust:1.93.1-slim-bookworm AS builder

WORKDIR /build
COPY .cargo ./.cargo
COPY vendor ./vendor
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --offline

# Stage 2: Ubuntu 24.04 runtime image (matching section 4.2.1)
FROM centos:7

WORKDIR /
COPY --from=builder --chmod=755 minivectordb /minivectordb

RUN mkdir -p /data && chmod 1777 /data
VOLUME ["/data"]

ENTRYPOINT ["/minivectordb"]
CMD ["--data-dir", "/data"]
