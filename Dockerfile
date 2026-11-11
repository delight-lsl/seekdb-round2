# Multi-stage Dockerfile building MiniVectorDB from source on Ubuntu 24.04
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    (apt-get update && apt-get install -y --no-install-recommends ca-certificates cargo rustc gcc libc6-dev) || { \
        sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g; s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; \
        apt-get update && apt-get install -y --no-install-recommends ca-certificates cargo rustc gcc libc6-dev; \
    }; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY .cargo ./.cargo
COPY vendor ./vendor
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --offline

# Runtime stage matching section 4.2.1 (Ubuntu 24.04)
FROM ubuntu:24.04

WORKDIR /app
COPY --from=builder /build/target/release/minivectordb /app/minivectordb
RUN chmod 755 /app/minivectordb && \
    ln -s /app/minivectordb /usr/local/bin/minivectordb && \
    mkdir -p /data && chmod 1777 /data

VOLUME ["/data"]

ENTRYPOINT ["/app/minivectordb"]
CMD ["--data-dir", "/data"]
