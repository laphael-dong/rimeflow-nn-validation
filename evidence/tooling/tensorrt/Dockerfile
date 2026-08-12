FROM docker.io/library/rust:1.85.1-alpine3.21@sha256:813376c206852d4250641eb86720c04fd20fbc6547d1a030c93a45b22a1303a3 AS rust-builder
WORKDIR /build
COPY evidence/tooling/tensorrt-postprocess evidence/tooling/tensorrt-postprocess
COPY src/postprocess.rs src/postprocess.rs
RUN CARGO_TARGET_DIR=/build/tensorrt-target CARGO_INCREMENTAL=0 \
    cargo build --locked --offline --release \
      --manifest-path evidence/tooling/tensorrt-postprocess/Cargo.toml \
      --bin rimeflow-tensorrt-postprocess

FROM docker.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04@sha256:2a015be069bda4de48d677b6e3f271a2794560c7d788a39a18ecf218cae0751d

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=10.9.0.34-1+cuda12.8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg python3.10 python3-pip \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia-cuda.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libnvinfer10=${TRT_VERSION} libnvinfer-plugin10=${TRT_VERSION} \
      libnvonnxparsers10=${TRT_VERSION} python3-libnvinfer=${TRT_VERSION} \
 && rm -rf /var/lib/apt/lists/*

COPY evidence/tooling/tensorrt-requirements.lock /tmp/requirements.lock
RUN python3.10 -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY --from=rust-builder /build/tensorrt-target/release/rimeflow-tensorrt-postprocess /usr/local/bin/rimeflow-tensorrt-postprocess
WORKDIR /workspace
