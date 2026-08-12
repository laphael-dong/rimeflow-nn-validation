# Linux x86_64 CUDA EP harness

本目录只验证官方 ONNX Runtime `CUDAExecutionProvider`，不包含 adapter、TensorRT、OpenVINO、ARM64 或性能基线。

兼容链固定为 ONNX Runtime GPU 1.26.0（官方 CUDA 12.8 / cuDNN 9.x 构建）、CUDA runtime 12.8.90、cuDNN 9.10.2.21。ORT 官方兼容表说明 1.26.x 使用 CUDA 12.8 和 cuDNN 9.x；NVIDIA CUDA minor-version compatibility 说明 CUDA 12.x 至少需要 525 系列 driver。所有 Python/NVIDIA wheel 均由 `evidence/tooling/cuda-requirements.lock` 固定 PyPI SHA-256。

在带 NVIDIA GPU、Linux x86_64、driver >= 525 的干净 runner 上执行：

```bash
python3.12 -m venv .evidence/cuda/venv
.evidence/cuda/venv/bin/pip install --require-hashes -r evidence/tooling/cuda-requirements.lock
bun install --cwd evidence/tooling/web --frozen-lockfile
bun evidence/scripts/export_web_reference_tensors.mjs .evidence/cuda/web-reference
cargo build --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml
.evidence/cuda/venv/bin/python evidence/scripts/run_cuda_ep.py \
  --workspace .evidence/cuda/run \
  --web-reference-dir .evidence/cuda/web-reference \
  --output .evidence/cuda/cuda-ep-report.json
node evidence/scripts/validate_cuda_evidence.mjs \
  --report .evidence/cuda/cuda-ep-report.json \
  --manifest evidence/conversions/cuda-ep-spike-manifest.json
```

runner 只请求 `CUDAExecutionProvider`。`get_available_providers()`、`session.get_providers()`、`session.get_provider_options()` 只能证明可配置；必须同时存在 ORT profile 中 `provider=CUDAExecutionProvider` 的 Node event，且已加载 `libonnxruntime_providers_cuda.so`、`libcuda.so`、`libcudart.so`、`libcudnn.so` 和 `libcublas.so`，才会设置 `runtimeExecuted=true`。CPU provider 或仅有 CUDA 配置但没有 CUDA profile 节点都会失败。

成功报告为可重算证据索引，而不是结果声明。每个 profile、`/proc/self/maps` 快照、shared library、input、Web reference raw、CUDA raw 和 decoded JSON 均记录 path、bytes 与 SHA-256；shared library 还记录 canonical realpath、精确 basename、ELF64 x86-64 identity、SONAME、component version 和固定 version source。validator 会重新读取 profile 统计 Node provider，校验持久化 maps 快照与库 realpath/ELF/SONAME/版本身份，按 `abs(actual-reference) <= 1e-5 + 1e-4 * abs(reference)` 逐元素重算 raw，并重新执行锁定的 `evidence/tooling/raw-golden` 后比较类别、置信度、bbox 与 IoU，不信任报告中的 `passed` 或计数。离线 validator 能证明快照与持久化二进制相互一致，不能重新证明采集时瞬时 `/proc/self/maps` 的真实性；真实 runner 的隔离、命令日志和后续不可变 evidence 存储仍是运行时来源边界。

生产后处理 runner 不能由 report 自选路径：它固定为 `evidence/tooling/raw-golden/target/debug/rimeflow-raw-golden`。validator 将其 Cargo manifest/lock、Rust source 和生产 `src/postprocess.rs` 与当前 `HEAD` blob 绑定，现场执行 `cargo build --offline` 后再核对 canonical binary digest。Web input/reference 也不能由 report 自选：`.evidence/cuda` 下 exporter manifest 的 SHA、`sourceReferenceSha256`、fixture 顺序、图片尺寸以及逐 fixture input/raw identity 必须与 tracked `evidence/golden/web-reference.json` 三方一致。

五个图片 fixture 的 raw tensor 由 CUDA session 产生，decoded 结果仅通过 `evidence/tooling/raw-golden` 调用生产 `src/postprocess.rs`；本 harness 不实现 CUDA 专用 decode/NMS。临时 venv、profile、输入张量和 raw 输出必须留在 `.evidence/cuda/`，不得提交或进入 release/dist/publish 目录。

普通 blocked replay 只能写 `.evidence/cuda/`，并对 CUDA manifest/report/replay、golden manifest、task replay 和 conversion spikes 逐文件记录真实 bytes/SHA/mtime before/after。只有证据维护时显式执行 `node evidence/scripts/replay_cuda_ep.mjs --record`，才会在两轮成功且保护检查通过后以同目录 staging 原子替换 tracked replay report。聚合 golden manifest 不哈希 replay report，以避免 replay 自身保护与聚合哈希之间产生不可解的循环身份；tracked replay 由专项 validator 独立验证。

官方依据：

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://pypi.org/project/onnxruntime-gpu/1.26.0/
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
