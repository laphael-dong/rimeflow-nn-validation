# YOLOv8n 契约与验收证据

本目录是 `rimeflow-yolov8n` 的任务 1 证据根目录。所有 JSON 使用稳定字段顺序和 UTF-8 编码；模型、图片 fixture、raw tensor fixture、Web 参考结果和转换 spike 均有独立 SHA-256。`.evidence/` 仅用于本机临时工具环境，已被忽略，不属于证据。

`/home/raffael/下载/yolov8n_ios_benchmark_handoff/Assets` 中的 `.pt` 与 ONNX 只作为内部 `onnx-base` 框架升级验证输入，并通过指定 SHA-256 锁定。本证据按模型文件 metadata 如实记录许可证声明，但不在任务 1.4 中评估 RimeCut 商业授权；模型及临时转换产物不进入 RimeCut 产品包或发布目录。

## 锁定输入

- 源 commit：`eacbcf00dfc2fba941b494e2955e87fffd707382`
- 模型：`models/yolov8n.onnx`
- 模型 SHA-256：`9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad`
- Web runtime：`onnxruntime-web@1.27.0`，WASM EP 是确定性参考；WebGPU 仅单独记录，不能与 WASM 性能混用。

## 干净环境复现

在仓库根目录执行：

```sh
bun install --frozen-lockfile --cwd evidence/tooling/web
node evidence/scripts/lock_python_tooling.mjs
node evidence/scripts/prepare_python_tooling.mjs
node evidence/scripts/prepare_mindspore_lite.mjs
bun run evidence/scripts/generate_all.mjs
sha256sum models/yolov8n.onnx evidence/model/model-contract.json evidence/fixtures/manifest.json evidence/golden/web-reference.json evidence/reports/preprocess-conformance.json evidence/reports/model-provenance.json evidence/conversions/conversion-spikes.json
cargo test --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml
node evidence/scripts/test_validator_negative.mjs
node evidence/scripts/generate_task1_replay_manifest.mjs
bun run evidence/scripts/validate_evidence.mjs
git diff --check
```

`generate_all.mjs` 会先从固定 `ultralytics/assets@42ef8a125df038dcca49f6216f446fe9112946c1` 获取并校验源文件，再读取模型与 ORT session 的真实 metadata，生成 contract、PPM fixture、raw tensor fixture、三次 WASM 推理和真实生产 WGSL conformance 报告。图片解码与裁剪固定为 `sharp@0.34.4`，其包完整性由 `evidence/tooling/web/bun.lock` 锁定。

Python 转换工具固定为 CPython 3.12/Linux x86_64 wheel，并由 `requirements.lock` 逐包记录 PyPI 官方 SHA-256。`prepare_python_tooling.mjs` 使用 `--require-hashes --no-deps` 安装后执行 `pip check`。LiteRT 2.1.6 对 `backports.strenum` 的 metadata 仍是无条件依赖，但该包声明 Python `<3.11`；Python 3.12 的 LiteRT 实现实际使用标准库 `enum.StrEnum`，因此准备命令显式使用 `--ignore-requires-python` 安装该已锁纯 Python wheel，不能省略或隐藏该上游元数据例外。MindSpore Lite 2.7.0 tarball 使用官方发布页给出的 SHA-256 校验。

Android LiteRT 转换使用独立的 `litert-requirements.lock`。该文件锁定 CPython 3.12/Linux x86_64 环境的全部传递依赖和单个 wheel SHA-256；`torch==2.12.1+cpu`、`torchvision==0.27.1+cpu` 来自 PyTorch 官方 CPU index，其余包来自 PyPI。实际官方路径为 `ultralytics==8.4.104` 的 `YOLO.export(format="litert")`，由 `litert-torch==0.9.3`、`litert-converter==0.3.1` 直接把 PyTorch graph 降级为 TFLite；该路径不安装、不调用 TensorFlow 或 onnx2tf。重新建立环境和执行两轮 replay：

```sh
node evidence/scripts/lock_litert_tooling.mjs
RIMEFLOW_LITERT_VENV=.evidence/litert/verify-venv node evidence/scripts/prepare_litert_tooling.mjs
HANDOFF_ASSETS=/home/raffael/下载/yolov8n_ios_benchmark_handoff/Assets
.evidence/litert/verify-venv/bin/python evidence/scripts/run_litert_replay.py \
  --pt "$HANDOFF_ASSETS/yolov8n.pt" \
  --workspace .evidence/litert/replay
```

转换 worker 在解析 checkpoint 前验证 SHA-256，并在结束后再次验证源文件 bytes/hash/mtime 均未改变。它只把 exporter 的逻辑输出前缀重定向到 `.evidence/`，不复制 `.pt`。Ultralytics 在 FlatBuffer 后附加 `metadata.json` ZIP entry；worker 只把该 entry 的两个 DOS timestamp 字段固定为 `2026-08-11T00:00:00`，同时记录未改动的 FlatBuffer prefix bytes/SHA-256。模型图、权重、tensor 和 metadata 内容不做改写。

候选产物固定为 `.evidence/litert/artifacts/yolov8n-fp32.tflite`，12,841,227 字节，SHA-256 `794e17d9a2795084787e5125bcfada6cb501c6afc4f708e7e58e96fd8dc84be1`。该目录被 Git 忽略；重新生成方法、完整命令/时间/退出码/stdout/stderr、artifact manifest 和 golden 结果分别在 `evidence/scripts/run_litert_replay.py`、`evidence/reports/litert-conversion-report.json`、`evidence/conversions/litert-artifact-manifest.json` 与 `evidence/reports/litert-golden-report.json`。

外部模型 handoff 的 `.pt`/ONNX 审计使用独立、隔离的 CPU 环境，顶层版本固定在 `evidence/tooling/model-audit-requirements.lock`。先从 PyTorch CPU index 安装 `torch==2.12.1+cpu` 与 `torchvision==0.27.1+cpu`，再安装锁文件中的其余版本；随后执行：

```sh
$AUDIT_PYTHON evidence/scripts/audit_handoff_models.py --pt "$HANDOFF_ASSETS/yolov8n.pt" --onnx "$HANDOFF_ASSETS/yolov8n.onnx" --reference-onnx models/yolov8n.onnx --exported-onnx "$WORK/yolov8n.onnx" --output evidence/reports/handoff-model-audit.json
```

审计确认 `.pt` 与 Ultralytics `assets` v8.3.0 release asset 的当前字节一致，并对 checkpoint、候选 ONNX 和仓库参考 ONNX 执行固定输入两轮推理与 initializer 比较。`$WORK/yolov8n.onnx` 必须由报告中的锁定 `yolo export` 命令在隔离目录生成，只是临时失败/比对 artifact，不得提交。上游 URL 不作为产品发布来源；本任务只以指定 SHA-256 标识内部框架验证输入。

外部源文件逐文件记录在 `evidence/fixtures/manifest.json`：`im/bus.jpg` 和 `docs/ultralytics-dogs.avif` 均锁定 upstream commit、Git blob SHA、内容 SHA-256、准确转换与 upstream 根 `AGPL-3.0-only` LICENSE。完整 AGPL-3.0 文本保存为 `evidence/fixtures/licenses/ultralytics-assets-AGPL-3.0.txt`，固定来源与文件清单保存在 `evidence/fixtures/THIRD_PARTY_NOTICES.md`；validator 要求两者存在且由 Git 跟踪。仓库的 MIT LICENSE 不覆盖这些图片，所有外部图片只允许用于测试/evidence，禁止进入 RimeCut 产品安装包。无检测图由脚本生成并标记 CC0；不使用本机 `素材/` 或私人媒体。

模型级图片只覆盖无检测、单目标、多类别、边界框和极端宽高比。重叠框/NMS 由 `evidence/fixtures/raw/overlap-nms.json` 与隔离的 `evidence/tooling/raw-golden` harness 通过 `#[path]` 直接编译生产 `src/postprocess.rs`，并调用真实 `decode_yolo_output`/`nms` 验证；该 raw tensor 明确不来自图片推理。分层关系见 `evidence/golden/coverage-matrix.json`。

性能重新采样使用 `RIMEFLOW_RECORD_PERFORMANCE=1 bun run evidence/scripts/run_web_golden.mjs`，随后运行 `bun run evidence/scripts/finalize_manifest.mjs`。计时与 RSS 是环境测量值，重新采样预期会变化；contract、fixture、raw tensor、Web 原始 tensor 与 decode reference 则必须重复生成相同 digest。

## 状态判定

Core ML 9.0 仍不接受 ONNX 作为直接转换源。Android 已通过官方 PyTorch/Ultralytics `format=litert` 路径生成 FP32 TFLite，并由 `ai-edge-litert==2.1.6` 在 host 完成真实 Load/Run 和五个图片 golden。runtime 实际输入为 `serving_default_args_0`/index 0/NCHW FP32 `[1,3,640,640]`，输出为 `serving_default_output_0_output`/index 414/attributes-first FP32 `[1,84,8400]`，量化 scale/zero-point 均为 `0/0`。官方 exporter 把输出 attributes 0..3 除以 640；测试层明确乘回 640 后比较，其他轴和值不映射。模型 op 列表无 NMS，letterbox/RGB/normalize 和 decode/NMS 仍由 adapter/operator 负责。官方 MindSpore Lite 2.7.0 `converter_lite` 已进入 ONNX graph optimization 并记录失败算子。Windows x64/ARM64 缺真实 runner，不能以 Linux 上的兼容命令替代加载证据。

`evidence/reports/model-provenance.json` 由 ONNX 1.22.0 读取真实 metadata，并结合 Git 历史和 `handoff-model-audit.json` 生成。原始 `.pt` 的来源、准确 SHA 和与两个 ONNX 的权重/推理等价性已经验证。用途限定为内部 `onnx-base` 框架验证，产品打包明确排除；商业授权判断不属于本次技术验证。Android conversion 子项已达到 `artifact-verified` 和 `host-inference-verified`，但尚未完成 Android arm64 真机 runner、adapter、性能、包加载或 fallback 验证，因此 `supported` 保持 `false`。任务 1.4 与大任务 1 仍保持 blocked，且 1.4 不勾选。
