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

Apple Core ML 转换使用独立的 `coreml-requirements.lock`，锁定 CPython 3.12/Linux x86_64 的全部传递依赖和逐 wheel SHA-256。PyTorch `2.7.0+cpu` 是 coremltools 9.0 明确测试过的最新版本，配套 torchvision `0.22.0+cpu`；两者来自 PyTorch 官方 CPU index。其余依赖来自 PyPI，其中 Ultralytics 固定为 `8.4.104`、coremltools 固定为 `9.0`、NumPy 固定为 `2.3.5`。重新建立环境并执行两轮 scoped replay：

```sh
node evidence/scripts/lock_coreml_tooling.mjs
RIMEFLOW_COREML_VENV=.evidence/coreml/verify-venv node evidence/scripts/prepare_coreml_tooling.mjs
HANDOFF_ASSETS=/home/raffael/下载/yolov8n_ios_benchmark_handoff/Assets
.evidence/coreml/verify-venv/bin/python evidence/scripts/run_coreml_replay.py \
  --pt "$HANDOFF_ASSETS/yolov8n.pt" \
  --workspace .evidence/coreml/replay \
  --record
node evidence/scripts/run_conversion_spikes.mjs --coreml-only
```

只有 `--record` 模式可以创建或替换 `.evidence/coreml/artifacts/yolov8n-fp32.mlpackage`，并同步刷新 tracked artifact manifest 与 conversion report。普通 replay 必须省略 `--record` 并使用独立 `--workspace`；它只在 workspace 生成 package，读取 tracked manifest/report 验证 normalized spec、normalized package manifest、`weight.bin`、I/O、FLOAT32、预处理、坐标、NMS、source 和工具链锁，绝不创建、删除或修改固定候选。若固定候选存在，replay 保存其前后精确 tree digest 并要求一致；若不存在，则报告 `recorded-artifact-unavailable`，不能声称本机已验证精确 artifact identity。

Core ML worker 在导入 Ultralytics/PyTorch 前校验 `.pt` 的 6,549,796 字节和锁定 SHA-256，并在转换结束后复核 size、SHA-256、mtime_ns 均未变化。正式路径为 `YOLO.export(format="coreml", batch=1, imgsz=640, dynamic=false, nms=false, device="cpu", half=false, quantize=None)`，内部执行 `torch.jit.trace` 和 coremltools MIL/ML Program conversion。转换器未显式接收 minimum deployment target；真实 spec 是 specificationVersion 6/CoreML5，对应 iOS 15、macOS 12、watchOS 8、tvOS 15。证据明确区分 `recordedArtifactTreeDigest`（固定候选逐字节身份）与 `semanticReplayDigests`（忽略合法 UUID/date 波动后的可重复转换语义）；后者不得作为 artifact digest。

候选产物位于 `.evidence/coreml/artifacts/yolov8n-fp32.mlpackage`，只含 `model.mlmodel`、`weight.bin` 和 `Manifest.json` 三个文件，总计 12,825,402 字节；该路径被 Git 忽略。每轮均按排序后的 POSIX 相对路径、文件字节数和逐文件 SHA-256 计算 canonical tree digest，不读取目录时间戳。两轮原始 tree digest 不同：`weight.bin` 完全一致，差异仅为 `Manifest.json` 的随机 UUID 和 `model.mlmodel` 中 `description.metadata.userDefined.date`。比较层只为诊断计算移除这些字段后的 digest，未修改任何 package、模型图或权重；归一化 spec/manifest digest、I/O、精度和坐标契约均一致。

HarmonyOS 转换 spike 固定使用官方 MindSpore Lite 2.7.0 Linux x64 archive，archive SHA-256 为 `8bb1097100c9fec12675670ba2d4264a2cd6da3a9be093eb56631d00fc0c455b`，内部 commit 为 `d2b243f75f33a7a896483b09e567d845155cad06`。`converter_lite` 没有可用的 `--version` 子命令，因此版本证据由官方 archive 文件名/SHA、`.commit_id`、converter 二进制 SHA 和 `--help` 输出共同组成。执行两轮有限矩阵、Linux host Load/Run 与生产 Rust golden：

```sh
node evidence/scripts/prepare_mindspore_lite.mjs
HANDOFF_ASSETS=/home/raffael/下载/yolov8n_ios_benchmark_handoff/Assets
RIMEFLOW_MINDSPORE_VENV=.evidence/mindspore/python-venv node evidence/scripts/prepare_mindspore_python.mjs
MINDSPORE_PYTHON=.evidence/mindspore/python-venv/bin/python
$MINDSPORE_PYTHON evidence/scripts/run_mindspore_replay.py \
  --pt "$HANDOFF_ASSETS/yolov8n.pt" \
  --handoff-onnx "$HANDOFF_ASSETS/yolov8n.onnx" \
  --workspace .evidence/mindspore/record-replay \
  --record
node evidence/scripts/run_conversion_spikes.mjs --mindspore-only
$MINDSPORE_PYTHON evidence/scripts/run_mindspore_replay.py \
  --pt "$HANDOFF_ASSETS/yolov8n.pt" \
  --handoff-onnx "$HANDOFF_ASSETS/yolov8n.onnx" \
  --workspace .evidence/mindspore/replay
```

只有 `--record` 可以在两轮转换、host Load/Run、生产 Rust golden 与确定性检查全部通过后，通过同目录 staging 和原子替换创建或更新 `.evidence/mindspore/artifacts/yolov8n-fp32.ms`，并同步刷新 tracked manifest、golden report 和 conversion report。普通 replay 必须使用与 `.evidence/mindspore/artifacts/` 不重叠的独立 workspace；它只生成临时 ONNX、`.ms` 和 workspace report，不创建、删除、覆盖或触碰固定候选时间戳。普通 replay 在转换前校验 tracked record 与已存在候选的 bytes/SHA，在结束后校验候选 bytes/SHA/mtime 和三个 tracked JSON 的 SHA 均未变化；候选缺失时明确记录 `recorded-artifact-unavailable`，不得声称验证了本机固定候选身份。报告分别使用 `recordedArtifactSha256` 和 `replayArtifactSha256`，即使 digest 相同也不混淆固定候选存在性与 workspace 确定性。

脚本在每轮开始和结束验证 `.pt`、handoff ONNX、规范 ONNX 与 archive 的字节数和锁定 SHA，源文件只读且不得覆盖。矩阵只包含六条有依据的路径：规范 ONNX 原命令重放、规范 ONNX 加静态 `inputShape`、handoff ONNX 加静态 `inputShape`、从锁定 `.pt` 以 `imgsz=640,batch=1,opset=17,dynamic=false,simplify=false,nms=false,optimize=false` 重导出后转换、规范 ONNX 加 `optimize=none`，以及对重导出图进行 DFL 等价改写后转换。前五条均稳定失败于 `legacy_optimizer/InferSubgraph -> Conv2DFusion infer-shape -> graph pass` 的 `/model.22/dfl/conv/Conv`；输入、权重和输出 Shape 分别为 `[1,16,4,8400]`、`[1,16,1,1]`、`[1,1,4,8400]`，并保留 `/model.10/Resize` 与 `/model.13/Resize` 的可选空输入警告。

唯一成功路径使用 `derive_mindspore_onnx.py` 的结构化 ONNX API，把权重已断言精确为 `0..15` 的单个无 bias DFL `1x1 Conv` 替换为 `Mul + ReduceSum(axis=1, keepdims=1)`。重导出 ONNX 为 12,824,178 字节、SHA-256 `8718af53d53b6336f301ef7eacb529376f29f0c04bec415815fde7d734b9def2`；派生 ONNX 为 12,824,387 字节、SHA-256 `a5a73dd7a25245eb47f7de8d35fa1f612212b38587d88494bb67ad6e0753b6ea`。重导出图与规范 ONNX 在三个合成输入和五个图片 fixture 上逐元素一致，派生图在相同八个输入上通过冻结 raw 容差；两图均无 NMS。两轮转换得到完全相同的 MindIR Lite `.ms`，12,832,800 字节、SHA-256 `7ceeca31471d772c0ccf426b856a2533fcf66735aa3c3c90726c2e459c83e6a5`，只保存在 ignored `.evidence/mindspore/artifacts/`。

官方 `benchmark` 与官方 C++ MindSpore Lite runtime 已在 Linux x64 host 对五个图片 fixture 真实 Load/Run。实际输入为 `images`/index 0/NHWC FP32 `[1,640,640,3]`，输出为 `output0`/index 0/attributes-first FP32 `[1,84,8400]`，输入输出 quantization 均为空。未来 adapter 负责 letterbox、RGB、`/255` 和 NCHW 到 NHWC 转置；模型输出仍是 640×640 letterbox 输入像素单位的 `xywh`，NMS 仍由 operator 的单份生产 `src/postprocess.rs` 负责。本子项未实现 adapter，也没有 HarmonyOS 真机、性能、fallback 或包加载证据，因此仅为 `host-inference-verified`，`supported=false`，任务 1.4 保持未完成。

外部模型 handoff 的 `.pt`/ONNX 审计使用独立、隔离的 CPU 环境，顶层版本固定在 `evidence/tooling/model-audit-requirements.lock`。先从 PyTorch CPU index 安装 `torch==2.12.1+cpu` 与 `torchvision==0.27.1+cpu`，再安装锁文件中的其余版本；随后执行：

```sh
$AUDIT_PYTHON evidence/scripts/audit_handoff_models.py --pt "$HANDOFF_ASSETS/yolov8n.pt" --onnx "$HANDOFF_ASSETS/yolov8n.onnx" --reference-onnx models/yolov8n.onnx --exported-onnx "$WORK/yolov8n.onnx" --output evidence/reports/handoff-model-audit.json
```

审计确认 `.pt` 与 Ultralytics `assets` v8.3.0 release asset 的当前字节一致，并对 checkpoint、候选 ONNX 和仓库参考 ONNX 执行固定输入两轮推理与 initializer 比较。`$WORK/yolov8n.onnx` 必须由报告中的锁定 `yolo export` 命令在隔离目录生成，只是临时失败/比对 artifact，不得提交。上游 URL 不作为产品发布来源；本任务只以指定 SHA-256 标识内部框架验证输入。

外部源文件逐文件记录在 `evidence/fixtures/manifest.json`：`im/bus.jpg` 和 `docs/ultralytics-dogs.avif` 均锁定 upstream commit、Git blob SHA、内容 SHA-256、准确转换与 upstream 根 `AGPL-3.0-only` LICENSE。完整 AGPL-3.0 文本保存为 `evidence/fixtures/licenses/ultralytics-assets-AGPL-3.0.txt`，固定来源与文件清单保存在 `evidence/fixtures/THIRD_PARTY_NOTICES.md`；validator 要求两者存在且由 Git 跟踪。仓库的 MIT LICENSE 不覆盖这些图片，所有外部图片只允许用于测试/evidence，禁止进入 RimeCut 产品安装包。无检测图由脚本生成并标记 CC0；不使用本机 `素材/` 或私人媒体。

模型级图片只覆盖无检测、单目标、多类别、边界框和极端宽高比。重叠框/NMS 由 `evidence/fixtures/raw/overlap-nms.json` 与隔离的 `evidence/tooling/raw-golden` harness 通过 `#[path]` 直接编译生产 `src/postprocess.rs`，并调用真实 `decode_yolo_output`/`nms` 验证；该 raw tensor 明确不来自图片推理。分层关系见 `evidence/golden/coverage-matrix.json`。

性能重新采样使用 `RIMEFLOW_RECORD_PERFORMANCE=1 bun run evidence/scripts/run_web_golden.mjs`，随后运行 `bun run evidence/scripts/finalize_manifest.mjs`。计时与 RSS 是环境测量值，重新采样预期会变化；contract、fixture、raw tensor、Web 原始 tensor 与 decode reference 则必须重复生成相同 digest。

## 状态判定

Linux x86_64 OpenVINO provider spike 使用官方 `onnxruntime-openvino==1.24.1` CPython 3.12 manylinux wheel；该 wheel 内置 OpenVINO Runtime `2025.4.1`。完整传递依赖、每个 wheel 的 SHA-256 与精确版本固定在 `evidence/tooling/openvino-requirements.lock`，官方 PyPI 下载 URL、许可证字段、ORT build commit、实际映射的 ORT/OpenVINO shared library bytes/SHA、host/CPU/kernel/glibc、OpenVINO CPU device/plugin introspection 则记录在 `evidence/conversions/openvino-ep-manifest.json`。本 spike 不做模型转换，只读取唯一 `models/yolov8n.onnx`；runtime、wheel、cache、profile 和 raw tensor 仅进入 ignored `.evidence/openvino/`，不进入 Git、RimeCut、产品包或发布目录。

建立并执行 record：

```sh
RIMEFLOW_OPENVINO_VENV=.evidence/openvino/venv node evidence/scripts/prepare_openvino_tooling.mjs
.evidence/openvino/venv/bin/python evidence/scripts/run_openvino_replay.py \
  --workspace .evidence/openvino/record \
  --record
node evidence/scripts/run_conversion_spikes.mjs --openvino-only
```

只有 `--record` 在两轮、每轮五个 fixture 各两次 OpenVINO inference、真实 ORT profile、冻结 raw/decoded 比较、生产 Rust decode/NMS 与确定性全部通过后，才通过下述 journal/recovery 协议 durable publish tracked manifest/report。session 明确按 `OpenVINOExecutionProvider`、`CPUExecutionProvider` 顺序创建，并指定 OpenVINO `device_type=CPU`。record 的每轮 profile 包含 10 次 `OpenVINOExecutionProvider` execution event，唯一图节点统计为 OpenVINO 1、CPU 0，所以该模型/host/run 的 `executionPlan=full`；若以后 profile 出现 CPU 节点必须报告 `partitioned`，无 OpenVINO 节点则保持 `blocked`。

独立普通 replay 使用第二个 hash-locked venv，且不得修改 tracked evidence：

```sh
RIMEFLOW_OPENVINO_VENV=.evidence/openvino/replay-venv node evidence/scripts/prepare_openvino_tooling.mjs
.evidence/openvino/replay-venv/bin/python evidence/scripts/run_openvino_replay.py \
  --workspace .evidence/openvino/replay
node evidence/scripts/test_openvino_replay_guards.mjs
```

普通 replay 只写 ignored workspace，并在执行前后比较 tracked manifest/report 的 bytes/SHA；record 与 replay 的语义 digest 必须一致。五个图片 fixture 都通过 runtime metadata、Shape/dtype/元素数、全有限值、同轮重复、两轮 replay、冻结 raw tolerance、class/confidence/bbox/IoU 以及生产 `src/postprocess.rs` decode/NMS。OpenVINO 专用 decode/NMS 未增加，NMS 仍由 operator 负责。该结果只将 Linux x86_64 OpenVINO 子项提升到 `host-inference-verified`；adapter、性能、打包和任务 1.4 全平台闭环均未完成，因此 `supported=false`、`task14Complete=false`，OpenSpec 1.4 仍不勾选。

OpenVINO validator 不信任 report 的 `passed`、`allClose` 或数值摘要。它逐个读取每次推理的 705600 个 little-endian FP32 值及同 fixture 的冻结 Web reference 文件，独立检查 Shape/dtype/长度/digest/有限值，并逐元素重算 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`、最大差异位置（attribute/anchor/两侧值/该位置容差）、均值与 reference absolute `<1e-6` 的 near-zero 统计；随后将全部重算字段与 report 逐项比较。decoded 结果也从生产 Rust harness 重新生成，并相对冻结 Web decoded reference 独立重算 detection count、class exact、confidence absolute、bbox maximum absolute 和 IoU，不能用 report 自报的 `decodedComparison` 代替。

`--record` 对 manifest/report 使用 durable journal 协调的双文件发布。单个文件通过其目标目录内的 staging 与同目录 `os.replace()` 原子替换；由于两个目标位于不同目录，本实现不声称两次 rename 在任意进程中断点构成一个文件系统原子操作。两文件一致性由 ignored `.evidence/openvino/transaction/` 内的 durable journal、staging、backup 和每次 record/replay/validator 读取前的启动恢复保证：`prepared`/`publishing` 恢复完整 old generation，`committed` 完成完整 new generation。journal 记录 schema、transaction ID、phase、intended generation、两个 canonical target、old existence/bytes/SHA、new bytes/SHA、staging、backup 和 recovery copy 路径；文件写入、journal phase 更新、target replace 和恢复均执行适用的文件及父目录 `fsync`。只有确认两个目标处于同一 generation 且目录同步完成后才删除恢复材料；恢复不完整时保留 journal/backup/staging/recovery copy，并用 `ExceptionGroup` 保留发布和恢复错误。双目录故障注入与真实 `os._exit` 崩溃恢复由 `python3 evidence/scripts/test_openvino_publish_transaction.py` 执行，并由 `node evidence/scripts/test_openvino_replay_guards.mjs` 统一调用。

OpenVINO validator 不读取或执行仓库中 ignored `evidence/tooling/raw-golden/target/` 的预存程序。每次 record evidence 或 ordinary replay validation 都先验证 `Cargo.toml`、`Cargo.lock`、`src/main.rs`、`src/lib.rs` 与生产 `src/postprocess.rs` 的 canonical path、bytes、SHA-256 及当前内容对应的 Git blob 与 `HEAD` 一致，再于独立临时 `CARGO_TARGET_DIR` 执行 `cargo build --offline --locked --manifest-path evidence/tooling/raw-golden/Cargo.toml`，并且只运行该临时目录中新构建的 runner。成功或异常路径都会清理临时 target 和 decoded JSON；Node、shell 或复制既有 decoded JSON 的假 executable、source/lock 漂移、build 失败及 runner 缺失均会阻断验证。

Apple 已通过官方 `.pt`/Ultralytics/Core ML 路径生成 ML Program `.mlpackage` 并由 coremltools spec API 完成真实结构检查。输入由 ONNX FLOAT MultiArray 语义变为 Core ML RGB Image feature `image`/index 0/640×640，ML Program 函数张量仍为 NCHW FLOAT32 `[1,3,640,640]`；模型图融合 RGB image conversion 和 `1/255` 缩放，不融合 letterbox resize/padding。输出为 `var_911`/index 0/FLOAT32 `[1,84,8400]`，仍是 attributes-first；真实 stride blob 为 6400 个 8、1600 个 16、400 个 32，bbox 直接乘 stride 后作为 640×640 输入像素单位 `xywh` 输出。129 个外置 blob constant 和全部浮点 op 输出均为 FLOAT32，图中不存在 FLOAT16 或 NMS op。Linux 只能保存/解析 `.mlpackage`，不能调用 Core ML runtime，因此 macOS/iOS Load/Run、golden、包加载和性能仍未验证，Apple 仅为 `artifact-spec-verified`，`supported=false`。

Android 已通过官方 PyTorch/Ultralytics `format=litert` 路径生成 FP32 TFLite，并由 `ai-edge-litert==2.1.6` 在 host 完成真实 Load/Run 和五个图片 golden。runtime 实际输入为 `serving_default_args_0`/index 0/NCHW FP32 `[1,3,640,640]`，输出为 `serving_default_output_0_output`/index 414/attributes-first FP32 `[1,84,8400]`，量化 scale/zero-point 均为 `0/0`。官方 exporter 把输出 attributes 0..3 除以 640；测试层明确乘回 640 后比较，其他轴和值不映射。模型 op 列表无 NMS，letterbox/RGB/normalize 和 decode/NMS 仍由 adapter/operator 负责。HarmonyOS MindSpore Lite 已达到 Linux host `host-inference-verified`，但没有 HarmonyOS 真机证据。Windows x64/ARM64 缺真实 runner，不能以 Linux 上的兼容命令替代加载证据。

`evidence/reports/model-provenance.json` 由 ONNX 1.22.0 读取真实 metadata，并结合 Git 历史和 `handoff-model-audit.json` 生成。原始 `.pt` 的来源、准确 SHA 和与两个 ONNX 的权重/推理等价性已经验证。用途限定为内部 `onnx-base` 框架验证，产品打包明确排除；商业授权判断不属于本次技术验证。Apple conversion 子项达到 `artifact-spec-verified`，Android 与 HarmonyOS conversion 子项达到 Linux host `host-inference-verified`；三者均缺真实目标平台 runner，且未实现对应 adapter、性能、包加载或 fallback 验证，因此 `supported` 均保持 `false`。任务 1.4 与大任务 1 仍保持 blocked，且 1.4 不勾选。
