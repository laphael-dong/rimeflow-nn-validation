# YOLOv8n 契约与验收证据

本目录是 `rimeflow-yolov8n` 的任务 1 证据根目录。所有 JSON 使用稳定字段顺序和 UTF-8 编码；模型、图片 fixture、raw tensor fixture、Web 参考结果和转换 spike 均有独立 SHA-256。`.evidence/` 仅用于本机临时工具环境，已被忽略，不属于证据。

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

外部模型 handoff 的 `.pt`/ONNX 审计使用独立、隔离的 CPU 环境，顶层版本固定在 `evidence/tooling/model-audit-requirements.lock`。先从 PyTorch CPU index 安装 `torch==2.12.1+cpu` 与 `torchvision==0.27.1+cpu`，再安装锁文件中的其余版本；随后执行：

```sh
$AUDIT_PYTHON evidence/scripts/audit_handoff_models.py --pt "$HANDOFF_ASSETS/yolov8n.pt" --onnx "$HANDOFF_ASSETS/yolov8n.onnx" --reference-onnx models/yolov8n.onnx --exported-onnx "$WORK/yolov8n.onnx" --output evidence/reports/handoff-model-audit.json
```

审计确认 `.pt` 与 Ultralytics `assets` v8.3.0 release asset 的当前字节一致，并对 checkpoint、候选 ONNX 和仓库参考 ONNX 执行固定输入两轮推理与 initializer 比较。`$WORK/yolov8n.onnx` 必须由报告中的锁定 `yolo export` 命令在隔离目录生成，只是临时失败/比对 artifact，不得提交。官方 GitHub release 标记为可变且未提供 asset digest，采用前还需将 `.pt` 镜像到自有不可变制品库。

外部源文件逐文件记录在 `evidence/fixtures/manifest.json`：`im/bus.jpg` 和 `docs/ultralytics-dogs.avif` 均锁定 upstream commit、Git blob SHA、内容 SHA-256、准确转换与 upstream 根 `AGPL-3.0-only` LICENSE。完整 AGPL-3.0 文本保存为 `evidence/fixtures/licenses/ultralytics-assets-AGPL-3.0.txt`，固定来源与文件清单保存在 `evidence/fixtures/THIRD_PARTY_NOTICES.md`；validator 要求两者存在且由 Git 跟踪。仓库的 MIT LICENSE 不覆盖这些图片，所有外部图片只允许用于测试/evidence，禁止进入 RimeCut 产品安装包。无检测图由脚本生成并标记 CC0；不使用本机 `素材/` 或私人媒体。

模型级图片只覆盖无检测、单目标、多类别、边界框和极端宽高比。重叠框/NMS 由 `evidence/fixtures/raw/overlap-nms.json` 与隔离的 `evidence/tooling/raw-golden` harness 通过 `#[path]` 直接编译生产 `src/postprocess.rs`，并调用真实 `decode_yolo_output`/`nms` 验证；该 raw tensor 明确不来自图片推理。分层关系见 `evidence/golden/coverage-matrix.json`。

性能重新采样使用 `RIMEFLOW_RECORD_PERFORMANCE=1 bun run evidence/scripts/run_web_golden.mjs`，随后运行 `bun run evidence/scripts/finalize_manifest.mjs`。计时与 RSS 是环境测量值，重新采样预期会变化；contract、fixture、raw tensor、Web 原始 tensor 与 decode reference 则必须重复生成相同 digest。

## 状态判定

Core ML 9.0 与 LiteRT 2.1.6 已调用真实官方 API：前者不接受 ONNX 作为直接转换源，后者是 TFLite runtime 而不是 ONNX converter。官方 MindSpore Lite 2.7.0 `converter_lite` 已进入 ONNX graph optimization 并记录失败算子。Windows x64/ARM64 缺真实 runner，不能以 Linux 上的兼容命令替代加载证据。转换报告保留完整命令、工具版本、stdout/stderr、退出码、失败阶段、I/O、量化、NMS、许可和再分发结论；失败或授权不明 artifact 不进入发布目录。

`evidence/reports/model-provenance.json` 由 ONNX 1.22.0 读取真实 metadata，并结合 Git 历史和 `handoff-model-audit.json` 生成。原始 `.pt` 的官方来源、准确 SHA 和与两个 ONNX 的权重/推理等价性已经验证；但 GitHub release 仍需不可变镜像，且没有 Ultralytics Enterprise 授权编号或获批的 AGPL 分发合规结论。当前禁止在 RimeCut 包中再分发 ONNX 或转换产物，任务 1.4 与大任务 1 保持 blocked；Core ML、LiteRT、Windows ML、MindSpore Lite 和其余真实平台证据由外部负责人回传后再验收。
