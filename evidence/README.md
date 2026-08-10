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
bun run evidence/scripts/generate_all.mjs
sha256sum models/yolov8n.onnx evidence/model/model-contract.json evidence/golden/web-reference.json
bun run evidence/scripts/validate_evidence.mjs
git diff --check
```

`generate_all.mjs` 会先读取模型文件和 ORT session 的真实 metadata，再生成 contract、合成 PPM 图片、raw tensor fixture 和三次 WASM 推理报告。图片是确定性程序生成的几何图案，无个人信息；raw tensor/NMS fixture 不冒充图片经过模型的结果。

性能重新采样使用 `RIMEFLOW_RECORD_PERFORMANCE=1 bun run evidence/scripts/run_web_golden.mjs`，随后运行 `bun run evidence/scripts/finalize_manifest.mjs`。计时与 RSS 是环境测量值，重新采样预期会变化；contract、fixture、raw tensor、Web 原始 tensor 与 decode reference 则必须重复生成相同 digest。

## 状态判定

本机没有 Core ML、LiteRT v2、Windows ML 或 MindSpore Lite converter/runner。转换报告保留实际命令、版本探测、失败阶段、I/O 影响、许可和再分发结论；失败 artifact 不会写入发布目录。只有真实目标设备通过全部 conformance、golden、fallback、性能和加载测试时才允许标记 `supported`。
