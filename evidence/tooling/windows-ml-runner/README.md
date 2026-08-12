# Windows ML 兼容性 spike runner

本目录只用于任务 1.4 的 Windows ML 兼容性验证，不是正式 base adapter。runner 直接加载仓库中的规范 ONNX，不复制、不转换、不改写模型，也不包含 Windows 专用 decode/NMS。

技术路径固定为 `Microsoft.WindowsAppSDK.ML@2.1.74` 的 self-contained Windows ML 部署：先调用 `Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog.RegisterCertifiedAsync()` 注册当前机器已有的认证 EP，再使用该包提供的 `Microsoft.ML.OnnxRuntime.InferenceSession`。两个直接 NuGet 引用都使用精确闭区间 `[2.1.74]`，项目没有引用普通 `Microsoft.ML.OnnxRuntime` NuGet 包。SDK 固定为 `8.0.423`，运行命令通过 `--fx-version 8.0.29` 固定 .NET runtime patch。

## Linux 静态编译重放

静态编译证据必须从独立的空 `mkdtemp` workspace 开始；脚本只复制 `Program.cs`、`RunnerSupport.cs`、project、`global.json` 和 `packages.lock.json`，确认不存在 `bin`、`obj` 或 `project.assets.json` 后再执行。lock 同时固定两个 RID，因此 restore 不传单个 `--runtime`；随后为 x64、ARM64 分别执行 `Compile` target。该 target 真实运行 `CoreCompile` 和 Roslyn `Csc` 并生成可哈希的 RID 专属 `WindowsMlSpike.dll`，不会预放 `app.manifest`、替换 `ManifestTool` 或跳过 compiler：

```bash
node evidence/scripts/replay_windows_ml_static_compile.mjs --report evidence/reports/windows-ml-static-compile-report.json
```

完整的逐 RID restore/compile 命令、时间、退出码、原始日志摘录和临时 DLL SHA 记录在该报告中。workspace 在取证后删除，DLL、binlog、`bin` 和 `obj` 均不保留。这个结果只表示 `staticCompileVerified=true`；由于 Linux 无法运行 Windows manifest/PRI 工具，也没有 Windows runner，`buildVerified` 和 `runtimeExecuted` 仍为 `false`。

## 失败与产物生命周期

failure report 保留失败发生时已经完成的真实阶段状态，不会把 catalog、session 或 inference 状态统一清零。`failureStage` 区分参数、平台、模型、输入、catalog、设备、session、metadata、inference、输出、provider、module、dependency、SDK/runtime 和 artifact publication。

raw output 仅在所有 runtime introspection 与 identity 检查通过后写入目标文件同目录的 staging 文件。success report 同样先写 staging，然后 output/report 作为可回滚的一对执行原子替换；任一步失败都恢复已有文件，且不会发布 success report。failure report 不替换已有 report，而是写唯一 sidecar。ORT profile 的返回路径与唯一 prefix 在最外层 `finally` 清理，覆盖 session、inference、introspection 和 publication 异常。

## Windows x64

在仓库根目录使用 PowerShell：

```powershell
bun run evidence/scripts/export_web_reference_tensors.mjs .evidence/windows-ml/web-reference
Push-Location evidence/tooling/windows-ml-runner
dotnet restore WindowsMlSpike.csproj --locked-mode
dotnet publish WindowsMlSpike.csproj --configuration Release --runtime win-x64 --no-restore --self-contained false --output ../../../.evidence/windows-ml/build/win-x64
Pop-Location
dotnet --fx-version 8.0.29 .evidence/windows-ml/build/win-x64/WindowsMlSpike.dll --model models/yolov8n.onnx --input .evidence/windows-ml/web-reference/single-target/input.f32le --output .evidence/windows-ml/outputs/win-x64-single-target.f32le --report .evidence/windows-ml/reports/win-x64-single-target.json --expected-model-sha256 9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad
cargo run --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml -- .evidence/windows-ml/outputs/win-x64-single-target.f32le 768 900 .evidence/windows-ml/decoded/win-x64-single-target.json
```

## Windows ARM64

必须在真实 Windows ARM64 runner 上执行；x64、ARM64EC、模拟器和交叉 publish 均不能替代：

```powershell
bun run evidence/scripts/export_web_reference_tensors.mjs .evidence/windows-ml/web-reference
Push-Location evidence/tooling/windows-ml-runner
dotnet restore WindowsMlSpike.csproj --locked-mode
dotnet publish WindowsMlSpike.csproj --configuration Release --runtime win-arm64 --no-restore --self-contained false --output ../../../.evidence/windows-ml/build/win-arm64
Pop-Location
dotnet --fx-version 8.0.29 .evidence/windows-ml/build/win-arm64/WindowsMlSpike.dll --model models/yolov8n.onnx --input .evidence/windows-ml/web-reference/single-target/input.f32le --output .evidence/windows-ml/outputs/win-arm64-single-target.f32le --report .evidence/windows-ml/reports/win-arm64-single-target.json --expected-model-sha256 9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad
cargo run --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml -- .evidence/windows-ml/outputs/win-arm64-single-target.f32le 768 900 .evidence/windows-ml/decoded/win-arm64-single-target.json
```

所有 `.evidence/windows-ml/` 输入、输出、profile、report 和 publish 目录均为临时验证文件，受 Git ignore 约束，不得复制到 RimeCut 或任何产品发布目录。runner 会删除 ORT profile 临时文件；raw output 固定写为 little-endian FP32，后续命令直接调用 `evidence/tooling/raw-golden`，从而复用生产 `src/postprocess.rs`。
