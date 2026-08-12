import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ortNative from '../tooling/web/node_modules/onnxruntime-node/dist/index.js';
import { preprocessCanonical, readPpm, tensorDigest } from './preprocess_contract.mjs';
import { summarizeOpenvinoForConversion } from './openvino_evidence_validation.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const modelPath = resolve(root, 'models/yolov8n.onnx');
const modelBytes = await readFile(modelPath);
const handoffAudit = JSON.parse(await readFile(resolve(root, 'evidence/reports/handoff-model-audit.json'), 'utf8'));
const litertManifestBytes = await readFile(resolve(root, 'evidence/conversions/litert-artifact-manifest.json'));
const litertManifest = JSON.parse(litertManifestBytes);
const litertGoldenBytes = await readFile(resolve(root, 'evidence/reports/litert-golden-report.json'));
const litertGolden = JSON.parse(litertGoldenBytes);
const litertConversionBytes = await readFile(resolve(root, 'evidence/reports/litert-conversion-report.json'));
const coremlManifestBytes = await readFile(resolve(root, 'evidence/conversions/coreml-artifact-manifest.json'));
const coremlManifest = JSON.parse(coremlManifestBytes);
const coremlConversionBytes = await readFile(resolve(root, 'evidence/reports/coreml-conversion-report.json'));
const mindsporeManifestBytes = await readFile(resolve(root, 'evidence/conversions/mindspore-artifact-manifest.json'));
const mindsporeManifest = JSON.parse(mindsporeManifestBytes);
const mindsporeGoldenBytes = await readFile(resolve(root, 'evidence/reports/mindspore-golden-report.json'));
const mindsporeGolden = JSON.parse(mindsporeGoldenBytes);
const mindsporeConversionBytes = await readFile(resolve(root, 'evidence/reports/mindspore-conversion-report.json'));
const mindsporeConversion = JSON.parse(mindsporeConversionBytes);
const reportPath = resolve(root, 'evidence/conversions/conversion-spikes.json');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => JSON.stringify(value, null, 2) + '\n';
const normalizeLog = (value) => value
  .replaceAll(root, '$REPO')
  .replace(/LITE\(\d+,[0-9a-f]+,converter_lite\):\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d+(?:\.\d+)?/g, 'LITE(<pid>,<thread>,converter_lite):<timestamp>');
function command(program, args, options = {}) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8', timeout: 120000, ...options });
  const unavailable = result.error?.code === 'ENOENT' || result.stderr?.includes('Executable not found in $PATH');
  return { command: [program, ...args], exitCode: result.status ?? null, signal: result.signal ?? null, stdout: normalizeLog((result.stdout || '').trim()), stderr: unavailable ? `executable unavailable: ${program}` : normalizeLog((result.stderr || result.error?.message || '').trim()), logNormalization: '仅将仓库绝对路径替换为 $REPO，并将 converter_lite 的时间戳/PID/thread 替换为占位符；其余 stdout/stderr 逐字保留' };
}

function tensorSummary(values) {
  let minimum = Infinity; let maximum = -Infinity; let sum = 0; let finiteCount = 0;
  for (const value of values) if (Number.isFinite(value)) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); sum += value; finiteCount++; }
  return { elementCount: values.length, finiteCount, minimum, maximum, mean: sum / finiteCount, sha256Float32Le: sha256(Buffer.from(values.buffer, values.byteOffset, values.byteLength)) };
}

function androidSpike(legacyOnnxAsRuntimeInputProbe) {
  return {
    platform: 'android',
    format: 'tflite-flatbuffer',
    state: 'host-inference-verified',
    supported: false,
    tool: {
      name: 'Ultralytics LiteRT exporter / litert-torch / ai-edge-litert runtime',
      version: litertManifest.toolchain['ai-edge-litert'],
      converterVersion: litertManifest.toolchain['litert-torch'],
      compilerVersion: litertManifest.toolchain['litert-converter'],
      officialSource: 'https://docs.ultralytics.com/integrations/tflite/',
      converterSource: 'https://github.com/google-ai-edge/litert-torch',
    },
    attempt: {
      command: litertManifest.conversion.command,
      conversionExecuted: true,
      exitCode: 0,
      hostLoadRunExecuted: true,
      goldenPassed: litertGolden.passed,
      report: {
        path: 'evidence/reports/litert-conversion-report.json',
        bytes: litertConversionBytes.length,
        sha256: sha256(litertConversionBytes),
      },
      goldenReport: {
        path: 'evidence/reports/litert-golden-report.json',
        bytes: litertGoldenBytes.length,
        sha256: sha256(litertGoldenBytes),
      },
      legacyOnnxAsRuntimeInputProbe,
    },
    artifact: litertManifest.artifact,
    artifactManifest: {
      path: 'evidence/conversions/litert-artifact-manifest.json',
      bytes: litertManifestBytes.length,
      sha256: sha256(litertManifestBytes),
    },
    sourceCheckpoint: { sha256: handoffAudit.sourceCheckpoint.sha256, auditPath: 'evidence/reports/handoff-model-audit.json' },
    ioChanges: '运行时实际 I/O Shape/layout 与规范 ONNX 相同；名称和索引改变；官方 exporter 将输出 attributes 0..3 除以 640，测试层确定性乘回 640',
    quantization: 'FP32；输入/输出 scale=0、zero-point=0；未执行 FP16/INT8/UINT8 量化',
    nmsResponsibility: 'operator；实际 LiteRT op 列表无 NMS',
    preprocessingResponsibility: 'runtime adapter；模型输入为 NCHW FP32 [0,1]，未融合 letterbox/RGB/normalize',
    failedOperator: null,
    androidRunnerVerified: false,
    license: 'litert-torch/ai-edge-litert Apache-2.0；Ultralytics exporter 与模型 checkpoint metadata 声明 AGPL-3.0',
    redistribution: '仅允许隔离的内部框架验证 evidence；.tflite 位于 ignored .evidence，不进入 Git、RimeCut 产品包或发布目录',
    conclusion: '官方 PyTorch/Ultralytics LiteRT 路径已生成 FP32 TFLite，并由 host ai-edge-litert 2.1.6 完成真实 Load/Run 与冻结 Web golden；尚无 Android arm64 真机 runner，不能标记 supported。',
  };
}

function appleSpike(legacyOnnxProbe) {
  return {
    platform: 'apple',
    format: 'coreml-mlprogram-mlpackage',
    state: 'artifact-spec-verified',
    supported: false,
    tool: {
      name: 'Ultralytics Core ML exporter / coremltools',
      version: coremlManifest.toolchain.coremltools,
      torchVersion: coremlManifest.toolchain.torch,
      ultralyticsVersion: coremlManifest.toolchain.ultralytics,
      officialSource: 'https://docs.ultralytics.com/integrations/coreml/',
      converterSource: 'https://github.com/apple/coremltools',
    },
    attempt: {
      command: coremlManifest.conversion.command,
      conversionExecuted: true,
      exitCode: 0,
      specInspectionExecuted: true,
      macosRuntimeExecuted: false,
      iosRuntimeExecuted: false,
      report: {
        path: 'evidence/reports/coreml-conversion-report.json',
        bytes: coremlConversionBytes.length,
        sha256: sha256(coremlConversionBytes),
      },
      legacyOnnxProbe,
    },
    artifact: {
      location: coremlManifest.artifact.location,
      format: coremlManifest.artifact.format,
      fileCount: coremlManifest.artifact.tree.fileCount,
      totalFileBytes: coremlManifest.artifact.tree.totalFileBytes,
      treeDigest: coremlManifest.artifact.tree.digest,
      trackedByGit: false,
    },
    artifactManifest: {
      path: 'evidence/conversions/coreml-artifact-manifest.json',
      bytes: coremlManifestBytes.length,
      sha256: sha256(coremlManifestBytes),
    },
    sourceCheckpoint: {
      bytes: coremlManifest.source.before.bytes,
      sha256: coremlManifest.source.before.sha256,
      auditPath: 'evidence/reports/handoff-model-audit.json',
    },
    ioChanges: 'ONNX NCHW FLOAT MultiArray 输入变为 RGB Image feature；ML Program 函数张量仍为 NCHW FLOAT32 [1,3,640,640]。输出名称变为 var_911，Shape/layout/dtype 仍为 [1,84,8400]/attributes-first/FLOAT32。',
    computePrecision: coremlManifest.spec.computePrecision,
    minimumDeploymentTarget: coremlManifest.spec.minimumDeploymentTarget,
    preprocessingResponsibility: 'Core ML 图融合 RGB Image feature 到 NCHW 及 1/255 缩放；letterbox resize/padding 仍由 runtime adapter 负责',
    coordinateContract: coremlManifest.spec.coordinates,
    nmsResponsibility: 'operator；实际 ML Program op 列表无 NMS',
    failedOperator: null,
    macosRuntimeVerified: false,
    iosRuntimeVerified: false,
    license: 'coremltools BSD-3-Clause；Ultralytics exporter 与模型 checkpoint metadata 声明 AGPL-3.0',
    redistribution: '仅允许隔离的内部框架验证 evidence；.mlpackage 位于 ignored .evidence，不进入 Git、RimeCut 产品包或发布目录',
    conclusion: '官方 .pt -> Ultralytics YOLO.export(format=coreml) -> coremltools ML Program 路径已生成并检查真实 .mlpackage spec；Linux 无 Core ML runtime，尚未执行 macOS/iOS Load/Run，不能标记 supported。',
  };
}

function mindsporeSpike() {
  const finalRound = mindsporeConversion.rounds.at(-1);
  const failedPaths = finalRound.matrix.filter((item) => item.result === 'failed').map((item) => ({
    command: item.command,
    exitCode: item.exitCode,
    failureSignature: item.failureSignature,
    id: item.id,
    rationale: item.rationale,
  }));
  return {
    platform: 'harmonyos',
    format: 'mindir-lite-ms',
    state: 'host-inference-verified',
    supported: false,
    harmonyOsDeviceVerified: false,
    tool: {
      name: 'MindSpore Lite converter_lite / benchmark / C++ runtime',
      version: mindsporeManifest.toolchain.converterVersion,
      archiveSha256: mindsporeManifest.toolchain.archive.sha256,
      commitId: mindsporeManifest.toolchain.commitId,
      officialSource: mindsporeManifest.toolchain.officialDownloadPage,
    },
    attempt: {
      command: mindsporeManifest.conversion.command,
      conversionExecuted: true,
      exitCode: 0,
      hostBenchmarkExecuted: true,
      hostLoadRunExecuted: true,
      goldenPassed: mindsporeGolden.passed,
      failedPaths,
      replayComparison: mindsporeConversion.comparison,
      report: { path: 'evidence/reports/mindspore-conversion-report.json', bytes: mindsporeConversionBytes.length, sha256: sha256(mindsporeConversionBytes) },
      goldenReport: { path: 'evidence/reports/mindspore-golden-report.json', bytes: mindsporeGoldenBytes.length, sha256: sha256(mindsporeGoldenBytes) },
    },
    artifact: mindsporeManifest.artifact,
    artifactManifest: { path: 'evidence/conversions/mindspore-artifact-manifest.json', bytes: mindsporeManifestBytes.length, sha256: sha256(mindsporeManifestBytes) },
    sourceCheckpoint: { sha256: handoffAudit.sourceCheckpoint.sha256, auditPath: 'evidence/reports/handoff-model-audit.json' },
    derivedOnnx: mindsporeManifest.derivedOnnx,
    ioChanges: mindsporeManifest.differencesFromReferenceOnnx,
    quantization: mindsporeManifest.quantization,
    preprocessingResponsibility: mindsporeManifest.ownership.preprocessing,
    coordinateContract: mindsporeManifest.ownership.coordinates,
    nmsResponsibility: mindsporeManifest.ownership.nms,
    failedOperator: null,
    license: 'MindSpore Apache-2.0；Ultralytics exporter 与模型 checkpoint metadata 声明 AGPL-3.0',
    redistribution: '仅允许隔离的内部框架验证 evidence；.ms 位于 ignored .evidence，不进入 Git、RimeCut 产品包或发布目录',
    conclusion: '结构化改写 DFL Conv 后，官方 converter_lite 2.7.0 生成 MindIR Lite .ms；官方 benchmark 与 C++ runtime 在 Linux host 对五个 fixture 完成真实 Load/Run，冻结 golden 和生产 Rust decode/NMS 全部通过。尚无 HarmonyOS 真机证据，supported=false。',
  };
}

if (process.argv.includes('--coreml-only')) {
  const previous = JSON.parse(await readFile(reportPath, 'utf8'));
  const previousApple = previous.spikes.find((item) => item.platform === 'apple');
  if (!previousApple) throw new Error('conversion-spikes.json 缺少 Apple 条目');
  const legacyOnnxProbe = previousApple.attempt.legacyOnnxProbe ?? {
    acceptedSources: previousApple.attempt.acceptedSources,
    error: previousApple.attempt.error,
    errorType: previousApple.attempt.errorType,
    exitCode: previousApple.attempt.exitCode,
    purpose: '历史 ONNX 直接输入负向证据；不作为本轮 .pt 转换结论',
  };
  const nonAppleBefore = stable(previous.spikes.filter((item) => item.platform !== 'apple'));
  previous.spikes = previous.spikes.map((item) => (
    item.platform === 'apple' ? appleSpike(legacyOnnxProbe) : item
  ));
  const nonAppleAfter = stable(previous.spikes.filter((item) => item.platform !== 'apple'));
  if (nonAppleBefore !== nonAppleAfter) throw new Error('CoreML-only 更新改变了其他平台证据');
  const bytes = Buffer.from(stable(previous));
  await writeFile(reportPath, bytes);
  console.log(JSON.stringify({ mode: 'coreml-only', sha256: sha256(bytes) }));
  process.exit(0);
}

if (process.argv.includes('--litert-only')) {
  const previous = JSON.parse(await readFile(reportPath, 'utf8'));
  const previousAndroid = previous.spikes.find((item) => item.platform === 'android');
  if (!previousAndroid) throw new Error('conversion-spikes.json 缺少 Android 条目');
  const previousProbe = previousAndroid.attempt.legacyOnnxAsRuntimeInputProbe ?? {
    acceptedModelFormat: previousAndroid.attempt.acceptedModelFormat,
    error: previousAndroid.attempt.error,
    exitCode: previousAndroid.attempt.exitCode,
    purpose: '历史负向证据；不作为本轮转换结论',
  };
  const nonAndroidBefore = stable(previous.spikes.filter((item) => item.platform !== 'android'));
  previous.spikes = previous.spikes.map((item) => (
    item.platform === 'android' ? androidSpike(previousProbe) : item
  ));
  const nonAndroidAfter = stable(previous.spikes.filter((item) => item.platform !== 'android'));
  if (nonAndroidBefore !== nonAndroidAfter) throw new Error('LiteRT-only 更新改变了其他平台证据');
  const bytes = Buffer.from(stable(previous));
  await writeFile(reportPath, bytes);
  console.log(JSON.stringify({ mode: 'litert-only', sha256: sha256(bytes) }));
  process.exit(0);
}

if (process.argv.includes('--mindspore-only')) {
  const previous = JSON.parse(await readFile(reportPath, 'utf8'));
  if (!previous.spikes.some((item) => item.platform === 'harmonyos')) throw new Error('conversion-spikes.json 缺少 HarmonyOS 条目');
  const nonMindsporeBefore = stable(previous.spikes.filter((item) => item.platform !== 'harmonyos'));
  previous.spikes = previous.spikes.map((item) => (
    item.platform === 'harmonyos' ? mindsporeSpike() : item
  ));
  const nonMindsporeAfter = stable(previous.spikes.filter((item) => item.platform !== 'harmonyos'));
  if (nonMindsporeBefore !== nonMindsporeAfter) throw new Error('MindSpore-only 更新改变了其他平台证据');
  const bytes = Buffer.from(stable(previous));
  await writeFile(reportPath, bytes);
  console.log(JSON.stringify({ mode: 'mindspore-only', sha256: sha256(bytes) }));
  process.exit(0);
}

if (process.argv.includes('--openvino-only')) {
  const previous = JSON.parse(await readFile(reportPath, 'utf8'));
  const manifest = JSON.parse(await readFile(resolve(root, 'evidence/conversions/openvino-ep-manifest.json'), 'utf8'));
  const openvinoReport = JSON.parse(await readFile(resolve(root, 'evidence/reports/openvino-ep-report.json'), 'utf8'));
  if (!previous.spikes.some((item) => item.platform === 'linux-x86_64-openvino')) throw new Error('conversion-spikes.json 缺少 Linux x86_64 OpenVINO 条目');
  const otherBefore = stable(previous.spikes.filter((item) => item.platform !== 'linux-x86_64-openvino'));
  previous.spikes = previous.spikes.map((item) => item.platform === 'linux-x86_64-openvino' ? summarizeOpenvinoForConversion(manifest, openvinoReport) : item);
  const otherAfter = stable(previous.spikes.filter((item) => item.platform !== 'linux-x86_64-openvino'));
  if (otherBefore !== otherAfter) throw new Error('OpenVINO-only 更新改变了其他平台证据');
  const bytes = Buffer.from(stable(previous));
  await writeFile(reportPath, bytes);
  console.log(JSON.stringify({ mode: 'openvino-only', sha256: sha256(bytes) }));
  process.exit(0);
}

const pythonProbe = command('.evidence/python-tools/bin/python', ['evidence/scripts/probe_python_converters.py', 'models/yolov8n.onnx']);
if (pythonProbe.exitCode !== 0) throw new Error(`Python converter probe failed: ${pythonProbe.stderr}`);
const pythonResult = JSON.parse(pythonProbe.stdout.split('\n').at(-1));
const windowsMl = command('dotnet', ['--info']);
const fixtureManifest = JSON.parse(await readFile(resolve(root, 'evidence/fixtures/manifest.json'), 'utf8'));
const inferenceFixture = fixtureManifest.images.find((item) => item.id === 'single-target');
const image = readPpm(await readFile(resolve(root, inferenceFixture.path)));
const canonical = preprocessCanonical(image);
const supportedBackends = ortNative.listSupportedBackends();
async function linuxProvider(provider) {
  const attempt = { command: ['onnxruntime-node@1.24.3', 'InferenceSession.create+run', `--execution-provider=${provider}`], runtimeVersion: ortNative.env.versions.node, requestedProvider: provider, configuredProviders: [provider], bundledBackends: supportedBackends, fixtureId: inferenceFixture.id, canonicalInputSha256Float32Le: tensorDigest(canonical.tensor) };
  try {
    const session = await ortNative.InferenceSession.create(modelPath, { executionProviders: [provider] });
    const outputs = await session.run({ images: new ortNative.Tensor('float32', canonical.tensor, [1, 3, 640, 640]) });
    const raw = outputs.output0.data;
    Object.assign(attempt, { exitCode: 0, inputNames: session.inputNames, outputNames: session.outputNames, inputMetadata: session.inputMetadata, outputMetadata: session.outputMetadata, output: tensorSummary(raw), inferenceExecuted: true });
    await session.release();
  } catch (error) {
    Object.assign(attempt, { exitCode: 1, inferenceExecuted: false, error: String(error) });
  }
  return attempt;
}
const linuxProviders = [];
for (const provider of ['cpu', 'openvino', 'cuda', 'tensorrt']) linuxProviders.push(await linuxProvider(provider));
const report = {
  schemaVersion: 1,
  source: { commit: 'eacbcf00dfc2fba941b494e2955e87fffd707382', modelPath: 'models/yolov8n.onnx', modelSha256: sha256(modelBytes) },
  usageScope: { intendedUse: 'internal-onnx-base-framework-validation-only', authorizationEvaluation: 'out-of-scope', artifactHandling: 'isolated-test-evidence-only', productPackaging: 'excluded' },
  spikes: [
    appleSpike({
      acceptedSources: pythonResult.coremltools.acceptedSources,
      error: pythonResult.coremltools.attempt.error,
      errorType: pythonResult.coremltools.attempt.errorType,
      exitCode: pythonResult.coremltools.attempt.exitCode,
      purpose: '历史 ONNX 直接输入负向证据；不作为本轮 .pt 转换结论',
    }),
    androidSpike({
      acceptedModelFormat: pythonResult.litert.acceptedModelFormat,
      error: pythonResult.litert.attempt.error,
      exitCode: pythonResult.litert.attempt.exitCode,
      purpose: '历史负向证据；不作为本轮转换结论',
    }),
    { platform: 'windows-x86_64-and-arm64', format: 'onnx', state: 'blocked', conversion: '无格式转换：Windows ML 随 Windows App SDK 提供 ONNX Runtime API，原 ONNX 应由 Microsoft.ML.OnnxRuntime.InferenceSession 实际加载并执行固定输入', tool: { name: 'Windows ML / Windows App SDK', version: '1.8 target; unavailable on Linux host', officialSource: 'https://learn.microsoft.com/windows/ai/new-windows-ml/run-onnx-models' }, attempt: { ...windowsMl, requiredRunnerCommand: 'dotnet run --configuration Release --framework net8.0-windows10.0.26100.0 -- models/yolov8n.onnx single-target.nchw-f32le.bin', requiredApi: 'Microsoft.ML.OnnxRuntime.InferenceSession(modelPath, sessionOptions)' }, artifact: { path: 'models/yolov8n.onnx', sha256: sha256(modelBytes) }, ioChanges: 'none', quantization: 'none', nmsResponsibility: 'operator', failedOperator: 'Windows runner/toolchain discovery before Windows ML model load', license: 'Windows ML follows Windows App SDK terms；模型 checkpoint/ONNX metadata 声明 AGPL-3.0', redistribution: '仅允许隔离的内部框架验证 evidence，不进入 RimeCut 产品发布目录', conclusion: '没有 Windows x64 或 ARM64 runner；未实际加载，两个架构均保持 blocked，不能由 Linux ORT 推断。' },
    mindsporeSpike(),
    ...linuxProviders.map((attempt) => ({ platform: `linux-x86_64-${attempt.requestedProvider}`, format: 'onnx', state: attempt.exitCode === 0 ? 'inference-verified' : 'blocked', tool: { name: 'onnxruntime-node', version: '1.24.3' }, attempt, artifact: { path: 'models/yolov8n.onnx', sha256: sha256(modelBytes) }, ioChanges: 'none', quantization: 'none', nmsResponsibility: 'operator', failedOperator: attempt.exitCode === 0 ? null : 'provider/session initialization before graph execution', license: 'ONNX Runtime MIT；ONNX metadata 声明 AGPL-3.0', redistribution: '仅允许隔离的内部框架验证 evidence，不进入 RimeCut 产品发布目录', conclusion: attempt.exitCode === 0 ? `${attempt.requestedProvider} provider 已用固定 canonical 输入完成真实 inference；仅为本机 build-verified 证据。` : `${attempt.requestedProvider} provider 未能进入固定输入 inference，保持 blocked。` })),
  ],
};
await writeFile(reportPath, stable(report));
console.log(sha256(Buffer.from(stable(report))));
