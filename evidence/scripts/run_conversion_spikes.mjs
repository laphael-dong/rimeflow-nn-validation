import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ortNative from '../tooling/web/node_modules/onnxruntime-node/dist/index.js';
import { preprocessCanonical, readPpm, tensorDigest } from './preprocess_contract.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const modelPath = resolve(root, 'models/yolov8n.onnx');
const modelBytes = await readFile(modelPath);
const handoffAudit = JSON.parse(await readFile(resolve(root, 'evidence/reports/handoff-model-audit.json'), 'utf8'));
const litertManifestBytes = await readFile(resolve(root, 'evidence/conversions/litert-artifact-manifest.json'));
const litertManifest = JSON.parse(litertManifestBytes);
const litertGoldenBytes = await readFile(resolve(root, 'evidence/reports/litert-golden-report.json'));
const litertGolden = JSON.parse(litertGoldenBytes);
const litertConversionBytes = await readFile(resolve(root, 'evidence/reports/litert-conversion-report.json'));
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

const pythonProbe = command('.evidence/python-tools/bin/python', ['evidence/scripts/probe_python_converters.py', 'models/yolov8n.onnx']);
if (pythonProbe.exitCode !== 0) throw new Error(`Python converter probe failed: ${pythonProbe.stderr}`);
const pythonResult = JSON.parse(pythonProbe.stdout.split('\n').at(-1));
const windowsMl = command('dotnet', ['--info']);
const mindsporeRoot = resolve(root, '.evidence/mindspore/mindspore-lite-2.7.0-linux-x64');
const mindsporeBinary = resolve(mindsporeRoot, 'tools/converter/converter/converter_lite');
const mindsporeOutput = resolve(root, '.evidence/mindspore/yolov8n');
const mindspore = command(mindsporeBinary, ['--fmk=ONNX', '--modelFile=models/yolov8n.onnx', `--outputFile=${mindsporeOutput}`, '--optimize=general'], { env: { ...process.env, LD_LIBRARY_PATH: `${resolve(mindsporeRoot, 'tools/converter/lib')}:${resolve(mindsporeRoot, 'runtime/lib')}` } });
mindspore.command[0] = '.evidence/mindspore/mindspore-lite-2.7.0-linux-x64/tools/converter/converter/converter_lite';
mindspore.command[3] = '--outputFile=.evidence/mindspore/yolov8n';
const mindsporeArtifactPath = `${mindsporeOutput}.ms`;
const mindsporeArtifact = await stat(mindsporeArtifactPath).then(async (info) => ({ path: '.evidence/mindspore/yolov8n.ms', bytes: info.size, sha256: sha256(await readFile(mindsporeArtifactPath)) }), () => null);
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
    { platform: 'apple', format: 'coreml', state: 'blocked', tool: { name: 'coremltools', version: pythonResult.coremltools.version, officialSource: 'https://apple.github.io/coremltools/docs-guides/source/convert-learning-models.html' }, attempt: { command: pythonProbe.command, exitCode: pythonResult.coremltools.attempt.exitCode, stdout: pythonProbe.stdout, stderr: pythonProbe.stderr, apiSignature: pythonResult.coremltools.convertSignature, acceptedSources: pythonResult.coremltools.acceptedSources, errorType: pythonResult.coremltools.attempt.errorType, error: pythonResult.coremltools.attempt.error }, artifact: null, sourceCheckpoint: { sha256: handoffAudit.sourceCheckpoint.sha256, auditPath: 'evidence/reports/handoff-model-audit.json' }, ioChanges: '未生成 artifact；原 ONNX I/O 保持 [1,3,640,640] -> [1,84,8400]', quantization: '未执行', nmsResponsibility: 'operator', failedOperator: 'delegated Apple conversion/runner evidence not returned', license: 'coremltools BSD；模型 checkpoint/ONNX metadata 声明 AGPL-3.0', redistribution: '仅允许隔离的内部框架验证 evidence，不进入 RimeCut 产品发布目录', conclusion: 'coremltools 9.0 官方入口不直接接受 ONNX；同源 .pt 已由独立审计确认，但本机不是 macOS，Core ML 转换与真实 runner spike 已委托外部负责人，证据回传前保持 blocked。' },
    androidSpike({
      acceptedModelFormat: pythonResult.litert.acceptedModelFormat,
      error: pythonResult.litert.attempt.error,
      exitCode: pythonResult.litert.attempt.exitCode,
      purpose: '历史负向证据；不作为本轮转换结论',
    }),
    { platform: 'windows-x86_64-and-arm64', format: 'onnx', state: 'blocked', conversion: '无格式转换：Windows ML 随 Windows App SDK 提供 ONNX Runtime API，原 ONNX 应由 Microsoft.ML.OnnxRuntime.InferenceSession 实际加载并执行固定输入', tool: { name: 'Windows ML / Windows App SDK', version: '1.8 target; unavailable on Linux host', officialSource: 'https://learn.microsoft.com/windows/ai/new-windows-ml/run-onnx-models' }, attempt: { ...windowsMl, requiredRunnerCommand: 'dotnet run --configuration Release --framework net8.0-windows10.0.26100.0 -- models/yolov8n.onnx single-target.nchw-f32le.bin', requiredApi: 'Microsoft.ML.OnnxRuntime.InferenceSession(modelPath, sessionOptions)' }, artifact: { path: 'models/yolov8n.onnx', sha256: sha256(modelBytes) }, ioChanges: 'none', quantization: 'none', nmsResponsibility: 'operator', failedOperator: 'Windows runner/toolchain discovery before Windows ML model load', license: 'Windows ML follows Windows App SDK terms；模型 checkpoint/ONNX metadata 声明 AGPL-3.0', redistribution: '仅允许隔离的内部框架验证 evidence，不进入 RimeCut 产品发布目录', conclusion: '没有 Windows x64 或 ARM64 runner；未实际加载，两个架构均保持 blocked，不能由 Linux ORT 推断。' },
    { platform: 'harmonyos', format: 'mindir/ms', state: mindspore.exitCode === 0 && mindsporeArtifact ? 'converted-not-load-verified' : 'blocked', tool: { name: 'MindSpore Lite converter_lite', version: '2.7.0', archiveSha256: '8bb1097100c9fec12675670ba2d4264a2cd6da3a9be093eb56631d00fc0c455b', officialSource: 'https://www.mindspore.cn/lite/docs/en/r2.7.0/use/downloads.html' }, attempt: mindspore, artifact: mindsporeArtifact, ioChanges: mindsporeArtifact ? '需要 HarmonyOS runner 检查实际 I/O；原始 I/O 为 [1,3,640,640] -> [1,84,8400]' : '转换失败，无 artifact；原始 I/O 为 [1,3,640,640] -> [1,84,8400]', quantization: '命令未请求量化，FP32', nmsResponsibility: 'operator', failedOperator: mindspore.exitCode === 0 ? null : '/model.22/dfl/conv/Conv (Conv2DFusion infer-shape/graph pass)', license: 'MindSpore Apache-2.0；模型 checkpoint/ONNX metadata 声明 AGPL-3.0', redistribution: '仅允许隔离的内部框架验证 evidence，不进入 RimeCut 产品发布目录', conclusion: mindspore.exitCode === 0 ? '已进入真实 ONNX 转换并生成本地 artifact；缺 HarmonyOS runner，保持 test-evidence-only。' : '官方 converter_lite 2.7.0 已进入 ONNX parse/graph optimization，因 Conv2DFusion infer-shape 失败；缺 HarmonyOS runner。' },
    ...linuxProviders.map((attempt) => ({ platform: `linux-x86_64-${attempt.requestedProvider}`, format: 'onnx', state: attempt.exitCode === 0 ? 'inference-verified' : 'blocked', tool: { name: 'onnxruntime-node', version: '1.24.3' }, attempt, artifact: { path: 'models/yolov8n.onnx', sha256: sha256(modelBytes) }, ioChanges: 'none', quantization: 'none', nmsResponsibility: 'operator', failedOperator: attempt.exitCode === 0 ? null : 'provider/session initialization before graph execution', license: 'ONNX Runtime MIT；ONNX metadata 声明 AGPL-3.0', redistribution: '仅允许隔离的内部框架验证 evidence，不进入 RimeCut 产品发布目录', conclusion: attempt.exitCode === 0 ? `${attempt.requestedProvider} provider 已用固定 canonical 输入完成真实 inference；仅为本机 build-verified 证据。` : `${attempt.requestedProvider} provider 未能进入固定输入 inference，保持 blocked。` })),
  ],
};
await writeFile(reportPath, stable(report));
console.log(sha256(Buffer.from(stable(report))));
