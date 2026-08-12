import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => `${JSON.stringify(value, null, 2)}\n`;
const conversionPath = 'evidence/conversions/conversion-spikes.json';
const replayPath = 'evidence/replay/task1-replay.json';
const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
const conversion = await readJson(conversionPath);
const replay = await readJson(replayPath);
const cudaReport = await readJson('evidence/reports/cuda-ep-spike-report.json');
const tensorrtReport = await readJson('evidence/reports/tensorrt-ep-report.json');

const closure = {
  allRequiredSpikesRecorded: true,
  technicalSpikeClosure: true,
  publicationVerified: false,
  task14Complete: false,
  openspecTask1_4Checked: false,
  allPlatformsSupported: false,
};
const providerByPlatform = {
  apple: 'coreml',
  android: 'litert-v2',
  windows: 'windows-ml',
  harmonyos: 'mindspore-lite',
  'linux-x86_64-cpu': 'cpuexecutionprovider',
  'linux-x86_64-openvino': 'openvinoexecutionprovider',
  'linux-x86_64-cuda': 'cudaexecutionprovider',
  'linux-x86_64-tensorrt': 'tensorrtexecutionprovider',
};
const closureByPlatform = {
  apple: { runtimeExecuted: false, hostInferenceVerified: false, goldenExecuted: false, blocker: '缺少 macOS/iOS Core ML Load/Run 真实 runner。' },
  android: { runtimeExecuted: true, hostInferenceVerified: true, goldenExecuted: true, blocker: '缺少 Android arm64 LiteRT v2 真机 runner。' },
  windows: { runtimeExecuted: false, hostInferenceVerified: false, goldenExecuted: false, blocker: 'x64/ARM64 runner contract 与静态编译已验证；缺少真实 Windows Load/Run。' },
  harmonyos: { runtimeExecuted: true, hostInferenceVerified: true, goldenExecuted: true, blocker: 'Linux host conversion/Load/Run/golden 已验证；缺少 HarmonyOS 真机。' },
  'linux-x86_64-cpu': { runtimeExecuted: true, hostInferenceVerified: true, goldenExecuted: false, blocker: '仅作为 canonical ONNX host inference 基线；adapter、性能和包加载属于后续任务。' },
  'linux-x86_64-openvino': { runtimeExecuted: true, hostInferenceVerified: true, goldenExecuted: true, blocker: 'OpenVINO profile 已证明真实执行；adapter、性能、fallback 与包加载属于后续任务。' },
  'linux-x86_64-cuda': { runtimeExecuted: cudaReport.runtimeExecuted, hostInferenceVerified: cudaReport.hostInferenceVerified, goldenExecuted: cudaReport.goldenExecuted, blocker: cudaReport.failure.message },
  'linux-x86_64-tensorrt': { runtimeExecuted: tensorrtReport.runtimeExecuted, hostInferenceVerified: tensorrtReport.hostInferenceVerified, goldenExecuted: tensorrtReport.goldenExecuted, blocker: tensorrtReport.failureReason },
};

conversion.spikes = conversion.spikes.map((item) => ({
  ...item,
  provider: providerByPlatform[item.platform],
  spikeClosure: true,
  supported: false,
  task14Complete: false,
  openspecTask1_4Checked: false,
  ...closureByPlatform[item.platform],
}));
conversion.closure = closure;
await writeFile(resolve(root, conversionPath), stable(conversion));

const reportByPlatform = {
  apple: 'evidence/reports/coreml-conversion-report.json',
  android: 'evidence/reports/litert-golden-report.json',
  windows: 'evidence/reports/windows-ml-spike-report.json',
  harmonyos: 'evidence/reports/mindspore-golden-report.json',
  'linux-x86_64-cpu': conversionPath,
  'linux-x86_64-openvino': 'evidence/reports/openvino-ep-report.json',
  'linux-x86_64-cuda': 'evidence/reports/cuda-ep-spike-report.json',
  'linux-x86_64-tensorrt': 'evidence/reports/tensorrt-ep-report.json',
};
const aggregate = {
  schemaVersion: 1,
  task: 'OpenSpec rimeflow-backend-contract 1.4 technical spike aggregate',
  source: {
    aggregationBaseline: '7eba039ef1c55408216f0f54d543f0fdcbf1693b',
    providerCommits: {
      openvino: '7eba039ef1c55408216f0f54d543f0fdcbf1693b',
      cuda: 'f0a4e7fc9e412d040c2a1bbc6db75def39d1acd7',
      tensorrt: '228956ea9992baa279bf71a57164b383e4823878',
    },
    canonicalOnnx: { path: 'models/yolov8n.onnx', bytes: modelBytes.length, sha256: sha256(modelBytes) },
  },
  closure,
  publication: { pushed: false, upstreamConfigured: false, remoteRefVerified: false, blocker: '唯一聚合提交尚未推送，远端 ref 尚未验证；OpenSpec 任务 1.4 必须保持未勾选。' },
  spikes: conversion.spikes.map((item) => ({
    platform: item.platform,
    provider: item.provider,
    state: item.state,
    spikeClosure: item.spikeClosure,
    supported: item.supported,
    runtimeExecuted: item.runtimeExecuted,
    hostInferenceVerified: item.hostInferenceVerified,
    goldenExecuted: item.goldenExecuted,
    blocker: item.blocker,
    report: reportByPlatform[item.platform],
  })),
  remainingScope: ['远端 push/ref 可达性与 publication closure', '真实目标 runner 的 supported 闭环', 'adapter、性能、fallback、包加载与后续 OpenSpec 任务'],
};
await writeFile(resolve(root, 'evidence/conversions/task1-4-aggregate.json'), stable(aggregate));

// Golden manifest hashes the completed provider union. Replay hashes are computed only after it is stable.
await import('./finalize_manifest.mjs');

const providerSteps = aggregate.spikes.map((item) => {
  const existing = replay.steps.find((step) => step.platform === item.platform && step.provider === item.provider)
    ?? replay.steps.find((step) => item.platform === 'linux-x86_64-openvino' && step.id === 'linux-x86_64-openvino-provider-host-golden');
  const base = existing ?? { id: `spike-${item.platform}-${item.provider}`, command: `validate ${item.report}`, rounds: [] };
  return { ...base, platform: item.platform, provider: item.provider, state: item.state, spikeClosure: true, supported: false, blockedReason: item.blocker };
});
const nonProviderSteps = replay.steps.filter((step) => !step.provider && step.id !== 'linux-x86_64-openvino-provider-host-golden' && step.id !== 'delegated-platform-spikes' && step.id !== 'publication-and-platform-closure');
const cudaStep = providerSteps.find((step) => step.provider === 'cudaexecutionprovider');
cudaStep.command = 'node evidence/scripts/replay_cuda_ep.mjs';
cudaStep.rounds = [];
cudaStep.repeatComparison = { recordedBlockedConclusion: true, dedicatedReplayReport: 'evidence/reports/cuda-ep-replay-report.json' };
const tensorrtStep = providerSteps.find((step) => step.provider === 'tensorrtexecutionprovider');
tensorrtStep.command = 'node evidence/scripts/replay_tensorrt_ep.mjs .evidence/tensorrt/replay';
tensorrtStep.rounds = [{ run: 1, exitCode: 2, failureStage: tensorrtReport.failureStage, reportSha256: sha256(await readFile(resolve(root, 'evidence/reports/tensorrt-ep-report.json'))) }];
tensorrtStep.repeatComparison = { recordedBlockedConclusion: true, dedicatedReplayRequiredOnTargetRunner: true };
replay.steps = [...nonProviderSteps, ...providerSteps, {
  id: 'publication-and-platform-closure',
  command: 'push aggregate commit, verify remote ref, then close supported platform follow-up work',
  executed: false,
  blockedReason: aggregate.publication.blocker,
  rounds: [],
  repeatComparison: null,
}];
replay.closure = closure;
replay.task1_4Complete = false;
const replayOutputPaths = [...new Set([
  ...replay.outputs.map((item) => item.path).filter((path) => path !== 'evidence/reports/cuda-ep-replay-report.json'),
  'evidence/conversions/cuda-ep-spike-manifest.json',
  'evidence/conversions/task1-4-aggregate.json',
  'evidence/reports/cuda-ep-spike-report.json',
  'evidence/reports/tensorrt-ep-report.json',
  'evidence/tensorrt/contract.json',
])].sort();
replay.outputs = await Promise.all(replayOutputPaths.map(async (path) => {
  const bytes = await readFile(resolve(root, path));
  return { path, bytes: bytes.length, sha256: sha256(bytes) };
}));
await writeFile(resolve(root, replayPath), stable(replay));
console.log(JSON.stringify({ aggregate: 'evidence/conversions/task1-4-aggregate.json', spikes: aggregate.spikes.length, technicalSpikeClosure: true, publicationVerified: false }));
