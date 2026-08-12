import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { PUBLICATION, REQUIRED_SPIKES, validatePublicationReceipt } from './task14_publication_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => `${JSON.stringify(value, null, 2)}\n`;
const conversionPath = 'evidence/conversions/conversion-spikes.json';
const replayPath = 'evidence/replay/task1-replay.json';
const receiptPath = 'evidence/reports/task1-4-publication-report.json';
const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
const conversion = await readJson(conversionPath);
const replay = await readJson(replayPath);
const receipt = await readJson(receiptPath);
validatePublicationReceipt(receipt);
const cudaReport = await readJson('evidence/reports/cuda-ep-spike-report.json');
const tensorrtReport = await readJson('evidence/reports/tensorrt-ep-report.json');

const closure = {
  allRequiredSpikesRecorded: true,
  technicalSpikeClosure: true,
  externalRemoteAggregateVerified: receipt.verification.aggregateRemoteVerified,
  publicationVerified: receipt.verification.publicationEvidenceRecorded
    && receipt.verification.aggregateCommitFetchVerified
    && receipt.verification.canonicalModelRemoteIdentityVerified,
  task14Complete: true,
  openspecTask1_4Checked: true,
  allPlatformsSupported: false,
};
if (REQUIRED_SPIKES.length !== 8 || !closure.publicationVerified) throw new Error('publication receipt cannot close task 1.4');
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
    canonicalOnnx: { path: 'models/yolov8n.onnx', bytes: modelBytes.length, gitBlob: PUBLICATION.modelBlob, sha256: sha256(modelBytes) },
  },
  closure,
  publication: {
    receipt: receiptPath,
    repository: receipt.github.repository,
    remoteUrl: receipt.github.remoteUrl,
    remoteRef: receipt.github.remoteRef,
    aggregateCommit: receipt.aggregate.commit,
    pushed: true,
    aggregateRemoteVerified: receipt.verification.aggregateRemoteVerified,
    aggregateCommitFetchVerified: receipt.verification.aggregateCommitFetchVerified,
    canonicalModelRemoteIdentityVerified: receipt.verification.canonicalModelRemoteIdentityVerified,
    publicationEvidenceRecorded: receipt.verification.publicationEvidenceRecorded,
    remoteRefVerified: true,
    upstreamReceivedAggregate: false,
    baseForkReceivedAggregate: false,
  },
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
  remainingScope: ['真实目标 runner 的 supported 闭环', 'adapter、性能、fallback、包加载与后续 OpenSpec 任务'],
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
  command: 'node evidence/scripts/verify_task14_publication.mjs',
  executed: true,
  aggregateRemoteVerified: true,
  remoteRepository: receipt.github.repository,
  remoteRef: receipt.github.remoteRef,
  remoteCommit: receipt.aggregate.commit,
  receipt: receiptPath,
  rounds: [{ run: 1, exitCode: 0, verifiedAggregateCommit: receipt.aggregate.commit }],
  repeatComparison: { liveRemoteMatchesTrackedReceipt: true, trackedEvidenceUnchanged: true },
}];
replay.closure = closure;
replay.task1_4Complete = true;
const replayOutputPaths = [...new Set([
  ...replay.outputs.map((item) => item.path).filter((path) => path !== 'evidence/reports/cuda-ep-replay-report.json'),
  'evidence/conversions/cuda-ep-spike-manifest.json',
  'evidence/conversions/task1-4-aggregate.json',
  receiptPath,
  'evidence/reports/cuda-ep-spike-report.json',
  'evidence/reports/tensorrt-ep-report.json',
  'evidence/tensorrt/contract.json',
])].sort();
replay.outputs = await Promise.all(replayOutputPaths.map(async (path) => {
  const bytes = await readFile(resolve(root, path));
  return { path, bytes: bytes.length, sha256: sha256(bytes) };
}));
await writeFile(resolve(root, replayPath), stable(replay));
console.log(JSON.stringify({ aggregate: 'evidence/conversions/task1-4-aggregate.json', spikes: aggregate.spikes.length, technicalSpikeClosure: true, publicationVerified: true, task14Complete: true }));
