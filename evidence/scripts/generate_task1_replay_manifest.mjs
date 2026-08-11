import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { blockedStep, runRepeatedStep, runnerIdentity, sha256, toolVersion } from './task1_replay_execution.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const outputPath = resolve(root, 'evidence/replay/task1-replay.json');
const git = (args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const repositoryHeadAtReplay = git(['rev-parse', 'HEAD']);
const outputs = [
  'evidence/model/model-contract.json',
  'evidence/fixtures/manifest.json',
  'evidence/golden/coverage-matrix.json',
  'evidence/golden/manifest.json',
  'evidence/golden/web-reference.json',
  'evidence/reports/preprocess-conformance.json',
  'evidence/reports/handoff-model-audit.json',
  'evidence/reports/model-provenance.json',
  'evidence/conversions/conversion-spikes.json',
];
const commonInputs = ['models/yolov8n.onnx', 'evidence/fixtures/manifest.json'];
const steps = [];
steps.push(await runRepeatedStep({
  root,
  id: 'contract-fixture-golden',
  command: 'bun run evidence/scripts/generate_all.mjs',
  executable: 'bun',
  args: ['run', 'evidence/scripts/generate_all.mjs'],
  inputPaths: commonInputs,
  outputPaths: outputs,
}));
steps.push(await runRepeatedStep({
  root,
  id: 'production-raw-golden',
  command: 'cargo test --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml',
  executable: 'cargo',
  args: ['test', '--offline', '--manifest-path', 'evidence/tooling/raw-golden/Cargo.toml'],
  inputPaths: ['evidence/fixtures/raw/overlap-nms.json', 'src/postprocess.rs'],
  outputPaths: [],
}));
steps.push(await runRepeatedStep({
  root,
  id: 'conversion-report-regeneration',
  command: 'node evidence/scripts/run_conversion_spikes.mjs',
  executable: 'node',
  args: ['evidence/scripts/run_conversion_spikes.mjs'],
  inputPaths: ['models/yolov8n.onnx', 'evidence/reports/handoff-model-audit.json', 'evidence/tooling/requirements.lock'],
  outputPaths: ['evidence/conversions/conversion-spikes.json'],
}));
steps.push(blockedStep(
  'delegated-platform-spikes',
  'Core ML/LiteRT/Windows ML/MindSpore/Linux accelerated provider platform commands',
  '源 .pt 与 ONNX 同源性已验证；模型用途限定为内部框架验证。外部负责人尚未回传 Core ML、LiteRT、Windows ML、MindSpore Lite 与加速 Linux provider 的完整 spike evidence。',
));
const artifacts = [];
for (const path of outputs) {
  const bytes = await readFile(resolve(root, path));
  artifacts.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
}
const manifest = {
  schemaVersion: 2,
  repository: 'rimeflow-yolov8n',
  repositoryHeadAtReplay: { commit: repositoryHeadAtReplay, kind: 'evidence-input-head', finalEvidenceCommitRecordedByGit: true },
  inputs: {
    operatorSourceCommit: 'eacbcf00dfc2fba941b494e2955e87fffd707382',
    modelSha256: '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad',
    fixtureSourceCommit: '42ef8a125df038dcca49f6216f446fe9112946c1',
  },
  runner: runnerIdentity(),
  tools: { node: toolVersion(root, 'node', ['--version']), bun: toolVersion(root, 'bun', ['--version']), cargo: toolVersion(root, 'cargo', ['--version']), onnxruntimeWeb: '1.27.0' },
  immutableLogEvidence: { kind: 'embedded-in-manifest', path: 'evidence/replay/task1-replay.json', ciJobUrl: process.env.CI_JOB_URL ?? null },
  steps,
  outputs: artifacts,
  task1_7OwnershipReplayComplete: steps.filter((step) => step.executed).every((step) => step.repeatComparison.allExitCodesZero),
  task1_4Complete: false,
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', outputCount: artifacts.length }));
