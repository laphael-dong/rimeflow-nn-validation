import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { blockedStep, runRepeatedStep, runnerIdentity, sha256, toolVersion } from './task1_replay_execution.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const outputPath = resolve(root, 'evidence/replay/task1-replay.json');
const git = (args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const repositoryHeadAtReplay = git(['rev-parse', 'HEAD']);
const litertOnly = process.argv.includes('--litert-only');
const coremlOnly = process.argv.includes('--coreml-only');
if (litertOnly && coremlOnly) throw new Error('一次只能选择一个 scoped replay 模式');
const scopedOnly = litertOnly || coremlOnly;
const previousReplay = scopedOnly
  ? JSON.parse(await readFile(outputPath, 'utf8'))
  : null;
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
  'evidence/conversions/coreml-artifact-manifest.json',
  'evidence/conversions/litert-artifact-manifest.json',
  'evidence/reports/coreml-conversion-report.json',
  'evidence/reports/litert-conversion-report.json',
  'evidence/reports/litert-golden-report.json',
  'evidence/tooling/coreml-requirements.lock',
  'evidence/tooling/litert-requirements.lock',
];
const commonInputs = ['models/yolov8n.onnx', 'evidence/fixtures/manifest.json'];
const steps = [];
if (scopedOnly) {
  const previousContractReplay = previousReplay.steps.find((step) => step.id === 'contract-fixture-golden');
  if (!previousContractReplay) throw new Error('历史 task1 replay 缺少 contract-fixture-golden');
  steps.push(previousContractReplay);
} else {
  steps.push(await runRepeatedStep({
    root,
    id: 'contract-fixture-golden',
    command: 'bun run evidence/scripts/generate_all.mjs',
    executable: 'bun',
    args: ['run', 'evidence/scripts/generate_all.mjs'],
    inputPaths: commonInputs,
    outputPaths: outputs,
  }));
}
steps.push(await runRepeatedStep({
  root,
  id: 'production-raw-golden',
  command: 'cargo test --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml',
  executable: 'cargo',
  args: ['test', '--offline', '--manifest-path', 'evidence/tooling/raw-golden/Cargo.toml'],
  inputPaths: ['evidence/fixtures/raw/overlap-nms.json', 'src/postprocess.rs'],
  outputPaths: [],
}));
const conversionMode = coremlOnly ? '--coreml-only' : '--litert-only';
const conversionInputs = coremlOnly
  ? [
      'evidence/conversions/conversion-spikes.json',
      'evidence/conversions/coreml-artifact-manifest.json',
      'evidence/reports/handoff-model-audit.json',
      'evidence/reports/coreml-conversion-report.json',
    ]
  : [
      'evidence/conversions/conversion-spikes.json',
      'evidence/conversions/litert-artifact-manifest.json',
      'evidence/reports/handoff-model-audit.json',
      'evidence/reports/litert-conversion-report.json',
      'evidence/reports/litert-golden-report.json',
    ];
steps.push(await runRepeatedStep({
  root,
  id: 'conversion-report-regeneration',
  command: `node evidence/scripts/run_conversion_spikes.mjs ${conversionMode}`,
  executable: 'node',
  args: ['evidence/scripts/run_conversion_spikes.mjs', conversionMode],
  inputPaths: conversionInputs,
  outputPaths: ['evidence/conversions/conversion-spikes.json'],
}));
if (coremlOnly) {
  const previousLitert = previousReplay.steps.find((step) => step.id === 'android-litert-conversion-and-host-golden');
  if (!previousLitert) throw new Error('历史 task1 replay 缺少 LiteRT replay');
  steps.push(previousLitert);
} else {
  const litertReplayPath = resolve(root, '.evidence/litert/replay/litert-replay.json');
  const litertReplay = JSON.parse(await readFile(litertReplayPath, 'utf8').catch(() => {
    throw new Error('缺少两轮 LiteRT replay；先执行 evidence/scripts/run_litert_replay.py');
  }));
  const litertLockBytes = await readFile(resolve(root, 'evidence/tooling/litert-requirements.lock'));
  const litertCommand = '$LITERT_PYTHON evidence/scripts/run_litert_replay.py --pt $HANDOFF_ASSETS/yolov8n.pt --workspace .evidence/litert/replay';
  steps.push({
  id: 'android-litert-conversion-and-host-golden',
  command: litertCommand,
  executed: true,
  blockedReason: null,
  rounds: litertReplay.rounds.map((round) => {
    const stdout = `${round.webReference.stdout}\n${round.conversion.stdout}\n${round.validation.stdout}`;
    const stderr = `${round.webReference.stderr}\n${round.conversion.stderr}\n${round.validation.stderr}`;
    return {
      run: round.round,
      actualCommand: litertCommand,
      startedAt: round.webReference.startedAt,
      endedAt: round.validation.endedAt,
      exitCode: Math.max(round.webReference.exitCode, round.conversion.exitCode, round.validation.exitCode),
      signal: null,
      repositoryHead: repositoryHeadAtReplay,
      runnerId: runnerIdentity().id,
      worktreeBefore: round.worktreeBefore,
      worktreeAfter: round.worktreeAfter,
      inputs: [
        { path: '$HANDOFF_ASSETS/yolov8n.pt', exists: true, bytes: 6549796, sha256: 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36' },
        { path: 'evidence/tooling/litert-requirements.lock', exists: true, bytes: litertLockBytes.length, sha256: sha256(litertLockBytes) },
      ],
      outputs: [round.artifact, round.artifactManifest, round.goldenReport],
      log: {
        storage: 'embedded-in-replay-manifest',
        stdout,
        stderr,
        bytes: Buffer.byteLength(stdout) + Buffer.byteLength(stderr),
        sha256: sha256(Buffer.from(`${stdout}\0${stderr}`)),
      },
    };
  }),
  repeatComparison: {
    runs: 2,
    allExitCodesZero: litertReplay.rounds.every((round) => round.webReference.exitCode === 0 && round.conversion.exitCode === 0 && round.validation.exitCode === 0),
    deterministicOutputDigestsEqual: Object.values(litertReplay.comparison).every(Boolean),
    details: litertReplay.comparison,
  },
  });
}

if (litertOnly) {
  const previousCoreml = previousReplay.steps.find((step) => step.id === 'apple-coreml-conversion-and-spec-inspection');
  if (previousCoreml) steps.push(previousCoreml);
} else {
  const coremlReplayPath = resolve(root, '.evidence/coreml/replay/coreml-replay.json');
  const coremlReplay = JSON.parse(await readFile(coremlReplayPath, 'utf8').catch(() => {
    throw new Error('缺少两轮 Core ML replay；先执行 evidence/scripts/run_coreml_replay.py');
  }));
  const coremlLockBytes = await readFile(resolve(root, 'evidence/tooling/coreml-requirements.lock'));
  const coremlCommand = '$COREML_PYTHON evidence/scripts/run_coreml_replay.py --pt $HANDOFF_ASSETS/yolov8n.pt --workspace .evidence/coreml/replay';
  steps.push({
    id: 'apple-coreml-conversion-and-spec-inspection',
    command: coremlCommand,
    executed: true,
    blockedReason: null,
    rounds: coremlReplay.rounds.map((round) => ({
      run: round.round,
      actualCommand: coremlCommand,
      startedAt: round.conversion.startedAt,
      endedAt: round.conversion.endedAt,
      exitCode: round.conversion.exitCode,
      signal: null,
      repositoryHead: repositoryHeadAtReplay,
      runnerId: runnerIdentity().id,
      worktreeBefore: round.worktreeBefore,
      worktreeAfter: round.worktreeAfter,
      inputs: [
        { path: '$HANDOFF_ASSETS/yolov8n.pt', exists: true, bytes: 6549796, sha256: 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36' },
        { path: 'evidence/tooling/coreml-requirements.lock', exists: true, bytes: coremlLockBytes.length, sha256: sha256(coremlLockBytes) },
      ],
      outputs: [{ bytes: round.artifactTree.totalFileBytes, sha256: round.artifactTree.digest }],
      log: {
        storage: 'embedded-in-replay-manifest',
        stdout: round.conversion.stdout,
        stderr: round.conversion.stderr,
        bytes: Buffer.byteLength(round.conversion.stdout) + Buffer.byteLength(round.conversion.stderr),
        sha256: sha256(Buffer.from(`${round.conversion.stdout}\0${round.conversion.stderr}`)),
      },
    })),
    repeatComparison: {
      runs: 2,
      allExitCodesZero: coremlReplay.rounds.every((round) => round.conversion.exitCode === 0),
      deterministicOutputDigestsEqual: coremlReplay.comparison.packageTreeDigestEqual,
      semanticDeterminismVerified: coremlReplay.comparison.normalizedSpecDigestEqual
        && coremlReplay.comparison.normalizedPackageManifestDigestEqual
        && coremlReplay.comparison.weightBlobDigestEqual,
      details: coremlReplay.comparison,
    },
  });
}
steps.push(blockedStep(
  'delegated-platform-spikes',
  'Windows ML/MindSpore/Linux accelerated provider platform commands and real Apple runtime commands',
  'Core ML 已完成 Linux host artifact/spec 验证但缺 macOS/iOS Load/Run；LiteRT 缺 Android runner；Windows ML、MindSpore Lite 与加速 Linux provider 仍缺完整平台证据。',
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
  tools: { node: toolVersion(root, 'node', ['--version']), bun: toolVersion(root, 'bun', ['--version']), cargo: toolVersion(root, 'cargo', ['--version']), onnxruntimeWeb: '1.27.0', coremltools: '9.0', litertRuntime: '2.1.6', litertTorch: '0.9.3' },
  immutableLogEvidence: { kind: 'embedded-in-manifest', path: 'evidence/replay/task1-replay.json', ciJobUrl: process.env.CI_JOB_URL ?? null },
  steps,
  outputs: artifacts,
  task1_7OwnershipReplayComplete: scopedOnly
    ? previousReplay.task1_7OwnershipReplayComplete
    : steps.filter((step) => step.executed).every((step) => step.repeatComparison.allExitCodesZero),
  task1_4Complete: false,
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', outputCount: artifacts.length }));
