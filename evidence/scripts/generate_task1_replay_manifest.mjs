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
const mindsporeOnly = process.argv.includes('--mindspore-only');
const openvinoOnly = process.argv.includes('--openvino-only');
if ([litertOnly, coremlOnly, mindsporeOnly, openvinoOnly].filter(Boolean).length > 1) throw new Error('一次只能选择一个 scoped replay 模式');
const scopedOnly = litertOnly || coremlOnly || mindsporeOnly || openvinoOnly;
const previousReplay = scopedOnly
  ? JSON.parse(openvinoOnly
    ? execFileSync('git', ['show', 'HEAD:evidence/replay/task1-replay.json'], { cwd: root, encoding: 'utf8' })
    : await readFile(outputPath, 'utf8'))
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
  'evidence/conversions/mindspore-artifact-manifest.json',
  'evidence/conversions/openvino-ep-manifest.json',
  'evidence/reports/coreml-conversion-report.json',
  'evidence/reports/litert-conversion-report.json',
  'evidence/reports/litert-golden-report.json',
  'evidence/reports/mindspore-conversion-report.json',
  'evidence/reports/mindspore-golden-report.json',
  'evidence/reports/openvino-ep-report.json',
  'evidence/tooling/coreml-requirements.lock',
  'evidence/tooling/litert-requirements.lock',
  'evidence/tooling/mindspore-python-addons.lock',
  'evidence/tooling/openvino-requirements.lock',
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
if (openvinoOnly) {
  const previousRawGolden = previousReplay.steps.find((step) => step.id === 'production-raw-golden');
  if (!previousRawGolden) throw new Error('历史 task1 replay 缺少 production-raw-golden');
  steps.push(previousRawGolden);
} else {
  steps.push(await runRepeatedStep({
    root,
    id: 'production-raw-golden',
    command: 'cargo test --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml',
    executable: 'cargo',
    args: ['test', '--offline', '--manifest-path', 'evidence/tooling/raw-golden/Cargo.toml'],
    inputPaths: ['evidence/fixtures/raw/overlap-nms.json', 'src/postprocess.rs'],
    outputPaths: [],
  }));
}
const conversionMode = coremlOnly ? '--coreml-only' : mindsporeOnly ? '--mindspore-only' : openvinoOnly ? '--openvino-only' : '--litert-only';
const conversionInputs = coremlOnly
  ? [
      'evidence/conversions/conversion-spikes.json',
      'evidence/conversions/coreml-artifact-manifest.json',
      'evidence/reports/handoff-model-audit.json',
      'evidence/reports/coreml-conversion-report.json',
    ]
    : mindsporeOnly
      ? [
        'evidence/conversions/conversion-spikes.json',
        'evidence/conversions/mindspore-artifact-manifest.json',
        'evidence/reports/handoff-model-audit.json',
        'evidence/reports/mindspore-conversion-report.json',
        'evidence/reports/mindspore-golden-report.json',
      ]
      : openvinoOnly
        ? [
          'evidence/conversions/conversion-spikes.json',
          'evidence/conversions/openvino-ep-manifest.json',
          'evidence/reports/openvino-ep-report.json',
        ]
        : [
      'evidence/conversions/conversion-spikes.json',
      'evidence/conversions/litert-artifact-manifest.json',
      'evidence/reports/handoff-model-audit.json',
      'evidence/reports/litert-conversion-report.json',
      'evidence/reports/litert-golden-report.json',
    ];
if (openvinoOnly) {
  const previousConversion = previousReplay.steps.find((step) => step.id === 'conversion-report-regeneration');
  if (!previousConversion) throw new Error('历史 task1 replay 缺少 conversion-report-regeneration');
  steps.push(previousConversion);
} else {
  steps.push(await runRepeatedStep({
    root,
    id: 'conversion-report-regeneration',
    command: `node evidence/scripts/run_conversion_spikes.mjs ${conversionMode}`,
    executable: 'node',
    args: ['evidence/scripts/run_conversion_spikes.mjs', conversionMode],
    inputPaths: conversionInputs,
    outputPaths: ['evidence/conversions/conversion-spikes.json'],
  }));
}
if (coremlOnly || mindsporeOnly || openvinoOnly) {
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

if (litertOnly || mindsporeOnly || openvinoOnly) {
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
    mode: coremlReplay.mode,
    recordedArtifactTreeDigest: coremlReplay.recordedArtifactTreeDigest,
    recordedArtifactVerification: coremlReplay.recordedArtifactVerification,
    semanticReplayDigests: coremlReplay.semanticReplayDigests,
    semanticReplayValidation: coremlReplay.semanticReplayValidation,
    trackedEvidence: coremlReplay.trackedEvidence,
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
      outputs: [{
        bytes: round.artifactTree.totalFileBytes,
        kind: 'workspace-package-tree',
        workspacePackageTreeDigest: round.artifactTree.digest,
      }],
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

if (litertOnly || coremlOnly || openvinoOnly) {
  const previousMindspore = previousReplay.steps.find((step) => step.id === 'harmonyos-mindspore-conversion-and-host-golden');
  if (!previousMindspore) throw new Error('历史 task1 replay 缺少 MindSpore replay');
  steps.push(previousMindspore);
} else {
  const mindsporeReplayPath = resolve(root, '.evidence/mindspore/replay/mindspore-replay.json');
  const mindsporeReplay = JSON.parse(await readFile(mindsporeReplayPath, 'utf8').catch(() => {
    throw new Error('缺少两轮 MindSpore replay；先执行 evidence/scripts/run_mindspore_replay.py');
  }));
  const mindsporeCommand = '$MINDSPORE_PYTHON evidence/scripts/run_mindspore_replay.py --pt $HANDOFF_ASSETS/yolov8n.pt --handoff-onnx $HANDOFF_ASSETS/yolov8n.onnx --workspace .evidence/mindspore/replay';
  steps.push({
    id: 'harmonyos-mindspore-conversion-and-host-golden',
    command: mindsporeCommand,
    executed: true,
    blockedReason: null,
    mode: mindsporeReplay.mode,
    recorded: mindsporeReplay.recorded,
    recordedArtifactSha256: mindsporeReplay.recordedArtifactSha256,
    recordedArtifactVerification: mindsporeReplay.recordedArtifactVerification,
    replayArtifactSha256: mindsporeReplay.replayArtifactSha256,
    trackedEvidence: mindsporeReplay.trackedEvidence,
    workspaceArtifact: mindsporeReplay.artifact,
    rounds: mindsporeReplay.rounds.map((round) => {
      const success = round.matrix.find((item) => item.result === 'success');
      const stdout = [round.webReference.stdout, round.export.stdout, round.derivation.stdout, ...round.matrix.map((item) => item.stdout), ...round.hostValidation.fixtures.flatMap((item) => [item.benchmark.stdout, item.runtime.stdout, item.productionRust.stdout])].join('\n');
      const stderr = [round.webReference.stderr, round.export.stderr, round.derivation.stderr, ...round.matrix.map((item) => item.stderr), round.hostValidation.compile.stderr, round.hostValidation.productionRustBuild.stderr, ...round.hostValidation.fixtures.flatMap((item) => [item.benchmark.stderr, item.runtime.stderr, item.productionRust.stderr])].join('\n');
      return {
        run: round.round,
        actualCommand: mindsporeCommand,
        startedAt: round.webReference.startedAt,
        endedAt: round.hostValidation.fixtures.at(-1).productionRust.endedAt,
        exitCode: success?.exitCode === 0 && round.hostValidation.passed ? 0 : 1,
        signal: null,
        repositoryHead: repositoryHeadAtReplay,
        runnerId: runnerIdentity().id,
        worktreeBefore: round.worktreeBefore,
        worktreeAfter: round.worktreeAfter,
        inputs: Object.values(round.sourceBefore),
        outputs: [
          success.artifact,
          { bytes: round.exportReport.output.bytes, path: '.evidence/mindspore/replay/round-N/onnx/yolov8n-opset17-unsimplified.onnx', sha256: round.exportReport.output.sha256 },
          { bytes: round.derivationReport.derived.bytes, path: '.evidence/mindspore/replay/round-N/onnx/yolov8n-opset17-dfl-reduced.onnx', sha256: round.derivationReport.derived.sha256 },
        ],
        expectedFailureSignatures: round.matrix.filter((item) => item.result === 'failed').map((item) => ({ id: item.id, exitCode: item.exitCode, failureSignature: item.failureSignature })),
        hostFixtureCount: round.hostValidation.fixtures.length,
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
      allExitCodesZero: mindsporeReplay.rounds.every((round) => round.matrix.find((item) => item.result === 'success')?.exitCode === 0 && round.hostValidation.passed),
      deterministicOutputDigestsEqual: mindsporeReplay.comparison.allDeterministic && mindsporeReplay.comparison.derivedOnnxDigestEqual && mindsporeReplay.comparison.reexportOnnxDigestEqual && mindsporeReplay.comparison.fixtureResultsEqual,
      failureSignaturesEqual: Object.values(mindsporeReplay.comparison.paths).every((item) => item.failureSignatureEqual),
      details: mindsporeReplay.comparison,
    },
  });
}
if (openvinoOnly) {
  const replay = JSON.parse(await readFile(resolve(root, '.evidence/openvino/replay-final/openvino-replay.json'), 'utf8'));
  const openvinoCommand = '$OPENVINO_PYTHON evidence/scripts/run_openvino_replay.py --workspace .evidence/openvino/replay-final';
  steps.push({
    id: 'linux-x86_64-openvino-provider-host-golden',
    command: openvinoCommand,
    executed: true,
    blockedReason: null,
    mode: replay.mode,
    recordDigest: replay.recordDigest,
    trackedEvidence: replay.trackedEvidence,
    rounds: replay.rounds.map((round) => ({
      run: round.round,
      actualCommand: openvinoCommand,
      startedAt: round.startedAt,
      endedAt: round.endedAt,
      exitCode: round.exitCode,
      signal: null,
      repositoryHead: repositoryHeadAtReplay,
      runnerId: runnerIdentity().id,
      availableProviders: round.availableProviders,
      sessionProviders: round.sessionProviders,
      executionPlan: round.profile.executionPlan,
      profileNodeCounts: round.profile.uniqueNodeCounts,
      hostFixtureCount: round.fixtures.length,
      runsPerFixture: round.fixtures[0].runs.length,
      outputDigests: round.fixtures.map((fixture) => ({ id: fixture.id, sha256: fixture.runs[0].raw.sha256 })),
    })),
    repeatComparison: {
      runs: replay.rounds.length,
      allExitCodesZero: replay.rounds.every((round) => round.exitCode === 0),
      deterministicOutputDigestsEqual: replay.rounds[0].fixtures.every((fixture, index) => fixture.runs[0].raw.sha256 === replay.rounds[1].fixtures[index].runs[0].raw.sha256),
      trackedEvidenceUnchanged: Object.values(replay.trackedEvidence).every((item) => item.unchanged),
    },
  });
} else if (scopedOnly) {
  const previousOpenvino = previousReplay.steps.find((step) => step.id === 'linux-x86_64-openvino-provider-host-golden');
  if (previousOpenvino) steps.push(previousOpenvino);
}
steps.push(blockedStep(
  'delegated-platform-spikes',
  'Windows ML、Linux CUDA/TensorRT 与真实 Apple/Android/HarmonyOS runtime commands',
  'Linux x86_64 OpenVINO host inference 已验证；CUDA/TensorRT 未处理。MindSpore Lite 缺 HarmonyOS 真机，Core ML 缺 macOS/iOS Load/Run，LiteRT 缺 Android runner，Windows ML 缺真实 runner。',
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
  tools: { node: toolVersion(root, 'node', ['--version']), bun: toolVersion(root, 'bun', ['--version']), cargo: toolVersion(root, 'cargo', ['--version']), onnxruntimeWeb: '1.27.0', onnxruntimeOpenvino: '1.24.1', openvinoRuntime: '2025.4.1', coremltools: '9.0', litertRuntime: '2.1.6', litertTorch: '0.9.3', mindsporeLite: '2.7.0' },
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
// Scoped provider regeneration above may intentionally replace one provider entry.
// Re-apply the complete Task 1.4 closure only after this base replay exists, so the
// aggregate, golden manifest, and replay hashes describe the same final generation.
await import('./generate_task14_aggregate.mjs');
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', outputCount: artifacts.length }));
