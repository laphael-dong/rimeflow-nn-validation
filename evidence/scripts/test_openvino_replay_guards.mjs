import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, writeFile, mkdtemp, rm, symlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { recoverOpenvinoPublication, resolveTrustedOpenvinoLibraryPath, resolveTrustedOpenvinoProfilePath, validateOpenvinoEvidence, validateOpenvinoReplayEvidence } from './openvino_evidence_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
recoverOpenvinoPublication(root);
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const resolveReplayArtifact = (path) => resolve(root, path.replace('.evidence/openvino/record-fix03-final/', '.evidence/openvino/replay-final/'));
const manifest = await readJson('evidence/conversions/openvino-ep-manifest.json');
const report = await readJson('evidence/reports/openvino-ep-report.json');
const frozen = await readJson('evidence/golden/web-reference.json');
const fixtures = await readJson('evidence/fixtures/manifest.json');
const replay = await readJson('.evidence/openvino/replay-final/openvino-replay.json');
await validateOpenvinoEvidence(root, manifest, report, frozen, fixtures);
await validateOpenvinoReplayEvidence(root, replay, manifest, report);

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function rejected(name, mutate) {
  const copiedManifest = structuredClone(manifest);
  const copiedReport = structuredClone(report);
  const copiedFrozen = structuredClone(frozen);
  await mutate(copiedManifest, copiedReport, copiedFrozen);
  const caseDirectory = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-case-'));
  try {
    const manifestPath = join(caseDirectory, 'manifest.json');
    const reportPath = join(caseDirectory, 'report.json');
    const frozenPath = join(caseDirectory, 'frozen.json');
    await Promise.all([
      writeFile(manifestPath, JSON.stringify(copiedManifest)),
      writeFile(reportPath, JSON.stringify(copiedReport)),
      writeFile(frozenPath, JSON.stringify(copiedFrozen)),
    ]);
    const [filesystemManifest, filesystemReport, filesystemFrozen] = await Promise.all(
      [manifestPath, reportPath, frozenPath].map(async (path) => JSON.parse(await readFile(path, 'utf8'))),
    );
    await validateOpenvinoEvidence(root, filesystemManifest, filesystemReport, filesystemFrozen, fixtures);
  } catch {
    return name;
  } finally {
    await rm(caseDirectory, { recursive: true, force: true });
  }
  throw new Error(`OpenVINO negative case unexpectedly passed: ${name}`);
}

const cases = [];
await Promise.all(manifest.runtime.libraries.map((library) => resolveTrustedOpenvinoLibraryPath(root, library)));
cases.push(await rejected('absolute library path injection', async (candidate) => {
  candidate.runtime.libraries[0].actualPath = '/tmp/injected/libopenvino.so';
}));
cases.push(await rejected('library path traversal', async (candidate) => {
  candidate.runtime.libraries[0].actualPath = '$OPENVINO_VENV/lib/python3.12/site-packages/onnxruntime/capi/../../../../../../tmp/injected.so';
}));
cases.push(await rejected('wrong library capi directory', async (candidate) => {
  candidate.runtime.libraries[0].actualPath = '$OPENVINO_VENV/lib/python3.12/site-packages/onnxruntime/wrong-capi/libonnxruntime_providers_openvino.so';
}));
cases.push(await rejected('library basename substitution', async (candidate) => {
  candidate.runtime.libraries[0].actualPath = '$OPENVINO_VENV/lib/python3.12/site-packages/onnxruntime/capi/libopenvino_c.so';
}));
cases.push(await rejected('absolute profile path injection', async (_candidate, evidence) => {
  evidence.rounds[0].profile.path = '/tmp/injected-profile.json';
}));
cases.push(await rejected('profile path traversal', async (_candidate, evidence) => {
  evidence.rounds[0].profile.path = '$OPENVINO_WORKSPACE/round-1/../../injected-profile.json';
}));
cases.push(await rejected('wrong profile directory', async (_candidate, evidence) => {
  evidence.rounds[0].profile.path = '$OPENVINO_WORKSPACE/round-2/ort-profile.node-events.json';
}));
cases.push(await rejected('deterministic profile identity drift', async (_candidate, evidence) => {
  evidence.rounds[0].profile.sha256 = '9'.repeat(64);
}));
cases.push(await rejected('ordinary CPU ORT impersonates OpenVINO', async (candidate, evidence) => {
  candidate.runtime.onnxruntimeOpenvino = '1.24.1';
  evidence.rounds[0].availableProviders = ['CPUExecutionProvider'];
}));
cases.push(await rejected('available providers omit OpenVINO', async (_candidate, evidence) => { evidence.rounds[0].availableProviders = ['CPUExecutionProvider']; }));
cases.push(await rejected('session configured OpenVINO without OpenVINO profile node', async (_candidate, evidence) => {
  evidence.rounds[0].profile.uniqueNodeCounts.OpenVINOExecutionProvider = 0;
}));
cases.push(await rejected('CPU fallback hidden and partitioned claimed full', async (candidate, evidence) => {
  candidate.provider.fallbackVisible = false;
  evidence.rounds[0].profile.executionEventCounts.CPUExecutionProvider = 1;
}));
cases.push(await rejected('provider name drift', async (candidate) => { candidate.provider.requested = 'openvino'; }));
cases.push(await rejected('runtime version drift', async (candidate) => { candidate.runtime.openvino.runtime.buildNumber = '2025.4.0'; }));
cases.push(await rejected('shared library SHA drift', async (candidate) => { candidate.runtime.libraries[0].sha256 = '0'.repeat(64); }));
cases.push(await rejected('model SHA drift', async (candidate) => { candidate.artifact.sha256 = '0'.repeat(64); }));
cases.push(await rejected('I/O name drift', async (candidate) => { candidate.ioContract.inputs[0].name = 'input'; }));
cases.push(await rejected('I/O shape drift', async (candidate) => { candidate.ioContract.outputs[0].shape = [1, 8400, 84]; }));
cases.push(await rejected('I/O dtype drift', async (candidate) => { candidate.ioContract.outputs[0].dtype = 'float16'; }));
cases.push(await rejected('I/O element count drift', async (candidate) => { candidate.ioContract.outputs[0].elementCount = 1; }));
cases.push(await rejected('frozen tolerance relaxed', async (_candidate, evidence) => { evidence.tolerances.rawTensorAbsolute = 1; }));
cases.push(await rejected('supported without adapter/package/performance/target closure', async (candidate, evidence) => {
  candidate.status.supported = true;
  evidence.status.supported = true;
}));

const provenanceCases = [
  ['production source artifact is missing', (value) => { value.sourceArtifacts.pop(); }],
  ['production source canonical logical path drift', (value) => { value.sourceArtifacts[0].canonicalPath = '/tmp/injected'; }],
  ['production source bytes drift', (value) => { value.sourceArtifacts[0].bytes += 1; }],
  ['production source SHA drift', (value) => { value.sourceArtifacts[0].sha256 = '0'.repeat(64); }],
  ['production source HEAD blob drift', (value) => { value.sourceArtifacts[0].headBlobOid = '0'.repeat(40); }],
  ['production runner SHA drift', (value) => { value.runner.sha256 = '1'.repeat(64); }],
  ['production runner bytes drift', (value) => { value.runner.bytes += 1; }],
  ['production runner ELF identity drift', (value) => { value.runner.elf.magic = '00000000'; }],
  ['production build omits offline argv', (value) => { value.build.argv = value.build.argv.filter((item) => item !== '--offline'); }],
  ['production build offline claim is false', (value) => { value.build.offline = false; }],
  ['production build omits locked argv', (value) => { value.build.argv = value.build.argv.filter((item) => item !== '--locked'); }],
  ['production build locked claim is false', (value) => { value.build.locked = false; }],
  ['production build target is not fresh', (value) => { value.build.freshTarget = false; }],
  ['production build source mirror is not isolated', (value) => { value.build.isolatedSourceMirror = false; }],
  ['production build Cargo home is not isolated', (value) => { value.build.isolatedCargoHome = false; }],
  ['production build root Cargo config participates', (value) => { value.build.repositoryRootCargoConfigParticipated = true; }],
  ['production build cleared environment list drift', (value) => { value.build.clearedEnvironmentVariables.pop(); }],
  ['production build rejected environment list drift', (value) => { value.build.rejectedEnvironmentVariables.pop(); }],
  ['production build controlled environment drift', (value) => { value.build.controlledEnvironment.CARGO_INCREMENTAL = '1'; }],
  ['production Cargo version drift', (value) => { value.build.cargoVersion = 'cargo 0.0.0'; }],
  ['production Rustc version drift', (value) => { value.build.rustcVersion = 'rustc 0.0.0'; }],
  ['production fixture runner SHA drift', (_value, evidence) => { evidence.rounds[0].fixtures[0].runs[0].productionPostprocess.runner.sha256 = '2'.repeat(64); }],
  ['production fixture logical command drift', (_value, evidence) => { evidence.rounds[0].fixtures[0].runs[0].productionPostprocess.command[0] = '/tmp/fake-runner'; }],
];
for (const [name, mutate] of provenanceCases) {
  cases.push(await rejected(name, async (_candidate, evidence) => mutate(evidence.productionPostprocess, evidence)));
}

const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-negative-'));
try {
  cases.push(await rejected('absolute injected profile with no OpenVINO node', async (_candidate, evidence) => {
    const profile = join(temporary, 'cpu-profile.json');
    const bytes = Buffer.from(JSON.stringify([{ cat: 'Node', name: 'cpu', args: { provider: 'CPUExecutionProvider' } }]));
    await writeFile(profile, bytes);
    Object.assign(evidence.rounds[0].profile, { path: profile, bytes: bytes.length, sha256: (await import('node:crypto')).createHash('sha256').update(bytes).digest('hex') });
  }));
  cases.push(await rejected('raw tensor contains NaN', async (_candidate, evidence) => {
    const source = resolveReplayArtifact(evidence.rounds[0].fixtures[0].runs[0].raw.path);
    const bytes = Buffer.from(await readFile(source));
    bytes.writeFloatLE(Number.NaN, 0);
    const raw = join(temporary, 'nan.f32le');
    await writeFile(raw, bytes);
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { path: raw, sha256: digest(bytes) });
  }));
  cases.push(await rejected('raw tensor contains Infinity', async (_candidate, evidence) => {
    const source = resolveReplayArtifact(evidence.rounds[0].fixtures[0].runs[0].raw.path);
    const bytes = Buffer.from(await readFile(source));
    bytes.writeFloatLE(Number.POSITIVE_INFINITY, 0);
    const raw = join(temporary, 'infinity.f32le');
    await writeFile(raw, bytes);
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { path: raw, sha256: digest(bytes) });
  }));
  cases.push(await rejected('raw tensor byte length drift', async (_candidate, evidence) => {
    const run = evidence.rounds[0].fixtures[0].runs[0];
    const bytes = Buffer.from(await readFile(resolveReplayArtifact(run.raw.path))).subarray(0, 1024);
    const raw = join(temporary, 'truncated.f32le');
    await writeFile(raw, bytes);
    Object.assign(run.raw, { bytes: bytes.length, path: raw, sha256: digest(bytes) });
  }));
  cases.push(await rejected('raw tensor shape dtype and element count drift', async (_candidate, evidence) => {
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { dtype: 'float16', elementCount: 1, shape: [1, 8400, 84] });
  }));
  cases.push(await rejected('finite raw value exceeds frozen tolerance despite forged pass flags', async (_candidate, evidence) => {
    const run = evidence.rounds[0].fixtures[0].runs[0];
    const bytes = Buffer.from(await readFile(resolveReplayArtifact(run.raw.path)));
    bytes.writeFloatLE(bytes.readFloatLE(0) + 1, 0);
    const raw = join(temporary, 'finite-tamper.f32le');
    await writeFile(raw, bytes);
    Object.assign(run.raw, { path: raw, sha256: digest(bytes) });
    Object.assign(run.rawComparison, { allClose: true, sha256Float32Le: digest(bytes) });
    run.passed = true;
  }));
  cases.push(await rejected('synchronized finite raw tamper across every round and repeat', async (_candidate, evidence) => {
    const first = evidence.rounds[0].fixtures[0].runs[0];
    const bytes = Buffer.from(await readFile(resolveReplayArtifact(first.raw.path)));
    bytes.writeFloatLE(bytes.readFloatLE(0) + 1, 0);
    const raw = join(temporary, 'finite-tamper-all-runs.f32le');
    await writeFile(raw, bytes);
    const rawDigest = digest(bytes);
    for (const round of evidence.rounds) {
      for (const run of round.fixtures[0].runs) {
        Object.assign(run.raw, { path: raw, sha256: rawDigest });
        Object.assign(run.rawComparison, { allClose: true, sha256Float32Le: rawDigest });
        run.passed = true;
      }
    }
  }));
  cases.push(await rejected('reported maximum raw difference drift', async (_candidate, evidence) => {
    evidence.rounds[0].fixtures[0].runs[0].rawComparison.maxAbsoluteDifference = 0;
  }));
  cases.push(await rejected('reported maximum raw location drift', async (_candidate, evidence) => {
    evidence.rounds[0].fixtures[0].runs[0].rawComparison.maxAbsoluteDifferenceLocation.flatIndex += 1;
  }));
  cases.push(await rejected('reference raw digest drift', async (_candidate, evidence) => {
    const comparison = evidence.rounds[0].fixtures[0].runs[0].rawComparison;
    comparison.reference.sha256 = '3'.repeat(64);
    comparison.referenceSha256Float32Le = '3'.repeat(64);
  }));
  cases.push(await rejected('raw comparison passes but decoded class differs', async (_candidate, evidence, frozenEvidence) => {
    const frozenFixture = frozenEvidence.fixtures.find((item) => item.id === 'single-target');
    frozenFixture.runs[0].decoded[0].classId += 1;
    const run = evidence.rounds[0].fixtures.find((item) => item.id === 'single-target').runs[0];
    run.rawComparison.allClose = true;
    run.decodedComparison.comparisons[0].classEqual = true;
    run.decodedComparison.comparisons[0].passed = true;
    run.decodedComparison.passed = true;
    run.passed = true;
  }));
  cases.push(await rejected('raw comparison passes but decoded confidence exceeds tolerance', async (_candidate, evidence, frozenEvidence) => {
    const frozenFixture = frozenEvidence.fixtures.find((item) => item.id === 'single-target');
    frozenFixture.runs[0].decoded[0].score += 0.01;
    const run = evidence.rounds[0].fixtures.find((item) => item.id === 'single-target').runs[0];
    run.rawComparison.allClose = true;
    Object.assign(run.decodedComparison.comparisons[0], { confidenceAbsoluteDifference: 0, passed: true });
    run.decodedComparison.passed = true;
    run.passed = true;
  }));
  cases.push(await rejected('raw comparison passes but decoded bbox absolute and IoU exceed tolerance', async (_candidate, evidence, frozenEvidence) => {
    const frozenFixture = frozenEvidence.fixtures.find((item) => item.id === 'single-target');
    frozenFixture.runs[0].decoded[0].bbox[0] += 0.1;
    const run = evidence.rounds[0].fixtures.find((item) => item.id === 'single-target').runs[0];
    run.rawComparison.allClose = true;
    Object.assign(run.decodedComparison.comparisons[0], { bboxIou: 1, bboxMaxAbsoluteDifference: 0, passed: true });
    run.decodedComparison.passed = true;
    run.passed = true;
  }));
  const replayDrift = structuredClone(replay);
  replayDrift.trackedEvidence.report.after.sha256 = '4'.repeat(64);
  replayDrift.trackedEvidence.report.unchanged = true;
  try {
    await validateOpenvinoReplayEvidence(root, replayDrift, manifest, report);
    throw new Error('OpenVINO negative case unexpectedly passed: ordinary replay modifies tracked evidence');
  } catch (error) {
    if (String(error).includes('unexpectedly passed')) throw error;
    cases.push('ordinary replay modifies tracked evidence');
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}

const symlinkRoot = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-library-escape-'));
try {
  const capi = join(symlinkRoot, '.evidence/openvino/venv/lib/python3.12/site-packages/onnxruntime/capi');
  const outside = join(symlinkRoot, 'outside.so');
  await mkdir(capi, { recursive: true });
  await writeFile(outside, 'outside');
  await symlink(outside, join(capi, manifest.runtime.libraries[0].name));
  try {
    await resolveTrustedOpenvinoLibraryPath(symlinkRoot, manifest.runtime.libraries[0]);
    throw new Error('OpenVINO negative case unexpectedly passed: symlink escape');
  } catch (error) {
    if (String(error).includes('unexpectedly passed')) throw error;
    cases.push('symlink escape');
  }
} finally {
  await rm(symlinkRoot, { recursive: true, force: true });
}

const profileSymlinkRoot = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-profile-escape-'));
try {
  const workspace = join(profileSymlinkRoot, '.evidence/openvino/replay-final');
  const round = join(workspace, 'round-1');
  const outside = join(profileSymlinkRoot, 'outside-profile.json');
  await mkdir(round, { recursive: true });
  await writeFile(outside, '[]\n');
  await symlink(outside, join(round, 'ort-profile.node-events.json'));
  try {
    await resolveTrustedOpenvinoProfilePath(profileSymlinkRoot, '.evidence/openvino/replay-final', report.rounds[0].profile, 1);
    throw new Error('OpenVINO negative case unexpectedly passed: profile symlink escape');
  } catch (error) {
    if (String(error).includes('unexpectedly passed')) throw error;
    cases.push('profile symlink escape');
  }
} finally {
  await rm(profileSymlinkRoot, { recursive: true, force: true });
}

const transaction = spawnSync(
  resolve(root, '.evidence/openvino/venv/bin/python'),
  ['evidence/scripts/test_openvino_publish_transaction.py'],
  { cwd: root, encoding: 'utf8' },
);
if (transaction.status !== 0) throw new Error(`OpenVINO publication transaction tests failed: ${transaction.stderr}`);
const transactionResult = JSON.parse(transaction.stdout);
if (!transactionResult.ok || transactionResult.filesystemCases.length !== 22) throw new Error('OpenVINO publication transaction coverage drift');

const rustIdentity = spawnSync('node', ['evidence/scripts/test_openvino_rust_identity_guards.mjs'], { cwd: root, encoding: 'utf8' });
if (rustIdentity.status !== 0) throw new Error(`OpenVINO Rust source/binary identity guards failed: ${rustIdentity.stdout}\n${rustIdentity.stderr}`);
const rustIdentityResult = JSON.parse(rustIdentity.stdout);
if (!rustIdentityResult.ok || rustIdentityResult.cases.length < 34 || rustIdentityResult.environmentCases !== 15 || rustIdentityResult.sourceCount !== 5) throw new Error('OpenVINO Rust source/binary guard coverage drift');

console.log(JSON.stringify({ filesystemCases: transactionResult.filesystemCases, ok: true, negativeCases: cases, positiveCases: ['trusted token resolves from current clean checkout venv root', 'real OpenVINO record files and runtime evidence'], rustIdentityCases: rustIdentityResult.cases }));
