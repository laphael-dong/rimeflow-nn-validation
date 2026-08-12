import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { recoverOpenvinoPublication, validateOpenvinoEvidence, validateOpenvinoReplayEvidence } from './openvino_evidence_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
recoverOpenvinoPublication(root);
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
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

const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-negative-'));
try {
  cases.push(await rejected('profile file has no OpenVINO node', async (_candidate, evidence) => {
    const profile = join(temporary, 'cpu-profile.json');
    const bytes = Buffer.from(JSON.stringify([{ cat: 'Node', name: 'cpu', args: { provider: 'CPUExecutionProvider' } }]));
    await writeFile(profile, bytes);
    Object.assign(evidence.rounds[0].profile, { path: profile, bytes: bytes.length, sha256: (await import('node:crypto')).createHash('sha256').update(bytes).digest('hex') });
  }));
  cases.push(await rejected('raw tensor contains NaN', async (_candidate, evidence) => {
    const source = resolve(root, evidence.rounds[0].fixtures[0].runs[0].raw.path);
    const bytes = Buffer.from(await readFile(source));
    bytes.writeFloatLE(Number.NaN, 0);
    const raw = join(temporary, 'nan.f32le');
    await writeFile(raw, bytes);
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { path: raw, sha256: digest(bytes) });
  }));
  cases.push(await rejected('raw tensor contains Infinity', async (_candidate, evidence) => {
    const source = resolve(root, evidence.rounds[0].fixtures[0].runs[0].raw.path);
    const bytes = Buffer.from(await readFile(source));
    bytes.writeFloatLE(Number.POSITIVE_INFINITY, 0);
    const raw = join(temporary, 'infinity.f32le');
    await writeFile(raw, bytes);
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { path: raw, sha256: digest(bytes) });
  }));
  cases.push(await rejected('raw tensor byte length drift', async (_candidate, evidence) => {
    const run = evidence.rounds[0].fixtures[0].runs[0];
    const bytes = Buffer.from(await readFile(resolve(root, run.raw.path))).subarray(0, 1024);
    const raw = join(temporary, 'truncated.f32le');
    await writeFile(raw, bytes);
    Object.assign(run.raw, { bytes: bytes.length, path: raw, sha256: digest(bytes) });
  }));
  cases.push(await rejected('raw tensor shape dtype and element count drift', async (_candidate, evidence) => {
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { dtype: 'float16', elementCount: 1, shape: [1, 8400, 84] });
  }));
  cases.push(await rejected('finite raw value exceeds frozen tolerance despite forged pass flags', async (_candidate, evidence) => {
    const run = evidence.rounds[0].fixtures[0].runs[0];
    const bytes = Buffer.from(await readFile(resolve(root, run.raw.path)));
    bytes.writeFloatLE(bytes.readFloatLE(0) + 1, 0);
    const raw = join(temporary, 'finite-tamper.f32le');
    await writeFile(raw, bytes);
    Object.assign(run.raw, { path: raw, sha256: digest(bytes) });
    Object.assign(run.rawComparison, { allClose: true, sha256Float32Le: digest(bytes) });
    run.passed = true;
  }));
  cases.push(await rejected('synchronized finite raw tamper across every round and repeat', async (_candidate, evidence) => {
    const first = evidence.rounds[0].fixtures[0].runs[0];
    const bytes = Buffer.from(await readFile(resolve(root, first.raw.path)));
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
if (!rustIdentityResult.ok || rustIdentityResult.cases.length !== 9 || rustIdentityResult.sourceCount !== 5) throw new Error('OpenVINO Rust source/binary guard coverage drift');

console.log(JSON.stringify({ filesystemCases: transactionResult.filesystemCases, ok: true, negativeCases: cases, positiveCases: ['real OpenVINO record files and runtime evidence'], rustIdentityCases: rustIdentityResult.cases }));
