import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { validateOpenvinoEvidence, validateOpenvinoReplayEvidence } from './openvino_evidence_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const manifest = await readJson('evidence/conversions/openvino-ep-manifest.json');
const report = await readJson('evidence/reports/openvino-ep-report.json');
const frozen = await readJson('evidence/golden/web-reference.json');
const fixtures = await readJson('evidence/fixtures/manifest.json');
const replay = await readJson('.evidence/openvino/replay-final/openvino-replay.json');
await validateOpenvinoEvidence(root, manifest, report, frozen, fixtures);
await validateOpenvinoReplayEvidence(root, replay, manifest, report);

async function rejected(name, mutate) {
  const copiedManifest = structuredClone(manifest);
  const copiedReport = structuredClone(report);
  await mutate(copiedManifest, copiedReport);
  try {
    await validateOpenvinoEvidence(root, copiedManifest, copiedReport, frozen, fixtures);
  } catch {
    return name;
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
    const digest = (await import('node:crypto')).createHash('sha256').update(bytes).digest('hex');
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { path: raw, sha256: digest });
  }));
  cases.push(await rejected('raw tensor contains Infinity', async (_candidate, evidence) => {
    const source = resolve(root, evidence.rounds[0].fixtures[0].runs[0].raw.path);
    const bytes = Buffer.from(await readFile(source));
    bytes.writeFloatLE(Number.POSITIVE_INFINITY, 0);
    const raw = join(temporary, 'infinity.f32le');
    await writeFile(raw, bytes);
    const digest = (await import('node:crypto')).createHash('sha256').update(bytes).digest('hex');
    Object.assign(evidence.rounds[0].fixtures[0].runs[0].raw, { path: raw, sha256: digest });
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

console.log(JSON.stringify({ ok: true, negativeCases: cases, positiveCases: ['real OpenVINO record files and runtime evidence'] }));
