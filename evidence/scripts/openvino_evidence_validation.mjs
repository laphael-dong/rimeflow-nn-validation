import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const MODEL_SHA = '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad';
const INPUT = { dtype: 'float32', elementCount: 1228800, name: 'images', shape: [1, 3, 640, 640] };
const OUTPUT = { dtype: 'float32', elementCount: 705600, name: 'output0', shape: [1, 84, 8400] };
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  return value;
};
const same = (left, right) => JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
const fail = (message) => { throw new Error(`OpenVINO evidence: ${message}`); };

function profileFacts(events) {
  const counts = { OpenVINOExecutionProvider: 0, CPUExecutionProvider: 0, unknown: 0 };
  const unique = { OpenVINOExecutionProvider: new Set(), CPUExecutionProvider: new Set(), unknown: new Set() };
  for (const event of events.filter((item) => item.cat === 'Node')) {
    const provider = event.args?.provider;
    const key = Object.hasOwn(counts, provider) ? provider : 'unknown';
    counts[key] += 1;
    unique[key].add(event.name);
  }
  const uniqueCounts = Object.fromEntries(Object.entries(unique).map(([key, values]) => [key, values.size]));
  let executionPlan = 'unknown';
  if (uniqueCounts.OpenVINOExecutionProvider > 0) executionPlan = uniqueCounts.CPUExecutionProvider > 0 ? 'partitioned' : 'full';
  return { counts, executionPlan, uniqueCounts };
}

async function validateRaw(root, raw, fixture, frozenFixture, tolerances, temporary) {
  const rawPath = resolve(root, raw.raw.path);
  const bytes = await readFile(rawPath);
  if (bytes.length !== 705600 * 4 || sha256(bytes) !== raw.raw.sha256) fail(`${fixture.id} raw file identity`);
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  let finiteCount = 0;
  for (const value of values) if (Number.isFinite(value)) finiteCount += 1;
  if (finiteCount !== 705600 || raw.rawComparison.finiteCount !== finiteCount || raw.rawComparison.elementCount !== 705600) fail(`${fixture.id} raw non-finite/element count`);
  if (!raw.rawComparison.allClose || !raw.passed) fail(`${fixture.id} raw frozen comparison`);
  const decodedPath = join(temporary, `${fixture.id}-${raw.repeat}.json`);
  const rust = spawnSync(
    resolve(root, 'evidence/tooling/raw-golden/target/debug/rimeflow-raw-golden'),
    [rawPath, String(fixture.width), String(fixture.height), decodedPath],
    { cwd: root, encoding: 'utf8' },
  );
  if (rust.status !== 0) fail(`${fixture.id} production Rust decode failed: ${rust.stderr}`);
  const decoded = JSON.parse(await readFile(decodedPath, 'utf8'));
  if (!same(decoded, raw.decoded)) fail(`${fixture.id} report decoded output differs from real Rust harness`);
  if (!raw.decodedComparison.passed || raw.decodedComparison.actualCount !== decoded.length || raw.decodedComparison.expectedCount !== frozenFixture.runs[0].decoded.length) fail(`${fixture.id} decoded frozen comparison`);
  for (const comparison of raw.decodedComparison.comparisons) {
    if (!comparison.classEqual || comparison.confidenceAbsoluteDifference > tolerances.confidenceAbsolute || comparison.bboxIou < tolerances.boxIouMinimum || comparison.bboxMaxAbsoluteDifference > tolerances.decodedBoxAbsolute) fail(`${fixture.id} decoded tolerance`);
  }
}

export async function validateOpenvinoEvidence(root, manifest, report, frozen, fixtures) {
  const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
  if (modelBytes.length !== 12851098 || sha256(modelBytes) !== MODEL_SHA) fail('canonical ONNX identity');
  if (!manifest.artifact.noConversion || !manifest.artifact.sameCanonicalFileLoaded || manifest.artifact.path !== 'models/yolov8n.onnx' || manifest.artifact.bytes !== modelBytes.length || manifest.artifact.sha256 !== sha256(modelBytes)) fail('no-conversion artifact contract');
  if (!same(report.modelBefore, report.modelAfter) || report.modelBefore.sha256 !== MODEL_SHA || report.modelBefore.bytes !== modelBytes.length || !report.noConversion) fail('model before/after evidence');
  const lockBytes = await readFile(resolve(root, manifest.toolchain.lock.path));
  if (lockBytes.length !== manifest.toolchain.lock.bytes || sha256(lockBytes) !== manifest.toolchain.lock.sha256) fail('toolchain lock identity');
  const lockLines = lockBytes.toString().split('\n').filter((line) => line && !line.startsWith('#') && !line.startsWith('--'));
  if (lockLines.length !== 7 || lockLines.some((line) => !/^[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}$/.test(line))) fail('complete hashed dependency lock');
  if (manifest.toolchain.wheels.length !== 7 || manifest.toolchain.wheels.some((wheel) => !wheel.source.startsWith('https://files.pythonhosted.org/') || !/^[0-9a-f]{64}$/.test(wheel.sha256) || !wheel.license)) fail('wheel source/hash/license metadata');
  if (manifest.runtime.onnxruntimeOpenvino !== '1.24.1' || manifest.runtime.openvino.runtime.buildNumber !== '2025.4.1-0-test' || manifest.runtime.numpy !== '2.5.2' || manifest.runtime.python !== '3.12.3' || manifest.runtime.pip !== '24.0') fail('runtime exact versions');
  if (!manifest.runtime.buildInfo.includes('git-commit-id=b5963e82c8') || manifest.runtime.device !== 'CPU-OPENVINO_CPU' || !manifest.runtime.openvino.availableDevices.includes('CPU') || manifest.runtime.openvino.requestedDevice !== 'CPU') fail('runtime build/device introspection');
  const capi = resolve(root, '.evidence/openvino/venv/lib/python3.12/site-packages/onnxruntime/capi');
  const requiredLibraries = new Set(['libonnxruntime_providers_openvino.so', 'libonnxruntime_providers_shared.so', 'libopenvino.so.2541', 'libopenvino_c.so', 'libopenvino_intel_cpu_plugin.so', 'libopenvino_onnx_frontend.so.2541', 'onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so']);
  if (manifest.runtime.libraries.length !== requiredLibraries.size) fail('loaded library count');
  for (const library of manifest.runtime.libraries) {
    if (!requiredLibraries.delete(library.name) || !library.mappedByProcess) fail(`loaded library declaration: ${library.name}`);
    const path = resolve(capi, basename(library.path));
    if (library.actualPath !== path || !['1.24.1', '2025.4.1'].includes(library.componentVersion)) fail(`loaded library path/version: ${library.name}`);
    const bytes = await readFile(path);
    if (bytes.length !== library.bytes || sha256(bytes) !== library.sha256) fail(`loaded library identity: ${library.name}`);
  }
  if (requiredLibraries.size) fail('missing loaded library identity');
  if (!same(manifest.ioContract.inputs, [INPUT]) || !same(manifest.ioContract.outputs, [OUTPUT]) || manifest.ioContract.outputLayout !== 'N_ATTRIBUTES_ANCHORS') fail('manifest I/O contract');
  if (manifest.provider.requested !== 'OpenVINOExecutionProvider' || manifest.provider.configured[0] !== 'OpenVINOExecutionProvider' || !manifest.provider.fallbackVisible) fail('provider request/fallback declaration');
  if (report.mode !== 'record' || report.rounds.length !== 2 || report.executionPlan !== 'full' || !same(report.tolerances, frozen.tolerances)) fail('record/tolerance/execution plan');
  if (report.status.state !== 'host-inference-verified' || !report.status.artifactVerified || !report.status.hostInferenceVerified || report.status.supported || report.status.task14Complete || report.status.adapterImplemented || report.status.packagingVerified || report.status.performanceVerified || report.status.targetPlatformClosed) fail('status closure semantics');
  if (report.productionPostprocess.implementation !== 'src/postprocess.rs' || report.productionPostprocess.platformSpecificImplementationAdded) fail('single production postprocess');
  const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-validator-'));
  try {
    for (const round of report.rounds) {
      if (!round.availableProviders.includes('OpenVINOExecutionProvider') || round.sessionProviders[0] !== 'OpenVINOExecutionProvider' || !same(round.inputs, [INPUT]) || !same(round.outputs, [OUTPUT])) fail(`round ${round.round} provider/I/O`);
      const profilePath = resolve(round.profile.path);
      const profileBytes = await readFile(profilePath);
      if (profileBytes.length !== round.profile.bytes || sha256(profileBytes) !== round.profile.sha256) fail(`round ${round.round} real profile identity`);
      const facts = profileFacts(JSON.parse(profileBytes));
      if (!same(facts.counts, round.profile.executionEventCounts) || !same(facts.uniqueCounts, round.profile.uniqueNodeCounts) || facts.executionPlan !== round.profile.executionPlan || facts.uniqueCounts.OpenVINOExecutionProvider < 1) fail(`round ${round.round} real profile provider counts`);
      if (round.fixtures.length !== 5) fail(`round ${round.round} fixture count`);
      for (const item of round.fixtures) {
        const fixture = fixtures.images.find((candidate) => candidate.id === item.id);
        const frozenFixture = frozen.fixtures.find((candidate) => candidate.id === item.id);
        if (!fixture || !frozenFixture || item.runs.length !== 2 || !item.deterministic || item.runs[0].raw.sha256 !== item.runs[1].raw.sha256 || !same(item.runs[0].decoded, item.runs[1].decoded)) fail(`round ${round.round} ${item.id} determinism`);
        for (const run of item.runs) await validateRaw(root, run, fixture, frozenFixture, frozen.tolerances, temporary);
      }
    }
    for (let index = 0; index < report.rounds[0].fixtures.length; index += 1) {
      const left = report.rounds[0].fixtures[index];
      const right = report.rounds[1].fixtures[index];
      if (left.id !== right.id || left.runs[0].raw.sha256 !== right.runs[0].raw.sha256 || !same(left.runs[0].decoded, right.runs[0].decoded)) fail(`${left.id} cross-round replay determinism`);
    }
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
  return { executionPlan: report.executionPlan, fixtureCount: 5, profileOpenvinoNodes: report.rounds[0].profile.uniqueNodeCounts.OpenVINOExecutionProvider, profileCpuNodes: report.rounds[0].profile.uniqueNodeCounts.CPUExecutionProvider };
}

export async function validateOpenvinoReplayEvidence(root, replay, manifest, report) {
  if (replay.schemaVersion !== 1 || replay.mode !== 'replay' || replay.recordDigest !== report.recordDigest || replay.rounds.length !== 2) fail('ordinary replay identity');
  for (const [key, path] of Object.entries({ manifest: 'evidence/conversions/openvino-ep-manifest.json', report: 'evidence/reports/openvino-ep-report.json' })) {
    const current = await readFile(resolve(root, path));
    const preservation = replay.trackedEvidence[key];
    if (!preservation?.unchanged || preservation.before.path !== path || preservation.after.path !== path || preservation.before.bytes !== current.length || preservation.after.bytes !== current.length || preservation.before.sha256 !== sha256(current) || preservation.after.sha256 !== sha256(current)) fail(`ordinary replay changed tracked ${key}`);
  }
  for (const round of replay.rounds) {
    const facts = profileFacts(JSON.parse(await readFile(resolve(round.profile.path), 'utf8')));
    if (facts.uniqueCounts.OpenVINOExecutionProvider < 1 || facts.executionPlan !== round.profile.executionPlan) fail(`ordinary replay round ${round.round} profile`);
  }
  if (manifest.status.supported || manifest.status.task14Complete) fail('ordinary replay overclaimed completion');
  return { recordDigest: replay.recordDigest, trackedEvidenceUnchanged: true };
}

export function summarizeOpenvinoForConversion(manifest, report) {
  return {
    artifact: manifest.artifact,
    artifactVerified: manifest.status.artifactVerified,
    attempt: {
      availableProviders: report.rounds[0].availableProviders,
      command: report.rounds[0].command,
      executionPlan: report.executionPlan,
      goldenPassed: report.rounds.every((round) => round.fixtures.every((fixture) => fixture.runs.every((run) => run.passed))),
      inferenceExecuted: true,
      profileNodeCounts: report.rounds[0].profile.uniqueNodeCounts,
      sessionProviders: report.rounds[0].sessionProviders,
    },
    conclusion: '官方 onnxruntime-openvino wheel 在 Linux x86_64 host 上实际执行 OpenVINO 图节点并通过五 fixture 冻结 golden；尚无 adapter、性能、打包和任务 1.4 全平台闭环。',
    format: 'onnx',
    hostInferenceVerified: manifest.status.hostInferenceVerified,
    ioChanges: 'none',
    license: manifest.usageScope.licenseAndRedistribution,
    nmsResponsibility: 'operator',
    platform: 'linux-x86_64-openvino',
    quantization: 'none / float32',
    state: manifest.status.state,
    supported: manifest.status.supported,
    task14Complete: manifest.status.task14Complete,
    tool: { name: 'onnxruntime-openvino', openvinoVersion: manifest.runtime.openvino.runtime.buildNumber, version: manifest.runtime.onnxruntimeOpenvino },
  };
}

export { profileFacts };
