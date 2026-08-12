import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { decodedFacts, profileFacts, rawFacts, validateCacheArtifactTimeBoundary, validateProfileEvidenceClaims, validateTensorRtReport } from './validate_tensorrt_ep.mjs';

const root = resolve(import.meta.dirname, '../..');
const workspace = resolve(root, '.evidence/tensorrt/attacks');
const blocked = JSON.parse(await readFile(resolve(root, 'evidence/reports/tensorrt-ep-report.json')));
const contract = JSON.parse(await readFile(resolve(root, 'evidence/tensorrt/contract.json')));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const clone = (value) => structuredClone(value);
const productionBuildCommand = 'CARGO_TARGET_DIR=/build/tensorrt-target CARGO_INCREMENTAL=0 cargo build --locked --offline --release --manifest-path evidence/tooling/tensorrt-postprocess/Cargo.toml --bin rimeflow-tensorrt-postprocess';
await mkdir(workspace, { recursive: true });

function syntheticSuccess() {
  const report = clone(blocked);
  Object.assign(report, {
    schemaVersion: 2,
    status: 'host-inference-verified',
    runtimeExecuted: true,
    hostInferenceVerified: true,
    goldenExecuted: true,
    supported: false,
    task14Complete: false,
    openspecTask1_4Checked: false,
    failureStage: null,
    failureReason: null,
  });
  report.runner = { realNvidiaRunner: true, host: 'attack', os: 'Linux', arch: 'x86_64', gpu: { name: 'fake', uuid: 'fake' }, driver: 'fake', containerImageId: `sha256:${'a'.repeat(64)}` };
  report.versions = { onnxruntime: '1.22.0', tensorrt: '10.9.0.34', cuda: '12.8.0', cudnn: '9.7.0.66', python: '3.10.18', matchesContract: true };
  report.providers.available = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'];
  report.providers.session = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'];
  report.runtimeEvidence = {
    pid: 1,
    mapsArtifact: { path: 'missing/proc-self-maps.txt', bytes: 1, sha256: 'a'.repeat(64), mtimeNs: '1', ctimeNs: '1' },
    versionProbeArtifact: { path: 'missing/version-probes.json', bytes: 1, sha256: 'a'.repeat(64), mtimeNs: '1', ctimeNs: '1' },
    containerImageInspectArtifact: { path: 'missing/container-image-inspect.json', bytes: 1, sha256: 'a'.repeat(64), mtimeNs: '1', ctimeNs: '1' },
  };
  report.dynamicLibraries = [
    ['libonnxruntime_providers_tensorrt.so', 'onnxruntime'], ['libonnxruntime_providers_cuda.so', 'onnxruntime'],
    ['libnvinfer.so.10', 'tensorrt'], ['libnvonnxparser.so.10', 'tensorrt'], ['libcudart.so.12', 'cuda'], ['libcudnn.so.9', 'cudnn'],
  ].map(([requested, component], index) => ({ requested, component, componentVersion: report.versions[component], versionSource: 'fake', mappedPath: `/fake/${requested}`, realpath: `/fake/${requested}`, basename: requested, soname: requested, sourceDevice: 1, sourceInode: index + 1, mapLines: [], artifact: { path: `missing/${requested}`, bytes: 1, sha256: 'a'.repeat(64), mtimeNs: '1', ctimeNs: '1' } }));
  report.providers.profileArtifact = { path: 'missing/profile.json', bytes: 1, sha256: 'b'.repeat(64), mtimeNs: '1', ctimeNs: '1' };
  report.providers.profileExecutionEventCounts = { TensorrtExecutionProvider: 1, CUDAExecutionProvider: 0, CPUExecutionProvider: 0, unknown: 0 };
  report.providers.profileUniqueNodeCounts = { TensorrtExecutionProvider: 1, CUDAExecutionProvider: 0, CPUExecutionProvider: 0, unknown: 0 };
  report.providers.executionPlan = 'full';
  report.providers.fallback = { cudaExecutionEvents: 0, cpuExecutionEvents: 0, cudaUniqueNodes: 0, cpuUniqueNodes: 0, hidden: false };
  const transactionId = '11111111-1111-4111-8111-111111111111';
  const requestedOptions = { ...contract.providerOptions, trt_engine_cache_path: resolve(workspace, 'engine-cache', contract.canonicalModel.sha256, transactionId), trt_timing_cache_path: resolve(workspace, 'timing-cache', contract.canonicalModel.sha256, transactionId) };
  report.providers.options = { requested: requestedOptions, session: { TensorrtExecutionProvider: { ...requestedOptions } } };
  report.engineBuild = { attempted: true, succeeded: true, stage: 'profile-verified', runId: '22222222-2222-4222-8222-222222222222', transactionId, startedAt: '2026-01-01T00:00:00Z', startedAtEpochNs: '1', finishedAt: '2026-01-01T00:00:01Z', finishedAtEpochNs: '2', generatedAt: '2026-01-01T00:00:00Z', logArtifact: { path: 'missing/ort.log', bytes: 0, sha256: sha256(Buffer.alloc(0)), mtimeNs: '1', ctimeNs: '1' }, errors: { parserErrors: [], buildErrors: [], operatorErrors: [], profileErrors: [] } };
  const engine = { path: `engine-cache/${contract.canonicalModel.sha256}/${transactionId}/fake.engine`, bytes: 1, sha256: 'c'.repeat(64), mtimeNs: '1', ctimeNs: '1', runId: report.engineBuild.runId, transactionId, generatedAt: report.engineBuild.generatedAt };
  report.cache = { root: '.evidence/tensorrt/attacks', modelSha256: contract.canonicalModel.sha256, freshBuildRequired: true, discardedBeforeRun: [], before: [], after: [engine], generated: [engine] };
  report.webReference = { frozenReferenceSha256: 'd'.repeat(64), exporterManifest: { path: 'missing/manifest.json', bytes: 1, sha256: 'd'.repeat(64), mtimeNs: '1', ctimeNs: '1' } };
  report.productionPostprocess = { implementation: 'src/postprocess.rs', platformSpecificDecodeOrNmsAdded: false, sourceArtifacts: [], containerBinary: { path: 'missing/runner', bytes: 1, sha256: 'e'.repeat(64), mtimeNs: '1', ctimeNs: '1', containerImageId: report.runner.containerImageId }, build: { command: productionBuildCommand, rustBuilderIndex: contract.compatibility.rustBuilderIndex, rustBuilderAmd64Manifest: contract.compatibility.rustBuilderAmd64Manifest } };
  report.fixtures = contract.fixtures.map((id) => ({ id, image: { width: 640, height: 640 }, artifacts: { input: { path: `missing/${id}.input`, bytes: 4915200, sha256: 'f'.repeat(64), mtimeNs: '1', ctimeNs: '1', shape: [1, 3, 640, 640], dtype: 'float32', elementCount: 1228800 }, referenceRaw: { path: `missing/${id}.reference`, bytes: 2822400, sha256: 'f'.repeat(64), mtimeNs: '1', ctimeNs: '1', shape: [1, 84, 8400], dtype: 'float32', elementCount: 705600 }, actualRaw: { path: `missing/${id}.actual`, bytes: 2822400, sha256: 'f'.repeat(64), mtimeNs: '1', ctimeNs: '1', shape: [1, 84, 8400], dtype: 'float32', elementCount: 705600 }, decoded: { path: `missing/${id}.json`, bytes: 2, sha256: sha256(Buffer.from('[]')), mtimeNs: '1', ctimeNs: '1', dtype: 'json' } }, rawComparison: { passed: true }, decodedComparison: { passed: true } }));
  return report;
}

async function rejected(name, mutate, expected) {
  const report = syntheticSuccess();
  mutate(report);
  await assert.rejects(() => validateTensorRtReport(report, { evidenceRoot: workspace }), expected, name);
  return name;
}

await validateTensorRtReport(blocked);
const rejectedNames = [];
const forgedReportPath = resolve(workspace, 'synthetic-success.json');
await writeFile(forgedReportPath, `${JSON.stringify(syntheticSuccess(), null, 2)}\n`);
const forgedCli = spawnSync('node', ['evidence/scripts/validate_tensorrt_ep.mjs', forgedReportPath, workspace], { cwd: root, encoding: 'utf8' });
assert.notEqual(forgedCli.status, 0, '正式 TensorRT validator CLI 必须拒绝仅 passed=true 且 artifacts 不存在的报告');
assert.match(`${forgedCli.stdout}${forgedCli.stderr}`, /file missing/);
rejectedNames.push('仅 passed=true 且 artifacts 不存在（正式 CLI）');
const profile = profileFacts([
  { name: 'trt_kernel_time', cat: 'Node', args: { provider: 'TensorrtExecutionProvider', node_index: 1, op_name: 'TRT' } },
  { name: 'cuda_kernel_time', cat: 'Node', args: { provider: 'CUDAExecutionProvider', node_index: 2, op_name: 'CUDA' } },
  { name: 'cpu_kernel_time', cat: 'Node', args: { provider: 'CPUExecutionProvider', node_index: 3, op_name: 'CPU' } },
]);
assert.equal(profile.uniqueNodeCounts.TensorrtExecutionProvider, 1, 'profile unique TensorRT node recompute');
assert.equal(profile.executionEventCounts.CUDAExecutionProvider, 1, 'profile CUDA fallback recompute');
assert.throws(() => validateProfileEvidenceClaims(profile, { profileExecutionEventCounts: profile.executionEventCounts, profileUniqueNodeCounts: { ...profile.uniqueNodeCounts, TensorrtExecutionProvider: 99 }, executionPlan: profile.executionPlan, fallback: { cudaExecutionEvents: 1, cpuExecutionEvents: 1, cudaUniqueNodes: 1, cpuUniqueNodes: 1, hidden: false } }), /recomputed facts mismatch/, 'profile count forgery rejected by formal claim guard');
const cudaOnly = profileFacts([{ name: 'cuda_kernel_time', cat: 'Node', args: { provider: 'CUDAExecutionProvider', node_index: 2, op_name: 'CUDA' } }]);
assert.equal(cudaOnly.uniqueNodeCounts.TensorrtExecutionProvider, 0, 'CUDA cannot impersonate TensorRT');
assert.throws(() => validateProfileEvidenceClaims(cudaOnly, { profileExecutionEventCounts: cudaOnly.executionEventCounts, profileUniqueNodeCounts: cudaOnly.uniqueNodeCounts, executionPlan: cudaOnly.executionPlan, fallback: { cudaExecutionEvents: 1, cpuExecutionEvents: 0, cudaUniqueNodes: 1, cpuUniqueNodes: 0, hidden: false } }), /real TensorRT unique profile node/, 'CUDA-only profile rejected by formal claim guard');
rejectedNames.push('profile count 伪造', '仅 CUDA 冒充 TensorRT');
rejectedNames.push(await rejected('隐藏 fallback', (r) => { r.providers.fallback.hidden = true; }, /fallback/));
assert.throws(() => validateCacheArtifactTimeBoundary({ mtimeNs: '9', ctimeNs: '9' }, '10', '20'), /time boundary/, 'old engine rejected by formal time guard');
rejectedNames.push('旧 engine');
rejectedNames.push(await rejected('跨模型 SHA engine/cache', (r) => { r.cache.modelSha256 = '0'.repeat(64); }, /canonical-model TensorRT cache/));
rejectedNames.push(await rejected('library identity 漂移', (r) => { r.dynamicLibraries[1].realpath = r.dynamicLibraries[0].realpath; }, /impersonate/));
rejectedNames.push(await rejected('版本漂移', (r) => { r.versions.tensorrt = '10.9.0.35'; }, /version drift/));
rejectedNames.push(await rejected('engine build 失败却 verified', (r) => { r.engineBuild.succeeded = false; }, /fresh verified engine build/));
rejectedNames.push(await rejected('fake Rust runner', (r) => { r.productionPostprocess.build.command = 'true'; }, /build command/));
await assert.rejects(() => validateTensorRtReport(blocked, { trackedFiles: ['evidence/cache/fake.engine'] }), /不得由 Git 跟踪/);
await assert.rejects(() => validateTensorRtReport(blocked, { releaseFiles: ['/tmp/release/fake.plan'] }), /不得进入发布目录/);
rejectedNames.push('tracked engine/cache', '发布目录 engine/cache');
assert.throws(() => rawFacts(Buffer.alloc(4), Buffer.alloc(4), contract.frozenTolerances), /byte length/, 'raw truncation');
const zeroRaw = Buffer.alloc(2822400);
const nanRaw = Buffer.from(zeroRaw); nanRaw.writeFloatLE(Number.NaN, 0);
const infinityRaw = Buffer.from(zeroRaw); infinityRaw.writeFloatLE(Number.POSITIVE_INFINITY, 0);
const outOfToleranceRaw = Buffer.from(zeroRaw); outOfToleranceRaw.writeFloatLE(0.001, 0);
assert.equal(rawFacts(nanRaw, zeroRaw, contract.frozenTolerances).passed, false, 'raw NaN');
assert.equal(rawFacts(infinityRaw, zeroRaw, contract.frozenTolerances).passed, false, 'raw Infinity');
assert.equal(rawFacts(outOfToleranceRaw, zeroRaw, contract.frozenTolerances).mismatchCount, 1, 'raw finite out of tolerance');
const decoded = decodedFacts([{ classId: 1, score: 0.9, bbox: [0, 0, 10, 10] }], [{ classId: 1, score: 0.9, bbox: [0, 0, 10, 10] }], contract.frozenTolerances);
assert.equal(decoded.passed, true);
assert.equal(decodedFacts([{ classId: 2, score: 0.9, bbox: [0, 0, 10, 10] }], [{ classId: 1, score: 0.9, bbox: [0, 0, 10, 10] }], contract.frozenTolerances).passed, false, 'decoded tamper');
rejectedNames.push('raw 截断', 'raw NaN', 'raw Infinity', 'raw 有限越界', 'decoded 篡改');
rejectedNames.push(await rejected('supported=true', (r) => { r.supported = true; }, /cannot claim platform support/));
rejectedNames.push(await rejected('task14Complete=true', (r) => { r.task14Complete = true; }, /cannot claim platform support/));
rejectedNames.push(await rejected('OpenSpec 1.4 checked', (r) => { r.openspecTask1_4Checked = true; }, /OpenSpec/));

console.log(JSON.stringify({ ok: true, blockedReportAccepted: true, syntheticSuccessRejected: true, attacksRejected: rejectedNames.length, rejectedNames }));
