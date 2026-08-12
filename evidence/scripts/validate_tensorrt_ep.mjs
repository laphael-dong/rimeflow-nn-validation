import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { lstat, mkdtemp, readFile, readdir, realpath, rm, stat } from 'node:fs/promises';
import { basename, dirname, extname, isAbsolute, join, normalize, resolve, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const contractPath = resolve(root, 'evidence/tensorrt/contract.json');
const defaultReportPath = resolve(root, 'evidence/reports/tensorrt-ep-report.json');
const fail = (message) => { throw new Error(message); };
const json = async (path) => JSON.parse(await readFile(path, 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const cacheExtensions = new Set(['.engine', '.profile', '.timing', '.cache', '.plan']);
const providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider', 'unknown'];
const requiredLibraries = [
  ['libonnxruntime_providers_tensorrt.so', 'onnxruntime'],
  ['libonnxruntime_providers_cuda.so', 'onnxruntime'],
  ['libnvinfer.so.10', 'tensorrt'],
  ['libnvonnxparser.so.10', 'tensorrt'],
  ['libcudart.so.12', 'cuda'],
  ['libcudnn.so.9', 'cudnn'],
];
const output = { shape: [1, 84, 8400], dtype: 'float32', elementCount: 705600, bytes: 2822400 };
const productionBuildCommand = 'CARGO_TARGET_DIR=/build/tensorrt-target CARGO_INCREMENTAL=0 cargo build --locked --offline --release --manifest-path evidence/tooling/tensorrt-postprocess/Cargo.toml --bin rimeflow-tensorrt-postprocess';

function inside(path, parent) {
  const child = resolve(path);
  const base = resolve(parent);
  return child === base || child.startsWith(`${base}${sep}`);
}

async function listFiles(path) {
  const found = [];
  try {
    for (const entry of await readdir(path, { withFileTypes: true })) {
      const child = resolve(path, entry.name);
      if (entry.isDirectory()) found.push(...await listFiles(child));
      else if (entry.isFile()) found.push(child);
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  return found;
}

async function diskArtifacts(paths, evidenceRoot) {
  const artifacts = [];
  for (const path of paths.sort()) {
    const link = await lstat(path);
    if (link.isSymbolicLink() || !link.isFile()) fail(`cache artifact must be a regular non-symlink file: ${path}`);
    const canonical = await realpath(path);
    if (!inside(canonical, await realpath(evidenceRoot))) fail(`cache artifact escapes evidence root: ${path}`);
    const bytes = await readFile(path);
    const metadata = await stat(path, { bigint: true });
    artifacts.push({ path: path.slice(resolve(evidenceRoot).length + 1), bytes: bytes.length, sha256: sha256(bytes), mtimeNs: String(metadata.mtimeNs), ctimeNs: String(metadata.ctimeNs) });
  }
  return artifacts;
}

function validateTrackedArtifacts(trackedFiles) {
  for (const path of trackedFiles) {
    const lower = path.toLowerCase();
    if (cacheExtensions.has(extname(lower)) || /(^|\/)(engine-cache|timing-cache|tensorrt-cache)(\/|$)/.test(lower)) fail(`TensorRT engine/cache 不得由 Git 跟踪: ${path}`);
    if (/(^|\/)(release|dist|package|artifacts?)(\/|$)/.test(lower) && /(tensorrt|\.engine$|\.plan$|\.timing$|\.profile$)/.test(lower)) fail(`TensorRT engine/cache 不得进入发布目录: ${path}`);
  }
}

function artifactPath(artifact, evidenceRoot, label) {
  if (!artifact?.path || isAbsolute(artifact.path) || normalize(artifact.path).startsWith(`..${sep}`)) fail(`${label} path escapes evidence root`);
  const path = resolve(evidenceRoot, artifact.path);
  if (!inside(path, evidenceRoot)) fail(`${label} path escapes evidence root`);
  return path;
}

async function verifyArtifact(artifact, evidenceRoot, label, allowEmpty = false) {
  const path = artifactPath(artifact, evidenceRoot, label);
  const link = await lstat(path).catch(() => fail(`${label} file missing`));
  if (link.isSymbolicLink() || !link.isFile() || link.nlink !== 1) fail(`${label} must be a unique regular non-symlink file`);
  const canonical = await realpath(path);
  if (!inside(canonical, await realpath(evidenceRoot))) fail(`${label} symlink escape`);
  const bytes = await readFile(path).catch(() => fail(`${label} file missing`));
  if ((!allowEmpty && bytes.length === 0) || bytes.length !== artifact.bytes || sha256(bytes) !== artifact.sha256) fail(`${label} bytes/SHA mismatch`);
  const metadata = await stat(path, { bigint: true });
  if (artifact.mtimeNs !== undefined && (String(metadata.mtimeNs) !== String(artifact.mtimeNs) || String(metadata.ctimeNs) !== String(artifact.ctimeNs))) fail(`${label} mtime/ctime mismatch`);
  return { path, bytes, metadata };
}

export function profileFacts(events) {
  if (!Array.isArray(events)) fail('ORT profile JSON must be an array');
  const executionEventCounts = Object.fromEntries(providers.map((provider) => [provider, 0]));
  const unique = Object.fromEntries(providers.map((provider) => [provider, new Set()]));
  const nodes = [];
  for (const event of events) {
    const provider = event?.args?.provider;
    if (!provider || !String(event.name ?? '').endsWith('_kernel_time')) continue;
    const key = providers.includes(provider) ? provider : 'unknown';
    const nodeIdentity = `${event.args?.node_index ?? ''}:${event.args?.op_name ?? ''}:${String(event.name).replace(/_kernel_time$/, '')}`;
    executionEventCounts[key] += 1;
    unique[key].add(nodeIdentity);
    nodes.push({ name: event.name, opName: event.args?.op_name ?? null, provider });
  }
  const uniqueNodeCounts = Object.fromEntries(providers.map((provider) => [provider, unique[provider].size]));
  const accelerated = uniqueNodeCounts.TensorrtExecutionProvider;
  const fallback = uniqueNodeCounts.CUDAExecutionProvider + uniqueNodeCounts.CPUExecutionProvider;
  const executionPlan = accelerated > 0 ? (fallback > 0 ? 'partitioned' : 'full') : 'unknown';
  return { executionEventCounts, uniqueNodeCounts, executionPlan, nodes };
}
export const recomputeTensorRtProfile = profileFacts;

export function validateProfileEvidenceClaims(facts, reported) {
  if (!same(facts.executionEventCounts, reported.profileExecutionEventCounts) || !same(facts.uniqueNodeCounts, reported.profileUniqueNodeCounts) || facts.executionPlan !== reported.executionPlan) fail('ORT profile independently recomputed facts mismatch');
  if (facts.uniqueNodeCounts.TensorrtExecutionProvider < 1) fail('real TensorRT unique profile node required');
  const expectedFallback = { cudaExecutionEvents: facts.executionEventCounts.CUDAExecutionProvider, cpuExecutionEvents: facts.executionEventCounts.CPUExecutionProvider, cudaUniqueNodes: facts.uniqueNodeCounts.CUDAExecutionProvider, cpuUniqueNodes: facts.uniqueNodeCounts.CPUExecutionProvider, hidden: false };
  if (!same(expectedFallback, reported.fallback)) fail('CUDA/CPU fallback must be recomputed from profile');
}

export function validateCacheArtifactTimeBoundary(artifact, startedNs, finishedNs) {
  const started = BigInt(startedNs);
  const finished = BigInt(finishedNs);
  const mtime = BigInt(artifact.mtimeNs);
  const ctime = BigInt(artifact.ctimeNs);
  if (started <= 0n || finished < started || mtime < started || mtime > finished || ctime < started || ctime > finished) fail('generated cache outside run time boundary');
}

function readFloat32Le(bytes, label) {
  if (bytes.length !== output.bytes) fail(`${label} byte length`);
  const values = new Float64Array(output.elementCount);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) values[index] = view.getFloat32(index * 4, true);
  return values;
}

export function rawFacts(actualBytes, referenceBytes, tolerances) {
  const actual = readFloat32Le(actualBytes, 'TensorRT raw');
  const reference = readFloat32Le(referenceBytes, 'Web reference raw');
  let finiteCount = 0;
  let referenceFiniteCount = 0;
  let mismatchCount = 0;
  let differenceSum = 0;
  let maxAbsoluteDifference = -1;
  let maxIndex = 0;
  let nearZeroElementCount = 0;
  let nearZeroMaxAbsoluteDifference = 0;
  let nearZeroMismatchCount = 0;
  for (let index = 0; index < output.elementCount; index += 1) {
    const candidate = actual[index];
    const expected = reference[index];
    if (Number.isFinite(candidate)) finiteCount += 1;
    if (Number.isFinite(expected)) referenceFiniteCount += 1;
    const referenceNearZero = Number.isFinite(expected) && Math.abs(expected) < 1e-6;
    if (referenceNearZero) nearZeroElementCount += 1;
    if (!Number.isFinite(candidate) || !Number.isFinite(expected)) { mismatchCount += 1; if (referenceNearZero) nearZeroMismatchCount += 1; continue; }
    const difference = Math.abs(candidate - expected);
    const tolerance = tolerances.rawTensorAbsolute + tolerances.rawTensorRelative * Math.abs(expected);
    differenceSum += difference;
    if (difference > maxAbsoluteDifference) { maxAbsoluteDifference = difference; maxIndex = index; }
    if (difference > tolerance) mismatchCount += 1;
    if (referenceNearZero) {
      nearZeroMaxAbsoluteDifference = Math.max(nearZeroMaxAbsoluteDifference, difference);
      if (difference > tolerance) nearZeroMismatchCount += 1;
    }
  }
  const expectedAtMax = reference[maxIndex];
  return {
    passed: finiteCount === output.elementCount && referenceFiniteCount === output.elementCount && mismatchCount === 0,
    elementCount: output.elementCount,
    finiteCount,
    referenceFiniteCount,
    mismatchCount,
    maxAbsoluteDifference,
    meanAbsoluteDifference: differenceSum / output.elementCount,
    maxAbsoluteDifferenceLocation: {
      flatIndex: maxIndex,
      attribute: Math.floor(maxIndex / output.shape[2]) % output.shape[1],
      anchor: maxIndex % output.shape[2],
      actual: actual[maxIndex],
      reference: expectedAtMax,
      tolerance: tolerances.rawTensorAbsolute + tolerances.rawTensorRelative * Math.abs(expectedAtMax),
    },
    nearZero: { referenceAbsoluteThreshold: 1e-6, elementCount: nearZeroElementCount, maxAbsoluteDifference: nearZeroMaxAbsoluteDifference, mismatchCount: nearZeroMismatchCount },
  };
}
export const recomputeTensorRtRaw = rawFacts;

function bboxIou(left, right) {
  const intersection = Math.max(0, Math.min(left[2], right[2]) - Math.max(left[0], right[0])) * Math.max(0, Math.min(left[3], right[3]) - Math.max(left[1], right[1]));
  const union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection;
  return union <= 0 ? 0 : intersection / union;
}

export function decodedFacts(actual, expected, tolerances) {
  if (!Array.isArray(actual) || !Array.isArray(expected)) fail('decoded output/reference must be arrays');
  const detections = [];
  for (let index = 0; index < Math.min(actual.length, expected.length); index += 1) {
    const got = actual[index]; const want = expected[index];
    if (!got || !want || !Number.isInteger(got.classId) || !Number.isInteger(want.classId) || !Array.isArray(got.bbox) || !Array.isArray(want.bbox) || got.bbox.length !== 4 || want.bbox.length !== 4) fail('decoded detection structure invalid');
    const numeric = [got.score, want.score, ...got.bbox, ...want.bbox].map(Number);
    if (numeric.some((value) => !Number.isFinite(value))) fail('decoded detection contains non-finite value');
    const confidenceAbsoluteDifference = Math.abs(Number(got.score) - Number(want.score));
    const bboxMaxAbsoluteDifference = Math.max(...got.bbox.map((value, coordinate) => Math.abs(Number(value) - Number(want.bbox[coordinate]))));
    const iou = bboxIou(got.bbox, want.bbox);
    const classExact = got.classId === want.classId;
    detections.push({ classExact, confidenceAbsoluteDifference, bboxMaxAbsoluteDifference, bboxIou: iou, passed: classExact && confidenceAbsoluteDifference <= tolerances.confidenceAbsolute && bboxMaxAbsoluteDifference <= tolerances.decodedBoxAbsolute && iou >= tolerances.boxIouMinimum });
  }
  return { passed: actual.length === expected.length && detections.every((item) => item.passed), actualCount: actual.length, expectedCount: expected.length, detections };
}
export const recomputeTensorRtDecoded = decodedFacts;

function compareNumeric(actual, reported, label) {
  if (typeof actual === 'number' || typeof reported === 'number') {
    if (typeof actual !== 'number' || typeof reported !== 'number' || !Number.isFinite(actual) || !Number.isFinite(reported) || Math.abs(actual - reported) > Math.max(1e-12, Math.abs(actual) * 1e-10)) fail(`${label} numeric mismatch`);
  } else if (Array.isArray(actual) || Array.isArray(reported)) {
    if (!Array.isArray(actual) || !Array.isArray(reported) || actual.length !== reported.length) fail(`${label} array mismatch`);
    actual.forEach((value, index) => compareNumeric(value, reported[index], `${label}[${index}]`));
  } else if (actual && typeof actual === 'object') {
    if (!reported || typeof reported !== 'object' || !same(Object.keys(actual).sort(), Object.keys(reported).sort())) fail(`${label} fields mismatch`);
    for (const [key, value] of Object.entries(actual)) compareNumeric(value, reported[key], `${label}.${key}`);
  } else if (actual !== reported) fail(`${label} mismatch`);
}

function logFacts(text) {
  const lines = text.split('\n').filter(Boolean);
  const pick = (tokens) => lines.filter((line) => tokens.some((token) => line.toLowerCase().includes(token)));
  return {
    parserErrors: pick(['parser error', 'failed to parse', 'modelparser', 'onnxparser']),
    buildErrors: pick(['builder error', 'build engine', 'engine build', 'tactic', 'workspace size']),
    operatorErrors: pick(['unsupported operator', 'unsupported node', 'no importer registered', 'kernel not found']),
    profileErrors: pick(['profile parse', 'profiling file', 'end_profiling']),
  };
}

async function sourceIdentity(sourceRoot, path, commit) {
  const bytes = await readFile(resolve(sourceRoot, path));
  const blob = spawnSync('git', ['show', `${commit}:${path}`], { cwd: sourceRoot, encoding: null, maxBuffer: 10 * 1024 * 1024 });
  const oid = spawnSync('git', ['rev-parse', `${commit}:${path}`], { cwd: sourceRoot, encoding: 'utf8' });
  if (blob.status !== 0 || oid.status !== 0 || !blob.stdout.equals(bytes)) fail(`production Rust source differs from evidence source commit: ${path}`);
  return { path, bytes: bytes.length, sha256: sha256(bytes), gitBlobOid: oid.stdout.trim() };
}

async function buildRustRunner(sourceRoot, reported, evidenceSourceCommit) {
  const paths = ['evidence/tooling/tensorrt-postprocess/Cargo.toml', 'evidence/tooling/tensorrt-postprocess/Cargo.lock', 'evidence/tooling/tensorrt-postprocess/main.rs', 'src/postprocess.rs', 'evidence/tooling/tensorrt/Dockerfile'];
  const identities = await Promise.all(paths.map((path) => sourceIdentity(sourceRoot, path, evidenceSourceCommit)));
  if (!same(identities, reported.sourceArtifacts)) fail('production Rust source identity mismatch');
  if (reported.build?.command !== productionBuildCommand) fail('production Rust build command drift');
  const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-tensorrt-rust-'));
  const target = join(temporary, 'target');
  const build = spawnSync('cargo', ['build', '--locked', '--offline', '--release', '--manifest-path', 'evidence/tooling/tensorrt-postprocess/Cargo.toml', '--bin', 'rimeflow-tensorrt-postprocess'], { cwd: sourceRoot, encoding: 'utf8', env: { ...process.env, CARGO_TARGET_DIR: target, CARGO_INCREMENTAL: '0' } });
  if (build.status !== 0) { await rm(temporary, { recursive: true, force: true }); fail(`fresh production Rust build failed: ${build.stderr}`); }
  const runner = join(target, 'release/rimeflow-tensorrt-postprocess');
  const bytes = await readFile(runner);
  return { runner, bytes, temporary };
}

async function verifyLibraries(report, evidenceRoot, contract) {
  if (report.dynamicLibraries.length !== requiredLibraries.length) fail('required runtime library identities missing');
  const seenRealpaths = new Set();
  const seenArtifacts = new Set();
  for (let index = 0; index < requiredLibraries.length; index += 1) {
    const [requiredName, component] = requiredLibraries[index];
    const library = report.dynamicLibraries[index];
    const expectedVersion = component === 'driver' ? report.runner.driver : contract.compatibility[component];
    if (library.requested !== requiredName || library.component !== component || library.componentVersion !== expectedVersion || !library.versionSource) fail(`runtime library declaration: ${requiredName}`);
    if (seenRealpaths.has(library.realpath) || seenArtifacts.has(library.artifact?.path)) fail('one runtime library cannot impersonate multiple components');
    seenRealpaths.add(library.realpath); seenArtifacts.add(library.artifact?.path);
  }
  const maps = await verifyArtifact(report.runtimeEvidence.mapsArtifact, evidenceRoot, 'proc maps');
  const mapsText = maps.bytes.toString('utf8');
  for (let index = 0; index < requiredLibraries.length; index += 1) {
    const [requiredName] = requiredLibraries[index];
    const library = report.dynamicLibraries[index];
    const artifact = await verifyArtifact(library.artifact, evidenceRoot, `runtime library ${requiredName}`);
    const exactMapLines = mapsText.split('\n').filter((line) => line.split(/\s+/, 6)[5]?.replace(/ \(deleted\)$/, '') === library.mappedPath);
    if (!exactMapLines.length || !same(exactMapLines, library.mapLines) || basename(library.realpath) !== library.basename || !(library.basename === requiredName || library.basename.startsWith(`${requiredName}.`))) fail(`runtime maps/basename mismatch: ${requiredName}`);
    const mapIdentity = exactMapLines[0].trim().split(/\s+/);
    if (Number.parseInt(mapIdentity[4], 10) !== library.sourceInode) fail(`runtime map inode mismatch: ${requiredName}`);
    const [majorHex, minorHex] = mapIdentity[3].split(':');
    const major = BigInt(`0x${majorHex}`); const minor = BigInt(`0x${minorHex}`);
    const mappedDevice = (minor & 0xffn) | ((major & 0xfffn) << 8n) | ((minor & ~0xffn) << 12n) | ((major & ~0xfffn) << 32n);
    if (mappedDevice !== BigInt(library.sourceDevice)) fail(`runtime map device mismatch: ${requiredName}`);
    if (!inside(artifact.path, resolve(evidenceRoot, 'runtime-libraries'))) fail(`runtime library outside captured root: ${requiredName}`);
    if (artifact.bytes[0] !== 0x7f || artifact.bytes[1] !== 0x45 || artifact.bytes[2] !== 0x4c || artifact.bytes[3] !== 0x46 || artifact.bytes[4] !== 2 || artifact.bytes[5] !== 1 || artifact.bytes.readUInt16LE(18) !== 62) fail(`runtime library is not ELF64 little-endian x86_64: ${requiredName}`);
    const elf = spawnSync('readelf', ['-h', '-d', artifact.path], { encoding: 'utf8' });
    const soname = elf.stdout.match(/Library soname: \[([^\]]+)\]/)?.[1] ?? null;
    if (elf.status !== 0 || soname !== library.soname || !soname || !(soname === requiredName || soname.startsWith(`${requiredName}.`))) fail(`runtime library SONAME mismatch: ${requiredName}`);
    if (!elf.stdout.includes('Class:                             ELF64') || !elf.stdout.includes('Machine:                           Advanced Micro Devices X86-64')) fail(`runtime library ELF identity mismatch: ${requiredName}`);
  }
  const probes = await verifyArtifact(report.runtimeEvidence.versionProbeArtifact, evidenceRoot, 'runtime version probes');
  const probeJson = JSON.parse(probes.bytes.toString('utf8'));
  const smi = probeJson.commands?.nvidiaSmi;
  if (smi?.exitCode !== 0 || !String(smi.stdout ?? '').trim()) fail('nvidia-smi runtime probe missing');
  const gpu = String(smi.stdout).trim().split('\n')[0].split(',').map((value) => value.trim());
  if (gpu.length !== 4 || gpu[0] !== report.runner.gpu.name || gpu[1] !== report.runner.gpu.uuid || gpu[2] !== report.runner.driver || gpu[3] !== report.runner.gpu.computeCapability) fail('nvidia-smi runner identity mismatch');
  if (probeJson.commands?.dpkgQuery?.exitCode !== 0 || probeJson.commands?.nvcc?.exitCode !== 0) fail('TensorRT/CUDA/cuDNN runtime version probe failed');
  if (!same(probeJson.commands?.ort?.availableProviders, report.providers.available)) fail('ORT available providers probe mismatch');
  const packages = Object.fromEntries(String(probeJson.commands?.dpkgQuery?.stdout ?? '').split('\n').filter((line) => line.includes('=')).map((line) => line.split('=', 2)));
  const independentlyResolved = {
    onnxruntime: probeJson.commands?.ort?.version ?? null,
    tensorrt: String(packages.libnvinfer10 ?? '').split('-', 1)[0] || null,
    cuda: probeJson.commands?.cuda?.cuda?.version ?? null,
    cudnn: String(packages['libcudnn9-cuda-12'] ?? '').split('-', 1)[0] || null,
    driver: gpu[2],
    python: probeJson.commands?.python?.version ?? null,
    matchesContract: true,
  };
  if (!same(independentlyResolved, report.versions)) fail('runtime version probes do not independently match reported versions');
}

async function verifySuccessful(report, contract, frozen, evidenceRoot, sourceRoot) {
  if (report.runtimeExecuted !== true || report.hostInferenceVerified !== true || report.goldenExecuted !== true || report.supported !== false || report.task14Complete !== false || report.openspecTask1_4Checked !== false) fail('TensorRT status closure semantics');
  if (!report.runner.realNvidiaRunner || !report.runner.gpu?.name || !report.runner.gpu?.uuid || !report.runner.driver || !/^sha256:[0-9a-f]{64}$/.test(report.runner.containerImageId ?? '')) fail('real NVIDIA runner/container identity required');
  if (!report.versions.matchesContract) fail('version contract must be verified');
  for (const key of ['onnxruntime', 'tensorrt', 'cuda', 'cudnn']) if (report.versions[key] !== contract.compatibility[key]) fail(`${key} version drift`);
  if (!String(report.versions.python).startsWith(`${contract.compatibility.python}.`)) fail('Python version drift');
  if (!report.providers.available.includes('TensorrtExecutionProvider') || report.providers.session[0] !== 'TensorrtExecutionProvider') fail('TensorRT provider must be available and first in session');
  if (report.providers.fallback?.hidden !== false) fail('CUDA/CPU fallback must never be hidden');
  if (report.cache.modelSha256 !== contract.canonicalModel.sha256 || report.cache.freshBuildRequired !== true || contract.artifactPolicy.freshEngineBuildRequired !== true) fail('fresh canonical-model TensorRT cache is required');
  if (!report.engineBuild.attempted || !report.engineBuild.succeeded || report.engineBuild.stage !== 'profile-verified' || !report.engineBuild.runId || !report.engineBuild.transactionId || !report.engineBuild.generatedAt) fail('successful inference requires a fresh verified engine build');
  if (report.productionPostprocess?.build?.command !== productionBuildCommand) fail('production Rust build command drift');
  const requestedOptions = report.providers.options?.requested;
  const effectiveOptions = report.providers.options?.session?.TensorrtExecutionProvider;
  if (!requestedOptions || !effectiveOptions) fail('TensorRT provider options missing');
  for (const [key, value] of Object.entries(contract.providerOptions)) {
    const expected = key.endsWith('_cache_path') ? resolve(evidenceRoot, key === 'trt_engine_cache_path' ? 'engine-cache' : 'timing-cache', contract.canonicalModel.sha256, report.engineBuild.transactionId) : value;
    if (String(requestedOptions[key]) !== String(expected) || String(effectiveOptions[key]) !== String(expected)) fail(`TensorRT provider option drift: ${key}`);
  }
  const lock = await readFile(resolve(sourceRoot, 'evidence/tooling/tensorrt-requirements.lock'), 'utf8');
  const lockLines = lock.split('\n').filter((line) => line && !line.startsWith('#') && !line.startsWith('--'));
  if (lockLines.length !== 9 || lockLines.some((line) => !/^[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}$/.test(line))) fail('TensorRT dependency lock must contain nine hash-locked entries');
  const dockerfile = await readFile(resolve(sourceRoot, 'evidence/tooling/tensorrt/Dockerfile'), 'utf8');
  if (!dockerfile.includes(contract.compatibility.rustBuilderAmd64Manifest) || !dockerfile.includes(contract.compatibility.containerAmd64Manifest)) fail('TensorRT Docker image manifest digest drift');
  await verifyLibraries(report, evidenceRoot, contract);
  const imageInspect = await verifyArtifact(report.runtimeEvidence.containerImageInspectArtifact, evidenceRoot, 'container image inspect');
  const imageInspectJson = JSON.parse(imageInspect.bytes.toString('utf8'));
  if (!Array.isArray(imageInspectJson) || imageInspectJson.length !== 1 || imageInspectJson[0]?.Id !== report.runner.containerImageId) fail('runtime container image inspect does not match the executed image ID');
  const profile = await verifyArtifact(report.providers.profileArtifact, evidenceRoot, 'ORT profile');
  const profileFactsValue = profileFacts(JSON.parse(profile.bytes.toString('utf8')));
  if (!same(profileFactsValue.executionEventCounts, report.providers.profileExecutionEventCounts) || !same(profileFactsValue.uniqueNodeCounts, report.providers.profileUniqueNodeCounts) || profileFactsValue.executionPlan !== report.providers.executionPlan) fail('ORT profile independently recomputed facts mismatch');
  if (profileFactsValue.uniqueNodeCounts.TensorrtExecutionProvider < 1) fail('real TensorRT unique profile node required');
  const expectedFallback = { cudaExecutionEvents: profileFactsValue.executionEventCounts.CUDAExecutionProvider, cpuExecutionEvents: profileFactsValue.executionEventCounts.CPUExecutionProvider, cudaUniqueNodes: profileFactsValue.uniqueNodeCounts.CUDAExecutionProvider, cpuUniqueNodes: profileFactsValue.uniqueNodeCounts.CPUExecutionProvider, hidden: false };
  if (!same(expectedFallback, report.providers.fallback)) fail('CUDA/CPU fallback must be recomputed from profile');
  const log = await verifyArtifact(report.engineBuild.logArtifact, evidenceRoot, 'ORT log', true);
  const errors = logFacts(log.bytes.toString('utf8'));
  if (!same(errors, report.engineBuild.errors) || Object.values(errors).some((items) => items.length)) fail('ORT log contains parser/build/operator/profile errors');
  if ((report.cache.before ?? []).length !== 0) fail('fresh engine namespace must be empty after pre-run cleanup');
  const cacheRoots = [
    resolve(evidenceRoot, 'engine-cache', contract.canonicalModel.sha256, report.engineBuild.transactionId),
    resolve(evidenceRoot, 'timing-cache', contract.canonicalModel.sha256, report.engineBuild.transactionId),
  ];
  for (const path of cacheRoots) if (!inside(path, evidenceRoot)) fail('cache transaction root escapes evidence root');
  const diskCache = await diskArtifacts((await Promise.all(cacheRoots.map(listFiles))).flat(), evidenceRoot);
  const stripped = (items) => items.map(({ runId: _runId, transactionId: _transactionId, generatedAt: _generatedAt, ...item }) => item);
  if (!report.cache.generated?.length || !same(report.cache.after, report.cache.generated) || !same(diskCache, stripped(report.cache.after))) fail('engine after/generated snapshot mismatch');
  const startedNs = BigInt(report.engineBuild.startedAtEpochNs);
  const finishedNs = BigInt(report.engineBuild.finishedAtEpochNs);
  if (startedNs <= 0n || finishedNs < startedNs) fail('engine build time boundary invalid');
  let engineCount = 0;
  for (const artifact of report.cache.generated) {
    const verified = await verifyArtifact(artifact, evidenceRoot, `generated cache ${artifact.path}`);
    const extension = extname(artifact.path.toLowerCase());
    if (!inside(verified.path, cacheRoots[0]) && !inside(verified.path, cacheRoots[1])) fail('generated cache outside model SHA/transaction namespace');
    if (artifact.runId !== report.engineBuild.runId || artifact.transactionId !== report.engineBuild.transactionId || !artifact.generatedAt) fail('generated cache run identity mismatch');
    validateCacheArtifactTimeBoundary({ mtimeNs: verified.metadata.mtimeNs, ctimeNs: verified.metadata.ctimeNs }, startedNs, finishedNs);
    const created = Date.parse(artifact.generatedAt); const started = Date.parse(report.engineBuild.startedAt); const finished = Date.parse(report.engineBuild.finishedAt);
    if (!Number.isFinite(created) || created < started || created > finished) fail('generated cache timestamp outside run boundary');
    if (['.engine', '.plan'].includes(extension)) engineCount += 1;
  }
  if (engineCount < 1) fail('non-empty fresh TensorRT engine/plan required');
  const exporter = await verifyArtifact(report.webReference.exporterManifest, evidenceRoot, 'Web exporter manifest');
  const exporterJson = JSON.parse(exporter.bytes.toString('utf8'));
  const frozenBytes = await readFile(resolve(sourceRoot, 'evidence/golden/web-reference.json'));
  if (report.webReference.frozenReferenceSha256 !== sha256(frozenBytes) || exporterJson.sourceReferenceSha256 !== sha256(frozenBytes) || !same(exporterJson.fixtures.map((item) => item.id), contract.fixtures)) fail('Web reference/exporter binding mismatch');
  if (!/^[0-9a-f]{40}$/.test(report.evidenceSourceCommit ?? '')) fail('TensorRT evidence source commit missing');
  const rust = await buildRustRunner(sourceRoot, report.productionPostprocess, report.evidenceSourceCommit);
  const capturedRunner = await verifyArtifact(report.productionPostprocess.containerBinary, evidenceRoot, 'container production Rust runner');
  if (report.productionPostprocess.containerBinary.containerImageId !== report.runner.containerImageId || !/^[0-9a-f]{64}$/.test(report.productionPostprocess.containerBinary.sha256 ?? '')) fail('container production Rust binary/source identity mismatch');
  if (capturedRunner.bytes[0] !== 0x7f || capturedRunner.bytes[4] !== 2 || capturedRunner.bytes.readUInt16LE(18) !== 62) fail('production Rust runner is not ELF64 x86_64');
  if (report.productionPostprocess.build.rustBuilderIndex !== contract.compatibility.rustBuilderIndex || report.productionPostprocess.build.rustBuilderAmd64Manifest !== contract.compatibility.rustBuilderAmd64Manifest) fail('Rust builder digest identity');
  const containerBinary = await verifyArtifact(report.productionPostprocess.containerBinary, evidenceRoot, 'container postprocess binary');
  if (containerBinary.bytes[0] !== 0x7f || containerBinary.bytes[1] !== 0x45 || containerBinary.bytes[4] !== 2 || containerBinary.bytes[5] !== 1 || containerBinary.bytes.readUInt16LE(18) !== 62) fail('container postprocess binary ELF identity');
  const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-tensorrt-decode-'));
  try {
    if (report.fixtures.length !== contract.fixtures.length || !same(report.fixtures.map((item) => item.id), contract.fixtures)) fail('five canonical fixtures required');
    for (const fixture of report.fixtures) {
      const frozenFixture = frozen.fixtures.find((item) => item.id === fixture.id);
      const exportedFixture = exporterJson.fixtures.find((item) => item.id === fixture.id);
      if (!frozenFixture || !exportedFixture || !same(fixture.image, exportedFixture.image)) fail(`${fixture.id} exporter fixture identity`);
      for (const [name, expectedMetadata] of [['input', { shape: [1, 3, 640, 640], dtype: 'float32', elementCount: 1228800 }], ['referenceRaw', output], ['actualRaw', output], ['decoded', { dtype: 'json' }]]) {
        const declared = fixture.artifacts[name];
        if (!declared || !Object.entries(expectedMetadata).every(([key, value]) => same(declared[key], value))) fail(`${fixture.id} ${name} metadata`);
      }
      const input = await verifyArtifact(fixture.artifacts.input, evidenceRoot, `${fixture.id} input`);
      const reference = await verifyArtifact(fixture.artifacts.referenceRaw, evidenceRoot, `${fixture.id} reference raw`);
      const actual = await verifyArtifact(fixture.artifacts.actualRaw, evidenceRoot, `${fixture.id} actual raw`);
      const decodedArtifact = await verifyArtifact(fixture.artifacts.decoded, evidenceRoot, `${fixture.id} decoded`);
      if (sha256(input.bytes) !== frozenFixture.canonicalInput.sha256Float32Le || sha256(reference.bytes) !== frozenFixture.runs[0].rawTensor.sha256Float32Le) fail(`${fixture.id} frozen input/reference digest`);
      const raw = rawFacts(actual.bytes, reference.bytes, frozen.tolerances);
      compareNumeric(raw, fixture.rawComparison, `${fixture.id} raw comparison`);
      if (!raw.passed) fail(`${fixture.id} raw frozen tolerance`);
      const regeneratedPath = join(temporary, `${fixture.id}.json`);
      const decode = spawnSync(rust.runner, [actual.path, String(fixture.image.width), String(fixture.image.height), regeneratedPath], { cwd: sourceRoot, encoding: 'utf8' });
      if (decode.status !== 0) fail(`${fixture.id} production Rust decode failed`);
      const decoded = JSON.parse(decodedArtifact.bytes.toString('utf8'));
      const regenerated = JSON.parse(await readFile(regeneratedPath, 'utf8'));
      if (!same(decoded, regenerated)) fail(`${fixture.id} decoded differs from fresh production Rust runner`);
      const decodedComparison = decodedFacts(regenerated, frozenFixture.runs[0].decoded, frozen.tolerances);
      compareNumeric(decodedComparison, fixture.decodedComparison, `${fixture.id} decoded comparison`);
      if (!decodedComparison.passed) fail(`${fixture.id} decoded frozen tolerance`);
    }
  } finally {
    await rm(temporary, { recursive: true, force: true });
    await rm(rust.temporary, { recursive: true, force: true });
  }
}

async function verifyRealRunnerBlocked(report, contract, evidenceRoot) {
  if (report.runtimeExecuted !== false || report.hostInferenceVerified !== false || report.goldenExecuted !== false || report.engineBuild.succeeded !== false) fail('blocked TensorRT runner cannot claim inference or golden success');
  if (!report.failureReason || report.failureStage !== report.engineBuild.stage) fail('blocked TensorRT runner failure stage/reason mismatch');
  const stages = new Set(['parser', 'build', 'operator', 'profile', 'golden-compare', 'runtime-unknown']);
  if (!stages.has(report.failureStage)) fail('blocked TensorRT runner failure stage is not classified');
  const log = await verifyArtifact(report.engineBuild.logArtifact, evidenceRoot, 'blocked ORT log', true);
  const recomputedLogErrors = logFacts(log.bytes.toString('utf8'));
  for (const [kind, lines] of Object.entries(recomputedLogErrors)) {
    if (!Array.isArray(report.engineBuild.errors?.[kind]) || lines.some((line) => !report.engineBuild.errors[kind].includes(line))) fail(`blocked ORT ${kind} disclosure mismatch`);
  }
  if ((report.cache.before ?? []).length !== 0) fail('blocked TensorRT run must use an empty fresh cache namespace');
  const cacheRoots = [
    resolve(evidenceRoot, 'engine-cache', contract.canonicalModel.sha256, report.engineBuild.transactionId),
    resolve(evidenceRoot, 'timing-cache', contract.canonicalModel.sha256, report.engineBuild.transactionId),
  ];
  const diskCache = await diskArtifacts((await Promise.all(cacheRoots.map(listFiles))).flat(), evidenceRoot);
  const stripped = (items) => items.map(({ runId: _runId, transactionId: _transactionId, generatedAt: _generatedAt, ...item }) => item);
  if (!same(diskCache, stripped(report.cache.after ?? []))) fail('blocked TensorRT cache after snapshot mismatch');
  if (!same(report.cache.generated ?? [], report.cache.after ?? [])) fail('blocked TensorRT cache generated snapshot mismatch');
  const startedNs = BigInt(report.engineBuild.startedAtEpochNs);
  const finishedNs = BigInt(report.engineBuild.finishedAtEpochNs);
  for (const artifact of report.cache.after ?? []) {
    const verified = await verifyArtifact(artifact, evidenceRoot, `blocked cache ${artifact.path}`);
    if (!inside(verified.path, cacheRoots[0]) && !inside(verified.path, cacheRoots[1])) fail('blocked cache outside model SHA/transaction namespace');
    validateCacheArtifactTimeBoundary({ mtimeNs: verified.metadata.mtimeNs, ctimeNs: verified.metadata.ctimeNs }, startedNs, finishedNs);
  }
}

export async function validateTensorRtReport(report, options = {}) {
  const sourceRoot = resolve(options.sourceRoot ?? root);
  const allowedEvidenceRoot = resolve(sourceRoot, '.evidence/tensorrt');
  if ((await lstat(allowedEvidenceRoot).catch(() => null))?.isSymbolicLink()) fail('TensorRT evidence root must not be a symlink');
  const evidenceRoot = resolve(options.evidenceRoot ?? resolve(sourceRoot, report.cache?.root ?? '.evidence/tensorrt'));
  if (!inside(evidenceRoot, allowedEvidenceRoot)) fail('TensorRT evidence root must stay under .evidence/tensorrt');
  const existingEvidenceRoot = await realpath(evidenceRoot).catch(() => evidenceRoot);
  if (!inside(existingEvidenceRoot, await realpath(allowedEvidenceRoot).catch(() => allowedEvidenceRoot))) fail('TensorRT evidence root symlink escape');
  const contract = options.contract ?? await json(resolve(sourceRoot, 'evidence/tensorrt/contract.json'));
  const frozen = options.frozen ?? await json(resolve(sourceRoot, 'evidence/golden/web-reference.json'));
  const trackedFiles = options.trackedFiles ?? execFileSync('git', ['ls-files'], { cwd: sourceRoot, encoding: 'utf8' }).trim().split('\n').filter(Boolean);
  validateTrackedArtifacts(trackedFiles);
  for (const path of trackedFiles) {
    const trackedPath = resolve(sourceRoot, path);
    if (inside(trackedPath, evidenceRoot)) fail(`TensorRT workspace overlaps tracked evidence: ${path}`);
  }
  const releaseFiles = options.releaseFiles ?? (await Promise.all(['release', 'dist', 'package', 'artifacts'].map((directory) => listFiles(resolve(sourceRoot, directory))))).flat();
  for (const path of releaseFiles) {
    const lower = path.toLowerCase();
    if (cacheExtensions.has(extname(lower)) || /(^|\/)(engine-cache|timing-cache|tensorrt-cache)(\/|$)/.test(lower)) fail(`TensorRT engine/cache 不得进入发布目录: ${path}`);
  }
  if (report.schemaVersion !== 2 || report.taskId !== 'T14-LNX-TRT-01') fail('TensorRT report identity drift');
  if (report.sourceCommit !== contract.sourceCommit) fail('TensorRT source commit drift');
  const modelBytes = await readFile(resolve(sourceRoot, contract.canonicalModel.path));
  if (modelBytes.length !== contract.canonicalModel.bytes || sha256(modelBytes) !== contract.canonicalModel.sha256 || report.model.path !== contract.canonicalModel.path || report.model.bytes !== contract.canonicalModel.bytes || report.model.sha256 !== contract.canonicalModel.sha256) fail('canonical ONNX identity drift');
  if (!same(contract.frozenTolerances, frozen.tolerances)) fail('TensorRT tolerance drift');
  if (!same(report.providers.requested, contract.officialPath.requestedProviders)) fail('TensorRT provider request/order drift');
  if (report.supported !== false || report.task14Complete !== false) fail('TensorRT spike cannot claim platform support/task completion');
  if (report.openspecTask1_4Checked !== false) fail('TensorRT spike cannot claim OpenSpec task 1.4 checked');
  if (report.status === 'host-inference-verified') await verifySuccessful(report, contract, frozen, evidenceRoot, sourceRoot);
  else if (report.status === 'blocked') {
    if (report.runtimeExecuted !== false || report.hostInferenceVerified !== false || report.goldenExecuted !== false || !report.failureStage || !report.failureReason) fail('blocked report execution fields must remain false');
    if (!report.runner.realNvidiaRunner && (report.engineBuild.attempted || report.engineBuild.succeeded || report.fixtures.length)) fail('runner-preflight blocked report cannot contain runtime execution evidence');
    if (report.runner.realNvidiaRunner && report.engineBuild.attempted) await verifyRealRunnerBlocked(report, contract, evidenceRoot);
  } else fail(`unsupported TensorRT status: ${report.status}`);
  if (report.protectedTrackedEvidence) {
    if (report.protectedTrackedEvidence.unchanged !== true || !same(Object.keys(report.protectedTrackedEvidence.before).sort(), Object.keys(report.protectedTrackedEvidence.after).sort())) fail('ordinary replay tracked evidence set drift');
    for (const [path, snapshot] of Object.entries(report.protectedTrackedEvidence.before)) {
      const absolute = resolve(sourceRoot, path); const current = await stat(absolute, { bigint: true }); const bytes = await readFile(absolute); const canonical = await realpath(absolute);
      const recomputed = { path, canonicalPath: canonical, exists: true, bytes: bytes.length, sha256: sha256(bytes), inode: String(current.ino), ctimeNs: String(current.ctimeNs), mtimeNs: String(current.mtimeNs) };
      if (!same(snapshot, report.protectedTrackedEvidence.after[path]) || !same(snapshot, recomputed)) fail(`ordinary replay changed tracked evidence: ${path}`);
    }
  }
  return true;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = await json(resolve(root, process.argv[2] ?? defaultReportPath));
  const evidenceRoot = process.argv[3] ? resolve(root, process.argv[3]) : undefined;
  await validateTensorRtReport(report, { evidenceRoot });
  console.log(`TensorRT EP evidence valid: ${report.status}`);
}
