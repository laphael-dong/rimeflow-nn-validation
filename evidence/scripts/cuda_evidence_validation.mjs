import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, readdir, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, resolve } from 'node:path';

const MODEL_SHA = '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad';
const MODEL_BYTES = 12851098;
const PACKAGES = {
  flatbuffers: '25.2.10', numpy: '2.2.3', 'onnxruntime-gpu': '1.26.0', packaging: '24.2', protobuf: '5.29.3',
  'nvidia-cublas-cu12': '12.8.4.1', 'nvidia-cuda-nvrtc-cu12': '12.8.93', 'nvidia-cuda-runtime-cu12': '12.8.90',
  'nvidia-cudnn-cu12': '9.10.2.21', 'nvidia-cufft-cu12': '11.3.3.83', 'nvidia-curand-cu12': '10.3.9.90', 'nvidia-nvjitlink-cu12': '12.8.93',
};
const INPUT = { name: 'images', shape: [1, 3, 640, 640], dtype: 'tensor(float)' };
const OUTPUT = { name: 'output0', shape: [1, 84, 8400], dtype: 'tensor(float)' };
const OUTPUT_COUNT = 84 * 8400;
const REQUIRED_LIBRARIES = ['libonnxruntime_providers_cuda.so', 'libcuda.so', 'libcudart.so', 'libcudnn.so', 'libcublas.so'];
const LIBRARY_VERSIONS = {
  'libonnxruntime_providers_cuda.so': '1.26.0', 'libcuda.so': null, 'libcudart.so': '12.8.90',
  'libcudnn.so': '9.10.2.21', 'libcublas.so': '12.8.4.1',
};
const LIBRARY_VERSION_SOURCES = {
  'libonnxruntime_providers_cuda.so': 'onnxruntime-gpu package',
  'libcuda.so': 'nvidia-smi driver_version',
  'libcudart.so': 'nvidia-cuda-runtime-cu12 package',
  'libcudnn.so': 'nvidia-cudnn-cu12 package',
  'libcublas.so': 'nvidia-cublas-cu12 package',
};
const LIBRARY_PATTERNS = {
  'libonnxruntime_providers_cuda.so': /^libonnxruntime_providers_cuda\.so(?:\.\d+)*$/,
  'libcuda.so': /^libcuda\.so(?:\.\d+)*$/,
  'libcudart.so': /^libcudart\.so(?:\.\d+)*$/,
  'libcudnn.so': /^libcudnn\.so(?:\.\d+)*$/,
  'libcublas.so': /^libcublas\.so(?:\.\d+)*$/,
};
const RUST_RUNNER_PATH = 'evidence/tooling/raw-golden/target/debug/rimeflow-raw-golden';
const RUST_SOURCE_PATHS = [
  'evidence/tooling/raw-golden/Cargo.toml', 'evidence/tooling/raw-golden/Cargo.lock',
  'evidence/tooling/raw-golden/src/main.rs', 'evidence/tooling/raw-golden/src/lib.rs', 'src/postprocess.rs',
];
const OPTIONS = { device_id: '0', arena_extend_strategy: 'kNextPowerOfTwo', cudnn_conv_algo_search: 'EXHAUSTIVE', cudnn_conv_use_max_workspace: '1', do_copy_in_default_stream: '1' };
export const TRACKED_CUDA_EVIDENCE = [
  'evidence/conversions/cuda-ep-spike-manifest.json',
  'evidence/reports/cuda-ep-spike-report.json',
  'evidence/reports/cuda-ep-replay-report.json',
  'evidence/golden/manifest.json',
  'evidence/replay/task1-replay.json',
  'evidence/conversions/conversion-spikes.json',
];
const RECORDED_CUDA_CURRENT_IDENTITY = [
  'evidence/conversions/cuda-ep-spike-manifest.json',
  'evidence/reports/cuda-ep-spike-report.json',
];

const equal = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const product = (shape) => shape.reduce((value, dimension) => value * dimension, 1);
const artifactPath = (root, path) => path.startsWith('/') ? resolve(path) : resolve(root, path);
const isCudaWorkspacePath = (root, path) => {
  const allowed = resolve(root, '.evidence/cuda'); const actual = artifactPath(root, path);
  return actual.startsWith(`${allowed}/`);
};

async function actualArtifact(root, claimed) {
  assert(claimed && typeof claimed.path === 'string' && Number.isInteger(claimed.bytes) && /^[0-9a-f]{64}$/.test(claimed.sha256), 'CUDA artifact identity incomplete');
  const path = artifactPath(root, claimed.path);
  const bytes = await readFile(path);
  assert(bytes.length === claimed.bytes && sha256(bytes) === claimed.sha256, `CUDA artifact bytes/SHA drift: ${claimed.path}`);
  return { path, bytes };
}

async function assertTrackedHeadIdentity(root, path, claim) {
  assert(claim.path === path, `CUDA tracked source path drift: ${claim.path}`);
  const actual = await actualArtifact(root, claim);
  const committed = spawnSync('git', ['show', `HEAD:${path}`], { cwd: root });
  assert(committed.status === 0 && actual.bytes.equals(committed.stdout), `CUDA tracked source differs from HEAD: ${path}`);
  return actual;
}

export function compareRawValues(actual, reference, absolute, relative) {
  assert(actual.length === reference.length, 'CUDA raw/reference element count differs');
  let mismatchCount = 0;
  let maximumAbsolute = 0;
  let maximumRelative = 0;
  let maximumDifferenceFlatIndex = actual.length ? 0 : null;
  let maximumExcess = Number.NEGATIVE_INFINITY;
  let maximumExcessFlatIndex = actual.length ? 0 : null;
  let nearZeroReferenceCount = 0;
  for (let index = 0; index < actual.length; index++) {
    assert(Number.isFinite(actual[index]) && Number.isFinite(reference[index]), `CUDA raw/reference non-finite at ${index}`);
    const difference = Math.abs(actual[index] - reference[index]);
    const allowed = absolute + relative * Math.abs(reference[index]);
    const relativeDifference = reference[index] === 0 ? 0 : difference / Math.abs(reference[index]);
    if (Math.abs(reference[index]) <= 1e-12) nearZeroReferenceCount++;
    if (difference > allowed) mismatchCount++;
    if (difference > maximumAbsolute) { maximumAbsolute = difference; maximumDifferenceFlatIndex = index; }
    if (relativeDifference > maximumRelative) maximumRelative = relativeDifference;
    if (difference - allowed > maximumExcess) { maximumExcess = difference - allowed; maximumExcessFlatIndex = index; }
  }
  return {
    passed: mismatchCount === 0, mismatchCount, maximumAbsolute, maximumRelative,
    maximumDifferenceFlatIndex, maximumExcessFlatIndex, nearZeroReferenceCount,
    rule: 'abs(actual-reference) <= rawTensorAbsolute + rawTensorRelative * abs(reference)',
  };
}

function readFloat32Le(bytes, claim, expectedShape) {
  const expectedCount = product(expectedShape);
  assert(claim.dtype === 'float32-le' && equal(claim.shape, expectedShape) && claim.elementCount === expectedCount, `CUDA FP32 artifact metadata drift: ${claim.path}`);
  assert(bytes.length === expectedCount * 4, `CUDA FP32 artifact byte count drift: ${claim.path}`);
  const values = new Float64Array(expectedCount);
  for (let index = 0; index < values.length; index++) values[index] = bytes.readFloatLE(index * 4);
  assert(values.every(Number.isFinite), `CUDA FP32 artifact contains NaN or Infinity: ${claim.path}`);
  return values;
}

export async function validateRawArtifacts(root, fixture, tolerances) {
  const raw = await actualArtifact(root, fixture.cudaRaw);
  const reference = await actualArtifact(root, fixture.referenceRaw);
  const actual = readFloat32Le(raw.bytes, fixture.cudaRaw, OUTPUT.shape);
  const expected = readFloat32Le(reference.bytes, fixture.referenceRaw, OUTPUT.shape);
  const comparison = compareRawValues(actual, expected, tolerances.rawTensorAbsolute, tolerances.rawTensorRelative);
  assert(equal(fixture.rawComparison, comparison), `${fixture.id}: CUDA raw comparison was not recomputed from bytes`);
  assert(comparison.passed, `${fixture.id}: CUDA raw tensor exceeds frozen tolerance`);
  return comparison;
}

function bboxIou(left, right) {
  const x1 = Math.max(left[0], right[0]); const y1 = Math.max(left[1], right[1]);
  const x2 = Math.min(left[2], right[2]); const y2 = Math.min(left[3], right[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area = (box) => (box[2] - box[0]) * (box[3] - box[1]);
  const union = area(left) + area(right) - intersection;
  return union <= 0 ? 0 : intersection / union;
}

export function compareDecodedValues(actual, expected, tolerances) {
  let classMismatchCount = Math.abs(actual.length - expected.length);
  let maximumConfidenceAbsolute = 0; let maximumDecodedBboxAbsolute = 0;
  let minimumBboxIou = actual.length || expected.length ? 1 : null;
  for (let index = 0; index < Math.min(actual.length, expected.length); index++) {
    const left = actual[index]; const right = expected[index];
    assert(Number.isInteger(left.classId) && Number.isFinite(left.score) && Array.isArray(left.bbox) && left.bbox.length === 4 && left.bbox.every(Number.isFinite), `CUDA decoded result invalid at ${index}`);
    classMismatchCount += Number(left.classId !== right.classId);
    maximumConfidenceAbsolute = Math.max(maximumConfidenceAbsolute, Math.abs(left.score - right.score));
    maximumDecodedBboxAbsolute = Math.max(maximumDecodedBboxAbsolute, ...left.bbox.map((value, axis) => Math.abs(value - right.bbox[axis])));
    minimumBboxIou = Math.min(minimumBboxIou, bboxIou(left.bbox, right.bbox));
  }
  return {
    passed: actual.length === expected.length && classMismatchCount === 0 && maximumConfidenceAbsolute <= tolerances.confidenceAbsolute && maximumDecodedBboxAbsolute <= tolerances.decodedBoxAbsolute && (minimumBboxIou === null || minimumBboxIou >= tolerances.boxIouMinimum),
    countMismatch: actual.length !== expected.length, classMismatchCount, maximumConfidenceAbsolute, maximumDecodedBboxAbsolute, minimumBboxIou,
  };
}

export async function validateProfileArtifact(root, profile) {
  assert(isCudaWorkspacePath(root, profile.artifact?.path), 'CUDA ORT profile must stay under .evidence/cuda');
  const artifact = await actualArtifact(root, profile.artifact);
  const events = JSON.parse(artifact.bytes.toString('utf8'));
  assert(Array.isArray(events), 'CUDA ORT profile must be an event array');
  const counts = {};
  for (const event of events) if (event?.cat === 'Node') {
    const provider = event.args?.provider ?? 'unassigned'; counts[provider] = (counts[provider] ?? 0) + 1;
  }
  assert(equal(profile.nodeProviderCounts, counts) && profile.cudaNodeCount === (counts.CUDAExecutionProvider ?? 0) && profile.cpuNodeCount === (counts.CPUExecutionProvider ?? 0), 'CUDA profile provider counts were not recomputed from profile bytes');
  assert((counts.CUDAExecutionProvider ?? 0) > 0, 'configured CUDA provider has no profiled CUDA Node');
  return counts;
}

export async function validateLibraryArtifacts(root, report) {
  assert(isCudaWorkspacePath(root, report.sharedLibraries.mapsArtifact?.path), 'CUDA maps snapshot must stay under .evidence/cuda');
  const maps = await actualArtifact(root, report.sharedLibraries.mapsArtifact);
  const mapsText = maps.bytes.toString('utf8');
  const mappedPaths = new Set(mapsText.split('\n').map((line) => line.trim().split(/\s+/).at(-1)).filter((path) => path?.startsWith('/')).map((path) => resolve(path)));
  for (const name of REQUIRED_LIBRARIES) {
    const entry = report.sharedLibraries.entries[name];
    assert(report.sharedLibraries.requiredLoaded[name] === true && entry, `CUDA required library claim missing: ${name}`);
    const entryPath = artifactPath(root, entry.path);
    const canonicalPath = await realpath(entryPath);
    assert(entry.path === canonicalPath && entry.realpath === canonicalPath && entry.basename === basename(canonicalPath), `CUDA library path is not canonical: ${name}`);
    assert(LIBRARY_PATTERNS[name].test(entry.basename), `CUDA library basename mismatch: ${name}`);
    const library = await actualArtifact(root, entry);
    assert(library.bytes[0] === 0x7f && library.bytes.subarray(1, 4).toString('ascii') === 'ELF' && library.bytes[4] === 2 && library.bytes[5] === 1 && library.bytes.readUInt16LE(18) === 62, `CUDA library is not ELF64 x86_64: ${name}`);
    const dynamic = spawnSync('readelf', ['-d', canonicalPath], { encoding: 'utf8' });
    const soname = dynamic.stdout.match(/\(SONAME\).*\[(.+?)\]/)?.[1];
    assert(dynamic.status === 0 && soname === entry.soname && LIBRARY_PATTERNS[name].test(soname), `CUDA library SONAME mismatch: ${name}`);
    assert(entry.elfClass === 'ELF64' && entry.elfMachine === 'Advanced Micro Devices X86-64', `CUDA library reported ELF identity drift: ${name}`);
    assert(mappedPaths.has(canonicalPath), `CUDA library is not backed by persisted maps snapshot: ${name}`);
    const expectedVersion = name === 'libcuda.so' ? report.versions.nvidiaDriver : LIBRARY_VERSIONS[name];
    assert(entry.componentVersion === expectedVersion && entry.versionSource === LIBRARY_VERSION_SOURCES[name], `CUDA library component version source drift: ${name}`);
  }
  const driverProbe = report.commands.find((item) => item.command?.[0] === 'nvidia-smi' && item.command.includes('--query-gpu=name,uuid,driver_version,compute_cap'));
  assert(driverProbe?.exitCode === 0 && driverProbe.stdout.split(',').map((item) => item.trim()).includes(report.versions.nvidiaDriver), 'CUDA driver version is not bound to nvidia-smi probe output');
}

export async function validateProductionRunner(root, report) {
  assert(report.productionPostprocess.runnerArtifact?.path === RUST_RUNNER_PATH, 'CUDA production runner path must be canonical');
  assert(equal(report.productionPostprocess.sourceArtifacts?.map((item) => item.path), RUST_SOURCE_PATHS), 'CUDA production runner source identity list drift');
  for (const [index, path] of RUST_SOURCE_PATHS.entries()) await assertTrackedHeadIdentity(root, path, report.productionPostprocess.sourceArtifacts[index]);
  const build = spawnSync('cargo', ['build', '--offline', '--manifest-path', 'evidence/tooling/raw-golden/Cargo.toml'], { cwd: root, encoding: 'utf8' });
  assert(build.status === 0, `CUDA canonical production runner build failed: ${build.stdout}\n${build.stderr}`);
  const runner = await actualArtifact(root, report.productionPostprocess.runnerArtifact);
  assert(runner.path === resolve(root, RUST_RUNNER_PATH), 'CUDA production runner resolved path drift');
  return runner;
}

export async function validateWebReferenceBindings(root, report, frozen) {
  const frozenBytes = await readFile(resolve(root, 'evidence/golden/web-reference.json'));
  assert(isCudaWorkspacePath(root, report.webReferenceManifest?.path), 'CUDA Web reference manifest must stay under .evidence/cuda');
  const exportedManifestArtifact = await actualArtifact(root, report.webReferenceManifest);
  const exportedManifest = JSON.parse(exportedManifestArtifact.bytes.toString('utf8'));
  assert(exportedManifest.sourceReferenceSha256 === sha256(frozenBytes), 'CUDA Web reference manifest is not bound to tracked frozen reference');
  assert(equal(exportedManifest.fixtures.map((item) => item.id), report.fixtures.map((item) => item.id)), 'CUDA exported/reference fixture order drift');
  const exportRoot = resolve(artifactPath(root, report.webReferenceManifest.path), '..');
  for (const fixture of report.fixtures) {
    const frozenFixture = frozen.fixtures.find((item) => item.id === fixture.id);
    const exportedFixture = exportedManifest.fixtures.find((item) => item.id === fixture.id);
    assert(frozenFixture && exportedFixture && equal(fixture.image, exportedFixture.image), `${fixture.id}: CUDA image dimensions are not bound to Web exporter manifest`);
    assert(artifactPath(root, fixture.input.path) === resolve(exportRoot, fixture.id, 'input.f32le') && artifactPath(root, fixture.referenceRaw.path) === resolve(exportRoot, fixture.id, 'raw.f32le'), `${fixture.id}: CUDA input/reference path is not canonical`);
    assert(fixture.input.bytes === exportedFixture.input.bytes && fixture.input.sha256 === exportedFixture.input.sha256 && fixture.input.sha256 === frozenFixture.canonicalInput.sha256Float32Le, `${fixture.id}: CUDA input identity is not bound to canonical Web reference`);
    assert(fixture.referenceRaw.bytes === exportedFixture.output.bytes && fixture.referenceRaw.sha256 === exportedFixture.output.sha256 && fixture.referenceRaw.sha256 === frozenFixture.runs[0].rawTensor.sha256Float32Le, `${fixture.id}: CUDA raw reference identity is not bound to canonical Web reference`);
  }
  return exportedManifest;
}

async function validateDecodedArtifact(root, report, fixture, expected, tolerances) {
  assert(isCudaWorkspacePath(root, fixture.cudaRaw?.path) && isCudaWorkspacePath(root, fixture.decoded?.path), `${fixture.id}: CUDA raw/decoded artifacts must stay under .evidence/cuda`);
  const raw = await actualArtifact(root, fixture.cudaRaw);
  const recorded = await actualArtifact(root, fixture.decoded);
  const recordedDecoded = JSON.parse(recorded.bytes.toString('utf8'));
  assert(Array.isArray(recordedDecoded) && fixture.decoded.detectionCount === recordedDecoded.length, `${fixture.id}: decoded artifact metadata drift`);
  const runner = await actualArtifact(root, report.productionPostprocess.runnerArtifact);
  const temporary = await mkdtemp(resolve(tmpdir(), 'rimeflow-cuda-validator-'));
  try {
    const output = resolve(temporary, `${fixture.id}.json`);
    const result = spawnSync(runner.path, [raw.path, String(fixture.image.width), String(fixture.image.height), output], { encoding: 'utf8' });
    assert(result.status === 0, `${fixture.id}: production raw-golden rerun failed: ${result.stderr}`);
    const recomputed = JSON.parse(await readFile(output, 'utf8'));
    assert(equal(recomputed, recordedDecoded), `${fixture.id}: recorded decoded artifact differs from production Rust rerun`);
    const comparison = compareDecodedValues(recomputed, expected, tolerances);
    assert(equal(fixture.decodedComparison, comparison), `${fixture.id}: decoded comparison was not recomputed`);
    assert(comparison.passed, `${fixture.id}: decoded result exceeds frozen tolerance`);
  } finally { await rm(temporary, { recursive: true, force: true }); }
}

export async function validateCudaEvidence(root, manifest, report) {
  assert(manifest.task === 'T14-LNX-CUDA-01' && report.task === manifest.task, 'CUDA task identity drift');
  assert(manifest.scope.os === 'linux' && manifest.scope.architecture === 'x86_64' && manifest.scope.executionProvider === 'CUDAExecutionProvider', 'CUDA target scope drift');
  assert(manifest.canonicalOnnx.path === 'models/yolov8n.onnx' && manifest.canonicalOnnx.bytes === MODEL_BYTES && manifest.canonicalOnnx.sha256 === MODEL_SHA && !manifest.canonicalOnnx.converted && !manifest.canonicalOnnx.copied && !manifest.canonicalOnnx.modified, 'CUDA model identity drift');
  assert(report.model.path === manifest.canonicalOnnx.path && report.model.bytes === MODEL_BYTES && report.model.sha256 === MODEL_SHA && !report.model.converted && !report.model.copied, 'CUDA report model identity drift');
  assert(manifest.compatibility.onnxruntimeGpu === '1.26.0' && manifest.compatibility.onnxruntimeBuildCuda === '12.8' && manifest.compatibility.cudaRuntimePackage === '12.8.90' && manifest.compatibility.cudnnPackage === '9.10.2.21' && manifest.compatibility.cudnnMajor === 9 && manifest.compatibility.minimumNvidiaDriverMajor === 525, 'CUDA/ORT/cuDNN compatibility drift');
  assert(equal(report.lockedPackages, PACKAGES), 'CUDA runtime package lock drift');
  assert(equal(manifest.provider.options, OPTIONS) && equal(report.providerOptionsRequested, OPTIONS), 'CUDA provider options drift');
  assert(equal(report.requestedProviders, ['CUDAExecutionProvider']) && !report.requestedProviders.includes('CPUExecutionProvider'), 'ordinary CPU ORT requested as CUDA');
  assert(equal({ name: manifest.ioContract.input.name, shape: manifest.ioContract.input.shape, dtype: manifest.ioContract.input.dtype }, INPUT) && equal({ name: manifest.ioContract.output.name, shape: manifest.ioContract.output.shape, dtype: manifest.ioContract.output.dtype }, OUTPUT), 'CUDA manifest I/O contract drift');
  assert(equal(report.ioContract, { input: INPUT, output: OUTPUT }), 'CUDA report I/O contract drift');
  assert(!manifest.ownership.cudaSpecificDecodeOrNms && manifest.ownership.nms.includes('src/postprocess.rs') && report.productionPostprocess.implementation === 'src/postprocess.rs' && !report.productionPostprocess.cudaSpecificDecodeOrNms, 'CUDA-specific decode/NMS is prohibited');
  const runtimeClaim = report.runtimeExecuted || report.hostInferenceVerified || report.goldenExecuted;
  if (runtimeClaim) {
    assert(report.versions.onnxruntime === '1.26.0' && report.versions.cudaRuntime === '12.8.90' && report.versions.cudnn === '9.10.2.21', 'CUDA runtime version introspection drift');
    assert(report.availableProviders.includes('CUDAExecutionProvider') && report.sessionProviders.includes('CUDAExecutionProvider'), 'configured/session CUDA provider missing');
    const actualOptions = report.sessionProviderOptions.CUDAExecutionProvider;
    assert(actualOptions && Object.entries(OPTIONS).every(([key, value]) => String(actualOptions[key]).toUpperCase() === value.toUpperCase()), 'CUDA session provider options introspection drift');
    await validateProfileArtifact(root, report.profile);
    await validateLibraryArtifacts(root, report);
  }
  if (report.hostInferenceVerified) assert(report.runtimeExecuted, 'host inference claimed without CUDA runtime execution');
  if (report.goldenExecuted) {
    assert(report.hostInferenceVerified && equal(report.fixtures.map((item) => item.id), ['no-detection', 'single-target', 'multi-class', 'boundary-box', 'extreme-aspect']), 'CUDA five-fixture identity/order incomplete');
    const frozen = JSON.parse(await readFile(resolve(root, 'evidence/golden/web-reference.json'), 'utf8'));
    const exportedManifest = await validateWebReferenceBindings(root, report, frozen);
    await validateProductionRunner(root, report);
    for (const fixture of report.fixtures) {
      const expected = frozen.fixtures.find((item) => item.id === fixture.id)?.runs[0]?.decoded;
      const exportedFixture = exportedManifest.fixtures.find((item) => item.id === fixture.id);
      assert(expected && fixture.passed === true && fixture.productionRust.exitCode === 0, `${fixture.id}: CUDA golden report claim incomplete`);
      const input = await actualArtifact(root, fixture.input);
      readFloat32Le(input.bytes, fixture.input, INPUT.shape);
      await validateRawArtifacts(root, fixture, frozen.tolerances);
      await validateDecodedArtifact(root, report, fixture, expected, frozen.tolerances);
    }
  } else assert(report.fixtures.length === 0, 'CUDA fixture result present while goldenExecuted=false');
  if (report.state === 'blocked') {
    for (const key of ['runtimeExecuted', 'hostInferenceVerified', 'goldenExecuted', 'supported', 'task14Complete']) assert(report[key] === false, `blocked CUDA report overclaims ${key}`);
    assert(report.failureStage && report.failure && Number.isInteger(report.commands[0]?.exitCode), 'blocked CUDA command/failure evidence incomplete');
    assert(report.commands.every((item) => item.command.length > 0 && item.startedAt && item.endedAt && Number.isInteger(item.exitCode) && typeof item.stdout === 'string' && typeof item.stderr === 'string'), 'blocked CUDA command timestamp/exit/stdout/stderr incomplete');
    assert(equal(report.failure.missing, ['NVIDIA GPU', 'NVIDIA driver / nvidia-smi', 'onnxruntime-gpu', ...REQUIRED_LIBRARIES]), 'blocked CUDA blocker list is not the final eight-item provenance');
  }
  assert(report.supported === false && report.task14Complete === false && manifest.status.supported === false && manifest.status.task14Complete === false && manifest.status.openspecTask1_4Checked === false, 'CUDA spike must not complete support or OpenSpec 1.4');
  assert(!manifest.artifactHandling.releaseDirectoryTouched && !manifest.artifactHandling.rimecutTouched && !manifest.artifactHandling.runtimeLibrariesTracked && !manifest.artifactHandling.rawOutputsTracked, 'CUDA publication boundary violated');
}

async function walk(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) result.push(...await walk(resolve(directory, entry.name), relative)); else result.push(relative);
  }
  return result;
}

export async function validateCudaRepositoryBoundary(root) {
  const tracked = spawnSync('git', ['ls-files'], { cwd: root, encoding: 'utf8' }).stdout.trim().split('\n').filter(Boolean);
  assert(equal(tracked.filter((path) => path.endsWith('.onnx')), ['models/yolov8n.onnx']), 'CUDA spike added a second tracked ONNX');
  assert(!tracked.some((path) => /(^|\/)(\.evidence\/cuda|release|dist|publish)(\/|$)/i.test(path)), 'CUDA temporary or release artifact is tracked');
  assert(!tracked.some((path) => /\.(so(?:\.\d+)*)$|\.raw\.f32le$|ort-profile/i.test(path)), 'CUDA runtime/raw/profile binary is tracked');
  const runnerFiles = await walk(resolve(root, 'evidence/tooling/cuda-runner'));
  assert(!runnerFiles.some((path) => /\.(onnx|so|bin|whl)$|(^|\/)(venv|target|raw)(\/|$)/i.test(path)), 'CUDA runner contains generated runtime/model/raw binary');
  const runner = await readFile(resolve(root, 'evidence/scripts/run_cuda_ep.py'), 'utf8');
  for (const token of ['providers=[("CUDAExecutionProvider", PROVIDER_OPTIONS)]', 'get_available_providers()', 'session.get_providers()', 'session.get_provider_options()', 'enable_profiling = True', 'end_profiling()', '/proc/self/maps', 'evidence/tooling/raw-golden', 'runtimeExecuted', 'absolute + relative * np.abs']) assert(runner.includes(token), `CUDA runner introspection missing: ${token}`);
  assert(!/providers\s*=\s*\[[^\]]*CPUExecutionProvider/s.test(runner), 'CUDA runner permits explicit CPU provider');
  const lock = await readFile(resolve(root, 'evidence/tooling/cuda-requirements.lock'), 'utf8');
  const locked = Object.fromEntries(lock.split('\n').filter((line) => line && !line.startsWith('#')).map((line) => {
    const match = line.match(/^([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})$/); assert(match, `CUDA unhashed or malformed dependency: ${line}`); return [match[1], match[2]];
  }));
  assert(equal(locked, PACKAGES), 'CUDA lockfile package/version drift');
}

export async function validateCudaReplay(root, replay) {
  assert(replay.task === 'T14-LNX-CUDA-01' && replay.rounds.length === 2, 'CUDA replay scope/round drift');
  assert((replay.recorded === true && replay.mode === 'recorded-blocked-host-replay') || (replay.recorded === false && replay.mode === 'blocked-host-replay'), 'CUDA replay record/mode boundary drift');
  assert(equal(replay.protectedPaths, TRACKED_CUDA_EVIDENCE), 'CUDA replay protected path ownership drift');
  assert(replay.trackedEvidenceUnchanged === true && replay.comparison.failureStageEqual && replay.comparison.exitCodeEqual && replay.comparison.stateEqual && replay.comparison.failureMissingEqual, 'CUDA replay determinism or tracked evidence drift');
  for (const round of replay.rounds) {
    assert(round.command.exitCode === 21 && round.report.state === 'blocked' && round.report.failureStage === 'nvidia-hardware-driver-preflight', 'CUDA replay blocked stage drift');
    assert(equal(round.report.failure.missing, ['NVIDIA GPU', 'NVIDIA driver / nvidia-smi', 'onnxruntime-gpu', ...REQUIRED_LIBRARIES]), 'CUDA replay blocker provenance is stale');
    assert(round.report.staticVerified && !round.report.buildVerified && !round.report.runtimeExecuted && !round.report.hostInferenceVerified && !round.report.goldenExecuted && !round.report.supported && !round.report.task14Complete, 'CUDA replay blocked status drift');
    assert(round.report.availableProviders.length === 0 && round.report.sessionProviders.length === 0 && round.report.profile.cudaNodeCount === 0 && Object.values(round.report.sharedLibraries.requiredLoaded).every((value) => value === false), 'CUDA replay introspection overclaim');
    assert(equal(round.trackedBefore, round.trackedAfter) && Object.values(round.trackedBefore).every((item) => item.exists && item.bytes > 0 && /^[0-9a-f]{64}$/.test(item.sha256)), 'CUDA replay modified tracked evidence bytes/SHA');
  }
  const replayReportPath = 'evidence/reports/cuda-ep-replay-report.json';
  const currentIdentityPaths = replay.recorded
    // A recorded replay is historical preservation evidence. The conversion summary,
    // golden manifest, and Task-1 replay are finalized after this CUDA step and are
    // independently bound by aggregate validation. Ordinary replay still binds every
    // protected path to its current bytes.
    ? RECORDED_CUDA_CURRENT_IDENTITY
    : TRACKED_CUDA_EVIDENCE;
  for (const path of currentIdentityPaths) {
    const bytes = await readFile(resolve(root, path));
    const identity = replay.rounds[0].trackedBefore[path];
    assert(identity.bytes === bytes.length && identity.sha256 === sha256(bytes), `CUDA protected tracked evidence current identity drift: ${path}`);
  }
  if (replay.recorded) {
    const path = replayReportPath;
    const current = await readFile(resolve(root, path));
    assert(equal(JSON.parse(current), replay), 'CUDA recorded replay object differs from the current tracked report bytes');
    const staged = spawnSync('git', ['show', `:${path}`], { cwd: root });
    assert(staged.status === 0 && current.equals(staged.stdout), 'CUDA recorded replay current bytes differ from the staged Git blob');
  }
}
