#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  compareDecodedValues, compareRawValues, validateCudaEvidence, validateCudaReplay,
  validateLibraryArtifacts, validateProductionRunner, validateProfileArtifact, validateRawArtifacts, validateWebReferenceBindings,
} from './cuda_evidence_validation.mjs';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const manifest = await readJson('evidence/conversions/cuda-ep-spike-manifest.json');
const blocked = await readJson('evidence/reports/cuda-ep-spike-report.json');
const replay = await readJson('evidence/reports/cuda-ep-replay-report.json');
const ordinaryReplayPath = process.env.RIMEFLOW_CUDA_ORDINARY_REPLAY ?? '.evidence/cuda/final-independent-replay.json';
const ordinaryReplay = await readJson(ordinaryReplayPath);
await validateCudaEvidence(root, manifest, blocked);
const expectFailure = async (name, operation, match) => {
  try { await operation(); } catch (error) {
    if (match && !String(error).includes(match)) throw new Error(`${name}: unexpected error: ${error}`);
    return { name, rejected: true, error: String(error) };
  }
  throw new Error(`CUDA guard unexpectedly accepted: ${name}`);
};
const cases = [];

const boundary = compareRawValues(new Float64Array([1.0001005]), new Float64Array([1]), 1e-5, 1e-4);
if (!boundary.passed || boundary.mismatchCount !== 0) throw new Error('加和容差边界正例被拒绝');
const outside = compareRawValues(new Float64Array([1.0001105]), new Float64Array([1]), 1e-5, 1e-4);
if (outside.passed || outside.mismatchCount !== 1) throw new Error('加和容差越界反例被接受');

const cpu = structuredClone(blocked); cpu.requestedProviders = ['CPUExecutionProvider'];
cases.push(await expectFailure('CPU ORT 冒充 CUDA', () => validateCudaEvidence(root, manifest, cpu), 'ordinary CPU ORT'));
const ortDrift = structuredClone(blocked); ortDrift.lockedPackages['onnxruntime-gpu'] = '1.27.0';
cases.push(await expectFailure('ORT/CUDA/cuDNN 版本漂移', () => validateCudaEvidence(root, manifest, ortDrift), 'package lock drift'));
const optionsDrift = structuredClone(manifest); optionsDrift.provider.options.device_id = '1';
cases.push(await expectFailure('provider options 漂移', () => validateCudaEvidence(root, optionsDrift, blocked), 'provider options drift'));
const modelDrift = structuredClone(blocked); modelDrift.model.sha256 = '0'.repeat(64);
cases.push(await expectFailure('模型 SHA 漂移', () => validateCudaEvidence(root, manifest, modelDrift), 'model identity drift'));
const shapeDrift = structuredClone(blocked); shapeDrift.ioContract.output.shape = [1, 8400, 84];
cases.push(await expectFailure('I/O Shape/dtype 漂移', () => validateCudaEvidence(root, manifest, shapeDrift), 'I/O contract drift'));
const supported = structuredClone(blocked); supported.supported = true;
cases.push(await expectFailure('无 NVIDIA runner 却 supported=true', () => validateCudaEvidence(root, manifest, supported), 'overclaims supported'));
const publication = structuredClone(manifest); publication.artifactHandling.runtimeLibrariesTracked = true;
cases.push(await expectFailure('临时 runtime/raw/二进制进入 Git 或发布边界', () => validateCudaEvidence(root, publication, blocked), 'publication boundary'));

const guardRoot = resolve(root, '.evidence/cuda');
await mkdir(guardRoot, { recursive: true });
const temporary = await mkdtemp(resolve(guardRoot, 'guards-'));
try {
  const profilePath = resolve(temporary, 'profile.json');
  const profileBytes = Buffer.from(JSON.stringify([{ cat: 'Node', args: { provider: 'CPUExecutionProvider' } }]));
  await writeFile(profilePath, profileBytes);
  const forgedProfile = { artifact: { path: profilePath, bytes: profileBytes.length, sha256: sha256(profileBytes), format: 'ort-chrome-trace-json' }, nodeProviderCounts: { CUDAExecutionProvider: 99 }, cudaNodeCount: 99, cpuNodeCount: 0 };
  cases.push(await expectFailure('伪造 CUDA node 计数但 profile 未变', () => validateProfileArtifact(root, forgedProfile), 'not recomputed'));

  const libraryNames = ['libonnxruntime_providers_cuda.so', 'libcuda.so', 'libcudart.so', 'libcudnn.so', 'libcublas.so'];
  const libraryVersions = { 'libonnxruntime_providers_cuda.so': '1.26.0', 'libcuda.so': '570.26', 'libcudart.so': '12.8.90', 'libcudnn.so': '9.10.2.21', 'libcublas.so': '12.8.4.1' };
  const entries = {};
  const mapLines = [];
  for (const name of libraryNames) {
    const libraryDirectory = resolve(temporary, name.replaceAll('.', '-'));
    await mkdir(libraryDirectory);
    const libraryPath = resolve(libraryDirectory, name);
    const libraryBytes = Buffer.from(`fixture-${name}`);
    await writeFile(libraryPath, libraryBytes);
    entries[name] = {
      path: libraryPath, realpath: libraryPath, basename: name,
      bytes: libraryBytes.length, sha256: sha256(libraryBytes),
      elfClass: 'ELF64', elfMachine: 'Advanced Micro Devices X86-64', soname: name,
      componentVersion: libraryVersions[name],
      versionSource: name === 'libcuda.so' ? 'nvidia-smi driver_version' : name === 'libonnxruntime_providers_cuda.so' ? 'onnxruntime-gpu package' : name === 'libcudart.so' ? 'nvidia-cuda-runtime-cu12 package' : name === 'libcudnn.so' ? 'nvidia-cudnn-cu12 package' : 'nvidia-cublas-cu12 package',
    };
    mapLines.push(`7f00-7f01 r-xp 0 00:00 0 ${libraryPath}`);
  }
  entries['libonnxruntime_providers_cuda.so'].sha256 = '0'.repeat(64);
  const mapsPath = resolve(temporary, 'proc-self-maps.txt');
  const mapsBytes = Buffer.from(`${mapLines.join('\n')}\n`);
  await writeFile(mapsPath, mapsBytes);
  const libraryReport = structuredClone(blocked);
  libraryReport.versions = { onnxruntime: '1.26.0', cudaRuntime: '12.8.90', cudnn: '9.10.2.21', nvidiaDriver: '570.26' };
  libraryReport.sharedLibraries = {
    mapsArtifact: { path: mapsPath, bytes: mapsBytes.length, sha256: sha256(mapsBytes) },
    requiredLoaded: Object.fromEntries(libraryNames.map((name) => [name, true])),
    entries,
  };
  cases.push(await expectFailure('伪造 shared library boolean/path/SHA', () => validateLibraryArtifacts(root, libraryReport), 'artifact bytes/SHA drift'));

  entries['libonnxruntime_providers_cuda.so'].sha256 = sha256(await readFile(entries['libonnxruntime_providers_cuda.so'].path));
  cases.push(await expectFailure('自洽 path/SHA/maps 的普通文本库冒充 CUDA ELF', () => validateLibraryArtifacts(root, libraryReport), 'not ELF64 x86_64'));

  const fakeRunnerPath = resolve(temporary, 'fake-node-runner');
  const fakeRunnerBytes = Buffer.from('#!/usr/bin/env node\nprocess.exit(0);\n');
  await writeFile(fakeRunnerPath, fakeRunnerBytes);
  const fakeRunnerReport = structuredClone(blocked);
  fakeRunnerReport.productionPostprocess.runnerArtifact = { path: fakeRunnerPath, bytes: fakeRunnerBytes.length, sha256: sha256(fakeRunnerBytes) };
  fakeRunnerReport.productionPostprocess.sourceArtifacts = [];
  cases.push(await expectFailure('report 自选假 executable 冒充生产 raw-golden', () => validateProductionRunner(root, fakeRunnerReport), 'path must be canonical'));

  const referencePath = resolve(temporary, 'reference.f32le');
  const rawPath = resolve(temporary, 'raw.f32le');
  const count = 84 * 8400;
  const referenceBytes = Buffer.alloc(count * 4); referenceBytes.writeFloatLE(1, 0);
  const rawBytes = Buffer.from(referenceBytes); rawBytes.writeFloatLE(1.001, 0);
  await writeFile(referencePath, referenceBytes); await writeFile(rawPath, rawBytes);
  const fixture = {
    id: 'finite-out-of-tolerance',
    referenceRaw: { path: referencePath, bytes: referenceBytes.length, sha256: sha256(referenceBytes), shape: [1, 84, 8400], dtype: 'float32-le', elementCount: count },
    cudaRaw: { path: rawPath, bytes: rawBytes.length, sha256: sha256(rawBytes), shape: [1, 84, 8400], dtype: 'float32-le', elementCount: count },
    rawComparison: { passed: true, mismatchCount: 0, maximumAbsolute: 0, maximumRelative: 0, maximumDifferenceFlatIndex: 0, maximumExcessFlatIndex: 0, nearZeroReferenceCount: count - 1, rule: 'abs(actual-reference) <= rawTensorAbsolute + rawTensorRelative * abs(reference)' },
  };
  cases.push(await expectFailure('有限 raw 越界且同步更新 SHA/passed', () => validateRawArtifacts(root, fixture, { rawTensorAbsolute: 1e-5, rawTensorRelative: 1e-4 }), 'not recomputed'));

  const forgedReferenceDir = resolve(root, '.evidence/cuda/guards-forged-reference');
  await mkdir(forgedReferenceDir, { recursive: true });
  const forgedManifestPath = resolve(forgedReferenceDir, 'manifest.json');
  const forgedManifestBytes = Buffer.from(JSON.stringify({ sourceReferenceSha256: '0'.repeat(64), fixtures: [] }));
  await writeFile(forgedManifestPath, forgedManifestBytes);
  const forgedReferenceReport = { webReferenceManifest: { path: forgedManifestPath, bytes: forgedManifestBytes.length, sha256: sha256(forgedManifestBytes) }, fixtures: [] };
  const frozen = await readJson('evidence/golden/web-reference.json');
  cases.push(await expectFailure('自造 reference 不能替代 canonical Web reference', () => validateWebReferenceBindings(root, forgedReferenceReport, frozen), 'not bound to tracked frozen reference'));

  const decodedActual = [{ classId: 1, score: 0.9, bbox: [0, 0, 1, 1] }];
  const decodedExpected = [{ classId: 2, score: 0.9, bbox: [0, 0, 1, 1] }];
  const decoded = compareDecodedValues(decodedActual, decodedExpected, { confidenceAbsolute: 1e-4, decodedBoxAbsolute: 1e-4, boxIouMinimum: 0.999 });
  if (decoded.passed || decoded.classMismatchCount !== 1) throw new Error('decoded 自相矛盾未被重算拒绝');

  const trackedReplay = structuredClone(replay);
  const protectedPath = 'evidence/conversions/cuda-ep-spike-manifest.json';
  const forged = { exists: true, bytes: 1, sha256: 'f'.repeat(64), mtimeNs: '1' };
  trackedReplay.rounds[0].trackedBefore[protectedPath] = forged;
  trackedReplay.rounds[0].trackedAfter[protectedPath] = structuredClone(forged);
  cases.push(await expectFailure('before/after 状态相同但 tracked bytes/SHA 变化', () => validateCudaReplay(root, trackedReplay), 'current identity drift'));

  const forgedOrdinaryReplay = structuredClone(ordinaryReplay);
  const replayReportPath = 'evidence/reports/cuda-ep-replay-report.json';
  for (const round of forgedOrdinaryReplay.rounds) {
    round.trackedBefore[replayReportPath] = structuredClone(forged);
    round.trackedAfter[replayReportPath] = structuredClone(forged);
  }
  cases.push(await expectFailure('普通 replay 的 tracked replay report before/after 同值伪造', () => validateCudaReplay(root, forgedOrdinaryReplay), `current identity drift: ${replayReportPath}`));
} finally { await rm(temporary, { recursive: true, force: true }); }

console.log(JSON.stringify({ ok: true, toleranceBoundary: { reference: 1, deltaAccepted: 0.0001005, deltaRejected: 0.0001105, absolute: 1e-5, relative: 1e-4 }, negativeCases: cases }));
