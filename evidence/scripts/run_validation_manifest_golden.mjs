import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import * as ortNative from '../tooling/web/node_modules/onnxruntime-node/dist/index.js';
import * as ortWeb from '../tooling/web/node_modules/onnxruntime-web/dist/ort.node.min.mjs';
import { preprocessCanonical, readPpm, tensorDigest } from './preprocess_contract.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const matrix = await readJson('evidence/requirements/validation-requirement-test-matrix.json');
const manifestBytes = await readFile(resolve(root, matrix.validationManifestPath));
const manifest = JSON.parse(manifestBytes);
const fixtureManifestBytes = await readFile(resolve(root, manifest.fixtures.manifestPath));
const fixtureManifest = JSON.parse(fixtureManifestBytes);
const frozenBytes = await readFile(resolve(root, manifest.fixtures.goldenPath));
const frozen = JSON.parse(frozenBytes);
const modelBytes = await readFile(resolve(root, manifest.model.path));
const temporaryRoot = await mkdtemp(join(tmpdir(), 'rimeflow-validation-golden-'));

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => `${JSON.stringify(value, null, 2)}\n`;
const round = (value, digits = 9) => Number(value.toFixed(digits));

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

async function readJson(path) {
  return JSON.parse(await readFile(resolve(root, path), 'utf8'));
}

function artifact(id) {
  const found = manifest.artifacts.find((entry) => entry.id === id);
  invariant(found, `manifest artifact missing: ${id}`);
  return found;
}

function runtimeName(session, artifactEntry, direction) {
  const binding = artifactEntry[direction];
  const runtimeNames = direction === 'input' ? session.inputNames : session.outputNames;
  invariant(binding.role === (direction === 'input' ? 'image' : 'detections'), `${artifactEntry.id}: invalid ${direction} role`);
  invariant(runtimeNames.includes(binding.runtimeName), `${artifactEntry.id}: manifest ${direction} binding not present at runtime`);
  return binding.runtimeName;
}

function tensorSummary(values) {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let finiteCount = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    finiteCount += 1;
  }
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  return {
    role: 'detections',
    shape: [1, 84, 8400],
    elementCount: values.length,
    finiteCount,
    min: round(minimum),
    max: round(maximum),
    mean: round(sum / finiteCount),
    sha256Float32Le: sha256(bytes),
  };
}

function compareRaw(reference, candidate, tolerances) {
  invariant(reference.length === candidate.length, 'raw tensor length mismatch');
  let maxAbsoluteDifference = 0;
  let maxRelativeDifference = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index];
    const actual = candidate[index];
    invariant(Number.isFinite(expected) && Number.isFinite(actual), `raw tensor contains non-finite value at ${index}`);
    const absolute = Math.abs(expected - actual);
    const relative = absolute / Math.max(Math.abs(expected), tolerances.rawTensorAbsolute);
    maxAbsoluteDifference = Math.max(maxAbsoluteDifference, absolute);
    maxRelativeDifference = Math.max(maxRelativeDifference, relative);
    invariant(
      absolute <= tolerances.rawTensorAbsolute + tolerances.rawTensorRelative * Math.abs(expected),
      `raw tensor exceeds frozen tolerance at ${index}: absolute=${absolute} relative=${relative}`,
    );
  }
  return {
    passed: true,
    maxAbsoluteDifference,
    maxRelativeDifference,
    tolerance: {
      absolute: tolerances.rawTensorAbsolute,
      relative: tolerances.rawTensorRelative,
    },
  };
}

function boxIou(left, right) {
  const x1 = Math.max(left[0], right[0]);
  const y1 = Math.max(left[1], right[1]);
  const x2 = Math.min(left[2], right[2]);
  const y2 = Math.min(left[3], right[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left[2] - left[0]) * Math.max(0, left[3] - left[1]);
  const rightArea = Math.max(0, right[2] - right[0]) * Math.max(0, right[3] - right[1]);
  const union = leftArea + rightArea - intersection;
  return union === 0 ? 0 : intersection / union;
}

function compareDecoded(reference, candidate, tolerances) {
  invariant(reference.length === candidate.length, 'decoded detection count mismatch');
  let minimumIou = 1;
  let maximumConfidenceDifference = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index];
    const actual = candidate[index];
    invariant(expected.classId === actual.classId, `decoded class mismatch at ${index}`);
    const confidenceDifference = Math.abs(expected.score - actual.score);
    const iou = boxIou(expected.bbox, actual.bbox);
    maximumConfidenceDifference = Math.max(maximumConfidenceDifference, confidenceDifference);
    minimumIou = Math.min(minimumIou, iou);
    invariant(confidenceDifference <= tolerances.confidenceAbsolute, `decoded confidence exceeds tolerance at ${index}`);
    invariant(iou >= tolerances.boxIouMinimum, `decoded box IoU below tolerance at ${index}`);
  }
  return {
    passed: true,
    detectionCount: candidate.length,
    maximumConfidenceDifference,
    minimumIou: candidate.length === 0 ? null : minimumIou,
  };
}

async function productionPostprocess(values, image, fixtureId, runtimeId, repeat) {
  const rawPath = resolve(temporaryRoot, `${fixtureId}-${runtimeId}-${repeat}.f32le`);
  const decodedPath = resolve(temporaryRoot, `${fixtureId}-${runtimeId}-${repeat}.json`);
  await writeFile(rawPath, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
  const execution = spawnSync('cargo', [
    'run', '--quiet', '--offline', '--locked',
    '--manifest-path', 'evidence/tooling/raw-golden/Cargo.toml',
    '--', rawPath, String(image.width), String(image.height), decodedPath,
  ], { cwd: root, encoding: 'utf8' });
  invariant(execution.status === 0, `production postprocess failed: ${execution.stderr}`);
  return JSON.parse(await readFile(decodedPath, 'utf8'));
}

async function inferRepeated(session, artifactEntry, canonicalInput, image, fixtureId) {
  const inputName = runtimeName(session, artifactEntry, 'input');
  const outputName = runtimeName(session, artifactEntry, 'output');
  const runs = [];
  for (let repeat = 1; repeat <= manifest.fixtures.repeatCount; repeat += 1) {
    const outputs = await session.run({
      [inputName]: new (artifactEntry.id === 'web-onnx-wasm' ? ortWeb.Tensor : ortNative.Tensor)(
        'float32', canonicalInput, artifactEntry.input.shape,
      ),
    });
    const values = outputs[outputName].data;
    invariant(values instanceof Float32Array, `${artifactEntry.id}: detections tensor must be Float32Array`);
    invariant(values.length === 84 * 8400, `${artifactEntry.id}: detections element count mismatch`);
    runs.push({
      repeat,
      summary: tensorSummary(values),
      values: new Float32Array(values),
      decoded: await productionPostprocess(values, image, fixtureId, artifactEntry.id, repeat),
    });
  }
  const rawDigest = runs[0].summary.sha256Float32Le;
  const decoded = stable(runs[0].decoded);
  invariant(runs.every((run) => run.summary.sha256Float32Le === rawDigest), `${artifactEntry.id}/${fixtureId}: raw output is not deterministic`);
  invariant(runs.every((run) => stable(run.decoded) === decoded), `${artifactEntry.id}/${fixtureId}: decoded output is not deterministic`);
  return runs;
}

function runGreenTests() {
  const tests = matrix.requirements.flatMap((requirement) =>
    requirement.scenarios.flatMap((scenario) => scenario.tests),
  );
  return tests.map((test) => {
    const execution = spawnSync('cargo', [
      'test', '--config', matrix.cargoDependencyInput.configPath, '--offline', '--locked',
      '--manifest-path', matrix.cargoDependencyInput.manifestPath,
      `tests::${test.testFunction}`, '--', '--exact',
    ], { cwd: root, encoding: 'utf8' });
    invariant(execution.status === 0, `${test.testId} failed:\n${execution.stdout}\n${execution.stderr}`);
    return {
      testId: test.testId,
      testFunction: test.testFunction,
      targetAssertion: test.expectedResult.assertion,
      outcome: 'green',
      environmentFailure: false,
      processExitCode: execution.status,
    };
  });
}

try {
  invariant(sha256(modelBytes) === manifest.model.sha256, 'model SHA mismatch');
  invariant(sha256(fixtureManifestBytes) === manifest.fixtures.manifestSha256, 'fixture manifest SHA mismatch');
  invariant(sha256(frozenBytes) === manifest.fixtures.goldenSha256, 'frozen golden SHA mismatch');

  const matrixValidation = spawnSync('node', ['evidence/scripts/validate_validation_matrix.mjs'], { cwd: root, encoding: 'utf8' });
  invariant(matrixValidation.status === 0, `Validation matrix failed:\n${matrixValidation.stdout}\n${matrixValidation.stderr}`);
  const tests = runGreenTests();

  ortWeb.env.wasm.numThreads = 1;
  ortWeb.env.wasm.proxy = false;
  const webSession = await ortWeb.InferenceSession.create(modelBytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const nativeSession = await ortNative.InferenceSession.create(modelBytes, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  });

  const webArtifact = artifact('web-onnx-wasm');
  const nativeArtifact = artifact('legacy-native-ort');
  const hostArtifact = artifact('host-nchw');
  invariant(stable(hostArtifact.input.shape) === stable(nativeArtifact.input.shape), 'host NCHW shape does not match Legacy ORT');
  invariant(hostArtifact.input.role === 'image', 'host NCHW image role missing');

  const fixtureReports = [];
  for (const fixture of fixtureManifest.images) {
    const expected = frozen.fixtures.find((entry) => entry.id === fixture.id);
    invariant(expected, `${fixture.id}: frozen Web fixture missing`);
    const image = readPpm(await readFile(resolve(root, fixture.path)));
    const prep = preprocessCanonical(image);
    invariant(tensorDigest(prep.tensor) === expected.canonicalInput.sha256Float32Le, `${fixture.id}: host NCHW input digest mismatch`);
    const webRuns = await inferRepeated(webSession, webArtifact, prep.tensor, image, fixture.id);
    const nativeRuns = await inferRepeated(nativeSession, nativeArtifact, prep.tensor, image, fixture.id);
    invariant(webRuns[0].summary.sha256Float32Le === expected.runs[0].rawTensor.sha256Float32Le, `${fixture.id}: Web raw digest drift`);
    const decodedComparison = compareDecoded(expected.runs[0].decoded, webRuns[0].decoded, frozen.tolerances);
    const nativeComparison = compareRaw(webRuns[0].values, nativeRuns[0].values, frozen.tolerances);
    const nativeDecodedComparison = compareDecoded(webRuns[0].decoded, nativeRuns[0].decoded, frozen.tolerances);
    fixtureReports.push({
      id: fixture.id,
      imageSha256: fixture.sha256,
      hostNchw: {
        role: hostArtifact.input.role,
        shape: hostArtifact.input.shape,
        dtype: hostArtifact.input.dtype,
        sha256Float32Le: tensorDigest(prep.tensor),
      },
      web: {
        role: webArtifact.output.role,
        shape: webArtifact.output.shape,
        runs: webRuns.map(({ repeat, summary, decoded }) => ({ repeat, summary, decoded })),
        deterministic: true,
        frozenRawDigestMatches: true,
        decodedComparison,
      },
      legacyNativeOrt: {
        role: nativeArtifact.output.role,
        shape: nativeArtifact.output.shape,
        runs: nativeRuns.map(({ repeat, summary, decoded }) => ({ repeat, summary, decoded })),
        deterministic: true,
        rawComparisonToWeb: nativeComparison,
        decodedComparisonToWeb: nativeDecodedComparison,
      },
    });
  }

  await webSession.release();
  await nativeSession.release();
  const report = {
    schemaVersion: 1,
    change: matrix.change,
    phase: matrix.phase,
    phase1DependencyCommit: matrix.phase1DependencyCommit,
    identities: {
      baseRuntime: manifest.baseRuntime,
      validationStartCommit: 'eaf4dc1f8e5a88d4a235bc85f6844e6ff29900f2',
      model: { path: manifest.model.path, sha256: manifest.model.sha256 },
      fixtureManifest: { path: manifest.fixtures.manifestPath, sha256: manifest.fixtures.manifestSha256 },
      frozenGolden: { path: manifest.fixtures.goldenPath, sha256: manifest.fixtures.goldenSha256 },
      validationManifest: { path: matrix.validationManifestPath, sha256: sha256(manifestBytes) },
    },
    postprocess: {
      ...manifest.postprocess,
      execution: 'evidence/tooling/raw-golden delegates directly to src/postprocess.rs',
      duplicateArtifactSpecificImplementations: 0,
    },
    runtime: {
      web: { name: 'onnxruntime-web', version: ortWeb.env.versions.web, executionProvider: 'wasm', threads: 1 },
      legacyNativeOrt: { name: 'onnxruntime-node', version: ortNative.env.versions.node, executionProvider: 'cpu' },
    },
    summary: {
      testCount: tests.length,
      green: tests.length,
      fixtureCount: fixtureReports.length,
      repeatCount: manifest.fixtures.repeatCount,
      allRuntimeComparisonsPassed: true,
      environmentFailures: 0,
    },
    tests,
    fixtures: fixtureReports,
  };
  const reportFlag = process.argv.indexOf('--write-report');
  const reportPath = reportFlag === -1 ? matrix.reportPath : process.argv[reportFlag + 1];
  invariant(reportPath, '--write-report requires a repository-relative path');
  const outputPath = resolve(root, reportPath);
  invariant(outputPath.startsWith(`${root}/`), 'report path must stay inside the repository');
  await writeFile(outputPath, stable(report));
  process.stdout.write(`${JSON.stringify(report.summary)}\n`);
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}
