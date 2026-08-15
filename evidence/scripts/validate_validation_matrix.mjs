import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const matrixPath = resolve(root, 'evidence/requirements/validation-requirement-test-matrix.json');
const matrix = JSON.parse(await readFile(matrixPath, 'utf8'));
const expectedCargoDependencyInput = {
  manifestPath: 'evidence/tooling/validation-contract/Cargo.toml',
  lockPath: 'evidence/tooling/validation-contract/Cargo.lock',
  configPath: 'evidence/tooling/validation-contract/cargo-config.toml',
  vendorDirectory: 'evidence/tooling/validation-contract/vendor',
  policy: 'all Cargo.lock registry sources are vendored with Cargo checksum metadata',
};

const requiredScenarios = new Set([
  'Input semantics are explicit::Layout and dtype differ by artifact',
  'Input semantics are explicit::Input contract is unsupported',
  'Output semantics are mapped by logical role::Converted artifact renames outputs',
  'Output semantics are mapped by logical role::Output role is missing',
  'Model conversion preserves a golden behavior baseline::Detection artifact matches baseline',
  'Model conversion preserves a golden behavior baseline::Conversion changes behavior beyond tolerance',
  'Preprocessing and postprocessing ownership is explicit::Preprocessing remains outside the model',
  'Preprocessing and postprocessing ownership is explicit::NMS is fused into one artifact',
  'Golden fixtures and tolerances are versioned before adapter implementation::A platform artifact is compared with the baseline',
  'Golden fixtures and tolerances are versioned before adapter implementation::A tolerance is missing or changed after seeing a result',
]);

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function sorted(values) {
  return [...values].sort((left, right) => left.localeCompare(right));
}

function sameSet(actual, expected, label) {
  const actualSorted = sorted(actual);
  const expectedSorted = sorted(expected);
  invariant(
    JSON.stringify(actualSorted) === JSON.stringify(expectedSorted),
    `${label}: expected ${JSON.stringify(expectedSorted)}, got ${JSON.stringify(actualSorted)}`,
  );
}

async function readJson(path) {
  return JSON.parse(await readFile(resolve(root, path), 'utf8'));
}

async function sha256(path) {
  const bytes = await readFile(resolve(root, path));
  return createHash('sha256').update(bytes).digest('hex');
}

invariant(matrix.schemaVersion === 1, 'matrix schemaVersion must be 1');
invariant(matrix.repository === 'github/rimeflow-nn-validation', 'matrix repository mismatch');
invariant(matrix.ownership === 'Validation', 'matrix ownership mismatch');
invariant(matrix.phase === '5-validation-manifest-golden', 'matrix phase mismatch');
invariant(
  JSON.stringify(matrix.cargoDependencyInput) === JSON.stringify(expectedCargoDependencyInput),
  'hermetic Cargo dependency input mismatch',
);
invariant(/^[0-9a-f]{40}$/.test(matrix.originalBaseCommit), 'invalid original base commit');
invariant(/^[0-9a-f]{40}$/.test(matrix.phase1DependencyCommit), 'invalid Phase 1 dependency commit');
invariant(/^[0-9a-f]{64}$/.test(matrix.model.sha256), 'invalid model SHA-256');
invariant(matrix.validationManifestPath === 'evidence/manifest/validation-runtime-manifest.json', 'Validation manifest path mismatch');
invariant(matrix.baseRuntime?.repository === 'github/rimeflow-nn-base', 'Base repository mismatch');
for (const field of ['commit', 'tree', 'parent']) {
  invariant(/^[0-9a-f]{40}$/.test(matrix.baseRuntime?.[field] ?? ''), `invalid Base ${field}`);
}

const ancestry = spawnSync(
  'git',
  ['merge-base', '--is-ancestor', matrix.phase1DependencyCommit, 'HEAD'],
  { cwd: root, encoding: 'utf8' },
);
invariant(ancestry.status === 0, 'Phase 1 dependency commit is not an ancestor of HEAD');

const scenarios = new Set();
const tests = [];
for (const requirement of matrix.requirements) {
  invariant(typeof requirement.requirement === 'string', 'requirement name must be a string');
  for (const scenario of requirement.scenarios ?? []) {
    const key = `${requirement.requirement}::${scenario.scenario}`;
    invariant(!scenarios.has(key), `duplicate scenario mapping: ${key}`);
    scenarios.add(key);
    invariant(Array.isArray(scenario.tests) && scenario.tests.length > 0, `scenario has no tests: ${key}`);
    for (const test of scenario.tests) tests.push({ ...test, scenario: key });
  }
}
sameSet(scenarios, requiredScenarios, 'Validation OpenSpec scenario coverage');

const testIds = new Set();
const testFunctions = new Set();
for (const test of tests) {
  invariant(/^RFB-VAL-[A-Z0-9-]+$/.test(test.testId), `invalid Validation test ID: ${test.testId}`);
  invariant(!testIds.has(test.testId), `duplicate Validation test ID: ${test.testId}`);
  invariant(!testFunctions.has(test.testFunction), `duplicate Validation test function: ${test.testFunction}`);
  testIds.add(test.testId);
  testFunctions.add(test.testFunction);
  const expectedCommand = `cargo test --config ${matrix.cargoDependencyInput.configPath} --offline --locked --manifest-path ${matrix.cargoDependencyInput.manifestPath} tests::${test.testFunction} -- --exact`;
  invariant(test.command === expectedCommand, `non-reproducible command for ${test.testId}`);
  invariant(test.expectedResult?.outcome === 'green', `invalid expected result for ${test.testId}`);
  invariant(typeof test.expectedResult?.assertion === 'string' && test.expectedResult.assertion.length > 0, `missing green assertion for ${test.testId}`);
}

const cargoMetadata = spawnSync(
  'cargo',
  [
    'metadata',
    '--config', matrix.cargoDependencyInput.configPath,
    '--offline',
    '--locked',
    '--manifest-path', matrix.cargoDependencyInput.manifestPath,
    '--format-version', '1',
  ],
  { cwd: root, encoding: 'utf8' },
);
invariant(
  cargoMetadata.status === 0,
  `hermetic Cargo dependency input is incomplete: ${cargoMetadata.stderr.trim()}`,
);

const testSource = await readFile(resolve(root, matrix.testSource), 'utf8');
const registrations = [...testSource.matchAll(/rfb_val_test!\(\s*"(RFB-VAL-[A-Z0-9-]+)",\s*([a-z0-9_]+),/gs)]
  .map((match) => ({ testId: match[1], testFunction: match[2] }));
sameSet(registrations.map((item) => item.testId), testIds, 'orphan or missing RFB-VAL test IDs');
sameSet(registrations.map((item) => item.testFunction), testFunctions, 'orphan or missing Rust tests');
invariant(new Set(registrations.map((item) => item.testId)).size === registrations.length, 'test source repeats an RFB-VAL ID');

const fixtureManifest = await readJson('evidence/fixtures/manifest.json');
const rawFixtureIds = fixtureManifest.rawTensorFixtures.map((fixture) => fixture.id);
const imageFixtureIds = fixtureManifest.images.map((fixture) => fixture.id);
sameSet(rawFixtureIds, matrix.fixtureCoverage.requiredScenarioIds, 'raw fixture coverage');
sameSet(imageFixtureIds, matrix.fixtureCoverage.webInferenceFixtureIds, 'image fixture coverage');

const webReference = await readJson('evidence/golden/web-reference.json');
sameSet(webReference.fixtures.map((fixture) => fixture.id), imageFixtureIds, 'Web golden fixture coverage');
for (const fixture of webReference.fixtures) {
  invariant(fixture.runs.length === matrix.fixtureCoverage.repeatCount, `${fixture.id}: repeat count mismatch`);
  invariant(fixture.determinism?.allRawDigestsEqual === true, `${fixture.id}: raw output is not deterministic`);
  invariant(fixture.determinism?.allDecodedEqual === true, `${fixture.id}: decoded output is not deterministic`);
}

const modelContract = await readJson('evidence/model/model-contract.json');
invariant(modelContract.source.commit === matrix.originalBaseCommit, 'model contract source commit mismatch');
invariant(modelContract.source.modelSha256 === matrix.model.sha256, 'model contract SHA mismatch');
invariant(await sha256(matrix.model.path) === matrix.model.sha256, 'model bytes do not match the frozen SHA');

const validationManifest = await readJson(matrix.validationManifestPath);
invariant(validationManifest.schemaVersion === 1, 'Validation manifest schemaVersion mismatch');
invariant(JSON.stringify(validationManifest.baseRuntime) === JSON.stringify(matrix.baseRuntime), 'Validation manifest Base identity mismatch');
invariant(validationManifest.model?.path === matrix.model.path, 'Validation manifest model path mismatch');
invariant(validationManifest.model?.sha256 === matrix.model.sha256, 'Validation manifest model SHA mismatch');
invariant(validationManifest.postprocess?.owner === 'src/postprocess.rs', 'postprocess owner mismatch');
invariant(validationManifest.postprocess?.decode === 'operator', 'decode ownership mismatch');
invariant(validationManifest.postprocess?.threshold === 0.25, 'threshold mismatch');
invariant(validationManifest.postprocess?.nms === 'operator', 'NMS ownership mismatch');
invariant(await sha256(validationManifest.postprocess.owner) === validationManifest.postprocess.sourceSha256, 'production postprocess SHA mismatch');
sameSet(validationManifest.artifacts.map((artifact) => artifact.id), ['web-onnx-wasm', 'legacy-native-ort', 'host-nchw'], 'runtime artifact coverage');
for (const artifact of validationManifest.artifacts) {
  invariant(artifact.input?.role === 'image', `${artifact.id}: missing image role`);
  invariant(artifact.input?.layout === 'NCHW' && artifact.input?.dtype === 'float32', `${artifact.id}: invalid input contract`);
  invariant(artifact.output?.role === 'detections', `${artifact.id}: missing detections role`);
  invariant(JSON.stringify(artifact.output?.shape) === '[1,84,8400]', `${artifact.id}: invalid detections shape`);
  invariant(artifact.output?.nmsFused === false, `${artifact.id}: unexpected fused NMS`);
}
invariant(await sha256(validationManifest.fixtures.manifestPath) === validationManifest.fixtures.manifestSha256, 'fixture manifest SHA mismatch');
invariant(await sha256(validationManifest.fixtures.goldenPath) === validationManifest.fixtures.goldenSha256, 'golden reference SHA mismatch');
invariant(validationManifest.fixtures.repeatCount === matrix.fixtureCoverage.repeatCount, 'manifest repeat count mismatch');

for (const scriptPath of [
  'evidence/scripts/run_web_golden.mjs',
  'evidence/scripts/run_validation_manifest_golden.mjs',
]) {
  const source = await readFile(resolve(root, scriptPath), 'utf8');
  invariant(!/function\s+(decode|nms|iou)\s*\(/.test(source), `${scriptPath}: artifact-specific postprocess duplicate`);
  invariant(source.includes('evidence/tooling/raw-golden/Cargo.toml'), `${scriptPath}: production postprocess delegation missing`);
}

const goldenManifest = await readJson('evidence/golden/manifest.json');
const ownedArtifacts = goldenManifest.artifacts.filter(({ path }) =>
  path === matrix.model.path
  || path.startsWith('evidence/model/')
  || path.startsWith('evidence/fixtures/')
  || path.startsWith('evidence/golden/'),
);
invariant(ownedArtifacts.length > 0, 'golden manifest has no Validation-owned artifacts');
for (const artifact of ownedArtifacts) {
  invariant(await sha256(artifact.path) === artifact.sha256, `artifact SHA mismatch: ${artifact.path}`);
}

if (process.argv.includes('--require-report')) {
  const report = await readJson(matrix.reportPath);
  invariant(report.schemaVersion === 1, 'Validation report schemaVersion mismatch');
  invariant(report.phase1DependencyCommit === matrix.phase1DependencyCommit, 'report dependency commit mismatch');
  sameSet(report.tests.map((result) => result.testId), testIds, 'Validation report test coverage');
  for (const result of report.tests) {
    invariant(result.outcome === 'green', `${result.testId}: report outcome is not green`);
    invariant(result.processExitCode === 0, `${result.testId}: green test did not exit cleanly`);
    invariant(result.environmentFailure === false, `${result.testId}: environment failure cannot be green evidence`);
  }
  invariant(report.summary?.green === tests.length, 'Validation report green count mismatch');
  invariant(report.summary?.allRuntimeComparisonsPassed === true, 'runtime golden comparison did not pass');
  invariant(report.summary?.environmentFailures === 0, 'Validation report contains environment failures');
  invariant(report.postprocess?.sourceSha256 === validationManifest.postprocess.sourceSha256, 'report postprocess identity mismatch');
  invariant(report.postprocess?.duplicateArtifactSpecificImplementations === 0, 'report contains duplicate postprocess implementations');
  sameSet(report.fixtures.map((fixture) => fixture.id), imageFixtureIds, 'runtime golden fixture coverage');
  for (const fixture of report.fixtures) {
    invariant(fixture.web?.deterministic === true, `${fixture.id}: Web result is not deterministic`);
    invariant(fixture.legacyNativeOrt?.deterministic === true, `${fixture.id}: Legacy result is not deterministic`);
    invariant(fixture.web?.runs?.length === matrix.fixtureCoverage.repeatCount, `${fixture.id}: Web report repeat mismatch`);
    invariant(fixture.legacyNativeOrt?.runs?.length === matrix.fixtureCoverage.repeatCount, `${fixture.id}: Legacy report repeat mismatch`);
    invariant(fixture.legacyNativeOrt?.rawComparisonToWeb?.passed === true, `${fixture.id}: Legacy raw comparison failed`);
    invariant(fixture.legacyNativeOrt?.decodedComparisonToWeb?.passed === true, `${fixture.id}: Legacy decoded comparison failed`);
  }
}

process.stdout.write(`${JSON.stringify({
  status: 'passed',
  matrix: 'evidence/requirements/validation-requirement-test-matrix.json',
  scenarioCount: scenarios.size,
  testCount: tests.length,
  fixtureScenarioCount: matrix.fixtureCoverage.requiredScenarioIds.length,
  environmentFailures: 0,
})}\n`);
