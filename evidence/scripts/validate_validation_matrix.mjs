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
invariant(matrix.phase === '2-test-first', 'matrix phase mismatch');
invariant(
  JSON.stringify(matrix.cargoDependencyInput) === JSON.stringify(expectedCargoDependencyInput),
  'hermetic Cargo dependency input mismatch',
);
invariant(/^[0-9a-f]{40}$/.test(matrix.originalBaseCommit), 'invalid original base commit');
invariant(/^[0-9a-f]{40}$/.test(matrix.phase1DependencyCommit), 'invalid Phase 1 dependency commit');
invariant(/^[0-9a-f]{64}$/.test(matrix.model.sha256), 'invalid model SHA-256');

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
  invariant(test.expectedFailure?.classification === 'target-assertion', `invalid failure classification for ${test.testId}`);
  invariant(test.expectedFailure?.marker === `${test.testId}: target_assertion`, `invalid failure marker for ${test.testId}`);
  invariant(typeof test.expectedFailure?.assertion === 'string' && test.expectedFailure.assertion.length > 0, `missing target assertion for ${test.testId}`);
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
const registrations = [...testSource.matchAll(/rfb_val_red_test!\(\s*"(RFB-VAL-[A-Z0-9-]+)",\s*([a-z0-9_]+),/gs)]
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
  invariant(report.schemaVersion === 1, 'red-test report schemaVersion mismatch');
  invariant(report.phase1DependencyCommit === matrix.phase1DependencyCommit, 'report dependency commit mismatch');
  sameSet(report.results.map((result) => result.testId), testIds, 'red-test report coverage');
  for (const result of report.results) {
    invariant(result.outcome === 'expected-red', `${result.testId}: report outcome is not expected-red`);
    invariant(result.actualFailureClassification === 'target-assertion', `${result.testId}: invalid report failure classification`);
    invariant(result.environmentFailure === false, `${result.testId}: environment failure cannot be red evidence`);
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
