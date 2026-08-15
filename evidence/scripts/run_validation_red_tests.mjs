import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const matrixPath = resolve(root, 'evidence/requirements/validation-requirement-test-matrix.json');
const matrixBytes = await readFile(matrixPath);
const matrix = JSON.parse(matrixBytes.toString('utf8'));
const cargoDependencyArgs = ['--config', matrix.cargoDependencyInput.configPath];
const tests = matrix.requirements.flatMap((requirement) =>
  requirement.scenarios.flatMap((scenario) => scenario.tests),
);

const validator = spawnSync('node', ['evidence/scripts/validate_validation_matrix.mjs'], {
  cwd: root,
  encoding: 'utf8',
});
if (validator.status !== 0) {
  process.stderr.write(validator.stdout);
  process.stderr.write(validator.stderr);
  throw new Error('Validation matrix failed before red-test execution');
}

const results = [];
for (const test of tests) {
  const args = [
    'test',
    ...cargoDependencyArgs,
    '--offline',
    '--locked',
    '--manifest-path',
    'evidence/tooling/validation-contract/Cargo.toml',
    `tests::${test.testFunction}`,
    '--',
    '--exact',
  ];
  const execution = spawnSync('cargo', args, { cwd: root, encoding: 'utf8' });
  const output = `${execution.stdout ?? ''}\n${execution.stderr ?? ''}`;
  const markerObserved = output.includes(test.expectedFailure.marker);
  const notImplementedObserved = output.includes('actual=not_implemented(');
  const expectedRed = execution.status !== 0 && markerObserved && notImplementedObserved;
  results.push({
    testId: test.testId,
    testFunction: test.testFunction,
    command: test.command,
    targetAssertion: test.expectedFailure.assertion,
    outcome: expectedRed ? 'expected-red' : 'invalid-red',
    actualFailureClassification: expectedRed ? 'target-assertion' : 'compile-fixture-environment-or-unexpected-result',
    environmentFailure: !expectedRed,
    processExitCode: execution.status,
    markerObserved,
    notImplementedObserved,
  });
  if (!expectedRed) {
    process.stderr.write(output);
    break;
  }
}

const head = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
if (head.status !== 0) throw new Error('cannot resolve source commit for red-test report');
const allExpectedRed = results.length === tests.length && results.every((result) => result.outcome === 'expected-red');
const report = {
  schemaVersion: 1,
  sourceCommit: head.stdout.trim(),
  phase1DependencyCommit: matrix.phase1DependencyCommit,
  matrixPath: 'evidence/requirements/validation-requirement-test-matrix.json',
  matrixSha256: createHash('sha256').update(matrixBytes).digest('hex'),
  runnerCommand: 'node evidence/scripts/run_validation_red_tests.mjs --write-report evidence/reports/validation-test-first.json',
  summary: {
    expectedRed: tests.length,
    observedExpectedRed: results.filter((result) => result.outcome === 'expected-red').length,
    environmentFailures: results.filter((result) => result.environmentFailure).length,
    firstFailureClassification: results[0]?.actualFailureClassification ?? null,
  },
  results,
};

const reportFlag = process.argv.indexOf('--write-report');
if (reportFlag !== -1) {
  const reportArgument = process.argv[reportFlag + 1];
  if (!reportArgument) throw new Error('--write-report requires a repository-relative path');
  const requestedPath = resolve(root, reportArgument);
  invariantInsideRoot(requestedPath);
  await writeFile(requestedPath, `${JSON.stringify(report, null, 2)}\n`);
}

process.stdout.write(`${JSON.stringify(report.summary)}\n`);
if (!allExpectedRed) process.exitCode = 1;

function invariantInsideRoot(path) {
  if (path !== root && !path.startsWith(`${root}/`)) {
    throw new Error('report path must stay inside the repository');
  }
}
