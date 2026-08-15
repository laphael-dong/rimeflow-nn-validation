import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
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
  const green = execution.status === 0;
  results.push({
    testId: test.testId,
    testFunction: test.testFunction,
    command: test.command,
    targetAssertion: test.expectedResult.assertion,
    outcome: green ? 'green' : 'failed',
    environmentFailure: !green,
    processExitCode: execution.status,
  });
  if (!green) {
    process.stderr.write(output);
    break;
  }
}

const head = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
if (head.status !== 0) throw new Error('cannot resolve source commit for Validation report');
const allGreen = results.length === tests.length && results.every((result) => result.outcome === 'green');
const report = {
  schemaVersion: 1,
  sourceCommit: head.stdout.trim(),
  phase1DependencyCommit: matrix.phase1DependencyCommit,
  matrixPath: 'evidence/requirements/validation-requirement-test-matrix.json',
  matrixSha256: createHash('sha256').update(matrixBytes).digest('hex'),
  runnerCommand: 'node evidence/scripts/run_validation_manifest_golden.mjs --write-report evidence/reports/validation-manifest-golden-report.json',
  summary: {
    expectedGreen: tests.length,
    observedGreen: results.filter((result) => result.outcome === 'green').length,
    environmentFailures: results.filter((result) => result.environmentFailure).length,
    firstFailureClassification: results.find((result) => result.outcome !== 'green')?.testId ?? null,
  },
  results,
};

const reportFlag = process.argv.indexOf('--write-report');
if (reportFlag !== -1) {
  const reportArgument = process.argv[reportFlag + 1];
  if (!reportArgument) throw new Error('--write-report requires a repository-relative path');
  invariantInsideRoot(resolve(root, reportArgument));
  const golden = spawnSync('node', [
    'evidence/scripts/run_validation_manifest_golden.mjs',
    '--write-report', reportArgument,
  ], { cwd: root, encoding: 'utf8' });
  process.stdout.write(golden.stdout ?? '');
  process.stderr.write(golden.stderr ?? '');
  if (golden.status !== 0) process.exitCode = 1;
}

process.stdout.write(`${JSON.stringify(report.summary)}\n`);
if (!allGreen) process.exitCode = 1;

function invariantInsideRoot(path) {
  if (path !== root && !path.startsWith(`${root}/`)) {
    throw new Error('report path must stay inside the repository');
  }
}
