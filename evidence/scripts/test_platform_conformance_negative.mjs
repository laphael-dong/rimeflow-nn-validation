import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateReport } from './platform_conformance.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const original = JSON.parse(await readFile(resolve(root, 'evidence/reports/platform-conformance-report.json'), 'utf8'));
const clone = () => structuredClone(original);

function expectFailure(label, mutate) {
  const report = clone();
  mutate(report);
  try {
    validateReport(report);
  } catch {
    return;
  }
  throw new Error(`negative conformance case was accepted: ${label}`);
}

validateReport(original);
expectFailure('missing package-load field', (report) => { delete report.platforms[0].checks.packageLoad; });
expectFailure('invented macOS runner identity', (report) => { report.platforms[0].runner.identity = 'pretend-macos-host'; });
expectFailure('unsupported platform promoted', (report) => { report.platforms[1].supported = true; });
expectFailure('Windows runtime promoted without runner', (report) => { report.platforms[2].status = 'host-inference-verified'; });
expectFailure('logical output role omitted', (report) => { report.platforms[4].io.outputRole = 'unknown'; });
expectFailure('artifact digest removed', (report) => { report.platforms[5].artifact.digest = ''; });
expectFailure('Base tree drift', (report) => { report.baseRuntime.tree = '0'.repeat(40); });
expectFailure('Validation implementation commit drift', (report) => { report.validationImplementation.commit = '0'.repeat(40); });
process.stdout.write(`${JSON.stringify({ ok: true, negativeCases: 8 })}\n`);
