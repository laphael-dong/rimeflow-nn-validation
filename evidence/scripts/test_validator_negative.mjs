import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateCoverageEvidence, validateLitertEvidence, validateThirdPartyFixtureLicenses } from './evidence_validation.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const expectFailure = async (name, operation) => {
  try {
    await operation();
  } catch {
    return;
  }
  throw new Error(`negative validator case unexpectedly passed: ${name}`);
};

const coverage = await readJson('evidence/golden/coverage-matrix.json');
const missingPath = structuredClone(coverage);
missingPath.cases.find((item) => item.id === 'overlap-nms').decode.path = 'tests/task1_raw_golden.rs';
await expectFailure('missing coverage path', () => validateCoverageEvidence(root, missingPath));

const fixtures = await readJson('evidence/fixtures/manifest.json');
const missingLicense = structuredClone(fixtures);
missingLicense.images.find((item) => typeof item.source === 'object').source.license.localPath = 'evidence/fixtures/licenses/missing.txt';
await expectFailure('missing local fixture license', () => validateThirdPartyFixtureLicenses(root, missingLicense));

const litertManifest = await readJson('evidence/conversions/litert-artifact-manifest.json');
const litertGolden = await readJson('evidence/reports/litert-golden-report.json');
const litertReplay = await readJson('evidence/reports/litert-conversion-report.json');
const frozen = (await readJson('evidence/golden/web-reference.json')).tolerances;
validateLitertEvidence(litertManifest, litertGolden, litertReplay, frozen);
const supported = structuredClone(litertManifest);
supported.status.supported = true;
await expectFailure('LiteRT supported without Android runner', async () => validateLitertEvidence(supported, litertGolden, litertReplay, frozen));
const relaxed = structuredClone(litertGolden);
relaxed.tolerances.rawTensorAbsolute = 1;
await expectFailure('LiteRT relaxed frozen tolerance', async () => validateLitertEvidence(litertManifest, relaxed, litertReplay, frozen));
const nondeterministic = structuredClone(litertReplay);
nondeterministic.comparison.artifactSha256Equal = false;
await expectFailure('LiteRT replay digest mismatch', async () => validateLitertEvidence(litertManifest, litertGolden, nondeterministic, frozen));

console.log(JSON.stringify({ ok: true, positiveCases: ['LiteRT evidence with differently ordered tolerance keys'], negativeCases: ['missing coverage path', 'missing local fixture license', 'LiteRT supported without Android runner', 'LiteRT relaxed frozen tolerance', 'LiteRT replay digest mismatch'] }));
