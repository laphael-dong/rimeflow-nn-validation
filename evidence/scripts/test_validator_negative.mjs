import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateCoremlEvidence, validateCoremlReplayEvidence, validateCoverageEvidence, validateLitertEvidence, validateThirdPartyFixtureLicenses } from './evidence_validation.mjs';

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

const coremlManifest = await readJson('evidence/conversions/coreml-artifact-manifest.json');
const coremlReplay = await readJson('evidence/reports/coreml-conversion-report.json');
validateCoremlEvidence(coremlManifest, coremlReplay);
const coremlSupported = structuredClone(coremlManifest);
coremlSupported.status.supported = true;
await expectFailure('Core ML supported without macOS/iOS runner', async () => validateCoremlEvidence(coremlSupported, coremlReplay));
const coremlTreeDrift = structuredClone(coremlManifest);
coremlTreeDrift.artifact.tree.digest = '0'.repeat(64);
coremlTreeDrift.artifact.tree.files[0].sha256 = 'not-a-sha';
await expectFailure('Core ML package tree digest drift', async () => validateCoremlEvidence(coremlTreeDrift, coremlReplay));
const coremlFp16 = structuredClone(coremlManifest);
coremlFp16.spec.computePrecision.float16Present = true;
await expectFailure('Core ML FP16 precision drift', async () => validateCoremlEvidence(coremlFp16, coremlReplay));
const coremlNms = structuredClone(coremlManifest);
coremlNms.spec.nms.fused = true;
coremlNms.spec.nms.operators = ['non_maximum_suppression'];
await expectFailure('Core ML fused NMS overclaim', async () => validateCoremlEvidence(coremlNms, coremlReplay));

const taskReplay = await readJson('evidence/replay/task1-replay.json');
const coremlNonRecord = taskReplay.steps.find((item) => item.id === 'apple-coreml-conversion-and-spec-inspection');
validateCoremlReplayEvidence(coremlManifest, coremlNonRecord);
const changedRecordedArtifact = structuredClone(coremlNonRecord);
changedRecordedArtifact.recordedArtifactVerification.afterTreeDigest = '1'.repeat(64);
changedRecordedArtifact.recordedArtifactVerification.unchanged = false;
await expectFailure('Core ML non-record replay changed fixed artifact', async () => validateCoremlReplayEvidence(coremlManifest, changedRecordedArtifact));
const semanticAsArtifact = structuredClone(coremlNonRecord);
semanticAsArtifact.recordedArtifactTreeDigest = coremlManifest.semanticReplayDigests.normalizedSpecSha256;
semanticAsArtifact.recordedArtifactVerification.expectedTreeDigest = semanticAsArtifact.recordedArtifactTreeDigest;
await expectFailure('Core ML semantic digest used as artifact identity', async () => validateCoremlReplayEvidence(coremlManifest, semanticAsArtifact));
const weightDrift = structuredClone(coremlNonRecord);
weightDrift.semanticReplayDigests.rounds[1].weightBlobSha256 = '2'.repeat(64);
await expectFailure('Core ML non-record weight blob drift', async () => validateCoremlReplayEvidence(coremlManifest, weightDrift));
const missingArtifactOverclaim = structuredClone(coremlNonRecord);
Object.assign(missingArtifactOverclaim.recordedArtifactVerification, {
  availableBefore: false,
  availableAfter: false,
  beforeTreeDigest: null,
  afterTreeDigest: null,
  exactIdentityVerifiedBefore: true,
  exactIdentityVerifiedAfter: true,
  status: 'verified-preserved',
});
await expectFailure('Core ML missing artifact exact identity overclaim', async () => validateCoremlReplayEvidence(coremlManifest, missingArtifactOverclaim));

console.log(JSON.stringify({ ok: true, positiveCases: ['LiteRT evidence with differently ordered tolerance keys', 'Core ML artifact/spec evidence', 'Core ML non-record artifact preservation evidence'], negativeCases: ['missing coverage path', 'missing local fixture license', 'LiteRT supported without Android runner', 'LiteRT relaxed frozen tolerance', 'LiteRT replay digest mismatch', 'Core ML supported without macOS/iOS runner', 'Core ML package tree digest drift', 'Core ML FP16 precision drift', 'Core ML fused NMS overclaim', 'Core ML non-record replay changed fixed artifact', 'Core ML semantic digest used as artifact identity', 'Core ML non-record weight blob drift', 'Core ML missing artifact exact identity overclaim'] }));
