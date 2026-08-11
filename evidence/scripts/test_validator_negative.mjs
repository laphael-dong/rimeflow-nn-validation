import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateCoremlEvidence, validateCoremlReplayEvidence, validateCoverageEvidence, validateLitertEvidence, validateMindsporeEvidence, validateMindsporeReplayEvidence, validateThirdPartyFixtureLicenses } from './evidence_validation.mjs';

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
const mindsporePython = process.env.RIMEFLOW_MINDSPORE_PYTHON ?? '.evidence/mindspore/python-venv/bin/python';
const guardTests = spawnSync(mindsporePython, ['evidence/scripts/test_mindspore_replay_guards.py'], { cwd: root, encoding: 'utf8' });
if (guardTests.status !== 0) throw new Error(`MindSpore replay guard tests failed:\n${guardTests.stdout}\n${guardTests.stderr}`);
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

const mindsporeManifest = await readJson('evidence/conversions/mindspore-artifact-manifest.json');
const mindsporeGolden = await readJson('evidence/reports/mindspore-golden-report.json');
const mindsporeReplay = await readJson('evidence/reports/mindspore-conversion-report.json');
validateMindsporeEvidence(mindsporeManifest, mindsporeGolden, mindsporeReplay, frozen);
const mindsporeNonRecord = taskReplay.steps.find((item) => item.id === 'harmonyos-mindspore-conversion-and-host-golden');
validateMindsporeReplayEvidence(mindsporeManifest, mindsporeNonRecord);
const mindsporeSupported = structuredClone(mindsporeManifest);
mindsporeSupported.status.supported = true;
await expectFailure('MindSpore supported without HarmonyOS device', async () => validateMindsporeEvidence(mindsporeSupported, mindsporeGolden, mindsporeReplay, frozen));
const mindsporeLayout = structuredClone(mindsporeManifest);
mindsporeLayout.ioContract.inputs[0].shape = [1, 3, 640, 640];
await expectFailure('MindSpore runtime input layout drift', async () => validateMindsporeEvidence(mindsporeLayout, mindsporeGolden, mindsporeReplay, frozen));
const mindsporeArtifact = structuredClone(mindsporeManifest);
mindsporeArtifact.artifact.sha256 = '0'.repeat(64);
await expectFailure('MindSpore artifact digest drift', async () => validateMindsporeEvidence(mindsporeArtifact, mindsporeGolden, mindsporeReplay, frozen));
const mindsporeTolerance = structuredClone(mindsporeGolden);
mindsporeTolerance.tolerances.rawTensorAbsolute = 1;
await expectFailure('MindSpore relaxed frozen tolerance', async () => validateMindsporeEvidence(mindsporeManifest, mindsporeTolerance, mindsporeReplay, frozen));
const mindsporeFailure = structuredClone(mindsporeReplay);
mindsporeFailure.rounds[0].matrix[0].failureSignature.failedOperator = '/wrong/operator';
await expectFailure('MindSpore failure signature drift', async () => validateMindsporeEvidence(mindsporeManifest, mindsporeGolden, mindsporeFailure, frozen));
const mindsporePostprocess = structuredClone(mindsporeGolden);
mindsporePostprocess.productionPostprocess.platformSpecificImplementationAdded = true;
await expectFailure('MindSpore platform postprocess duplication', async () => validateMindsporeEvidence(mindsporeManifest, mindsporePostprocess, mindsporeReplay, frozen));

const mindsporeChangedRecorded = structuredClone(mindsporeNonRecord);
mindsporeChangedRecorded.recordedArtifactVerification.afterSha256 = '1'.repeat(64);
mindsporeChangedRecorded.recordedArtifactVerification.unchanged = false;
await expectFailure('MindSpore non-record replay changed fixed artifact', async () => validateMindsporeReplayEvidence(mindsporeManifest, mindsporeChangedRecorded));
const mindsporeMissingOverclaim = structuredClone(mindsporeNonRecord);
Object.assign(mindsporeMissingOverclaim.recordedArtifactVerification, {
  availableBefore: false,
  availableAfter: false,
  beforeBytes: null,
  afterBytes: null,
  beforeMtimeNs: null,
  afterMtimeNs: null,
  beforeSha256: null,
  afterSha256: null,
  exactIdentityVerifiedBefore: true,
  exactIdentityVerifiedAfter: true,
  status: 'verified-preserved',
});
await expectFailure('MindSpore missing artifact exact identity overclaim', async () => validateMindsporeReplayEvidence(mindsporeManifest, mindsporeMissingOverclaim));
const mindsporeAutoCreated = structuredClone(mindsporeNonRecord);
Object.assign(mindsporeAutoCreated.recordedArtifactVerification, {
  availableBefore: false,
  availableAfter: true,
  beforeBytes: null,
  beforeMtimeNs: null,
  beforeSha256: null,
  status: 'recorded-artifact-created-during-replay',
  unchanged: false,
});
await expectFailure('MindSpore non-record replay created fixed artifact', async () => validateMindsporeReplayEvidence(mindsporeManifest, mindsporeAutoCreated));
const mindsporeReplayAsRecorded = structuredClone(mindsporeNonRecord);
mindsporeReplayAsRecorded.replayArtifactSha256 = '2'.repeat(64);
mindsporeReplayAsRecorded.workspaceArtifact.sha256 = mindsporeReplayAsRecorded.replayArtifactSha256;
for (const round of mindsporeReplayAsRecorded.rounds) round.outputs[0].sha256 = mindsporeReplayAsRecorded.replayArtifactSha256;
await expectFailure('MindSpore replay digest impersonates recorded artifact identity', async () => validateMindsporeReplayEvidence(mindsporeManifest, mindsporeReplayAsRecorded));
const mindsporeTrackedDrift = structuredClone(mindsporeNonRecord);
mindsporeTrackedDrift.trackedEvidence.goldenReport.sha256After = '3'.repeat(64);
mindsporeTrackedDrift.trackedEvidence.goldenReport.unchanged = true;
await expectFailure('MindSpore non-record tracked report drift', async () => validateMindsporeReplayEvidence(mindsporeManifest, mindsporeTrackedDrift));

console.log(JSON.stringify({ ok: true, filesystemGuardTests: 7, positiveCases: ['LiteRT evidence with differently ordered tolerance keys', 'Core ML artifact/spec evidence', 'Core ML non-record artifact preservation evidence', 'MindSpore Lite record evidence', 'MindSpore Lite non-record artifact preservation evidence'], negativeCases: ['missing coverage path', 'missing local fixture license', 'LiteRT supported without Android runner', 'LiteRT relaxed frozen tolerance', 'LiteRT replay digest mismatch', 'Core ML supported without macOS/iOS runner', 'Core ML package tree digest drift', 'Core ML FP16 precision drift', 'Core ML fused NMS overclaim', 'Core ML non-record replay changed fixed artifact', 'Core ML semantic digest used as artifact identity', 'Core ML non-record weight blob drift', 'Core ML missing artifact exact identity overclaim', 'MindSpore supported without HarmonyOS device', 'MindSpore runtime input layout drift', 'MindSpore artifact digest drift', 'MindSpore relaxed frozen tolerance', 'MindSpore failure signature drift', 'MindSpore platform postprocess duplication', 'MindSpore non-record replay changed fixed artifact', 'MindSpore missing artifact exact identity overclaim', 'MindSpore non-record replay created fixed artifact', 'MindSpore replay digest impersonates recorded artifact identity', 'MindSpore non-record tracked report drift'] }));
