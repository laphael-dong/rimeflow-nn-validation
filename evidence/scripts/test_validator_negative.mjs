import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compareWindowsMlStaticCompileEvidence, validateCoremlEvidence, validateCoremlReplayEvidence, validateCoverageEvidence, validateLitertEvidence, validateMindsporeEvidence, validateMindsporeReplayEvidence, validateThirdPartyFixtureLicenses, validateWindowsMlEvidence, validateWindowsMlStaticCompileEvidence } from './evidence_validation.mjs';
import { validateCudaEvidence, validateCudaReplay } from './cuda_evidence_validation.mjs';
import { validateTensorRtReport } from './validate_tensorrt_ep.mjs';
import { validateTask14Aggregate } from './aggregate_evidence_validation.mjs';

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

const openvinoGuards = spawnSync('node', ['evidence/scripts/test_openvino_replay_guards.mjs'], { cwd: root, encoding: 'utf8' });
if (openvinoGuards.status !== 0) throw new Error(`OpenVINO replay guard tests failed:\n${openvinoGuards.stdout}\n${openvinoGuards.stderr}`);
const cudaGuards = spawnSync('node', ['evidence/scripts/test_cuda_ep_guards.mjs'], { cwd: root, encoding: 'utf8' });
if (cudaGuards.status !== 0) throw new Error(`CUDA replay guard tests failed:\n${cudaGuards.stdout}\n${cudaGuards.stderr}`);
const tensorrtGuards = spawnSync('node', ['evidence/scripts/test_tensorrt_ep_negative.mjs'], { cwd: root, encoding: 'utf8' });
if (tensorrtGuards.status !== 0) throw new Error(`TensorRT negative tests failed:\n${tensorrtGuards.stdout}\n${tensorrtGuards.stderr}`);

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

const conversion = await readJson('evidence/conversions/conversion-spikes.json');
const windowsConversion = conversion.spikes.find((item) => item.platform === 'windows');
const windowsManifest = await readJson('evidence/conversions/windows-ml-spike-manifest.json');
const windowsReport = await readJson('evidence/reports/windows-ml-spike-report.json');
const windowsStaticCompile = await readJson('evidence/reports/windows-ml-static-compile-report.json');
await validateWindowsMlEvidence(root, windowsConversion, windowsManifest, windowsReport);
await validateWindowsMlStaticCompileEvidence(root, windowsStaticCompile);

const runtimeEvidence = (target, architecture, sourcePackage = 'Microsoft.WindowsAppSDK.ML') => ({
  schemaVersion: 1,
  state: 'runtime-verified',
  target,
  runtimeExecuted: true,
  runtimeIntrospectionComplete: true,
  windowsMlApiCalled: true,
  failureStage: 'artifact-publication',
  catalogRegistrationAttempted: true,
  catalogRegistrationCompleted: true,
  sessionCreated: true,
  inferenceExecuted: true,
  outputPublished: true,
  model: { sha256: '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad', noConversion: true },
  input: { count: 1, name: 'images', dtype: 'float32', shape: [1, 3, 640, 640], elementCount: 1228800, finiteCount: 1228800 },
  output: { count: 1, name: 'output0', dtype: 'float32', shape: [1, 84, 8400], elementCount: 705600, finiteCount: 705600 },
  host: { os: 'windows', processArchitecture: architecture, osArchitecture: architecture },
  runtime: {
    sourcePackage: { id: sourcePackage, version: '2.1.74' },
    dotnetRuntimeVersion: '8.0.29',
    sdk: { version: '8.0.423', source: 'dotnet --version executed at runtime from the publish directory containing global.json' },
    loadedModules: [
      { name: 'onnxruntime.dll', sha256: '1'.repeat(64) },
      { name: 'Microsoft.Windows.AI.MachineLearning.dll', sha256: '2'.repeat(64) },
    ],
  },
  execution: {
    availableDevices: [{ epName: 'CPUExecutionProvider' }],
    selectedDevice: { epName: 'CPUExecutionProvider' },
    sessionInputDevices: [{ epName: 'CPUExecutionProvider' }],
    profileProviders: ['CPUExecutionProvider'],
  },
});

function runtimeClaim(targetId, evidence) {
  const manifest = structuredClone(windowsManifest);
  const report = structuredClone(windowsReport);
  const conversion = structuredClone(windowsConversion);
  for (const target of [manifest.targets[targetId], report.targets[targetId], conversion.targets[targetId]]) {
    Object.assign(target, { runtimeExecuted: true, runtimeVerified: true, goldenExecuted: true, supported: false });
  }
  report.targets[targetId].runtimeIntrospectionComplete = true;
  report.runtimeEvidence = { [targetId]: evidence };
  return { manifest, report, conversion };
}

const validX64RuntimeClaim = runtimeClaim('win-x64', runtimeEvidence('win-x64', 'X64'));
await validateWindowsMlEvidence(root, validX64RuntimeClaim.conversion, validX64RuntimeClaim.manifest, validX64RuntimeClaim.report);

const linuxOrt = runtimeClaim('win-x64', runtimeEvidence('win-x64', 'X64'));
linuxOrt.report.runtimeEvidence['win-x64'].host.os = 'linux';
await expectFailure('Linux ORT success claimed as Windows ML', () => validateWindowsMlEvidence(root, linuxOrt.conversion, linuxOrt.manifest, linuxOrt.report));

const unexecutedRuntime = runtimeClaim('win-x64', runtimeEvidence('win-x64', 'X64'));
for (const target of [unexecutedRuntime.manifest.targets['win-x64'], unexecutedRuntime.report.targets['win-x64'], unexecutedRuntime.conversion.targets['win-x64']]) target.runtimeExecuted = false;
await expectFailure('Windows runtime verified without runner execution', () => validateWindowsMlEvidence(root, unexecutedRuntime.conversion, unexecutedRuntime.manifest, unexecutedRuntime.report));

const x64AsArm64 = runtimeClaim('win-arm64', runtimeEvidence('win-x64', 'X64'));
await expectFailure('x64 runtime evidence claimed as ARM64', () => validateWindowsMlEvidence(root, x64AsArm64.conversion, x64AsArm64.manifest, x64AsArm64.report));

const mergedArchitectures = structuredClone(windowsManifest);
mergedArchitectures.targets = { windows: { runtimeVerified: true, supported: true } };
await expectFailure('Windows architectures merged into one verified flag', () => validateWindowsMlEvidence(root, windowsConversion, mergedArchitectures, windowsReport));

const windowsModelDrift = structuredClone(windowsManifest);
windowsModelDrift.canonicalOnnx.sha256 = '0'.repeat(64);
await expectFailure('Windows ML model SHA drift', () => validateWindowsMlEvidence(root, windowsConversion, windowsModelDrift, windowsReport));

const windowsShapeDrift = structuredClone(windowsManifest);
windowsShapeDrift.ioContract.outputs[0].shape = [1, 8400, 84];
await expectFailure('Windows ML I/O shape drift', () => validateWindowsMlEvidence(root, windowsConversion, windowsShapeDrift, windowsReport));

const windowsDtypeDrift = structuredClone(windowsManifest);
windowsDtypeDrift.ioContract.inputs[0].dtype = 'float16';
await expectFailure('Windows ML dtype drift', () => validateWindowsMlEvidence(root, windowsConversion, windowsDtypeDrift, windowsReport));

const windowsProviderDrift = structuredClone(windowsManifest);
windowsProviderDrift.runner.officialApi.deviceApis = ['compile-time-provider-name'];
await expectFailure('Windows ML provider introspection drift', () => validateWindowsMlEvidence(root, windowsConversion, windowsProviderDrift, windowsReport));

const ordinaryCpuOrt = runtimeClaim('win-x64', runtimeEvidence('win-x64', 'X64', 'Microsoft.ML.OnnxRuntime'));
await expectFailure('ordinary CPU ORT claimed as Windows ML provider', () => validateWindowsMlEvidence(root, ordinaryCpuOrt.conversion, ordinaryCpuOrt.manifest, ordinaryCpuOrt.report));

const floatingPackage = structuredClone(windowsManifest);
floatingPackage.dependencies.windowsMlPackage.version = '2.1.*';
await expectFailure('Windows ML floating package version', () => validateWindowsMlEvidence(root, windowsConversion, floatingPackage, windowsReport));

const missingSdk = structuredClone(windowsManifest);
missingSdk.dependencies.dotnetSdk = null;
await expectFailure('Windows ML missing exact SDK version', () => validateWindowsMlEvidence(root, windowsConversion, missingSdk, windowsReport));

const runtimeVersionDrift = structuredClone(windowsReport);
runtimeVersionDrift.dependencies.dotnetRuntime = '8.0.x';
await expectFailure('Windows ML missing exact .NET runtime version', () => validateWindowsMlEvidence(root, windowsConversion, windowsManifest, runtimeVersionDrift));

const runtimeSdkDrift = runtimeClaim('win-x64', runtimeEvidence('win-x64', 'X64'));
runtimeSdkDrift.report.runtimeEvidence['win-x64'].runtime.sdk.version = '8.0.424';
await expectFailure('Windows ML runtime SDK introspection drift', () => validateWindowsMlEvidence(root, runtimeSdkDrift.conversion, runtimeSdkDrift.manifest, runtimeSdkDrift.report));

const missingIntrospection = runtimeClaim('win-x64', runtimeEvidence('win-x64', 'X64'));
missingIntrospection.report.targets['win-x64'].runtimeIntrospectionComplete = false;
delete missingIntrospection.report.runtimeEvidence['win-x64'].execution.profileProviders;
await expectFailure('Windows ML verification without runtime introspection', () => validateWindowsMlEvidence(root, missingIntrospection.conversion, missingIntrospection.manifest, missingIntrospection.report));

const supportedManifest = structuredClone(windowsManifest);
const supportedReport = structuredClone(windowsReport);
const supportedConversion = structuredClone(windowsConversion);
for (const evidence of [supportedManifest, supportedReport, supportedConversion]) {
  evidence.state = 'supported';
  evidence.supported = true;
  evidence.task14Complete = true;
}
await expectFailure('Windows ML supported without real x64 and ARM64 runners', () => validateWindowsMlEvidence(root, supportedConversion, supportedManifest, supportedReport));

const publicationLeak = structuredClone(windowsManifest);
publicationLeak.canonicalOnnx.path = 'release/models/yolov8n.onnx';
publicationLeak.publicationExclusions = publicationLeak.publicationExclusions.filter((item) => !item.includes('release'));
await expectFailure('Windows ML ONNX or build output added to release directory', () => validateWindowsMlEvidence(root, windowsConversion, publicationLeak, windowsReport));

const missingAssembly = structuredClone(windowsStaticCompile);
delete missingAssembly.targets['win-x64'].assembly;
await expectFailure('Windows ML static compile missing assembly', () => validateWindowsMlStaticCompileEvidence(root, missingAssembly));
const forgedAssembly = structuredClone(windowsStaticCompile);
forgedAssembly.targets['win-x64'].assembly.sha256 = '0'.repeat(64);
await expectFailure('Windows ML static compile forged assembly SHA', () => compareWindowsMlStaticCompileEvidence(forgedAssembly, windowsStaticCompile));
const wrongRidAssembly = structuredClone(windowsStaticCompile);
wrongRidAssembly.targets['win-arm64'].assembly.targetRidFromPath = 'win-x64';
await expectFailure('Windows ML static compile wrong RID assembly', () => validateWindowsMlStaticCompileEvidence(root, wrongRidAssembly));
const dirtyCompile = structuredClone(windowsStaticCompile);
dirtyCompile.targets['win-x64'].cleanBefore.objAbsent = false;
await expectFailure('Windows ML static compile reused obj', () => validateWindowsMlStaticCompileEvidence(root, dirtyCompile));
const missingCoreCompile = structuredClone(windowsStaticCompile);
missingCoreCompile.targets['win-x64'].compile.coreCompileExecuted = false;
await expectFailure('Windows ML static compile skipped CoreCompile', () => validateWindowsMlStaticCompileEvidence(root, missingCoreCompile));
const missingRoslyn = structuredClone(windowsStaticCompile);
missingRoslyn.targets['win-x64'].compile.roslynCscExecuted = false;
await expectFailure('Windows ML static compile skipped Roslyn Csc', () => validateWindowsMlStaticCompileEvidence(root, missingRoslyn));
const missingProgram = structuredClone(windowsStaticCompile);
missingProgram.targets['win-arm64'].compile.programCsCompiled = false;
await expectFailure('Windows ML static compile omitted Program.cs', () => validateWindowsMlStaticCompileEvidence(root, missingProgram));
const manifestBypass = structuredClone(windowsStaticCompile);
manifestBypass.targets['win-x64'].compile.command += ' -property:ManifestTool=/bin/true app.manifest';
await expectFailure('Windows ML static compile manifest tool bypass', () => validateWindowsMlStaticCompileEvidence(root, manifestBypass));

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
const portableFixture = structuredClone(coremlNonRecord);
portableFixture.semanticReplayDigests.rounds[0].normalizedSpecSha256 = '1'.repeat(64);
portableFixture.semanticReplayDigests.rounds[1].normalizedSpecSha256 = '1'.repeat(64);
portableFixture.semanticReplayValidation.portableSpec = {
  recordedArtifact: { normalization: 'coreml-spec-portable-v2: remove only description.metadata.userDefined date and com.github.apple.coremltools.conversion_date', removedMetadata: { date: '2026-08-11T00:00:00', 'com.github.apple.coremltools.conversion_date': '2026-08-11' }, sha256: '2'.repeat(64) },
  rounds: [
    { normalization: 'coreml-spec-portable-v2: remove only description.metadata.userDefined date and com.github.apple.coremltools.conversion_date', removedMetadata: { date: '2026-08-12T00:00:00', 'com.github.apple.coremltools.conversion_date': '2026-08-12' }, sha256: '2'.repeat(64) },
    { normalization: 'coreml-spec-portable-v2: remove only description.metadata.userDefined date and com.github.apple.coremltools.conversion_date', removedMetadata: { date: '2026-08-12T00:00:01', 'com.github.apple.coremltools.conversion_date': '2026-08-12' }, sha256: '2'.repeat(64) },
  ],
};
validateCoremlReplayEvidence(coremlManifest, portableFixture);
const portableSpecDrift = structuredClone(portableFixture);
portableSpecDrift.semanticReplayValidation.portableSpec.rounds[1].sha256 = '3'.repeat(64);
await expectFailure('Core ML portable spec drift', async () => validateCoremlReplayEvidence(coremlManifest, portableSpecDrift));
const unknownVolatileMetadata = structuredClone(portableFixture);
unknownVolatileMetadata.semanticReplayValidation.portableSpec.rounds[1].removedMetadata.unreviewed = 'ignored';
await expectFailure('Core ML unknown metadata cannot be normalized away', async () => validateCoremlReplayEvidence(coremlManifest, unknownVolatileMetadata));
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

const cudaManifest = await readJson('evidence/conversions/cuda-ep-spike-manifest.json');
const cudaReport = await readJson('evidence/reports/cuda-ep-spike-report.json');
const cudaReplay = await readJson(process.env.RIMEFLOW_CUDA_ORDINARY_REPLAY ?? 'evidence/reports/cuda-ep-replay-report.json');
await validateCudaEvidence(root, cudaManifest, cudaReport);
await validateCudaReplay(root, cudaReplay);
const cudaReplayWorkingTreeDrift = await readJson('evidence/reports/cuda-ep-replay-report.json');
cudaReplayWorkingTreeDrift.rounds[0].command.stdout = 'unstaged mutation';
await expectFailure('CUDA recorded replay working tree differs from staged Git blob', async () => validateCudaReplay(root, cudaReplayWorkingTreeDrift));
const tensorrtReport = await readJson('evidence/reports/tensorrt-ep-report.json');
await validateTensorRtReport(tensorrtReport, { sourceRoot: root });
const conversionAggregate = await readJson('evidence/conversions/conversion-spikes.json');
const taskReplayAggregate = await readJson('evidence/replay/task1-replay.json');
const task14Aggregate = await readJson('evidence/conversions/task1-4-aggregate.json');
const publicationReceipt = await readJson('evidence/reports/task1-4-publication-report.json');
const artifactManifest = await readJson('evidence/golden/manifest.json');
await validateTask14Aggregate(root, task14Aggregate, conversionAggregate, taskReplayAggregate, publicationReceipt, artifactManifest);
const aggregateNegativeCases = [];
const aggregateFailure = async (name, mutate) => {
  const aggregate = structuredClone(task14Aggregate);
  const conversion = structuredClone(conversionAggregate);
  const replay = structuredClone(taskReplayAggregate);
  const receipt = structuredClone(publicationReceipt);
  const artifacts = structuredClone(artifactManifest);
  mutate(aggregate, conversion, replay, receipt, artifacts);
  await expectFailure(name, () => validateTask14Aggregate(root, aggregate, conversion, replay, receipt, artifacts));
  aggregateNegativeCases.push(name);
};
await aggregateFailure('缺少必需 spike 却声称 technicalSpikeClosure', (aggregate) => aggregate.spikes.pop());
await aggregateFailure('重复 provider 条目', (aggregate) => aggregate.spikes[1] = structuredClone(aggregate.spikes[0]));
await aggregateFailure('错误 provider 身份', (aggregate) => { aggregate.spikes[0].provider = 'wrong-provider'; });
await aggregateFailure('CUDA blocked 报告伪造执行', (aggregate) => { aggregate.spikes.find((item) => item.provider === 'cudaexecutionprovider').runtimeExecuted = true; });
await aggregateFailure('TensorRT blocked 报告伪造 golden', (aggregate) => { aggregate.spikes.find((item) => item.provider === 'tensorrtexecutionprovider').goldenExecuted = true; });
await aggregateFailure('OpenVINO 仅配置未执行', (aggregate) => { aggregate.spikes.find((item) => item.provider === 'openvinoexecutionprovider').runtimeExecuted = false; });
await aggregateFailure('technical closure 冒充 allPlatformsSupported', (aggregate, conversion, replay) => { aggregate.closure.allPlatformsSupported = true; conversion.closure.allPlatformsSupported = true; replay.closure.allPlatformsSupported = true; });
await aggregateFailure('publicationVerified=false 却 task14Complete=true', (aggregate, conversion, replay) => { aggregate.closure.publicationVerified = false; conversion.closure.publicationVerified = false; replay.closure.publicationVerified = false; });
await aggregateFailure('task14Complete=true 但未勾选 OpenSpec 1.4', (aggregate, conversion, replay) => { aggregate.closure.openspecTask1_4Checked = false; conversion.closure.openspecTask1_4Checked = false; replay.closure.openspecTask1_4Checked = false; });
await aggregateFailure('共享 aggregate 丢失 provider', (_aggregate, conversion) => { conversion.spikes.pop(); });
await aggregateFailure('共享 replay 丢失 provider', (_aggregate, _conversion, replay) => { replay.steps = replay.steps.filter((item) => item.provider !== 'tensorrtexecutionprovider'); });
await aggregateFailure('provider source provenance 漂移', (aggregate) => { aggregate.source.providerCommits.cuda = '0'.repeat(40); });

console.log(JSON.stringify({ ok: true, filesystemGuardTests: 7, aggregateNegativeCases, providerGuardSuites: { openvino: true, cuda: true, tensorrt: true }, positiveCases: ['LiteRT evidence with differently ordered tolerance keys', 'Windows ML blocked x64/ARM64 harness contract', 'Windows ML clean static compile report', 'Windows ML valid partial x64 runtime claim', 'Core ML artifact/spec evidence', 'Core ML non-record artifact preservation evidence', 'MindSpore Lite record evidence', 'MindSpore Lite non-record artifact preservation evidence', 'CUDA blocked spike evidence', 'TensorRT blocked spike evidence', 'task 1.4 aggregate technical closure'], negativeCases: ['missing coverage path', 'missing local fixture license', 'LiteRT supported without Android runner', 'LiteRT relaxed frozen tolerance', 'LiteRT replay digest mismatch', 'Linux ORT success claimed as Windows ML', 'Windows runtime verified without runner execution', 'x64 runtime evidence claimed as ARM64', 'Windows architectures merged into one verified flag', 'Windows ML model SHA drift', 'Windows ML I/O shape drift', 'Windows ML dtype drift', 'Windows ML provider introspection drift', 'ordinary CPU ORT claimed as Windows ML provider', 'Windows ML floating package version', 'Windows ML missing exact SDK version', 'Windows ML missing exact .NET runtime version', 'Windows ML runtime SDK introspection drift', 'Windows ML verification without runtime introspection', 'Windows ML supported without real x64 and ARM64 runners', 'Windows ML ONNX or build output added to release directory', 'Windows ML static compile missing assembly', 'Windows ML static compile forged assembly SHA', 'Windows ML static compile wrong RID assembly', 'Windows ML static compile reused obj', 'Windows ML static compile skipped CoreCompile', 'Windows ML static compile skipped Roslyn Csc', 'Windows ML static compile omitted Program.cs', 'Windows ML static compile manifest tool bypass', 'Core ML supported without macOS/iOS runner', 'Core ML package tree digest drift', 'Core ML FP16 precision drift', 'Core ML fused NMS overclaim', 'Core ML non-record replay changed fixed artifact', 'Core ML semantic digest used as artifact identity', 'Core ML non-record weight blob drift', 'Core ML missing artifact exact identity overclaim', 'MindSpore supported without HarmonyOS device', 'MindSpore runtime input layout drift', 'MindSpore artifact digest drift', 'MindSpore relaxed frozen tolerance', 'MindSpore failure signature drift', 'MindSpore platform postprocess duplication', 'MindSpore non-record replay changed fixed artifact', 'MindSpore missing artifact exact identity overclaim', 'MindSpore non-record replay created fixed artifact', 'MindSpore replay digest impersonates recorded artifact identity', 'MindSpore non-record tracked report drift'] }));
