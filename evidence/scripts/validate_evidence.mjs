import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv from '../tooling/web/node_modules/ajv/lib/ajv.js';
import { PREPROCESS_CONTRACT, preprocessCanonical, readPpm, tensorDigest } from './preprocess_contract.mjs';
import { validateCoverageEvidence, validateLitertEvidence, validateThirdPartyFixtureLicenses } from './evidence_validation.mjs';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fail = (message) => { throw new Error(message); };
const contract = await readJson('evidence/model/model-contract.json');
const schema = await readJson('evidence/schemas/model-contract.schema.json');
const ajv = new Ajv({ allErrors: true, strict: false });
const validateContract = ajv.compile(schema);
if (!validateContract(contract)) fail(`model-contract JSON Schema: ${JSON.stringify(validateContract.errors)}`);
for (const mutate of [
  (copy) => { copy.model.output.layout = 'CANDIDATE_MAJOR'; },
  (copy) => { copy.model.output.axes = ['batch', 'anchor', 'attribute']; },
  (copy) => { copy.model.output.shape = [1, 84]; },
  (copy) => { copy.model.input.quantization = { kind: 'per-tensor' }; },
  (copy) => { copy.model.input.preprocess.interpolation = 'nearest'; },
]) {
  const invalid = structuredClone(contract);
  mutate(invalid);
  if (validateContract(invalid)) fail('model-contract schema accepted an invalid layout/quantization/preprocess mutation');
}
if (contract.schemaVersion !== 1) fail('contract schemaVersion');
const actualModelSha = sha256(await readFile(resolve(root, contract.source.modelPath)));
const actualShaderSha = sha256(await readFile(resolve(root, 'shaders/preprocess.wgsl')));
const actualPostprocessSha = sha256(await readFile(resolve(root, 'src/postprocess.rs')));
const actualPreprocessContractSha = sha256(await readFile(resolve(root, 'evidence/scripts/preprocess_contract.mjs')));
if (contract.source.modelSha256 !== actualModelSha || contract.verification.inputModelSha256 !== actualModelSha) fail('model sha mismatch');
if (contract.verification.shaderSha256 !== actualShaderSha) fail('shader sha mismatch');
if (contract.verification.postprocessSha256 !== actualPostprocessSha || contract.model.output.postprocess.sourceSha256 !== actualPostprocessSha) fail('postprocess sha mismatch');
if (contract.verification.preprocessContractSha256 !== actualPreprocessContractSha) fail('preprocess contract sha mismatch');
if (contract.model.input.runtimeName !== contract.runtimeMetadata.inputMetadata[0].name) fail('input metadata mismatch');
if (JSON.stringify(contract.model.input.shape) !== JSON.stringify(contract.runtimeMetadata.inputMetadata[0].shape)) fail('input shape mismatch');
if (contract.model.output.runtimeName !== contract.runtimeMetadata.outputMetadata[0].name) fail('output metadata mismatch');
if (JSON.stringify(contract.model.output.shape) !== JSON.stringify(contract.runtimeMetadata.outputMetadata[0].shape)) fail('output shape mismatch');
if (contract.model.input.shape.length !== contract.model.input.axes.length || contract.model.output.shape.length !== contract.model.output.axes.length) fail('shape/axes rank mismatch');
if (contract.model.output.layout !== 'N_ATTRIBUTES_ANCHORS' || contract.model.output.axes.join(',') !== 'batch,attribute,anchor') fail('ambiguous output layout');
if (contract.model.output.semantics.anchorDimension.count !== contract.model.output.shape[2]) fail('anchor count mismatch');
if (contract.model.output.semantics.attributes.classScores.count + 4 !== contract.model.output.shape[1]) fail('attribute count mismatch');
const fixtures = await readJson('evidence/fixtures/manifest.json');
const scenarios = new Set(fixtures.images.map((item) => item.scenario));
for (const expected of ['无检测', '单目标', '多类别', '极端宽高比', '边界框']) if (!scenarios.has(expected)) fail(`missing image scenario ${expected}`);
if (scenarios.has('重叠框/NMS')) fail('overlap/NMS must not be claimed as image inference coverage');
const rawScenarios = new Set(fixtures.rawTensorFixtures.map((item) => item.scenario));
for (const expected of ['无检测', '单目标', '多类别', '重叠框/NMS', '极端宽高比', '边界框']) if (!rawScenarios.has(expected)) fail(`missing raw scenario ${expected}`);
for (const item of [...fixtures.images, ...fixtures.rawTensorFixtures]) {
  if (sha256(await readFile(resolve(root, item.path))) !== item.sha256) fail(`fixture sha mismatch: ${item.path}`);
}
if (fixtures.rawTensorFixtures.some((item) => item.sourceImage !== null)) fail('raw fixture must not claim image inference');
for (const item of fixtures.images) {
  if (!item.license?.spdx) fail(`fixture license missing: ${item.id}`);
  if (typeof item.source === 'object') {
    if (!item.source.commit || !item.source.upstreamPath || !item.source.gitBlobSha || !item.source.sha256 || !item.source.license?.url || !item.source.license?.localPath || !item.source.license?.noticePath) fail(`external source provenance missing: ${item.id}`);
    if (sha256(await readFile(resolve(root, item.source.path))) !== item.source.sha256) fail(`external source sha mismatch: ${item.id}`);
  }
}
await validateThirdPartyFixtureLicenses(root, fixtures);
const reference = await readJson('evidence/golden/web-reference.json');
if (reference.runtime.name !== 'onnxruntime-web' || reference.runtime.version !== '1.27.0' || reference.runtime.actualExecutionProvider !== 'wasm') fail('runtime/EP mismatch');
if (reference.fixtures.length !== fixtures.images.length) fail('reference fixture count');
for (const item of reference.fixtures) {
  const fixture = fixtures.images.find((candidate) => candidate.id === item.id);
  if (!fixture) fail(`reference without image fixture: ${item.id}`);
  const image = readPpm(await readFile(resolve(root, fixture.path)));
  const canonical = preprocessCanonical(image);
  if (item.imageSha256 !== fixture.sha256 || item.canonicalInput.sha256Float32Le !== tensorDigest(canonical.tensor)) fail(`canonical input digest mismatch: ${item.id}`);
  if (JSON.stringify(item.preprocessing.contract) !== JSON.stringify(PREPROCESS_CONTRACT)) fail(`preprocess contract mismatch: ${item.id}`);
  if (item.runs.length < 3) fail(`repeat count: ${item.id}`);
  if (!item.determinism.allRawDigestsEqual || !item.determinism.allDecodedEqual) fail(`non-deterministic: ${item.id}`);
  if (!item.coverage?.passed) fail(`coverage failed: ${item.id}`);
  for (const run of item.runs) {
    if (run.rawTensor.elementCount !== 84 * 8400 || run.rawTensor.finiteCount !== run.rawTensor.elementCount || !/^[0-9a-f]{64}$/.test(run.rawTensor.sha256Float32Le)) fail(`raw tensor summary invalid: ${item.id}`);
  }
}
if (reference.fixtures.find((item) => item.id === 'no-detection').runs[0].decoded.length !== 0) fail('no-detection image produced detections');
const single = reference.fixtures.find((item) => item.id === 'single-target').runs[0].decoded;
if (single.length !== 1 || single[0].classId !== 16) fail('single-target must produce exactly one class 16 detection');
const multiClasses = new Set(reference.fixtures.find((item) => item.id === 'multi-class').runs[0].decoded.map((item) => item.classId));
if (!multiClasses.has(0) || !multiClasses.has(5) || multiClasses.size < 2) fail('multi-class must include person and bus class IDs');
for (const value of [reference.tolerances.confidenceAbsolute, reference.tolerances.boxIouMinimum, reference.tolerances.rawTensorAbsolute, reference.tolerances.rawTensorRelative]) if (!Number.isFinite(value)) fail('non-finite tolerance');
if (!reference.tolerances.frozenBeforeNativeAdapterResults) fail('model tolerances not frozen');
const coverageMatrix = await readJson('evidence/golden/coverage-matrix.json');
await validateCoverageEvidence(root, coverageMatrix);
const overlap = coverageMatrix.cases.find((item) => item.id === 'overlap-nms');
if (overlap.sourceKind !== 'manually-constructed-raw-tensor' || overlap.modelInference.covered || overlap.preprocessing.covered || !overlap.decode.covered || !overlap.nms.covered) fail('coverage matrix conflates raw NMS with image inference');
const conformance = await readJson('evidence/reports/preprocess-conformance.json');
if (!conformance.adapter?.name || !conformance.adapter?.backend || conformance.fixtures.length !== fixtures.images.length) fail('CPU/WGSL adapter evidence missing');
if (conformance.fixtures.some((item) => !item.passed || item.mismatchCount !== 0)) fail('CPU/WGSL conformance failed');
const manifest = await readJson('evidence/golden/manifest.json');
for (const item of manifest.artifacts) if (sha256(await readFile(resolve(root, item.path))) !== item.sha256) fail(`artifact sha mismatch: ${item.path}`);
const conversion = await readJson('evidence/conversions/conversion-spikes.json');
if (conversion.spikes.some((item) => item.state === 'supported')) fail('spike must not claim supported');
const apple = conversion.spikes.find((item) => item.platform === 'apple');
if (apple.tool.version !== '9.0' || apple.attempt.exitCode !== 1 || apple.attempt.acceptedSources.includes('onnx') || apple.artifact !== null) fail('Core ML spike evidence');
const android = conversion.spikes.find((item) => item.platform === 'android');
const litertManifest = await readJson('evidence/conversions/litert-artifact-manifest.json');
const litertGolden = await readJson('evidence/reports/litert-golden-report.json');
const litertConversion = await readJson('evidence/reports/litert-conversion-report.json');
validateLitertEvidence(litertManifest, litertGolden, litertConversion, reference.tolerances);
if (android.state !== 'host-inference-verified' || android.supported || android.androidRunnerVerified || android.tool.version !== '2.1.6' || android.tool.converterVersion !== '0.9.3' || android.attempt.exitCode !== 0 || !android.attempt.conversionExecuted || !android.attempt.hostLoadRunExecuted || !android.attempt.goldenPassed || android.artifact.sha256 !== litertManifest.artifact.sha256) fail('LiteRT spike evidence');
const litertLock = await readFile(resolve(root, 'evidence/tooling/litert-requirements.lock'), 'utf8');
for (const line of litertLock.split('\n').filter((line) => line && !line.startsWith('#') && !line.startsWith('--'))) if (!/^[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}$/.test(line)) fail(`LiteRT unhashed dependency: ${line}`);
const windows = conversion.spikes.find((item) => item.platform === 'windows-x86_64-and-arm64');
if (windows.state !== 'blocked' || windows.conversion !== '无格式转换：Windows ML 随 Windows App SDK 提供 ONNX Runtime API，原 ONNX 应由 Microsoft.ML.OnnxRuntime.InferenceSession 实际加载并执行固定输入' || windows.artifact.sha256 !== actualModelSha) fail('Windows ML spike evidence');
const mindspore = conversion.spikes.find((item) => item.platform === 'harmonyos');
if (mindspore.tool.version !== '2.7.0' || mindspore.tool.archiveSha256 !== '8bb1097100c9fec12675670ba2d4264a2cd6da3a9be093eb56631d00fc0c455b' || !mindspore.attempt.command.includes('--fmk=ONNX') || mindspore.attempt.exitCode === null) fail('MindSpore spike evidence');
const linuxCpu = conversion.spikes.find((item) => item.platform === 'linux-x86_64-cpu');
if (linuxCpu.state !== 'inference-verified' || linuxCpu.attempt.inferenceExecuted !== true || linuxCpu.attempt.output.finiteCount !== 84 * 8400) fail('Linux ORT CPU inference evidence');
for (const provider of ['openvino', 'cuda', 'tensorrt']) if (conversion.spikes.find((item) => item.platform === `linux-x86_64-${provider}`).state !== 'blocked') fail(`${provider} must remain independently blocked`);
const provenance = await readJson('evidence/reports/model-provenance.json');
const handoffAudit = await readJson('evidence/reports/handoff-model-audit.json');
if (handoffAudit.sourceCheckpoint.sha256 !== 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36' || handoffAudit.candidateOnnx.sha256 !== '71002056f43781f2d26681c56e7ec3686d918951c5c8ae70ca55de10409a2a45') fail('handoff model digests');
if (handoffAudit.decision.ptAndOnnxTechnicalVerification !== 'passed' || handoffAudit.decision.task14Complete !== false || handoffAudit.licensing.authorizationBlocker !== false || handoffAudit.licensing.publication !== 'test-evidence-only' || handoffAudit.licensing.productPackaging !== 'excluded') fail('handoff audit test-only semantics');
if (handoffAudit.comparisons.initializers.mismatches.length !== 0 || handoffAudit.comparisons.initializers.exactEqualCount !== 143) fail('handoff initializer equivalence');
if (handoffAudit.comparisons.inference.some((item) => !item.torchRepeatExact || !item.candidateReferenceExact || !item.ptCandidateAllclose)) fail('handoff inference equivalence');
if (provenance.model.sha256 !== actualModelSha || provenance.model.embeddedMetadata.license !== 'AGPL-3.0 License (https://ultralytics.com/license)' || provenance.originalTrainingArtifact.state !== 'source-identified-and-weight-equivalent' || provenance.originalTrainingArtifact.sha256 !== handoffAudit.sourceCheckpoint.sha256) fail('model provenance');
if (provenance.licensing.originalWeights.authorizationState !== 'out-of-scope-test-only' || provenance.licensing.conversionArtifacts.redistributionAllowed !== false || provenance.licensing.rimecutPackageRedistribution.allowed !== false || provenance.decision.task14 !== 'blocked' || provenance.decision.publication !== 'test-evidence-only') fail('model test-only decision');
const replay = await readJson('evidence/replay/task1-replay.json');
if (replay.schemaVersion !== 2 || replay.repository !== 'rimeflow-yolov8n' || replay.repositoryHeadAtReplay.kind !== 'evidence-input-head' || replay.repositoryHeadAtReplay.finalEvidenceCommitRecordedByGit !== true || replay.immutableLogEvidence.kind !== 'embedded-in-manifest') fail('operator replay metadata');
for (const output of replay.outputs) {
  const bytes = await readFile(resolve(root, output.path));
  if (bytes.length !== output.bytes || sha256(bytes) !== output.sha256) fail(`operator replay output drift: ${output.path}`);
}
for (const id of ['contract-fixture-golden', 'production-raw-golden', 'conversion-report-regeneration']) {
  const step = replay.steps.find((item) => item.id === id);
  if (!step?.executed || step.rounds.length !== 2 || !step.repeatComparison.allExitCodesZero) fail(`operator replay execution semantics: ${id}`);
  for (const round of step.rounds) {
    if (!round.startedAt || !round.endedAt || round.exitCode !== 0 || !round.repositoryHead || !round.runnerId || round.worktreeBefore.tracked !== '' || round.worktreeAfter.tracked !== '' || !/^[0-9a-f]{64}$/.test(round.log.sha256)) fail(`operator replay round metadata: ${id}`);
  }
}
const litertReplayStep = replay.steps.find((item) => item.id === 'android-litert-conversion-and-host-golden');
if (!litertReplayStep?.executed || litertReplayStep.rounds.length !== 2 || !litertReplayStep.repeatComparison.allExitCodesZero || !litertReplayStep.repeatComparison.deterministicOutputDigestsEqual || Object.values(litertReplayStep.repeatComparison.details).some((value) => value !== true)) fail('LiteRT task replay semantics');
for (const round of litertReplayStep.rounds) if (round.exitCode !== 0 || round.worktreeBefore.tracked !== '' || round.worktreeAfter.tracked !== '' || round.outputs[0].sha256 !== litertManifest.artifact.sha256) fail(`LiteRT task replay round: ${round.run}`);
const conversionReplayStep = replay.steps.find((item) => item.id === 'conversion-report-regeneration');
if (conversionReplayStep.command !== 'node evidence/scripts/run_conversion_spikes.mjs --litert-only') fail('LiteRT-only conversion replay scope');
if (replay.steps.find((item) => item.id === 'delegated-platform-spikes').executed !== false || replay.task1_7OwnershipReplayComplete !== true || replay.task1_4Complete !== false) fail('operator replay blocked semantics');
console.log(JSON.stringify({ ok: true, schemaVersion: 1, checkedArtifacts: manifest.artifacts.length, checkedFixtures: fixtures.images.length + fixtures.rawTensorFixtures.length }));
