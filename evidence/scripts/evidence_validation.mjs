import { createHash } from 'node:crypto';
import { access, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const FIXTURE_LICENSE_SHA256 = '0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0';
const UPSTREAM_ASSETS_COMMIT = '42ef8a125df038dcca49f6216f446fe9112946c1';
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

export async function isGitTracked(root, path) {
  const result = spawnSync('git', ['ls-files', '--error-unmatch', '--', path], {
    cwd: root,
    encoding: 'utf8',
  });
  return result.status === 0;
}

export async function validateCoverageEvidence(root, matrix) {
  for (const item of matrix.cases) {
    for (const layer of matrix.layers) {
      const evidence = item[layer];
      if (!evidence?.covered) continue;
      if (!evidence.path) throw new Error(`coverage evidence path missing: ${item.id}/${layer}`);
      await access(resolve(root, evidence.path));
      if (evidence.kind === 'test') {
        if (!evidence.testId || !evidence.command) throw new Error(`test evidence metadata missing: ${item.id}/${layer}`);
        if (!(await isGitTracked(root, evidence.path))) throw new Error(`test evidence is not Git tracked: ${evidence.path}`);
      }
    }
  }
}

export async function validateThirdPartyFixtureLicenses(root, fixtures) {
  if (fixtures.license.productPackaging !== 'excluded') throw new Error('fixture product packaging policy must be excluded');
  for (const item of fixtures.images) {
    if (typeof item.source !== 'object') continue;
    const licensePath = item.source.license?.localPath;
    const noticePath = item.source.license?.noticePath;
    if (!licensePath || !noticePath) throw new Error(`local third-party license missing: ${item.id}`);
    for (const path of [licensePath, noticePath]) {
      const bytes = await readFile(resolve(root, path));
      if (bytes.length === 0) throw new Error(`empty third-party license evidence: ${path}`);
      if (!(await isGitTracked(root, path))) throw new Error(`third-party license evidence is not Git tracked: ${path}`);
    }
    const licenseBytes = await readFile(resolve(root, licensePath));
    const licenseText = licenseBytes.toString('utf8');
    if (sha256(licenseBytes) !== FIXTURE_LICENSE_SHA256 || !licenseText.includes('GNU AFFERO GENERAL PUBLIC LICENSE') || !licenseText.includes('Version 3, 19 November 2007')) throw new Error(`third-party license text drift: ${licensePath}`);
    const notice = await readFile(resolve(root, noticePath), 'utf8');
    if (!notice.includes(UPSTREAM_ASSETS_COMMIT) || !notice.includes(item.source.gitBlobSha) || !notice.includes(item.source.sha256)) throw new Error(`third-party NOTICE provenance drift: ${item.id}`);
    if (item.distribution?.scope !== 'test-and-evidence-only' || item.distribution?.rimecutProductPackage !== 'prohibited') {
      throw new Error(`unsafe external fixture distribution policy: ${item.id}`);
    }
  }
}

export function validateLitertEvidence(manifest, golden, replay, frozenTolerances) {
  const expectedPtSha = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36';
  if (manifest.source.sha256 !== expectedPtSha || manifest.source.logicalPath !== '$HANDOFF_ASSETS/yolov8n.pt') throw new Error('LiteRT source checkpoint drift');
  if (!manifest.artifact?.sha256?.match(/^[0-9a-f]{64}$/) || manifest.artifact.bytes <= 0 || manifest.artifact.trackedByGit !== false) throw new Error('LiteRT artifact metadata invalid');
  if (manifest.status.value !== 'host-inference-verified' || !manifest.status.artifactVerified || !manifest.status.hostInferenceVerified || manifest.status.androidRunnerVerified || manifest.status.supported) throw new Error('LiteRT support status overclaim');
  if (manifest.toolchain.ultralytics !== '8.4.104' || manifest.toolchain.torch !== '2.12.1+cpu' || manifest.toolchain['litert-torch'] !== '0.9.3' || manifest.toolchain['litert-converter'] !== '0.3.1' || manifest.toolchain['ai-edge-litert'] !== '2.1.6') throw new Error('LiteRT toolchain drift');
  if (manifest.toolchain.tensorflow.installed || manifest.toolchain.onnx2tf.installed) throw new Error('LiteRT direct PyTorch path dependency claim');
  const input = manifest.ioContract.input;
  const output = manifest.ioContract.output;
  if (input.name !== 'serving_default_args_0' || input.index !== 0 || input.layout !== 'NCHW' || input.dtype !== 'float32' || JSON.stringify(input.shape) !== '[1,3,640,640]' || input.quantization.scale !== 0 || input.quantization.zeroPoint !== 0) throw new Error('LiteRT input contract drift');
  if (output.name !== 'serving_default_output_0_output' || output.index !== 414 || output.layout !== 'N_ATTRIBUTES_ANCHORS' || output.dtype !== 'float32' || JSON.stringify(output.shape) !== '[1,84,8400]' || output.quantization.scale !== 0 || output.quantization.zeroPoint !== 0) throw new Error('LiteRT output contract drift');
  if (!manifest.mapping.outputCoordinates.includes('attributes 0..3 by 640') || manifest.runtime.nmsOperators.length !== 0 || manifest.ownership.nms !== 'operator postprocess; no NMS op is present in the model graph') throw new Error('LiteRT output/NMS mapping drift');
  if (!golden.passed || !golden.summary.allFinite || !golden.summary.allShapesMatched || !golden.summary.deterministic || golden.summary.rawToleranceMismatchCount !== 0 || golden.summary.classMismatchCount !== 0 || golden.fixtures.length !== 5) throw new Error('LiteRT golden failed');
  const frozenToleranceKeys = Object.keys(frozenTolerances);
  const reportedToleranceKeys = Object.keys(golden.tolerances);
  if (
    reportedToleranceKeys.length !== frozenToleranceKeys.length
    || frozenToleranceKeys.some((key) => !Object.hasOwn(golden.tolerances, key) || !Object.is(golden.tolerances[key], frozenTolerances[key]))
  ) throw new Error('LiteRT frozen tolerance drift');
  if (!replay.comparison || Object.values(replay.comparison).some((value) => value !== true) || replay.rounds.length !== 2) throw new Error('LiteRT replay determinism failed');
  for (const round of replay.rounds) {
    if (round.conversion.exitCode !== 0 || round.validation.exitCode !== 0 || round.webReference.exitCode !== 0 || !round.conversion.startedAt || !round.conversion.endedAt || !round.conversion.stdout || round.artifact.sha256 !== manifest.artifact.sha256 || round.worktreeBefore.tracked !== round.worktreeAfter.tracked) throw new Error(`LiteRT replay round invalid: ${round.round}`);
  }
}

export function validateMindsporeEvidence(manifest, golden, replay, frozenTolerances) {
  const expected = {
    archive: '8bb1097100c9fec12675670ba2d4264a2cd6da3a9be093eb56631d00fc0c455b',
    artifact: '7ceeca31471d772c0ccf426b856a2533fcf66735aa3c3c90726c2e459c83e6a5',
    derivedOnnx: 'a5a73dd7a25245eb47f7de8d35fa1f612212b38587d88494bb67ad6e0753b6ea',
    handoffOnnx: '71002056f43781f2d26681c56e7ec3686d918951c5c8ae70ca55de10409a2a45',
    pt: 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
    referenceOnnx: '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad',
    reexportOnnx: '8718af53d53b6336f301ef7eacb529376f29f0c04bec415815fde7d734b9def2',
  };
  if (manifest.sourceInputs.archive.sha256 !== expected.archive || manifest.sourceInputs.handoffOnnx.sha256 !== expected.handoffOnnx || manifest.sourceInputs.pt.sha256 !== expected.pt || manifest.sourceInputs.referenceOnnx.sha256 !== expected.referenceOnnx || Object.values(manifest.sourceInputs).some((item) => !item.verified || item.sha256 !== item.expectedSha256)) throw new Error('MindSpore locked input drift');
  if (manifest.artifact.sha256 !== expected.artifact || manifest.recordedArtifactSha256 !== expected.artifact || manifest.artifact.bytes !== 12832800 || manifest.artifact.trackedByGit || manifest.artifact.format !== 'MindIR Lite / MINDIR_LITE FlatBuffer (.ms)' || manifest.artifact.location !== '.evidence/mindspore/artifacts/yolov8n-fp32.ms' || manifest.artifact.path !== manifest.artifact.location) throw new Error('MindSpore artifact metadata drift');
  if (manifest.status.value !== 'host-inference-verified' || !manifest.status.artifactVerified || !manifest.status.hostInferenceVerified || manifest.status.harmonyOsDeviceVerified || manifest.status.supported || manifest.status.task14Complete) throw new Error('MindSpore support status overclaim');
  if (manifest.toolchain.converterVersion !== '2.7.0' || manifest.toolchain.archive.sha256 !== expected.archive || manifest.toolchain.commitId !== 'd2b243f75f33a7a896483b09e567d845155cad06' || manifest.toolchain.converterVersionProbe.exitCode === 0 || manifest.toolchain.converterHelpProbe.exitCode !== 0 || manifest.toolchain.pythonRequirements.path !== 'evidence/tooling/mindspore-python-addons.lock' || !/^[0-9a-f]{64}$/.test(manifest.toolchain.pythonRequirements.sha256)) throw new Error('MindSpore toolchain provenance drift');
  if (JSON.stringify(manifest.toolchain.pythonEnvironment) !== JSON.stringify({ numpy: '2.3.5', onnx: '1.22.0', onnxruntime: '1.27.0', python: '3.12.3', torch: '2.7.0+cpu', torchvision: '0.22.0+cpu', ultralytics: '8.4.104' })) throw new Error('MindSpore Python toolchain drift');
  if (manifest.derivedOnnx.graph.sha256 !== expected.derivedOnnx || manifest.derivedOnnx.graph.nmsInGraph || manifest.derivedOnnx.transform.removedNode !== '/model.22/dfl/conv/Conv' || JSON.stringify(manifest.derivedOnnx.transform.insertedNodes) !== '["/model.22/dfl/conv/Mul","/model.22/dfl/conv/ReduceSum"]' || !manifest.derivedOnnx.equivalence.allCasesWithinTolerance || manifest.derivedOnnx.equivalence.cases.length !== 8) throw new Error('MindSpore derived ONNX evidence drift');
  const input = manifest.ioContract.inputs[0];
  const output = manifest.ioContract.outputs[0];
  if (manifest.ioContract.inputs.length !== 1 || input.name !== 'images' || input.index !== 0 || input.dtype !== 'float32' || JSON.stringify(input.shape) !== '[1,640,640,3]' || input.bytes !== 4915200 || input.quantization.length !== 0) throw new Error('MindSpore input contract drift');
  if (manifest.ioContract.outputs.length !== 1 || output.name !== 'output0' || output.index !== 0 || output.dtype !== 'float32' || JSON.stringify(output.shape) !== '[1,84,8400]' || output.bytes !== 2822400 || output.quantization.length !== 0) throw new Error('MindSpore output contract drift');
  if (!manifest.ownership.preprocessing.includes('NCHW-to-NHWC') || !manifest.ownership.coordinates.includes('xywh') || !manifest.ownership.nms.startsWith('operator;') || manifest.quantization.mode !== 'FP32; converter input/output type defaults, fp16 off, no quantization requested') throw new Error('MindSpore preprocessing/coordinate/NMS contract drift');
  if (!golden.passed || golden.recordedArtifactSha256 !== expected.artifact || golden.artifact.sha256 !== expected.artifact || golden.artifact.bytes !== manifest.artifact.bytes || golden.summary.fixtureCount !== 5 || golden.summary.passedCount !== 5 || golden.productionPostprocess.implementation !== 'src/postprocess.rs' || golden.productionPostprocess.platformSpecificImplementationAdded) throw new Error('MindSpore golden/production postprocess drift');
  const frozenKeys = Object.keys(frozenTolerances);
  if (Object.keys(golden.tolerances).length !== frozenKeys.length || frozenKeys.some((key) => !Object.hasOwn(golden.tolerances, key) || !Object.is(golden.tolerances[key], frozenTolerances[key]))) throw new Error('MindSpore frozen tolerance drift');
  for (const fixture of golden.fixtures) {
    if (!fixture.passed || !fixture.rawComparison.passed || fixture.rawComparison.elementCount !== 84 * 8400 || fixture.rawComparison.finiteCount !== fixture.rawComparison.elementCount || !fixture.decodedComparison.passed || JSON.stringify(fixture.runtimeInput.shape) !== '[1,640,640,3]' || !fixture.runtimeInput.mapping.includes('NCHW')) throw new Error(`MindSpore fixture evidence drift: ${fixture.id}`);
  }
  const recordedVerification = replay.recordedArtifactVerification;
  if (replay.mode !== 'record' || !replay.recorded || replay.recordedArtifactSha256 !== expected.artifact || replay.replayArtifactSha256 !== expected.artifact || replay.artifact.sha256 !== expected.artifact || replay.artifact.bytes !== manifest.artifact.bytes || replay.artifact.path !== manifest.artifact.path || replay.trackedEvidence !== null) throw new Error('MindSpore record/replay identity drift');
  if (!recordedVerification || recordedVerification.status !== 'recorded' || recordedVerification.expectedSha256 !== expected.artifact || recordedVerification.expectedBytes !== manifest.artifact.bytes || !recordedVerification.availableAfter || recordedVerification.afterSha256 !== expected.artifact || recordedVerification.afterBytes !== manifest.artifact.bytes || !recordedVerification.exactIdentityVerifiedAfter) throw new Error('MindSpore recorded artifact verification drift');
  if (JSON.stringify(manifest.recordedArtifactVerification) !== JSON.stringify(recordedVerification)) throw new Error('MindSpore manifest/report artifact verification mismatch');
  if (replay.rounds.length !== 2 || !replay.comparison.allDeterministic || !replay.comparison.derivedOnnxDigestEqual || !replay.comparison.fixtureResultsEqual || !replay.comparison.reexportOnnxDigestEqual || !replay.comparison.trackedWorktreeStateStable) throw new Error('MindSpore replay determinism drift');
  const expectedIds = ['reference-baseline-general', 'reference-static-general', 'handoff-static-general', 'pt-reexport-opset17-unsimplified-static-general', 'reference-static-none', 'pt-reexport-opset17-dfl-reduced-static-general'];
  for (const round of replay.rounds) {
    if (round.exportReport.output.sha256 !== expected.reexportOnnx || round.derivationReport.derived.sha256 !== expected.derivedOnnx || !round.hostValidation.passed || round.hostValidation.fixtures.length !== 5 || round.worktreeBefore.tracked !== round.worktreeAfter.tracked || JSON.stringify(round.sourceBefore) !== JSON.stringify(round.sourceAfter)) throw new Error(`MindSpore replay round drift: ${round.round}`);
    if (JSON.stringify(round.matrix.map((item) => item.id)) !== JSON.stringify(expectedIds)) throw new Error(`MindSpore bounded matrix drift: ${round.round}`);
    for (const attempt of round.matrix) {
      if (!attempt.startedAt || !attempt.endedAt || !Array.isArray(attempt.command) || !attempt.command.some((item) => item === '--fmk=ONNX') || typeof attempt.stdout !== 'string' || typeof attempt.stderr !== 'string') throw new Error(`MindSpore attempt log metadata missing: ${attempt.id}`);
      if (attempt.id === expectedIds.at(-1)) {
        if (attempt.result !== 'success' || attempt.exitCode !== 0 || attempt.artifact.sha256 !== expected.artifact || attempt.failureSignature !== null) throw new Error('MindSpore successful path drift');
      } else {
        const failure = attempt.failureSignature;
        if (attempt.result !== 'failed' || attempt.exitCode !== 255 || attempt.artifact !== null || failure.failedOperator !== '/model.22/dfl/conv/Conv' || failure.operatorType !== 'Conv2DFusion' || JSON.stringify(failure.inputShape) !== '[1,16,4,8400]' || JSON.stringify(failure.outputShape) !== '[1,1,4,8400]' || JSON.stringify(failure.weightShape) !== '[1,16,1,1]' || JSON.stringify(failure.resizeOptionalEmptyInputWarnings) !== '["/model.10/Resize","/model.13/Resize"]') throw new Error(`MindSpore failure signature drift: ${attempt.id}`);
      }
    }
    for (const fixture of round.hostValidation.fixtures) if (fixture.benchmark.exitCode !== 0 || fixture.runtime.exitCode !== 0 || fixture.productionRust.exitCode !== 0 || !fixture.passed) throw new Error(`MindSpore host Load/Run drift: ${fixture.id}`);
  }
  if (Object.values(replay.comparison.paths).some((item) => Object.values(item).some((value) => value !== true))) throw new Error('MindSpore per-path replay comparison drift');
  if (JSON.stringify(replay).includes('/home/')) throw new Error('MindSpore report leaked a host absolute path');
}

export function validateMindsporeReplayEvidence(manifest, replay) {
  const recordedSha = manifest.recordedArtifactSha256;
  const recordedBytes = manifest.artifact.bytes;
  if (manifest.status.supported || manifest.status.harmonyOsDeviceVerified || manifest.status.task14Complete) throw new Error('MindSpore replay support status overclaim');
  if (replay.mode !== 'replay' || replay.recorded !== false || replay.recordedArtifactSha256 !== recordedSha) throw new Error('MindSpore non-record replay identity drift');
  if (replay.replayArtifactSha256 !== recordedSha) throw new Error('MindSpore replay artifact digest differs from recorded identity');
  const workspaceArtifact = replay.workspaceArtifact ?? replay.artifact;
  if (!workspaceArtifact || workspaceArtifact.sha256 !== replay.replayArtifactSha256 || workspaceArtifact.bytes !== recordedBytes || workspaceArtifact.path === manifest.artifact.path || !workspaceArtifact.path.startsWith('.evidence/mindspore/replay/')) throw new Error('MindSpore workspace artifact identity drift');
  const verification = replay.recordedArtifactVerification;
  if (!verification || verification.expectedSha256 !== recordedSha || verification.expectedBytes !== recordedBytes || !verification.unchanged || verification.availableBefore !== verification.availableAfter) throw new Error('MindSpore non-record artifact preservation failed');
  if (verification.availableBefore) {
    if (verification.status !== 'verified-preserved' || verification.beforeSha256 !== recordedSha || verification.afterSha256 !== recordedSha || verification.beforeBytes !== recordedBytes || verification.afterBytes !== recordedBytes || verification.beforeMtimeNs !== verification.afterMtimeNs || !verification.exactIdentityVerifiedBefore || !verification.exactIdentityVerifiedAfter) throw new Error('MindSpore non-record exact artifact identity failed');
  } else if (verification.status !== 'recorded-artifact-unavailable' || verification.beforeSha256 !== null || verification.afterSha256 !== null || verification.beforeBytes !== null || verification.afterBytes !== null || verification.beforeMtimeNs !== null || verification.afterMtimeNs !== null || verification.exactIdentityVerifiedBefore || verification.exactIdentityVerifiedAfter) {
    throw new Error('MindSpore missing recorded artifact was overclaimed');
  }
  const expectedTrackedPaths = {
    manifest: 'evidence/conversions/mindspore-artifact-manifest.json',
    goldenReport: 'evidence/reports/mindspore-golden-report.json',
    conversionReport: 'evidence/reports/mindspore-conversion-report.json',
  };
  for (const [key, path] of Object.entries(expectedTrackedPaths)) {
    const item = replay.trackedEvidence?.[key];
    if (!item || item.path !== path || item.bytes <= 0 || item.bytes !== item.bytesAfter || !/^[0-9a-f]{64}$/.test(item.sha256) || item.sha256 !== item.sha256After || item.unchanged !== true) throw new Error(`MindSpore non-record tracked ${key} drift`);
  }
  const comparison = replay.repeatComparison?.details ?? replay.comparison;
  if (!comparison?.allDeterministic || !comparison.derivedOnnxDigestEqual || !comparison.fixtureResultsEqual || !comparison.reexportOnnxDigestEqual || !comparison.trackedWorktreeStateStable || Object.values(comparison.paths).some((item) => Object.values(item).some((value) => value !== true))) throw new Error('MindSpore non-record replay determinism drift');
  if (!Array.isArray(replay.rounds) || replay.rounds.length !== 2) throw new Error('MindSpore non-record replay rounds missing');
  for (const round of replay.rounds) {
    const roundArtifact = round.outputs?.[0] ?? round.matrix?.find((item) => item.result === 'success')?.artifact;
    const roundPassed = round.exitCode === undefined ? round.hostValidation?.passed : round.exitCode === 0;
    if (!roundArtifact || roundArtifact.sha256 !== replay.replayArtifactSha256 || roundArtifact.bytes !== recordedBytes || !roundPassed || round.worktreeBefore.tracked !== round.worktreeAfter.tracked) throw new Error(`MindSpore non-record replay round drift: ${round.run ?? round.round}`);
  }
}

export function validateCoremlEvidence(manifest, replay) {
  const expectedPtSha = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36';
  if (manifest.source.logicalPath !== '$HANDOFF_ASSETS/yolov8n.pt' || manifest.source.before.sha256 !== expectedPtSha || manifest.source.before.bytes !== 6549796 || JSON.stringify(manifest.source.before) !== JSON.stringify(manifest.source.after)) throw new Error('Core ML source checkpoint drift');
  if (manifest.status.value !== 'artifact-spec-verified' || !manifest.status.artifactVerified || manifest.status.hostInferenceVerified || manifest.status.macosRuntimeVerified || manifest.status.iosRuntimeVerified || manifest.status.supported) throw new Error('Core ML support status overclaim');
  if (manifest.toolchain.python !== '3.12.3' || manifest.toolchain.ultralytics !== '8.4.104' || manifest.toolchain.torch !== '2.7.0+cpu' || manifest.toolchain.torchvision !== '0.22.0+cpu' || manifest.toolchain.coremltools !== '9.0' || manifest.toolchain.numpy !== '2.3.5') throw new Error('Core ML toolchain drift');
  if (manifest.artifact.trackedByGit || manifest.artifact.tree.fileCount !== 3 || manifest.artifact.tree.totalFileBytes <= 0 || !/^[0-9a-f]{64}$/.test(manifest.artifact.tree.digest)) throw new Error('Core ML artifact tree metadata invalid');
  if (sha256(Buffer.from(JSON.stringify(manifest.artifact.tree.files))) !== manifest.artifact.tree.digest) throw new Error('Core ML canonical tree digest mismatch');
  const paths = manifest.artifact.tree.files.map((item) => item.path);
  if (JSON.stringify(paths) !== JSON.stringify(['Data/com.apple.CoreML/model.mlmodel', 'Data/com.apple.CoreML/weights/weight.bin', 'Manifest.json']) || manifest.artifact.tree.files.some((item) => item.bytes <= 0 || !/^[0-9a-f]{64}$/.test(item.sha256))) throw new Error('Core ML package tree drift');
  const weightBlob = manifest.artifact.tree.files.find((item) => item.path === 'Data/com.apple.CoreML/weights/weight.bin');
  const semantic = manifest.semanticReplayDigests;
  if (manifest.recordedArtifactTreeDigest !== manifest.artifact.tree.digest) throw new Error('Core ML recorded artifact identity drift');
  if (!semantic || semantic.normalizedSpecSha256 !== manifest.spec.normalizedSpecSha256 || semantic.normalizedPackageManifestSha256 !== manifest.packageManifest.normalizedSha256 || semantic.weightBlobSha256 !== weightBlob.sha256) throw new Error('Core ML recorded semantic digest drift');
  if (Object.values(semantic).includes(manifest.recordedArtifactTreeDigest)) throw new Error('Core ML semantic digest impersonates artifact identity');
  const spec = manifest.spec;
  if (spec.modelType !== 'mlProgram' || spec.specificationVersion !== 6 || spec.opset !== 'CoreML5' || spec.minimumDeploymentTarget.iOS !== '15.0' || spec.minimumDeploymentTarget.macOS !== '12.0') throw new Error('Core ML model/deployment spec drift');
  if (spec.input.name !== 'image' || spec.input.index !== 0 || spec.input.featureType !== 'IMAGE' || spec.input.colorSpace !== 'RGB' || spec.input.functionTensorDtype !== 'FLOAT32' || JSON.stringify(spec.input.shape) !== '[1,3,640,640]' || spec.input.width !== 640 || spec.input.height !== 640 || Math.abs(spec.input.scale - 1 / 255) > 1e-9 || JSON.stringify(spec.input.bias) !== '[0,0,0]') throw new Error('Core ML input contract drift');
  if (spec.output.name !== 'var_911' || spec.output.index !== 0 || spec.output.count !== 1 || spec.output.featureType !== 'MULTI_ARRAY' || spec.output.dtype !== 'FLOAT32' || spec.output.layout !== 'N_ATTRIBUTES_ANCHORS' || JSON.stringify(spec.output.shape) !== '[1,84,8400]') throw new Error('Core ML output contract drift');
  if (spec.computePrecision.actual !== 'FLOAT32' || spec.computePrecision.float16Present || spec.computePrecision.requestedHalf || spec.computePrecision.requestedQuantization !== null || spec.computePrecision.blobConstantDtypes.FLOAT32 <= 0 || Object.keys(spec.computePrecision.blobConstantDtypes).some((dtype) => dtype !== 'FLOAT32')) throw new Error('Core ML precision drift');
  if (spec.nms.fused || spec.nms.operators.length !== 0 || spec.nms.responsibility !== 'operator postprocess') throw new Error('Core ML NMS responsibility drift');
  if (!spec.preprocessing.fused.includes('multiply by 1/255') || !spec.preprocessing.notFused.includes('letterbox resize') || spec.coordinates.bboxEncoding !== 'xywh in 640x640 model-input pixel units' || JSON.stringify(spec.coordinates.strideTensor.uniqueValues) !== '[8,16,32]') throw new Error('Core ML preprocessing/coordinate contract drift');
  const comparison = replay.comparison;
  if (replay.mode !== 'record' || replay.recordedArtifactTreeDigest !== manifest.recordedArtifactTreeDigest) throw new Error('Core ML record report identity drift');
  if (JSON.stringify(replay.semanticReplayDigests?.recorded) !== JSON.stringify(semantic) || replay.semanticReplayDigests.rounds.some((item) => JSON.stringify(item) !== JSON.stringify(semantic)) || !replay.semanticReplayValidation?.allMatched) throw new Error('Core ML record report semantic digest drift');
  const recordedVerification = replay.recordedArtifactVerification;
  if (!recordedVerification?.availableAfter || !recordedVerification.exactIdentityVerifiedAfter || recordedVerification.afterTreeDigest !== manifest.recordedArtifactTreeDigest || recordedVerification.expectedTreeDigest !== manifest.recordedArtifactTreeDigest || recordedVerification.status !== 'recorded') throw new Error('Core ML recorded artifact verification drift');
  if (JSON.stringify(manifest.recordedArtifactVerification) !== JSON.stringify(recordedVerification)) throw new Error('Core ML manifest/report artifact verification mismatch');
  if (replay.rounds.length !== 2 || comparison.packageTreeDigestEqual || !comparison.weightBlobDigestEqual || !comparison.normalizedSpecDigestEqual || !comparison.normalizedPackageManifestDigestEqual || !comparison.precisionContractEqual || !comparison.sourceStateEqualAndUnchanged || Object.values(comparison.ioMetadataEqual).some((value) => value !== true)) throw new Error('Core ML replay determinism characterization drift');
  const changedPaths = comparison.changedFiles.map((item) => item.path).sort();
  if (JSON.stringify(changedPaths) !== JSON.stringify(['Data/com.apple.CoreML/model.mlmodel', 'Manifest.json'])) throw new Error('Core ML unexpected nondeterministic files');
  for (const round of replay.rounds) {
    if (round.conversion.exitCode !== 0 || !round.conversion.startedAt || !round.conversion.endedAt || !round.conversion.stdout || round.worktreeBefore.tracked !== round.worktreeAfter.tracked || round.source.before.sha256 !== expectedPtSha || JSON.stringify(round.source.before) !== JSON.stringify(round.source.after)) throw new Error(`Core ML replay round invalid: ${round.round}`);
  }
}

export function validateCoremlReplayEvidence(manifest, replay) {
  const recordedDigest = manifest.recordedArtifactTreeDigest;
  const semantic = manifest.semanticReplayDigests;
  if (replay.mode !== 'replay' || replay.recordedArtifactTreeDigest !== recordedDigest) throw new Error('Core ML non-record replay identity drift');
  if (Object.values(semantic).includes(recordedDigest)) throw new Error('Core ML semantic digest impersonates artifact identity');
  if (JSON.stringify(replay.semanticReplayDigests?.recorded) !== JSON.stringify(semantic) || replay.semanticReplayDigests.rounds.length !== 2 || replay.semanticReplayDigests.rounds.some((item) => JSON.stringify(item) !== JSON.stringify(semantic))) throw new Error('Core ML non-record semantic digest drift');
  if (!replay.semanticReplayValidation?.allMatched || !replay.semanticReplayValidation.scope?.includes('never an artifact identity')) throw new Error('Core ML non-record semantic validation drift');
  const verification = replay.recordedArtifactVerification;
  if (!verification || verification.expectedTreeDigest !== recordedDigest || !verification.unchanged || verification.availableBefore !== verification.availableAfter) throw new Error('Core ML non-record artifact preservation failed');
  if (verification.availableBefore) {
    if (verification.status !== 'verified-preserved' || verification.beforeTreeDigest !== recordedDigest || verification.afterTreeDigest !== recordedDigest || !verification.exactIdentityVerifiedBefore || !verification.exactIdentityVerifiedAfter) throw new Error('Core ML non-record exact artifact identity failed');
  } else if (verification.status !== 'recorded-artifact-unavailable' || verification.beforeTreeDigest !== null || verification.afterTreeDigest !== null || verification.exactIdentityVerifiedBefore || verification.exactIdentityVerifiedAfter) {
    throw new Error('Core ML missing recorded artifact was overclaimed');
  }
  for (const key of ['manifest', 'report']) {
    const item = replay.trackedEvidence?.[key];
    if (!item?.unchanged || item.sha256 !== item.sha256After || !/^[0-9a-f]{64}$/.test(item.sha256)) throw new Error(`Core ML non-record tracked ${key} drift`);
  }
  if (replay.rounds.length !== 2 || replay.rounds.some((round) => round.exitCode !== undefined && round.exitCode !== 0)) throw new Error('Core ML non-record replay rounds invalid');
}
