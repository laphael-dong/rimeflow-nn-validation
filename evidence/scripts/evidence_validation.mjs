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

export function validateCoremlEvidence(manifest, replay) {
  const expectedPtSha = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36';
  if (manifest.source.logicalPath !== '$HANDOFF_ASSETS/yolov8n.pt' || manifest.source.before.sha256 !== expectedPtSha || manifest.source.before.bytes !== 6549796 || JSON.stringify(manifest.source.before) !== JSON.stringify(manifest.source.after)) throw new Error('Core ML source checkpoint drift');
  if (manifest.status.value !== 'artifact-spec-verified' || !manifest.status.artifactVerified || manifest.status.hostInferenceVerified || manifest.status.macosRuntimeVerified || manifest.status.iosRuntimeVerified || manifest.status.supported) throw new Error('Core ML support status overclaim');
  if (manifest.toolchain.python !== '3.12.3' || manifest.toolchain.ultralytics !== '8.4.104' || manifest.toolchain.torch !== '2.7.0+cpu' || manifest.toolchain.torchvision !== '0.22.0+cpu' || manifest.toolchain.coremltools !== '9.0' || manifest.toolchain.numpy !== '2.3.5') throw new Error('Core ML toolchain drift');
  if (manifest.artifact.trackedByGit || manifest.artifact.tree.fileCount !== 3 || manifest.artifact.tree.totalFileBytes <= 0 || !/^[0-9a-f]{64}$/.test(manifest.artifact.tree.digest)) throw new Error('Core ML artifact tree metadata invalid');
  if (sha256(Buffer.from(JSON.stringify(manifest.artifact.tree.files))) !== manifest.artifact.tree.digest) throw new Error('Core ML canonical tree digest mismatch');
  const paths = manifest.artifact.tree.files.map((item) => item.path);
  if (JSON.stringify(paths) !== JSON.stringify(['Data/com.apple.CoreML/model.mlmodel', 'Data/com.apple.CoreML/weights/weight.bin', 'Manifest.json']) || manifest.artifact.tree.files.some((item) => item.bytes <= 0 || !/^[0-9a-f]{64}$/.test(item.sha256))) throw new Error('Core ML package tree drift');
  const spec = manifest.spec;
  if (spec.modelType !== 'mlProgram' || spec.specificationVersion !== 6 || spec.opset !== 'CoreML5' || spec.minimumDeploymentTarget.iOS !== '15.0' || spec.minimumDeploymentTarget.macOS !== '12.0') throw new Error('Core ML model/deployment spec drift');
  if (spec.input.name !== 'image' || spec.input.index !== 0 || spec.input.featureType !== 'IMAGE' || spec.input.colorSpace !== 'RGB' || spec.input.functionTensorDtype !== 'FLOAT32' || JSON.stringify(spec.input.shape) !== '[1,3,640,640]' || spec.input.width !== 640 || spec.input.height !== 640 || Math.abs(spec.input.scale - 1 / 255) > 1e-9 || JSON.stringify(spec.input.bias) !== '[0,0,0]') throw new Error('Core ML input contract drift');
  if (spec.output.name !== 'var_911' || spec.output.index !== 0 || spec.output.count !== 1 || spec.output.featureType !== 'MULTI_ARRAY' || spec.output.dtype !== 'FLOAT32' || spec.output.layout !== 'N_ATTRIBUTES_ANCHORS' || JSON.stringify(spec.output.shape) !== '[1,84,8400]') throw new Error('Core ML output contract drift');
  if (spec.computePrecision.actual !== 'FLOAT32' || spec.computePrecision.float16Present || spec.computePrecision.requestedHalf || spec.computePrecision.requestedQuantization !== null || spec.computePrecision.blobConstantDtypes.FLOAT32 <= 0 || Object.keys(spec.computePrecision.blobConstantDtypes).some((dtype) => dtype !== 'FLOAT32')) throw new Error('Core ML precision drift');
  if (spec.nms.fused || spec.nms.operators.length !== 0 || spec.nms.responsibility !== 'operator postprocess') throw new Error('Core ML NMS responsibility drift');
  if (!spec.preprocessing.fused.includes('multiply by 1/255') || !spec.preprocessing.notFused.includes('letterbox resize') || spec.coordinates.bboxEncoding !== 'xywh in 640x640 model-input pixel units' || JSON.stringify(spec.coordinates.strideTensor.uniqueValues) !== '[8,16,32]') throw new Error('Core ML preprocessing/coordinate contract drift');
  const comparison = replay.comparison;
  if (replay.rounds.length !== 2 || comparison.packageTreeDigestEqual || !comparison.weightBlobDigestEqual || !comparison.normalizedSpecDigestEqual || !comparison.normalizedPackageManifestDigestEqual || !comparison.precisionContractEqual || !comparison.sourceStateEqualAndUnchanged || Object.values(comparison.ioMetadataEqual).some((value) => value !== true)) throw new Error('Core ML replay determinism characterization drift');
  const changedPaths = comparison.changedFiles.map((item) => item.path).sort();
  if (JSON.stringify(changedPaths) !== JSON.stringify(['Data/com.apple.CoreML/model.mlmodel', 'Manifest.json'])) throw new Error('Core ML unexpected nondeterministic files');
  for (const round of replay.rounds) {
    if (round.conversion.exitCode !== 0 || !round.conversion.startedAt || !round.conversion.endedAt || !round.conversion.stdout || round.worktreeBefore.tracked !== round.worktreeAfter.tracked || round.source.before.sha256 !== expectedPtSha || JSON.stringify(round.source.before) !== JSON.stringify(round.source.after)) throw new Error(`Core ML replay round invalid: ${round.round}`);
  }
}
