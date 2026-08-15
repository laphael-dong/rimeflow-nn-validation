import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const outputPath = 'evidence/reports/platform-conformance-report.json';
const BASE_COMMIT = '96fcfa0a54a3db1978af5ec38fc3a92132be4ee3';
const MODEL_SHA = '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad';
const EVIDENCE_COMMIT = 'ef483004c57a48d8373171018f8b0ada8fb590d9';

const stable = (value) => `${JSON.stringify(value, null, 2)}\n`;
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const evidence = (path) => [path];
const passed = (reason, paths) => ({ outcome: 'passed', reason, evidence: paths });
const blocked = (reason, paths) => ({ outcome: 'blocked', reason, evidence: paths });
const fail = (message) => { throw new Error(message); };

async function json(path) {
  return JSON.parse(await readFile(resolve(root, path), 'utf8'));
}

function check(condition, message) {
  if (!condition) fail(message);
}

function io(inputLayout) {
  return { inputRole: 'image', inputLayout, outputRole: 'detections', outputLayout: 'N_ATTRIBUTES_ANCHORS', nmsFused: false };
}

function commonChecks(artifactPaths, ioPaths, { smoke, golden, diagnostics, packageLoad, fault }) {
  return {
    artifactManifest: passed('Tracked artifact manifest identifies the candidate and digest.', artifactPaths),
    logicalIo: passed('Logical image/detections roles and non-fused NMS are explicit.', ioPaths),
    timeout: blocked('No target-runner timeout evidence is recorded; this field cannot be inferred from host replay.', artifactPaths),
    smoke,
    golden,
    fault,
    diagnostics,
    performance: blocked('No target-platform performance sample is recorded in Validation evidence.', artifactPaths),
    packageLoad,
  };
}

export async function buildReport() {
  const [runtimeManifest, coreml, litert, windows, windowsReport, mindspore, validationReport, replay] = await Promise.all([
    json('evidence/manifest/validation-runtime-manifest.json'),
    json('evidence/conversions/coreml-artifact-manifest.json'),
    json('evidence/conversions/litert-artifact-manifest.json'),
    json('evidence/conversions/windows-ml-spike-manifest.json'),
    json('evidence/reports/windows-ml-spike-report.json'),
    json('evidence/conversions/mindspore-artifact-manifest.json'),
    json('evidence/reports/validation-manifest-golden-report.json'),
    json('evidence/replay/task1-replay.json'),
  ]);
  const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
  check(sha256(modelBytes) === MODEL_SHA, 'frozen model digest mismatch');
  check(runtimeManifest.baseRuntime?.commit === BASE_COMMIT, 'Base runtime identity drift');
  check(runtimeManifest.model?.sha256 === MODEL_SHA, 'runtime manifest model identity drift');
  check(coreml.status?.value === 'artifact-spec-verified' && coreml.status.supported === false, 'Core ML status overclaim');
  check(litert.status?.value === 'host-inference-verified' && litert.status.androidRunnerVerified === false && litert.status.supported === false, 'LiteRT status overclaim');
  check(windows.state === 'blocked' && windows.supported === false && windows.targets?.['win-x64']?.build?.staticCompileVerified === true && windows.targets?.['win-arm64']?.build?.staticCompileVerified === true, 'Windows static evidence drift');
  check(mindspore.status?.value === 'host-inference-verified' && mindspore.status.harmonyOsDeviceVerified === false && mindspore.status.supported === false, 'MindSpore status overclaim');
  check(validationReport.summary?.allRuntimeComparisonsPassed === true, 'Linux ORT golden evidence missing');
  check(replay.runner?.os === 'linux' && replay.runner?.architecture === 'x64', 'replay runner identity drift');

  const linuxRunner = replay.runner.id;
  const sourceCommit = EVIDENCE_COMMIT;
  const commonSource = { sourceCommit };
  const report = {
    schemaVersion: 1,
    change: 'rimeflow-backend-contract',
    ownership: 'Validation',
    baseRuntime: { commit: BASE_COMMIT },
    model: { path: 'models/yolov8n.onnx', sha256: MODEL_SHA },
    postprocess: { owner: 'src/postprocess.rs', decode: 'operator', nms: 'operator' },
    evidenceCommit: EVIDENCE_COMMIT,
    platforms: [
      {
        id: 'apple-coreml', section: '6.2', status: 'artifact-spec-verified', supported: false,
        runner: { identity: null, os: 'macos-or-ios', arch: 'arm64', availability: 'required-not-recorded', scope: 'target-runner' },
        sdkRuntime: { coreml: 'CoreML5', minimumMacos: coreml.spec.minimumDeploymentTarget.macOS, minimumIos: coreml.spec.minimumDeploymentTarget.iOS },
        converter: { name: 'coremltools', version: coreml.toolchain.coremltools, path: coreml.conversion.path },
        ...commonSource,
        artifact: { format: coreml.artifact.format, digest: coreml.artifact.tree.digest }, io: io('NCHW'),
        checks: commonChecks(evidence('evidence/conversions/coreml-artifact-manifest.json'), evidence('evidence/conversions/coreml-artifact-manifest.json'), {
          smoke: blocked('No real macOS or iOS Core ML runner has loaded the package.', evidence('evidence/conversions/coreml-artifact-manifest.json')),
          golden: blocked('No real macOS or iOS Core ML runner has produced golden comparisons.', evidence('evidence/conversions/coreml-artifact-manifest.json')),
          fault: blocked('No target Core ML fault-injection run is recorded.', evidence('evidence/conversions/coreml-artifact-manifest.json')),
          diagnostics: passed('Spec inspection and two-round semantic replay are recorded.', evidence('evidence/reports/coreml-conversion-report.json')),
          packageLoad: blocked('Package inspection is not a macOS/iOS Core ML Load/Run result.', evidence('evidence/conversions/coreml-artifact-manifest.json')),
        }),
        blockers: ['Missing real macOS/iOS Core ML Load/Run runner and target package-load, timeout, fault, golden, and performance evidence.'],
      },
      {
        id: 'android-litert-v2', section: '6.3', status: 'host-inference-verified', supported: false,
        runner: { identity: linuxRunner, os: 'linux', arch: 'x64', availability: 'recorded', scope: 'evidence-host' },
        sdkRuntime: { runtime: litert.runtime.name, version: litert.runtime.version, targetRunner: 'required Android arm64 LiteRT v2 runner' },
        converter: { name: 'litert-torch/litert-converter', version: litert.toolchain['litert-converter'], path: litert.conversion.path },
        ...commonSource,
        artifact: { format: litert.artifact.format, digest: litert.artifact.sha256 }, io: io(litert.ioContract.input.layout),
        checks: commonChecks(evidence('evidence/conversions/litert-artifact-manifest.json'), evidence('evidence/conversions/litert-artifact-manifest.json'), {
          smoke: passed('Linux x64 host interpreter replay ran all five image fixtures.', evidence('evidence/reports/litert-conversion-report.json')),
          golden: passed('Five-fixture frozen golden comparison passed deterministically on the recorded host.', evidence('evidence/conversions/litert-artifact-manifest.json')),
          fault: blocked('No Android runner fault-injection result is recorded.', evidence('evidence/conversions/litert-artifact-manifest.json')),
          diagnostics: passed('Two-round conversion and I/O metadata comparison are recorded.', evidence('evidence/reports/litert-conversion-report.json')),
          packageLoad: passed('Recorded Linux host interpreter loaded the TFLite candidate; this does not establish Android package loading.', evidence('evidence/reports/litert-conversion-report.json')),
        }),
        blockers: ['Missing Android arm64 LiteRT v2 runner, its timeout/fault/performance evidence, and Android package-load evidence.'],
      },
      ...['win-x64', 'win-arm64'].map((target) => ({
        id: `windows-ml-${target.slice(4)}`, section: '6.4', status: 'build-verified', supported: false,
        runner: { identity: null, os: 'windows', arch: target === 'win-x64' ? 'x64' : 'arm64', availability: 'required-not-recorded', scope: 'target-runner' },
        sdkRuntime: { dotnetSdk: windows.dependencies.dotnetSdk, dotnetRuntime: windows.dependencies.dotnetRuntime, windowsMl: windows.dependencies.windowsMlPackage.version },
        converter: { name: 'none', version: 'canonical ONNX is consumed directly', path: 'no conversion' },
        ...commonSource,
        artifact: { format: 'ONNX', digest: windows.canonicalOnnx.sha256 }, io: io('NCHW'),
        checks: commonChecks(evidence('evidence/conversions/windows-ml-spike-manifest.json'), evidence('evidence/conversions/windows-ml-spike-manifest.json'), {
          smoke: blocked(`No real ${target} Windows ML Load/Run runner is recorded.`, evidence('evidence/conversions/windows-ml-spike-manifest.json')),
          golden: blocked(`No real ${target} Windows ML golden output is recorded.`, evidence('evidence/conversions/windows-ml-spike-manifest.json')),
          fault: passed('Runner lifecycle defines failure stages and static compile guards exercise the contract.', ['evidence/conversions/windows-ml-spike-manifest.json', 'evidence/scripts/test_windows_ml_runner_guards.mjs']),
          diagnostics: passed('Static compile report and Windows ML runtime-introspection contract are recorded.', ['evidence/reports/windows-ml-static-compile-report.json', 'evidence/reports/windows-ml-spike-report.json']),
          packageLoad: blocked(`No real ${target} Windows ML package-load result is recorded.`, evidence('evidence/conversions/windows-ml-spike-manifest.json')),
        }),
        blockers: [windows.targets[target].blocker],
      })),
      {
        id: 'linux-ort-x64', section: '6.5', status: 'host-inference-verified', supported: false,
        runner: { identity: linuxRunner, os: 'linux', arch: 'x64', availability: 'recorded', scope: 'target-runner' },
        sdkRuntime: { runtime: 'onnxruntime-node-native', evidenceRuntime: 'onnxruntime-web/onnxruntime-node validation harness' },
        converter: { name: 'none', version: 'canonical ONNX is consumed directly', path: 'no conversion' },
        ...commonSource,
        artifact: { format: 'ONNX', digest: MODEL_SHA }, io: io('NCHW'),
        checks: commonChecks(evidence('evidence/manifest/validation-runtime-manifest.json'), evidence('evidence/manifest/validation-runtime-manifest.json'), {
          smoke: passed('Native ORT completed three runs for every frozen image fixture.', evidence('evidence/reports/validation-manifest-golden-report.json')),
          golden: passed('Native ORT raw and decoded outputs matched the frozen Web baseline.', evidence('evidence/reports/validation-manifest-golden-report.json')),
          fault: passed('Validation matrix includes rejection coverage for invalid I/O, roles, tolerance, and ownership.', evidence('evidence/reports/validation-manifest-golden-report.json')),
          diagnostics: passed('Per-fixture raw summaries and production postprocess identity are recorded.', evidence('evidence/reports/validation-manifest-golden-report.json')),
          packageLoad: passed('The native ONNX Runtime session loaded the canonical ONNX model.', evidence('evidence/reports/validation-manifest-golden-report.json')),
        }),
        blockers: ['Adapter, target package, timeout, and performance closure are outside this Validation evidence; supported remains false.'],
      },
      {
        id: 'harmonyos-mindspore-lite', section: '6.6', status: 'host-inference-verified', supported: false,
        runner: { identity: linuxRunner, os: 'linux', arch: 'x64', availability: 'recorded', scope: 'evidence-host' },
        sdkRuntime: { runtime: 'MindSpore Lite', version: mindspore.toolchain.converterVersion, targetRunner: 'required HarmonyOS device runner' },
        converter: { name: 'MindSpore Lite converter', version: mindspore.toolchain.converterVersion, path: mindspore.toolchain.converter.path },
        ...commonSource,
        artifact: { format: mindspore.artifact.format, digest: mindspore.artifact.sha256 }, io: io('NHWC'),
        checks: commonChecks(evidence('evidence/conversions/mindspore-artifact-manifest.json'), evidence('evidence/conversions/mindspore-artifact-manifest.json'), {
          smoke: passed('Official benchmark and C++ runtime loaded and ran five fixtures on recorded Linux x64 host.', evidence('evidence/reports/mindspore-golden-report.json')),
          golden: passed('Five fixture outputs passed the frozen golden and production postprocess checks.', evidence('evidence/reports/mindspore-golden-report.json')),
          fault: passed('Conversion matrix records expected failure signatures for unsupported source paths.', evidence('evidence/conversions/mindspore-artifact-manifest.json')),
          diagnostics: passed('Converter archive, binary, derived ONNX, and runtime I/O evidence are recorded.', evidence('evidence/conversions/mindspore-artifact-manifest.json')),
          packageLoad: passed('The official C++ MindSpore Lite runtime loaded the recorded .ms artifact on Linux x64.', evidence('evidence/reports/mindspore-golden-report.json')),
        }),
        blockers: ['Missing HarmonyOS device runner and target timeout/performance/package evidence; host evidence does not establish HarmonyOS support.'],
      },
    ],
  };
  return report;
}

export function validateReport(report) {
  check(report.schemaVersion === 1 && report.change === 'rimeflow-backend-contract' && report.ownership === 'Validation', 'report identity drift');
  check(report.baseRuntime?.commit === BASE_COMMIT && report.model?.sha256 === MODEL_SHA && report.evidenceCommit === EVIDENCE_COMMIT, 'report input identity drift');
  check(report.postprocess?.owner === 'src/postprocess.rs' && report.postprocess.decode === 'operator' && report.postprocess.nms === 'operator', 'postprocess ownership drift');
  const expected = ['apple-coreml', 'android-litert-v2', 'windows-ml-x64', 'windows-ml-arm64', 'linux-ort-x64', 'harmonyos-mindspore-lite'];
  check(JSON.stringify(report.platforms.map((item) => item.id)) === JSON.stringify(expected), 'platform coverage/order drift');
  for (const platform of report.platforms) {
    check(platform.supported === false, `${platform.id}: supported must remain false`);
    check(platform.sourceCommit === EVIDENCE_COMMIT, `${platform.id}: evidence commit drift`);
    check(/^[0-9a-f]{64}$/.test(platform.artifact?.digest ?? ''), `${platform.id}: artifact digest missing`);
    check(platform.io?.inputRole === 'image' && typeof platform.io.inputLayout === 'string' && platform.io.inputLayout.length > 0 && platform.io.outputRole === 'detections' && platform.io.outputLayout === 'N_ATTRIBUTES_ANCHORS' && platform.io.nmsFused === false, `${platform.id}: logical I/O drift`);
    for (const name of ['artifactManifest', 'logicalIo', 'timeout', 'smoke', 'golden', 'fault', 'diagnostics', 'performance', 'packageLoad']) {
      const item = platform.checks?.[name];
      check(item && ['passed', 'blocked', 'not-run', 'not-applicable'].includes(item.outcome) && typeof item.reason === 'string' && item.reason.length > 0 && Array.isArray(item.evidence) && item.evidence.length > 0, `${platform.id}: ${name} field missing`);
    }
    check(['evidence-host', 'target-runner'].includes(platform.runner.scope), `${platform.id}: runner scope missing`);
    if (platform.runner.availability === 'required-not-recorded') check(platform.runner.identity === null && platform.runner.scope === 'target-runner', `${platform.id}: unavailable runner must not have an invented identity`);
    if (platform.runner.availability === 'recorded') check(typeof platform.runner.identity === 'string' && platform.runner.identity.length > 0, `${platform.id}: recorded runner lacks identity`);
  }
  for (const id of ['apple-coreml', 'windows-ml-x64', 'windows-ml-arm64']) {
    const item = report.platforms.find((platform) => platform.id === id);
    check(item.checks.smoke.outcome === 'blocked' && item.checks.golden.outcome === 'blocked' && item.checks.packageLoad.outcome === 'blocked', `${id}: unavailable runner overclaim`);
  }
  check(report.platforms.find((item) => item.id === 'apple-coreml').status === 'artifact-spec-verified', 'Core ML status drift');
  check(report.platforms.find((item) => item.id === 'android-litert-v2').status === 'host-inference-verified', 'LiteRT status drift');
  check(report.platforms.filter((item) => item.id.startsWith('windows-ml-')).every((item) => item.status === 'build-verified'), 'Windows status drift');
  check(report.platforms.find((item) => item.id === 'linux-ort-x64').status === 'host-inference-verified', 'Linux ORT status drift');
  check(report.platforms.find((item) => item.id === 'harmonyos-mindspore-lite').status === 'host-inference-verified', 'MindSpore status drift');
  return { platformCount: report.platforms.length, statuses: Object.fromEntries(report.platforms.map((item) => [item.id, item.status])) };
}

const invokedAsScript = resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url);

if (invokedAsScript && process.argv.includes('--write')) {
  const report = await buildReport();
  validateReport(report);
  await writeFile(resolve(root, outputPath), stable(report));
  process.stdout.write(`${JSON.stringify({ output: outputPath, ...validateReport(report) })}\n`);
} else if (invokedAsScript && process.argv.includes('--check')) {
  const expected = await buildReport();
  const actual = await json(outputPath);
  check(stable(actual) === stable(expected), 'tracked platform conformance report is not derived from current evidence');
  process.stdout.write(`${JSON.stringify({ ok: true, ...validateReport(actual) })}\n`);
} else if (invokedAsScript) {
  process.stdout.write(`${stable(await buildReport())}`);
}
