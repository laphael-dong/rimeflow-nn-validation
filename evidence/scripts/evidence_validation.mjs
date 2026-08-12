import { createHash } from 'node:crypto';
import { access, readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const FIXTURE_LICENSE_SHA256 = '0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0';
const UPSTREAM_ASSETS_COMMIT = '42ef8a125df038dcca49f6216f446fe9112946c1';
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

const WINDOWS_MODEL_SHA256 = '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad';
const WINDOWS_PACKAGE_VERSION = '2.1.74';
const WINDOWS_DOTNET_RUNTIME_VERSION = '8.0.29';
const WINDOWS_LOCK_SHA256 = '4277499d381910ed0e268967b2949572ca24932aea27044dbce76113a128b255';
const WINDOWS_STATIC_SOURCE_FILES = ['Program.cs', 'RunnerSupport.cs', 'WindowsMlSpike.csproj', 'global.json', 'packages.lock.json'];

export async function validateWindowsMlStaticCompileEvidence(root, staticReport) {
  if (staticReport.schemaVersion !== 1 || staticReport.host?.os !== 'linux' || staticReport.host?.arch !== 'x64') throw new Error('Windows ML static compile report host/schema drift');
  if (staticReport.sdk?.version !== '8.0.423' || !staticReport.sdk.executable?.endsWith('/dotnet') || staticReport.lockFileSha256 !== WINDOWS_LOCK_SHA256) throw new Error('Windows ML static compile SDK/lock drift');
  if (JSON.stringify(Object.keys(staticReport.source)) !== JSON.stringify(WINDOWS_STATIC_SOURCE_FILES)) throw new Error('Windows ML static compile source set drift');
  for (const file of WINDOWS_STATIC_SOURCE_FILES) {
    const bytes = await readFile(resolve(root, 'evidence/tooling/windows-ml-runner', file));
    if (staticReport.source[file]?.bytes !== bytes.length || staticReport.source[file]?.sha256 !== sha256(bytes)) throw new Error(`Windows ML static compile source identity drift: ${file}`);
  }
  if (JSON.stringify(Object.keys(staticReport.targets)) !== '["win-x64","win-arm64"]') throw new Error('Windows ML static compile RIDs must remain separate');
  for (const rid of ['win-x64', 'win-arm64']) {
    const target = staticReport.targets[rid];
    if (!target || Object.values(target.cleanBefore ?? {}).some((value) => value !== true) || Object.keys(target.cleanBefore ?? {}).length !== 3) throw new Error(`${rid}: static compile did not begin without bin/obj/project.assets.json`);
    if (!target.workspacePolicy?.includes('mkdtemp workspace deleted') || target.restore?.exitCode !== 0 || !target.restore.startedAt || !target.restore.endedAt || Date.parse(target.restore.startedAt) > Date.parse(target.restore.endedAt)) throw new Error(`${rid}: locked restore evidence invalid`);
    if (!target.restore.command.includes('dotnet restore WindowsMlSpike.csproj --locked-mode') || /--runtime\s+win-/.test(target.restore.command)) throw new Error(`${rid}: restore command is not the locked all-RID restore`);
    const compile = target.compile;
    if (compile?.exitCode !== 0 || compile.target !== 'Compile' || !compile.startedAt || !compile.endedAt || Date.parse(compile.startedAt) > Date.parse(compile.endedAt) || !compile.coreCompileExecuted || !compile.roslynCscExecuted || !compile.programCsCompiled) throw new Error(`${rid}: CoreCompile/Roslyn evidence invalid`);
    for (const fragment of ['dotnet msbuild WindowsMlSpike.csproj', '-target:Compile', `-property:RuntimeIdentifier=${rid}`, '-property:OutputType=Library', '-property:WindowsAppSDKSelfContained=false', '-property:RestoreLockedMode=true', '-property:PathMap=', '=/_/windows-ml-runner']) if (!compile.command.includes(fragment)) throw new Error(`${rid}: static compile command drift: ${fragment}`);
    if (/ManifestTool|\/bin\/true|app\.manifest|SkipCompilerExecution|CoreCompile=false/i.test(compile.command)) throw new Error(`${rid}: static compile command bypasses compiler or reuses a manifest`);
    const log = compile.rawLogEvidence;
    if (!log?.coreCompileLine?.includes('Target "CoreCompile:') || !log.cscLine?.includes('Task "Csc"') || log.programSourceLine !== 'Program.cs' || log.supportSourceLine !== 'RunnerSupport.cs') throw new Error(`${rid}: raw Roslyn/source log evidence missing`);
    const assembly = target.assembly;
    const expectedPath = `obj/Release/net8.0-windows10.0.17763.0/${rid}/WindowsMlSpike.dll`;
    if (assembly?.logicalPath !== expectedPath || assembly.targetRidFromPath !== rid || assembly.bytes <= 0 || !/^[0-9a-f]{64}$/.test(assembly.sha256) || assembly.retained !== false) throw new Error(`${rid}: static compile assembly identity missing or forged`);
  }
}

export function compareWindowsMlStaticCompileEvidence(recorded, replayed) {
  for (const rid of ['win-x64', 'win-arm64']) {
    const expected = recorded.targets?.[rid]?.assembly;
    const actual = replayed.targets?.[rid]?.assembly;
    if (!expected || !actual || expected.logicalPath !== actual.logicalPath || expected.bytes !== actual.bytes || expected.sha256 !== actual.sha256 || actual.targetRidFromPath !== rid) throw new Error(`${rid}: clean static compile replay assembly differs from recorded evidence`);
  }
}

function validateWindowsRuntimeClaim(targetId, target, runtimeEvidence) {
  if (!target.runtimeVerified) return;
  if (!target.runtimeExecuted || !target.runtimeIntrospectionComplete || !target.goldenExecuted) throw new Error(`${targetId}: runtime verified without execution/introspection/golden`);
  if (!runtimeEvidence || runtimeEvidence.target !== targetId || runtimeEvidence.state !== 'runtime-verified') throw new Error(`${targetId}: runtime evidence missing or belongs to another architecture`);
  if (runtimeEvidence.failureStage !== 'artifact-publication' || !runtimeEvidence.catalogRegistrationAttempted || !runtimeEvidence.catalogRegistrationCompleted || !runtimeEvidence.sessionCreated || !runtimeEvidence.inferenceExecuted || !runtimeEvidence.runtimeIntrospectionComplete || !runtimeEvidence.outputPublished) throw new Error(`${targetId}: successful runtime evidence lacks completed lifecycle introspection/publication`);
  const expectedArchitecture = targetId === 'win-x64' ? 'X64' : 'Arm64';
  if (runtimeEvidence.host?.os !== 'windows' || runtimeEvidence.host?.processArchitecture !== expectedArchitecture || runtimeEvidence.host?.osArchitecture !== expectedArchitecture) throw new Error(`${targetId}: Linux ORT or wrong-architecture runtime evidence`);
  if (runtimeEvidence.runtime?.dotnetRuntimeVersion !== WINDOWS_DOTNET_RUNTIME_VERSION) throw new Error(`${targetId}: .NET runtime version drift`);
  if (!runtimeEvidence.windowsMlApiCalled || runtimeEvidence.runtime?.sourcePackage?.id !== 'Microsoft.WindowsAppSDK.ML' || runtimeEvidence.runtime.sourcePackage.version !== WINDOWS_PACKAGE_VERSION) throw new Error(`${targetId}: ordinary ORT claimed as Windows ML`);
  if (runtimeEvidence.runtime?.sdk?.version !== '8.0.423' || !runtimeEvidence.runtime.sdk.source?.includes('dotnet --version executed at runtime')) throw new Error(`${targetId}: .NET SDK runtime introspection drift`);
  const modules = runtimeEvidence.runtime.loadedModules;
  if (!Array.isArray(modules) || !modules.some((item) => item.name?.toLowerCase() === 'onnxruntime.dll' && /^[0-9a-f]{64}$/.test(item.sha256)) || !modules.some((item) => item.name?.toLowerCase() === 'microsoft.windows.ai.machinelearning.dll' && /^[0-9a-f]{64}$/.test(item.sha256))) throw new Error(`${targetId}: loaded Windows ML modules not introspected`);
  const input = runtimeEvidence.input;
  const output = runtimeEvidence.output;
  if (input?.count !== 1 || input.name !== 'images' || input.dtype !== 'float32' || JSON.stringify(input.shape) !== '[1,3,640,640]' || input.elementCount !== 1228800 || input.finiteCount !== input.elementCount) throw new Error(`${targetId}: runtime input introspection drift`);
  if (output?.count !== 1 || output.name !== 'output0' || output.dtype !== 'float32' || JSON.stringify(output.shape) !== '[1,84,8400]' || output.elementCount !== 705600 || output.finiteCount !== output.elementCount) throw new Error(`${targetId}: runtime output introspection drift`);
  if (!runtimeEvidence.model?.noConversion || runtimeEvidence.model.sha256 !== WINDOWS_MODEL_SHA256) throw new Error(`${targetId}: runtime model identity drift`);
  const execution = runtimeEvidence.execution;
  if (!Array.isArray(execution?.availableDevices) || execution.availableDevices.length === 0 || !execution.selectedDevice?.epName || !Array.isArray(execution.sessionInputDevices) || execution.sessionInputDevices.length !== 1 || !execution.sessionInputDevices[0]?.epName || !Array.isArray(execution.profileProviders) || execution.profileProviders.length === 0) throw new Error(`${targetId}: provider/device/profile introspection missing`);
  if (execution.claimedProvider && !execution.profileProviders.includes(execution.claimedProvider)) throw new Error(`${targetId}: claimed provider differs from ORT profile`);
  if (execution.profileProviders.every((provider) => provider === 'CPUExecutionProvider') && runtimeEvidence.runtime.sourcePackage.id !== 'Microsoft.WindowsAppSDK.ML') throw new Error(`${targetId}: ordinary CPU ORT claimed as Windows ML provider`);
}

export async function validateWindowsMlEvidence(root, conversion, manifest, report) {
  if (manifest.schemaVersion !== 1 || report.schemaVersion !== 1) throw new Error('Windows ML schema version drift');
  if (manifest.state !== report.state || manifest.supported !== report.supported || manifest.supported !== conversion.supported || manifest.task14Complete !== report.task14Complete || manifest.task14Complete !== conversion.task14Complete) throw new Error('Windows ML state/support/task evidence disagreement');
  if (manifest.supported ? manifest.state !== 'supported' || !manifest.task14Complete : manifest.state !== 'blocked' || manifest.task14Complete) throw new Error('Windows ML blocked/support/task state overclaim');
  if (!manifest.canonicalOnnx.noConversion || !report.noConversion || !report.canonicalOnnx || manifest.canonicalOnnx.convertedArtifacts.length !== 0 || manifest.canonicalOnnx.copyOrMutationAllowed || report.canonicalOnnx.copiesCreated !== 0 || report.canonicalOnnx.mutated) throw new Error('Windows ML must use canonical ONNX without conversion/copy/mutation');
  const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
  for (const identity of [manifest.canonicalOnnx, report.canonicalOnnx, conversion.artifact]) {
    if (identity.path !== 'models/yolov8n.onnx' || identity.bytes !== modelBytes.length || identity.sha256 !== WINDOWS_MODEL_SHA256 || identity.sha256 !== sha256(modelBytes)) throw new Error('Windows ML canonical ONNX identity drift');
  }

  const targetIds = ['win-x64', 'win-arm64'];
  if (JSON.stringify(Object.keys(manifest.targets)) !== JSON.stringify(targetIds) || JSON.stringify(Object.keys(report.targets)) !== JSON.stringify(targetIds) || JSON.stringify(Object.keys(conversion.targets)) !== JSON.stringify(targetIds)) throw new Error('Windows ML x64/ARM64 targets must be separate and complete');
  for (const targetId of targetIds) {
    const target = report.targets[targetId];
    const manifestTarget = manifest.targets[targetId];
    const conversionTarget = conversion.targets[targetId];
    if (!target.runnerPrepared || !manifestTarget.runner || !conversionTarget.runnerPrepared || !target.restoreExecuted || !target.staticCompileExecuted || !target.staticCompileVerified) throw new Error(`${targetId}: runner/restore/static compile preparation missing`);
    for (const field of ['runtimeExecuted', 'runtimeVerified', 'goldenExecuted', 'supported']) {
      if (target[field] !== manifestTarget[field] || target[field] !== conversionTarget[field]) throw new Error(`${targetId}: ${field} disagrees across report/manifest/conversion evidence`);
    }
    if (target.buildVerified !== manifestTarget.build.buildVerified || target.buildVerified !== conversionTarget.buildVerified) throw new Error(`${targetId}: build verification disagrees across report/manifest/conversion evidence`);
    if (target.supported && (!target.runtimeVerified || !target.goldenExecuted)) throw new Error(`${targetId}: supported without runtime/golden verification`);
    validateWindowsRuntimeClaim(targetId, target, report.runtimeEvidence?.[targetId]);
  }
  if (manifest.supported || report.supported || conversion.supported) {
    for (const targetId of targetIds) {
      const target = report.targets[targetId];
      if (!target.runtimeVerified || !target.goldenExecuted || !target.supported) throw new Error('Windows ML supported requires real x64 and ARM64 runner evidence');
    }
  }

  const input = manifest.ioContract.inputs[0];
  const output = manifest.ioContract.outputs[0];
  if (manifest.ioContract.inputs.length !== 1 || input.name !== 'images' || input.dtype !== 'float32' || input.layout !== 'NCHW' || JSON.stringify(input.shape) !== '[1,3,640,640]' || input.elementCount !== 1228800 || input.quantization !== null) throw new Error('Windows ML manifest input contract drift');
  if (manifest.ioContract.outputs.length !== 1 || output.name !== 'output0' || output.dtype !== 'float32' || output.layout !== 'N_ATTRIBUTES_ANCHORS' || JSON.stringify(output.shape) !== '[1,84,8400]' || output.elementCount !== 705600 || output.quantization !== null) throw new Error('Windows ML manifest output contract drift');
  if (manifest.ioContract.nms.fused || manifest.ioContract.nms.owner !== 'operator' || manifest.ioContract.nms.windowsSpecificImplementationAllowed || !manifest.ioContract.postprocessing.includes('src/postprocess.rs')) throw new Error('Windows ML NMS/postprocess ownership drift');
  const officialApi = manifest.runner.officialApi;
  if (manifest.runner.ordinaryOrtPackageReferenced || !manifest.runner.windowsMlApiRequired || officialApi.catalogNamespace !== 'Microsoft.Windows.AI.MachineLearning' || officialApi.catalogType !== 'ExecutionProviderCatalog' || officialApi.catalogMethod !== 'RegisterCertifiedAsync' || officialApi.sessionNamespace !== 'Microsoft.ML.OnnxRuntime' || officialApi.sessionType !== 'InferenceSession') throw new Error('Windows ML official catalog/session API drift');
  for (const api of ['OrtEnv.GetEpDevices', 'SessionOptions.AppendExecutionProvider', 'InferenceSession.GetEpDeviceForInputs', 'InferenceSession.EndProfiling']) if (!officialApi.deviceApis.includes(api)) throw new Error(`Windows ML provider introspection API drift: ${api}`);

  const project = await readFile(resolve(root, manifest.runner.project), 'utf8');
  if (!project.includes('<TargetFramework>net8.0-windows10.0.17763.0</TargetFramework>') || !project.includes('<RuntimeIdentifiers>win-x64;win-arm64</RuntimeIdentifiers>') || !project.includes('<WindowsAppSDKSelfContained>true</WindowsAppSDKSelfContained>') || !project.includes('<UseAppHost>false</UseAppHost>')) throw new Error('Windows ML project target/deployment drift');
  if (!project.includes('<PackageReference Include="Microsoft.WindowsAppSDK.ML" Version="[2.1.74]" />') || !project.includes('<PackageReference Include="Microsoft.Windows.AI.MachineLearning" Version="[2.1.74]" />') || /<PackageReference[^>]+Version="[^"]*[*+]/.test(project) || /PackageReference Include="Microsoft\.ML\.OnnxRuntime"/.test(project)) throw new Error('Windows ML project package pin/source drift');
  const globalJson = JSON.parse(await readFile(resolve(root, 'evidence/tooling/windows-ml-runner/global.json'), 'utf8'));
  if (globalJson.sdk.version !== '8.0.423' || globalJson.sdk.rollForward !== 'disable' || globalJson.sdk.allowPrerelease) throw new Error('Windows ML exact SDK pin missing');
  const lockPath = resolve(root, manifest.dependencies.lockFile.path);
  const lockBytes = await readFile(lockPath);
  if (sha256(lockBytes) !== WINDOWS_LOCK_SHA256 || manifest.dependencies.lockFile.sha256 !== WINDOWS_LOCK_SHA256 || report.dependencies.lockFile.sha256 !== WINDOWS_LOCK_SHA256) throw new Error('Windows ML lock file SHA drift');
  const lock = JSON.parse(lockBytes);
  for (const framework of ['net8.0-windows10.0.17763', 'net8.0-windows10.0.17763/win-x64', 'net8.0-windows10.0.17763/win-arm64']) if (!lock.dependencies[framework]) throw new Error(`Windows ML lock missing ${framework}`);
  const locked = lock.dependencies['net8.0-windows10.0.17763'];
  for (const [id, version] of [['Microsoft.WindowsAppSDK.ML', WINDOWS_PACKAGE_VERSION], ['Microsoft.Windows.AI.MachineLearning', WINDOWS_PACKAGE_VERSION]]) {
    if (locked[id]?.type !== 'Direct' || locked[id].requested !== `[${version}, ${version}]` || locked[id].resolved !== version || !/^[A-Za-z0-9+/]{86}==$/.test(locked[id].contentHash)) throw new Error(`Windows ML lock package drift: ${id}`);
  }
  if (manifest.dependencies.windowsMlPackage.version !== WINDOWS_PACKAGE_VERSION || manifest.dependencies.windowsMlRuntimePackage.version !== WINDOWS_PACKAGE_VERSION || manifest.dependencies.dotnetSdk !== '8.0.423' || manifest.dependencies.dotnetRuntime !== WINDOWS_DOTNET_RUNTIME_VERSION || report.dependencies.dotnetRuntime !== WINDOWS_DOTNET_RUNTIME_VERSION || /[*+]/.test(JSON.stringify(manifest.dependencies))) throw new Error('Windows ML manifest contains missing/floating package, SDK, or runtime version');
  const staticCompileReport = JSON.parse(await readFile(resolve(root, report.staticCompileReport), 'utf8'));
  await validateWindowsMlStaticCompileEvidence(root, staticCompileReport);
  for (const targetId of targetIds) {
    const staticEvidence = manifest.targets[targetId].staticCompileEvidence;
    if (staticEvidence?.report !== report.staticCompileReport || !staticEvidence.replayCommand?.includes('replay_windows_ml_static_compile.mjs') || !staticEvidence.cleanWorkspaceRequired || staticEvidence.compileTarget !== 'Compile' || staticEvidence.assemblyRetained) throw new Error(`${targetId}: static compile manifest contract drift`);
  }

  const requiredReferenceUrls = [
    'https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview',
    'https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/api-reference',
    'https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app',
    'https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers',
  ];
  if (!Array.isArray(report.officialReferences) || requiredReferenceUrls.some((url) => !report.officialReferences.some((item) => item.url === url && item.pageTitle && item.accessedAt.startsWith('2026-08-11T') && item.selectionFact))) throw new Error('Windows ML official reference/title/access time/selection basis missing');
  if (!report.selectionBasis || !report.previousDescriptionDifference?.old || !report.previousDescriptionDifference?.current) throw new Error('Windows ML selection basis or old/new API difference missing');
  if (report.currentHost.os !== 'Linux' || report.currentHost.arch !== 'x86_64' || report.currentHost.windowsRuntimeAvailable) throw new Error('Windows ML preparation host state drift');
  for (const target of Object.values(manifest.targets)) if (!target.runCommand.startsWith(`dotnet --fx-version ${WINDOWS_DOTNET_RUNTIME_VERSION} `)) throw new Error('Windows ML exact .NET runtime run command missing');
  if (report.artifactHandling.secondOnnxGenerated || report.artifactHandling.canonicalOnnxCopied || report.artifactHandling.canonicalOnnxModified || report.artifactHandling.productOrReleaseDirectoryTouched || report.artifactHandling.rimecutTouched) throw new Error('Windows ML artifact publication boundary violated');
  if (manifest.publicationExclusions.some((item) => /(^|\/)(release|dist)(\/|$)/i.test(item) && !item.includes('RimeCut'))) throw new Error('Windows ML artifact unexpectedly points into a publication directory');

  const runnerSource = await readFile(resolve(root, 'evidence/tooling/windows-ml-runner/Program.cs'), 'utf8');
  for (const symbol of ['ExecutionProviderCatalog.GetDefault', 'RegisterCertifiedAsync', 'OrtEnv.Instance', 'GetEpDevices', 'AppendExecutionProvider', 'new InferenceSession', 'InputMetadata', 'OutputMetadata', 'GetEpDeviceForInputs', 'EndProfiling', 'GetVersionString', 'Process.GetCurrentProcess().Modules', 'sourcePackage = new', 'os = OperatingSystem.IsWindows()', 'new ProcessStartInfo("dotnet", "--version")']) if (!runnerSource.includes(symbol)) throw new Error(`Windows ML runner introspection API missing: ${symbol}`);
  if (!runnerSource.includes('OperatingSystem.IsWindows()') || runnerSource.includes('AppendExecutionProvider_CPU') || runnerSource.includes('Microsoft.ML.OnnxRuntime.Gpu')) throw new Error('Windows ML runner permits ordinary/non-Windows ORT path');

  async function walk(directory, prefix = '') {
    const found = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) found.push(...await walk(resolve(directory, entry.name), relative)); else found.push(relative);
    }
    return found;
  }
  const runnerFiles = await walk(resolve(root, 'evidence/tooling/windows-ml-runner'));
  if (runnerFiles.some((path) => /(^|\/)(bin|obj|publish)(\/|$)|\.(dll|exe|nupkg|onnx)$/i.test(path))) throw new Error('Windows ML runner contains build/package/model binary output');
  const trackedOnnx = spawnSync('git', ['ls-files', '*.onnx'], { cwd: root, encoding: 'utf8' }).stdout.trim().split('\n').filter(Boolean);
  if (JSON.stringify(trackedOnnx) !== '["models/yolov8n.onnx"]') throw new Error('Windows ML spike added a second tracked ONNX');
}

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
