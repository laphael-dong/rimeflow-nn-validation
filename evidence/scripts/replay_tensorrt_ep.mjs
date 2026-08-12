import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const workspace = resolve(root, process.argv[2] ?? '.evidence/tensorrt/replay');
const allowedRoot = resolve(root, '.evidence/tensorrt');
const requiredWorkspace = resolve(allowedRoot, 'replay');
if (workspace !== requiredWorkspace) throw new Error('TensorRT replay has one fixed workspace: .evidence/tensorrt/replay');
if ((await lstat(allowedRoot).catch(() => null))?.isSymbolicLink()) throw new Error('TensorRT evidence root must not be a symlink');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const command = (executable, args) => spawnSync(executable, args, { cwd: root, encoding: 'utf8' });
const protectedPaths = [
  'evidence/tensorrt/contract.json',
  'evidence/reports/tensorrt-ep-report.json',
  'evidence/replay/task1-replay.json',
  'evidence/golden/manifest.json',
  'evidence/golden/web-reference.json',
  'evidence/fixtures/manifest.json',
  'evidence/scripts/run_tensorrt_ep.py',
  'evidence/scripts/run_tensorrt_runner.sh',
  'evidence/scripts/replay_tensorrt_ep.mjs',
  'evidence/scripts/validate_tensorrt_ep.mjs',
  'evidence/scripts/test_tensorrt_ep_negative.mjs',
  'evidence/scripts/tensorrt_record_publish.py',
  'evidence/scripts/test_tensorrt_record_publish.py',
  'evidence/tooling/tensorrt/Dockerfile',
  'evidence/tooling/tensorrt/Dockerfile.dockerignore',
  'evidence/tooling/tensorrt-postprocess/Cargo.toml',
  'evidence/tooling/tensorrt-postprocess/Cargo.lock',
  'evidence/tooling/tensorrt-postprocess/main.rs',
  'src/postprocess.rs',
];

async function snapshot(path) {
  const absolute = resolve(root, path);
  const canonical = await realpath(absolute);
  if (canonical !== absolute) throw new Error(`tracked evidence symlink is forbidden: ${path}`);
  const item = await stat(absolute, { bigint: true });
  const bytes = await readFile(absolute);
  return { path, canonicalPath: canonical, exists: true, bytes: bytes.length, sha256: sha256(bytes), inode: String(item.ino), ctimeNs: String(item.ctimeNs), mtimeNs: String(item.mtimeNs) };
}

async function snapshots() {
  return Object.fromEntries(await Promise.all(protectedPaths.map(async (path) => [path, await snapshot(path)])));
}

const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const inside = (path, parent) => path === parent || path.startsWith(`${parent}${sep}`);
const semantic = (report) => ({
  schemaVersion: report.schemaVersion,
  status: report.status,
  supported: report.supported,
  runtimeExecuted: report.runtimeExecuted,
  hostInferenceVerified: report.hostInferenceVerified,
  goldenExecuted: report.goldenExecuted,
  task14Complete: report.task14Complete,
  openspecTask1_4Checked: report.openspecTask1_4Checked,
  failureStage: report.failureStage,
  failureReason: report.failureReason,
  sourceCommit: report.sourceCommit,
  model: report.model,
  runner: { realNvidiaRunner: report.runner.realNvidiaRunner, arch: report.runner.arch, gpu: report.runner.gpu, driver: report.runner.driver },
  versions: report.versions,
  providers: report.providers,
  engineBuild: { attempted: report.engineBuild.attempted, succeeded: report.engineBuild.succeeded, stage: report.engineBuild.stage, errors: report.engineBuild.errors },
  fixtures: report.fixtures,
  commandResults: report.commands.map((item) => ({ command: item.command, exitCode: item.exitCode, stdout: item.stdout, stderr: item.stderr, stage: item.stage })),
});

await mkdir(workspace, { recursive: true });
if (!inside(await realpath(workspace), await realpath(allowedRoot))) throw new Error('TensorRT replay workspace symlink escape');
const rounds = [];
for (let round = 1; round <= 2; round += 1) {
  const roundWorkspace = resolve(workspace, `round-${round}`);
  await mkdir(roundWorkspace, { recursive: true });
  const reportPath = resolve(roundWorkspace, 'tensorrt-ep-report.json');
  const before = await snapshots();
  const startedAt = new Date().toISOString();
  const run = command('python3', ['evidence/scripts/run_tensorrt_ep.py', '--workspace', roundWorkspace]);
  if (run.status !== 2) throw new Error(`TensorRT replay runner round ${round} exit ${run.status}: ${run.stdout}\n${run.stderr}`);
  const after = await snapshots();
  if (!same(before, after)) throw new Error(`ordinary TensorRT replay changed tracked evidence in round ${round}`);
  const report = JSON.parse(await readFile(reportPath, 'utf8'));
  report.protectedTrackedEvidence = { before, after, unchanged: true };
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  const validate = command('node', ['evidence/scripts/validate_tensorrt_ep.mjs', reportPath, roundWorkspace]);
  const guards = command('node', ['evidence/scripts/test_tensorrt_ep_negative.mjs']);
  if (validate.status !== 0 || guards.status !== 0 || report.status !== 'blocked') throw new Error(`TensorRT replay round ${round} validation failed: ${validate.stderr}\n${guards.stderr}`);
  const value = semantic(report);
  rounds.push({ round, startedAt, endedAt: new Date().toISOString(), runnerExitCode: run.status, validatorExitCode: validate.status, guardExitCode: guards.status, semanticSha256: sha256(Buffer.from(JSON.stringify(value))), semantic: value, trackedEvidence: { before, after, unchanged: true }, stdout: `${run.stdout}${validate.stdout}${guards.stdout}`, stderr: `${run.stderr}${validate.stderr}${guards.stderr}` });
}
const result = { schemaVersion: 2, taskId: 'T14-LNX-TRT-01', rounds, trackedEvidenceUnchanged: rounds.every((item) => item.trackedEvidence.unchanged), deterministicSemanticEvidence: rounds[0].semanticSha256 === rounds[1].semanticSha256 };
if (!result.deterministicSemanticEvidence || !result.trackedEvidenceUnchanged) throw new Error('TensorRT blocked replay determinism/tracked evidence protection failed');
await writeFile(resolve(workspace, 'tensorrt-replay.json'), `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ rounds: 2, deterministicSemanticEvidence: true, trackedEvidenceUnchanged: true, protectedFiles: protectedPaths.length }));
