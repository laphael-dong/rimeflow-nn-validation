import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { validateTask14Aggregate } from './aggregate_evidence_validation.mjs';
import { validateOpenSpecTransition, validatePublicationReceipt } from './task14_publication_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const baseline = {
  aggregate: await readJson('evidence/conversions/task1-4-aggregate.json'),
  conversion: await readJson('evidence/conversions/conversion-spikes.json'),
  replay: await readJson('evidence/replay/task1-replay.json'),
  receipt: await readJson('evidence/reports/task1-4-publication-report.json'),
  manifest: await readJson('evidence/golden/manifest.json'),
};
const temp = await mkdtemp(join(tmpdir(), 'rimeflow-task14-publication-negative-'));
const cases = [];

async function expectFailure(name, mutate, validator = async (objects) => validateTask14Aggregate(root, objects.aggregate, objects.conversion, objects.replay, objects.receipt, objects.manifest)) {
  const objects = structuredClone(baseline);
  mutate(objects);
  const caseDir = join(temp, `${String(cases.length + 1).padStart(2, '0')}.json`);
  await writeFile(caseDir, `${JSON.stringify(objects)}\n`);
  const reread = JSON.parse(await readFile(caseDir, 'utf8'));
  let failed = false;
  try { await validator(reread); } catch { failed = true; }
  if (!failed) throw new Error(`negative case accepted: ${name}`);
  cases.push(name);
}

try {
  await expectFailure('1 missing required spike', (o) => { o.aggregate.spikes.pop(); });
  await expectFailure('2 duplicate provider', (o) => { o.aggregate.spikes[1] = structuredClone(o.aggregate.spikes[0]); });
  await expectFailure('3 technical closure false with task complete', (o) => { for (const x of [o.aggregate, o.conversion, o.replay]) x.closure.technicalSpikeClosure = false; });
  await expectFailure('4 aggregate remote false with publication true', (o) => { for (const x of [o.aggregate, o.conversion, o.replay]) x.closure.externalRemoteAggregateVerified = false; });
  await expectFailure('5 publication false with task complete', (o) => { for (const x of [o.aggregate, o.conversion, o.replay]) x.closure.publicationVerified = false; });
  await expectFailure('6 task complete without OpenSpec check', (o) => { for (const x of [o.aggregate, o.conversion, o.replay]) x.closure.openspecTask1_4Checked = false; });
  await expectFailure('7 OpenSpec checked without receipt', (o) => { o.aggregate.publication.receipt = 'evidence/reports/missing.json'; o.manifest.artifacts = o.manifest.artifacts.filter((x) => x.path !== 'evidence/reports/task1-4-publication-report.json'); });
  await expectFailure('8 repository drift to upstream', (o) => { o.receipt.github.repository = 'caozisheng/rimeflow-yolov8n'; });
  await expectFailure('9 remote ref drift to main', (o) => { o.receipt.github.remoteRef = 'refs/heads/main'; });
  await expectFailure('10 aggregate SHA drift', (o) => { o.receipt.aggregate.commit = '0'.repeat(40); });
  await expectFailure('11 GitHub object is not commit', (o) => { o.receipt.remoteVerification.githubApi.objectType = 'tag'; });
  await expectFailure('12 forged ls-remote tuple', (o) => { o.receipt.remoteVerification.lsRemote.sha = '1'.repeat(40); });
  await expectFailure('13 model identity drift', (o) => { o.receipt.canonicalModel.sha256 = '2'.repeat(64); });
  await expectFailure('14 main and aggregate model differ', (o) => { o.receipt.canonicalModel.validationMainGitBlob = '3'.repeat(40); });
  await expectFailure('15 aggregate changed models', (o) => { o.receipt.canonicalModel.aggregateCommitModifiedModels = true; });
  await expectFailure('16 base fork recorded as publication target', (o) => { o.receipt.boundaries.baseFork.pushTarget = true; });
  await expectFailure('17 upstream recorded as push target', (o) => { o.receipt.boundaries.upstream.pushTarget = true; });
  await expectFailure('18 tag PR release overclaim', (o) => { o.receipt.boundaries.githubSideEffects.tagCreated = true; });
  await expectFailure('19 all platforms supported', (o) => { for (const x of [o.aggregate, o.conversion, o.replay]) x.closure.allPlatformsSupported = true; });
  await expectFailure('20 blocked provider promoted', (o) => { o.aggregate.spikes.find((x) => x.platform === 'windows').state = 'host-inference-verified'; });
  await expectFailure('21 OpenVINO execution downgraded', (o) => { o.aggregate.spikes.find((x) => x.platform === 'linux-x86_64-openvino').runtimeExecuted = false; });
  await expectFailure('22 shared closure drift', (o) => { o.conversion.closure.task14Complete = false; });
  await expectFailure('23 replay publication missing', (o) => { o.replay.steps = o.replay.steps.filter((x) => x.id !== 'publication-and-platform-closure'); });
  await expectFailure('24 receipt absent from artifact manifest', (o) => { o.manifest.artifacts = o.manifest.artifacts.filter((x) => x.path !== 'evidence/reports/task1-4-publication-report.json'); });
  await expectFailure('27 closure commit self-reference', (o) => { o.receipt.closureCommitSha = '4'.repeat(40); }, async (o) => validatePublicationReceipt(o.receipt, o.aggregate));

  const generated = ['evidence/conversions/task1-4-aggregate.json', 'evidence/conversions/conversion-spikes.json', 'evidence/golden/manifest.json', 'evidence/replay/task1-replay.json'];
  const digestGenerated = async () => Object.fromEntries(await Promise.all(generated.map(async (path) => [path, sha256(await readFile(resolve(root, path)))])));
  execFileSync('node', ['evidence/scripts/generate_task14_aggregate.mjs'], { cwd: root, stdio: 'pipe' });
  const first = await digestGenerated();
  execFileSync('node', ['evidence/scripts/generate_task14_aggregate.mjs'], { cwd: root, stdio: 'pipe' });
  const second = await digestGenerated();
  if (JSON.stringify(first) !== JSON.stringify(second)) throw new Error('25 generator consecutive runs drifted');

  const tracked = execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard'], { cwd: root, encoding: 'utf8' }).trim().split('\n').filter(Boolean);
  const digestTracked = async () => Object.fromEntries(await Promise.all(tracked.map(async (path) => [path, sha256(await readFile(resolve(root, path)))])));
  const beforeVerify = await digestTracked();
  const verify = spawnSync('node', ['evidence/scripts/verify_task14_publication.mjs'], { cwd: root, encoding: 'utf8' });
  if (verify.status !== 0) throw new Error(`26 ordinary verifier failed: ${verify.stderr}`);
  const afterVerify = await digestTracked();
  if (JSON.stringify(beforeVerify) !== JSON.stringify(afterVerify)) throw new Error('26 ordinary verifier modified tracked evidence');

  const tasksBefore = await readFile('/home/raffael/algo/openspec/changes/rimeflow-backend-contract/tasks.md', 'utf8');
  const tasksAfter = tasksBefore.replace('- [ ] 1.4 ', '- [x] 1.4 ');
  validateOpenSpecTransition(tasksBefore, tasksAfter);
  await expectFailure('28 OpenSpec task outside 1.4 changed', () => {}, async () => validateOpenSpecTransition(tasksBefore, tasksAfter.replace('- [ ] 1.6 ', '- [x] 1.6 ')));

  console.log(JSON.stringify({ ok: true, negativeCases: cases.length, generatorDeterministicRuns: 2, generatedArtifacts: generated.length, verificationTrackedFilesUnchanged: tracked.length, openspecTransition: '1.4-only' }));
} finally {
  await rm(temp, { recursive: true, force: true });
}
