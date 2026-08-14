import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { validateTask14Aggregate } from './aggregate_evidence_validation.mjs';
import {
  CRITICAL_EVIDENCE_PATHS,
  PUBLICATION,
  validateLivePublicationFacts,
  validateOpenSpecTransition,
  validatePublicationReceipt,
} from './task14_publication_validation.mjs';

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
const syntheticHead = 'f'.repeat(40);
const syntheticTree = 'e'.repeat(40);
const syntheticCommitDigest = 'd'.repeat(64);
const syntheticBlobs = Object.fromEntries(CRITICAL_EVIDENCE_PATHS.map((path, index) => [path, `${index + 1}`.repeat(40)]));

const liveBaseline = {
  localHead: syntheticHead,
  remoteSha: syntheticHead,
  remoteRef: PUBLICATION.remoteRef,
  api: { ref: PUBLICATION.remoteRef, object: { type: 'commit', sha: syntheticHead } },
  actor: 'laphael-dong',
  canPush: true,
  localIdentity: { commit: syntheticHead, parent: PUBLICATION.previousClosureCommit, subject: PUBLICATION.fixSubject },
  previousClosureIdentity: { commit: PUBLICATION.previousClosureCommit, parent: PUBLICATION.aggregateCommit, subject: PUBLICATION.previousClosureSubject },
  aggregateIdentity: { commit: PUBLICATION.aggregateCommit, parent: PUBLICATION.aggregateParent, subject: PUBLICATION.aggregateSubject },
  freshFetchExecuted: true,
  fetchHead: syntheticHead,
  readbackIdentity: { commit: syntheticHead, parent: PUBLICATION.previousClosureCommit, subject: PUBLICATION.fixSubject },
  localCommitDigest: syntheticCommitDigest,
  readbackCommitDigest: syntheticCommitDigest,
  localTree: syntheticTree,
  readbackTree: syntheticTree,
  localRemoteDiff: '',
  localFirstParentChain: [[syntheticHead, PUBLICATION.previousClosureCommit], [PUBLICATION.previousClosureCommit, PUBLICATION.aggregateCommit]],
  readbackFirstParentChain: [[syntheticHead, PUBLICATION.previousClosureCommit], [PUBLICATION.previousClosureCommit, PUBLICATION.aggregateCommit]],
  previousClosureCriticalBlobs: structuredClone(syntheticBlobs),
  localCriticalBlobs: structuredClone(syntheticBlobs),
  readbackCriticalBlobs: structuredClone(syntheticBlobs),
  localModel: { path: PUBLICATION.modelPath, mode: '100644', bytes: PUBLICATION.modelBytes, blob: PUBLICATION.modelBlob },
  readbackModel: { path: PUBLICATION.modelPath, mode: '100644', bytes: PUBLICATION.modelBytes, blob: PUBLICATION.modelBlob },
  mainBlob: PUBLICATION.modelBlob,
  modelSha256: PUBLICATION.modelSha256,
  modelDiff: '',
  upstreamRef: '',
  baseRef: '',
  refsAtExpectedSha: [PUBLICATION.remoteRef],
  pullRequestCount: 0,
  releaseCount: 0,
};

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

async function expectLiveFailure(name, mutate) {
  const facts = structuredClone(liveBaseline);
  mutate(facts);
  let failed = false;
  try { validateLivePublicationFacts(facts); } catch { failed = true; }
  if (!failed) throw new Error(`live publication negative case accepted: ${name}`);
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

  validateLivePublicationFacts(structuredClone(liveBaseline));
  await expectLiveFailure('29 remote ref rolled back to aggregate commit', (f) => { f.remoteSha = PUBLICATION.aggregateCommit; f.api.object.sha = PUBLICATION.aggregateCommit; f.fetchHead = PUBLICATION.aggregateCommit; f.readbackIdentity.commit = PUBLICATION.aggregateCommit; });
  await expectLiveFailure('30 remote ref remains at previous closure', (f) => { f.remoteSha = PUBLICATION.previousClosureCommit; f.api.object.sha = PUBLICATION.previousClosureCommit; f.fetchHead = PUBLICATION.previousClosureCommit; f.readbackIdentity.commit = PUBLICATION.previousClosureCommit; });
  await expectLiveFailure('31 arbitrary forged 40-character closure SHA', (f) => { const forged = 'a'.repeat(40); f.remoteSha = forged; f.api.object.sha = forged; f.fetchHead = forged; f.readbackIdentity.commit = forged; });
  await expectLiveFailure('32 sibling commit with same parent and subject but different tree', (f) => { const sibling = 'b'.repeat(40); f.remoteSha = sibling; f.api.object.sha = sibling; f.fetchHead = sibling; f.readbackIdentity.commit = sibling; f.readbackTree = 'c'.repeat(40); });
  await expectLiveFailure('33 fresh FETCH_HEAD differs from remote SHA', (f) => { f.fetchHead = 'a'.repeat(40); });
  await expectLiveFailure('34 local HEAD differs from remote SHA', (f) => { f.remoteSha = 'a'.repeat(40); f.api.object.sha = 'a'.repeat(40); f.fetchHead = 'a'.repeat(40); f.readbackIdentity.commit = 'a'.repeat(40); });
  await expectLiveFailure('35 commit object content differs', (f) => { f.readbackCommitDigest = 'a'.repeat(64); });
  await expectLiveFailure('36 tree OID differs', (f) => { f.readbackTree = 'a'.repeat(40); });
  await expectLiveFailure('37 receipt blob drift', (f) => { f.readbackCriticalBlobs['evidence/reports/task1-4-publication-report.json'] = 'a'.repeat(40); });
  await expectLiveFailure('38 aggregate blob drift', (f) => { f.readbackCriticalBlobs['evidence/conversions/task1-4-aggregate.json'] = 'a'.repeat(40); });
  await expectLiveFailure('39 conversion summary blob drift', (f) => { f.readbackCriticalBlobs['evidence/conversions/conversion-spikes.json'] = 'a'.repeat(40); });
  await expectLiveFailure('40 replay ledger blob drift', (f) => { f.readbackCriticalBlobs['evidence/replay/task1-replay.json'] = 'a'.repeat(40); });
  await expectLiveFailure('41 artifact manifest blob drift', (f) => { f.readbackCriticalBlobs['evidence/golden/manifest.json'] = 'a'.repeat(40); });
  await expectLiveFailure('42 canonical model blob drift', (f) => { f.readbackModel.blob = 'a'.repeat(40); });
  await expectLiveFailure('43 canonical model bytes drift', (f) => { f.readbackModel.bytes += 1; });
  await expectLiveFailure('44 canonical model SHA-256 drift', (f) => { f.modelSha256 = 'a'.repeat(64); });
  await expectLiveFailure('45 fresh fetch not executed', (f) => { f.freshFetchExecuted = false; });
  await expectLiveFailure('46 merge introduced on first-parent path', (f) => { f.localFirstParentChain[0].push('a'.repeat(40)); });

  const generated = ['evidence/conversions/task1-4-aggregate.json', 'evidence/conversions/conversion-spikes.json', 'evidence/golden/manifest.json', 'evidence/replay/task1-replay.json'];
  const digestGenerated = async () => Object.fromEntries(await Promise.all(generated.map(async (path) => [path, sha256(await readFile(resolve(root, path)))])));
  execFileSync('node', ['evidence/scripts/generate_task14_aggregate.mjs'], { cwd: root, stdio: 'pipe' });
  const first = await digestGenerated();
  execFileSync('node', ['evidence/scripts/generate_task14_aggregate.mjs'], { cwd: root, stdio: 'pipe' });
  const second = await digestGenerated();
  if (JSON.stringify(first) !== JSON.stringify(second)) throw new Error('25 generator consecutive runs drifted');

  const tasksPath = '/home/raffael/algo/openspec/changes/rimeflow-backend-contract/tasks.md';
  const tasksAfter = await readFile(tasksPath, 'utf8');
  const currentTasksAfterSha256 = sha256(tasksAfter);
  let openspec;
  if (currentTasksAfterSha256 === PUBLICATION.tasksAfterSha256) {
    const checked = '- [x] 1.4 ';
    const unchecked = '- [ ] 1.4 ';
    if (tasksAfter.split(checked).length !== 2) throw new Error('OpenSpec task 1.4 checked line count drift');
    const tasksBefore = tasksAfter.replace(checked, unchecked);
    if (sha256(tasksBefore) !== PUBLICATION.tasksBeforeSha256) throw new Error('constructed OpenSpec tasks before SHA-256 drift');
    validateOpenSpecTransition(tasksBefore, tasksAfter);
    await expectFailure('28 OpenSpec task outside 1.4 changed', () => {}, async () => validateOpenSpecTransition(tasksBefore, tasksAfter.replace('- [ ] 1.6 ', '- [x] 1.6 ')));
    await expectFailure('47 OpenSpec task 2.1 changed', () => {}, async () => validateOpenSpecTransition(tasksBefore, tasksAfter.replace('- [ ] 2.1 ', '- [x] 2.1 ')));
    openspec = { classification: 'task-1.4-publication-snapshot-matched', currentTasksAfterSha256, expectedTasksAfterSha256: PUBLICATION.tasksAfterSha256, path: tasksPath, transition: '1.4-only-final-state' };
  } else {
    openspec = { classification: 'external-global-baseline-mismatch', currentTasksAfterSha256, expectedTasksAfterSha256: PUBLICATION.tasksAfterSha256, path: tasksPath, transition: 'not-replayed-against-drifted-global-file' };
  }

  console.log(JSON.stringify({ ok: true, negativeCases: cases.length, livePublicationAttacks: 18, generatorDeterministicRuns: 2, generatedArtifacts: generated.length, openspec }));
} finally {
  await rm(temp, { recursive: true, force: true });
}
