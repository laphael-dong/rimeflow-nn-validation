import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { collectLivePublicationFacts, PUBLICATION, REQUIRED_SPIKES, validateLivePublicationFacts, validatePublicationReceipt } from './task14_publication_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const output = resolve(root, 'evidence/reports/task1-4-publication-report.json');
const aggregate = JSON.parse(await readFile(resolve(root, 'evidence/conversions/task1-4-aggregate.json'), 'utf8'));
const facts = collectLivePublicationFacts(root);
validateLivePublicationFacts(facts);
const recordedAtUtc = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
const receipt = {
  schemaVersion: 1,
  openspec: { change: PUBLICATION.change, taskId: PUBLICATION.taskId, tasksBeforeSha256: PUBLICATION.tasksBeforeSha256, tasksAfterSha256: PUBLICATION.tasksAfterSha256, beforeProgress: '4/63', expectedAfterProgress: '5/63', onlyTask14CheckboxMayChange: true },
  recordedAtUtc,
  github: { repository: PUBLICATION.repository, remoteUrl: PUBLICATION.remoteUrl, remoteRef: PUBLICATION.remoteRef, actor: facts.actor, actorPushPermission: facts.canPush },
  aggregate: { commit: PUBLICATION.aggregateCommit, parent: PUBLICATION.aggregateParent, subject: PUBLICATION.aggregateSubject, singleParent: true, commitObjectVerified: true, modelsChanged: false },
  remoteVerification: { lsRemote: { sha: facts.remoteSha, ref: facts.remoteRef }, githubApi: { ref: facts.api.ref, objectType: facts.api.object.type, objectSha: facts.api.object.sha } },
  readback: { clonePath: PUBLICATION.validationClone, fetchHead: facts.fetchHead, commitObjectVerified: true, ...facts.readbackIdentity },
  canonicalModel: { path: PUBLICATION.modelPath, bytes: facts.modelBytes, gitBlob: facts.modelBlob, sha256: facts.modelSha256, validationMainGitBlob: facts.mainBlob, aggregateRefGitBlob: facts.aggregateBlob, validationMainMatchesAggregate: facts.mainBlob === facts.aggregateBlob, aggregateCommitModifiedModels: facts.modelDiff !== '' },
  requiredSpikes: REQUIRED_SPIKES.map(([platform, provider]) => ({ platform, provider, spikeClosure: true, supported: false })),
  boundaries: {
    upstream: { repository: PUBLICATION.upstreamRepository, remoteUrl: PUBLICATION.upstreamUrl, aggregateRefPresent: facts.upstreamRef !== '', pushTarget: false },
    baseFork: { repository: PUBLICATION.baseRepository, remoteUrl: PUBLICATION.baseUrl, aggregateRefPresent: facts.baseRef !== '', receivedValidationAggregate: false, pushTarget: false },
    githubSideEffects: { tagCreated: facts.validationTags !== '', pullRequestCreated: facts.pullRequestCount !== 0, releaseCreated: facts.releaseCount !== 0 },
  },
  verification: { aggregateRemoteVerified: true, aggregateCommitFetchVerified: true, canonicalModelRemoteIdentityVerified: true, publicationEvidenceRecorded: true, allRequiredSpikesRecorded: true, technicalSpikeClosure: true, allPlatformsSupported: false },
  remainingScope: ['真实目标 runner 的 supported 闭环', 'adapter、性能、fallback、包加载与后续 OpenSpec 任务'],
};
validatePublicationReceipt(receipt, aggregate);
if (!process.argv.includes('--record')) {
  console.log(JSON.stringify({ ok: true, mode: 'verify-only', output: 'evidence/reports/task1-4-publication-report.json', wouldWrite: false }));
  process.exit(0);
}
await mkdir(dirname(output), { recursive: true });
const staging = `${output}.staging-${process.pid}`;
try {
  const bytes = `${JSON.stringify(receipt, null, 2)}\n`;
  await writeFile(staging, bytes, { flag: 'wx' });
  const reread = JSON.parse(await readFile(staging, 'utf8'));
  validatePublicationReceipt(reread, aggregate);
  await rename(staging, output);
} finally {
  await rm(staging, { force: true });
}
console.log(JSON.stringify({ ok: true, mode: 'record', output: 'evidence/reports/task1-4-publication-report.json', aggregateCommit: PUBLICATION.aggregateCommit }));
