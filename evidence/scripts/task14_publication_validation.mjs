import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

export const PUBLICATION = Object.freeze({
  schemaVersion: 1,
  change: 'rimeflow-backend-contract',
  taskId: '1.4',
  repository: 'laphael-dong/rimeflow-nn-validation',
  remoteUrl: 'https://github.com/laphael-dong/rimeflow-nn-validation.git',
  remoteRef: 'refs/heads/feature/rimeflow-backend-contract-task1-aggregate',
  aggregateCommit: '19193a34f2fb2b36465538b02687a07608f7810e',
  aggregateParent: '7eba039ef1c55408216f0f54d543f0fdcbf1693b',
  aggregateSubject: '[ENH] Aggregate task 1.4 spike evidence',
  closureSubject: '[DOC] Close task 1.4 publication evidence',
  validationClone: '/home/raffael/algo/github/rimeflow-nn-validation',
  baseRepository: 'laphael-dong/rimeflow-nn-base',
  baseUrl: 'https://github.com/laphael-dong/rimeflow-nn-base.git',
  upstreamRepository: 'caozisheng/rimeflow-yolov8n',
  upstreamUrl: 'https://github.com/caozisheng/rimeflow-yolov8n.git',
  modelPath: 'models/yolov8n.onnx',
  modelBytes: 12851098,
  modelBlob: '22f19afe710dfa942b3e644c4e5a7ac5c42ac403',
  modelSha256: '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad',
  tasksBeforeSha256: '54db708645bca38150425c23b68d20869710c6269825ce6e4517d022bc7188c2',
  tasksAfterSha256: '08b07922fbf93dfb8bb265440205bb059109f3fb6f46bb3029dbccee2acc5f47',
});

export const REQUIRED_SPIKES = Object.freeze([
  ['apple', 'coreml'],
  ['android', 'litert-v2'],
  ['windows', 'windows-ml'],
  ['harmonyos', 'mindspore-lite'],
  ['linux-x86_64-cpu', 'cpuexecutionprovider'],
  ['linux-x86_64-openvino', 'openvinoexecutionprovider'],
  ['linux-x86_64-cuda', 'cudaexecutionprovider'],
  ['linux-x86_64-tensorrt', 'tensorrtexecutionprovider'],
]);

const fail = (message) => { throw new Error(message); };
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const git = (cwd, args, options = {}) => execFileSync('git', args, { cwd, encoding: 'utf8', ...options }).trim();
const ghJson = (args) => JSON.parse(execFileSync('gh', args, { encoding: 'utf8' }));
const exact = (actual, expected, label) => { if (actual !== expected) fail(`${label} drift`); };
const isUtc = (value) => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(value);
const spikeKey = (item) => `${item.platform}:${item.provider}`;

function assertNoClosureCommitIdentity(value, path = 'receipt') {
  if (!value || typeof value !== 'object') return;
  for (const [key, child] of Object.entries(value)) {
    const next = `${path}.${key}`;
    if (/^(closureCommit|closureCommitSha|finalClosureCommit|finalCommitSha)$/i.test(key)) fail(`tracked receipt must not contain closure commit identity: ${next}`);
    assertNoClosureCommitIdentity(child, next);
  }
}

export function validateOpenSpecTransition(before, after) {
  exact(sha256(before), PUBLICATION.tasksBeforeSha256, 'OpenSpec tasks before SHA-256');
  exact(sha256(after), PUBLICATION.tasksAfterSha256, 'OpenSpec tasks after SHA-256');
  const unchecked = '- [ ] 1.4 ';
  const checked = '- [x] 1.4 ';
  if (before.split(unchecked).length !== 2 || after.split(checked).length !== 2) fail('OpenSpec task 1.4 line count drift');
  if (after.replace(checked, unchecked) !== before) fail('OpenSpec transition changed more than task 1.4 checkbox');
  const beforeChecks = [...before.matchAll(/^- \[([ x])\] /gm)].map((match) => match[1]);
  const afterChecks = [...after.matchAll(/^- \[([ x])\] /gm)].map((match) => match[1]);
  if (beforeChecks.length !== 63 || afterChecks.length !== 63) fail('OpenSpec task count drift');
  const changed = beforeChecks.flatMap((value, index) => value === afterChecks[index] ? [] : [index]);
  if (changed.length !== 1 || changed[0] !== 3 || beforeChecks[3] !== ' ' || afterChecks[3] !== 'x') fail('OpenSpec checkbox drift outside task 1.4');
  return { beforeProgress: '4/63', afterProgress: '5/63', changedTask: '1.4' };
}

export function validatePublicationReceipt(receipt, aggregate = null) {
  if (!receipt || receipt.schemaVersion !== PUBLICATION.schemaVersion) fail('publication receipt schemaVersion');
  assertNoClosureCommitIdentity(receipt);
  exact(receipt.openspec?.change, PUBLICATION.change, 'OpenSpec change');
  exact(receipt.openspec?.taskId, PUBLICATION.taskId, 'OpenSpec task ID');
  exact(receipt.openspec?.tasksBeforeSha256, PUBLICATION.tasksBeforeSha256, 'OpenSpec before SHA');
  exact(receipt.openspec?.tasksAfterSha256, PUBLICATION.tasksAfterSha256, 'OpenSpec after SHA');
  if (receipt.openspec?.beforeProgress !== '4/63' || receipt.openspec?.expectedAfterProgress !== '5/63' || receipt.openspec?.onlyTask14CheckboxMayChange !== true) fail('OpenSpec transaction contract drift');
  if (!isUtc(receipt.recordedAtUtc)) fail('publication receipt UTC timestamp');
  exact(receipt.github?.repository, PUBLICATION.repository, 'publication repository');
  exact(receipt.github?.remoteUrl, PUBLICATION.remoteUrl, 'publication remote URL');
  exact(receipt.github?.remoteRef, PUBLICATION.remoteRef, 'publication remote ref');
  exact(receipt.github?.actor, 'laphael-dong', 'GitHub actor');
  if (receipt.github?.actorPushPermission !== true) fail('GitHub actor push permission missing');
  exact(receipt.aggregate?.commit, PUBLICATION.aggregateCommit, 'aggregate SHA');
  exact(receipt.aggregate?.parent, PUBLICATION.aggregateParent, 'aggregate parent');
  exact(receipt.aggregate?.subject, PUBLICATION.aggregateSubject, 'aggregate subject');
  if (receipt.aggregate?.singleParent !== true || receipt.aggregate?.commitObjectVerified !== true || receipt.aggregate?.modelsChanged !== false) fail('aggregate commit identity/model boundary drift');
  exact(receipt.remoteVerification?.lsRemote?.sha, PUBLICATION.aggregateCommit, 'ls-remote SHA');
  exact(receipt.remoteVerification?.lsRemote?.ref, PUBLICATION.remoteRef, 'ls-remote ref');
  exact(receipt.remoteVerification?.githubApi?.ref, PUBLICATION.remoteRef, 'GitHub API ref');
  exact(receipt.remoteVerification?.githubApi?.objectType, 'commit', 'GitHub API object type');
  exact(receipt.remoteVerification?.githubApi?.objectSha, PUBLICATION.aggregateCommit, 'GitHub API object SHA');
  exact(receipt.readback?.clonePath, PUBLICATION.validationClone, 'readback clone');
  exact(receipt.readback?.fetchHead, PUBLICATION.aggregateCommit, 'readback FETCH_HEAD');
  if (receipt.readback?.commitObjectVerified !== true) fail('readback commit object not verified');
  exact(receipt.readback?.commit, PUBLICATION.aggregateCommit, 'readback commit');
  exact(receipt.readback?.parent, PUBLICATION.aggregateParent, 'readback parent');
  exact(receipt.readback?.subject, PUBLICATION.aggregateSubject, 'readback subject');
  exact(receipt.canonicalModel?.path, PUBLICATION.modelPath, 'model path');
  exact(receipt.canonicalModel?.bytes, PUBLICATION.modelBytes, 'model bytes');
  exact(receipt.canonicalModel?.gitBlob, PUBLICATION.modelBlob, 'model Git blob');
  exact(receipt.canonicalModel?.sha256, PUBLICATION.modelSha256, 'model SHA-256');
  exact(receipt.canonicalModel?.validationMainGitBlob, PUBLICATION.modelBlob, 'validation main model blob');
  exact(receipt.canonicalModel?.aggregateRefGitBlob, PUBLICATION.modelBlob, 'aggregate ref model blob');
  if (receipt.canonicalModel?.validationMainMatchesAggregate !== true || receipt.canonicalModel?.aggregateCommitModifiedModels !== false) fail('remote canonical model identity not established');
  const expectedReceiptSpikes = REQUIRED_SPIKES.map(([platform, provider]) => `${platform}:${provider}`).sort();
  const actualReceiptSpikes = receipt.requiredSpikes?.map(spikeKey).sort() ?? [];
  if (new Set(actualReceiptSpikes).size !== actualReceiptSpikes.length || JSON.stringify(actualReceiptSpikes) !== JSON.stringify(expectedReceiptSpikes) || receipt.requiredSpikes.some((item) => item.spikeClosure !== true || item.supported !== false)) fail('publication receipt required spike set drift');
  exact(receipt.boundaries?.upstream?.repository, PUBLICATION.upstreamRepository, 'upstream repository');
  exact(receipt.boundaries?.upstream?.remoteUrl, PUBLICATION.upstreamUrl, 'upstream URL');
  if (receipt.boundaries?.upstream?.aggregateRefPresent !== false || receipt.boundaries?.upstream?.pushTarget !== false) fail('upstream publication overclaim');
  exact(receipt.boundaries?.baseFork?.repository, PUBLICATION.baseRepository, 'base fork repository');
  exact(receipt.boundaries?.baseFork?.remoteUrl, PUBLICATION.baseUrl, 'base fork URL');
  if (receipt.boundaries?.baseFork?.aggregateRefPresent !== false || receipt.boundaries?.baseFork?.receivedValidationAggregate !== false || receipt.boundaries?.baseFork?.pushTarget !== false) fail('base fork publication overclaim');
  for (const key of ['tagCreated', 'pullRequestCreated', 'releaseCreated']) if (receipt.boundaries?.githubSideEffects?.[key] !== false) fail(`GitHub ${key} overclaim`);
  const verification = receipt.verification;
  for (const key of ['aggregateRemoteVerified', 'aggregateCommitFetchVerified', 'canonicalModelRemoteIdentityVerified', 'publicationEvidenceRecorded', 'allRequiredSpikesRecorded', 'technicalSpikeClosure']) if (verification?.[key] !== true) fail(`publication verification missing: ${key}`);
  if (verification?.allPlatformsSupported !== false) fail('publication receipt must not claim all platforms supported');
  if (aggregate) {
    const expected = expectedReceiptSpikes;
    const actual = aggregate.spikes?.map(spikeKey).sort() ?? [];
    if (new Set(actual).size !== actual.length || JSON.stringify(actual) !== JSON.stringify(expected) || aggregate.spikes.some((item) => item.spikeClosure !== true || item.supported !== false)) fail('publication receipt is not backed by the eight required spike closures');
  }
  return { repository: PUBLICATION.repository, ref: PUBLICATION.remoteRef, aggregateCommit: PUBLICATION.aggregateCommit };
}

export async function readAndValidatePublicationReceipt(root, aggregate = null, receiptPath = 'evidence/reports/task1-4-publication-report.json') {
  const receipt = JSON.parse(await readFile(resolve(root, receiptPath), 'utf8'));
  validatePublicationReceipt(receipt, aggregate);
  return receipt;
}

function parseCommitIdentity(text) {
  const [commit, parent, subject] = text.split('\n');
  return { commit, parent, subject };
}

export function collectLivePublicationFacts(root) {
  const remoteLine = git(root, ['ls-remote', '--heads', PUBLICATION.remoteUrl, PUBLICATION.remoteRef.replace('refs/heads/', '')]);
  const [remoteSha, remoteRef] = remoteLine.split(/\s+/);
  const api = ghJson(['api', `repos/${PUBLICATION.repository}/git/ref/heads/${PUBLICATION.remoteRef.replace('refs/heads/', '')}`]);
  const actor = execFileSync('gh', ['api', 'user', '--jq', '.login'], { encoding: 'utf8' }).trim();
  const repo = ghJson(['api', `repos/${PUBLICATION.repository}`]);
  const localIdentity = parseCommitIdentity(git(root, ['show', '--no-patch', '--format=%H%n%P%n%s', PUBLICATION.aggregateCommit]));
  const fetchHead = git(PUBLICATION.validationClone, ['rev-parse', 'FETCH_HEAD']);
  const readbackIdentity = parseCommitIdentity(git(PUBLICATION.validationClone, ['show', '--no-patch', '--format=%H%n%P%n%s', 'FETCH_HEAD']));
  const modelTree = git(PUBLICATION.validationClone, ['ls-tree', '-l', 'FETCH_HEAD', PUBLICATION.modelPath]).split(/\s+/);
  const mainBlob = git(PUBLICATION.validationClone, ['rev-parse', `main:${PUBLICATION.modelPath}`]);
  const aggregateBlob = git(PUBLICATION.validationClone, ['rev-parse', `FETCH_HEAD:${PUBLICATION.modelPath}`]);
  const modelSha = execFileSync('git', ['show', `FETCH_HEAD:${PUBLICATION.modelPath}`], { cwd: PUBLICATION.validationClone, maxBuffer: 20 * 1024 * 1024 });
  const modelDiff = git(PUBLICATION.validationClone, ['diff', '--name-status', `${PUBLICATION.aggregateParent}..FETCH_HEAD`, '--', 'models']);
  const upstreamRef = git(root, ['ls-remote', '--heads', PUBLICATION.upstreamUrl, PUBLICATION.remoteRef.replace('refs/heads/', '')]);
  const baseRef = git(root, ['ls-remote', '--heads', PUBLICATION.baseUrl, PUBLICATION.remoteRef.replace('refs/heads/', '')]);
  const validationTags = git(root, ['ls-remote', '--tags', PUBLICATION.remoteUrl]);
  const prs = ghJson(['api', `repos/${PUBLICATION.repository}/pulls?state=all&head=laphael-dong:${PUBLICATION.remoteRef.replace('refs/heads/', '')}`]);
  const releases = ghJson(['api', `repos/${PUBLICATION.repository}/releases`]);
  return {
    remoteSha, remoteRef, api, actor, canPush: repo.permissions?.push === true,
    localIdentity, fetchHead, readbackIdentity,
    modelBytes: Number(modelTree[3]), modelBlob: modelTree[2], mainBlob, aggregateBlob, modelSha256: sha256(modelSha), modelDiff,
    upstreamRef, baseRef, validationTags, pullRequestCount: prs.length, releaseCount: releases.length,
  };
}

export function validateLivePublicationFacts(facts) {
  exact(facts.remoteRef, PUBLICATION.remoteRef, 'live remote ref');
  exact(facts.api?.ref, PUBLICATION.remoteRef, 'live GitHub API ref');
  exact(facts.api?.object?.type, 'commit', 'live GitHub API object type');
  exact(facts.api?.object?.sha, facts.remoteSha, 'live GitHub API object SHA');
  exact(facts.actor, 'laphael-dong', 'live GitHub actor');
  if (!facts.canPush) fail('live GitHub push permission missing');
  exact(facts.localIdentity.commit, PUBLICATION.aggregateCommit, 'aggregate commit object');
  exact(facts.localIdentity.parent, PUBLICATION.aggregateParent, 'aggregate parent object');
  exact(facts.localIdentity.subject, PUBLICATION.aggregateSubject, 'aggregate subject object');
  exact(facts.fetchHead, facts.remoteSha, 'live readback FETCH_HEAD');
  exact(facts.readbackIdentity.commit, facts.remoteSha, 'live readback commit');
  if (facts.remoteSha === PUBLICATION.aggregateCommit) {
    exact(facts.readbackIdentity.parent, PUBLICATION.aggregateParent, 'aggregate readback parent');
    exact(facts.readbackIdentity.subject, PUBLICATION.aggregateSubject, 'aggregate readback subject');
  } else {
    if (!/^[0-9a-f]{40}$/.test(facts.remoteSha)) fail('live closure SHA shape');
    exact(facts.readbackIdentity.parent, PUBLICATION.aggregateCommit, 'closure readback parent');
    exact(facts.readbackIdentity.subject, PUBLICATION.closureSubject, 'closure readback subject');
  }
  exact(facts.modelBytes, PUBLICATION.modelBytes, 'live model bytes');
  exact(facts.modelBlob, PUBLICATION.modelBlob, 'live model blob');
  exact(facts.mainBlob, PUBLICATION.modelBlob, 'live main model blob');
  exact(facts.aggregateBlob, PUBLICATION.modelBlob, 'live aggregate model blob');
  exact(facts.modelSha256, PUBLICATION.modelSha256, 'live model SHA-256');
  exact(facts.modelDiff, '', 'aggregate models diff');
  exact(facts.upstreamRef, '', 'upstream aggregate ref');
  exact(facts.baseRef, '', 'base aggregate ref');
  exact(facts.validationTags, '', 'validation publication tags');
  exact(facts.pullRequestCount, 0, 'validation publication PR count');
  exact(facts.releaseCount, 0, 'validation release count');
  return { remoteTipKind: facts.remoteSha === PUBLICATION.aggregateCommit ? 'aggregate' : 'closure', remoteSha: facts.remoteSha };
}
