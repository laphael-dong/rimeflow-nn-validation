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
  previousClosureCommit: 'a81b07f8b2b773f162d61cc82659f8c58ad3a832',
  previousClosureSubject: '[DOC] Close task 1.4 publication evidence',
  fixSubject: '[FIX] Bind task 1.4 closure to exact remote commit',
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

export const CRITICAL_EVIDENCE_PATHS = Object.freeze([
  'evidence/reports/task1-4-publication-report.json',
  'evidence/conversions/task1-4-aggregate.json',
  'evidence/conversions/conversion-spikes.json',
  'evidence/replay/task1-replay.json',
  'evidence/golden/manifest.json',
]);

const IMMUTABLE_HISTORICAL_EVIDENCE_PATHS = Object.freeze([
  'evidence/reports/task1-4-publication-report.json',
  'evidence/conversions/task1-4-aggregate.json',
  'evidence/conversions/conversion-spikes.json',
]);

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

function commitBytes(cwd, ref) {
  return execFileSync('git', ['cat-file', 'commit', ref], { cwd });
}

function parseTreeEntry(text, expectedPath) {
  const match = /^(\d+) (\w+) ([0-9a-f]{40})\s+(\d+)\t(.+)$/.exec(text);
  if (!match || match[2] !== 'blob' || match[5] !== expectedPath) fail(`tree entry missing: ${expectedPath}`);
  return { mode: match[1], blob: match[3], bytes: Number(match[4]), path: match[5] };
}

function firstParentChain(cwd, tip) {
  const text = git(cwd, ['rev-list', '--first-parent', '--parents', `${PUBLICATION.aggregateCommit}..${tip}`]);
  return text ? text.split('\n').map((line) => line.split(' ')) : [];
}

function criticalBlobs(cwd, ref) {
  return Object.fromEntries(CRITICAL_EVIDENCE_PATHS.map((path) => [path, git(cwd, ['rev-parse', `${ref}:${path}`])]));
}

export function collectLivePublicationFacts(root) {
  const localHead = git(root, ['rev-parse', 'HEAD']);
  const remoteLine = git(root, ['ls-remote', '--heads', PUBLICATION.remoteUrl, PUBLICATION.remoteRef.replace('refs/heads/', '')]);
  if (remoteLine.split('\n').length !== 1) fail('live remote ref missing or ambiguous');
  const [remoteSha, remoteRef] = remoteLine.split(/\s+/);
  const api = ghJson(['api', `repos/${PUBLICATION.repository}/git/ref/heads/${PUBLICATION.remoteRef.replace('refs/heads/', '')}`]);
  const actor = execFileSync('gh', ['api', 'user', '--jq', '.login'], { encoding: 'utf8' }).trim();
  const repo = ghJson(['api', `repos/${PUBLICATION.repository}`]);
  const localIdentity = parseCommitIdentity(git(root, ['show', '--no-patch', '--format=%H%n%P%n%s', 'HEAD']));
  const previousClosureIdentity = parseCommitIdentity(git(root, ['show', '--no-patch', '--format=%H%n%P%n%s', PUBLICATION.previousClosureCommit]));
  const aggregateIdentity = parseCommitIdentity(git(root, ['show', '--no-patch', '--format=%H%n%P%n%s', PUBLICATION.aggregateCommit]));
  const localCommitDigest = sha256(commitBytes(root, 'HEAD'));
  const localTree = git(root, ['rev-parse', 'HEAD^{tree}']);
  const localFirstParentChain = firstParentChain(root, 'HEAD');
  const localCriticalBlobs = criticalBlobs(root, 'HEAD');
  const previousClosureCriticalBlobs = criticalBlobs(root, PUBLICATION.previousClosureCommit);

  git(PUBLICATION.validationClone, ['fetch', '--no-tags', PUBLICATION.remoteUrl, PUBLICATION.remoteRef]);
  const fetchHead = git(PUBLICATION.validationClone, ['rev-parse', 'FETCH_HEAD']);
  const readbackIdentity = parseCommitIdentity(git(PUBLICATION.validationClone, ['show', '--no-patch', '--format=%H%n%P%n%s', 'FETCH_HEAD']));
  const readbackCommitDigest = sha256(commitBytes(PUBLICATION.validationClone, 'FETCH_HEAD'));
  const readbackTree = git(PUBLICATION.validationClone, ['rev-parse', 'FETCH_HEAD^{tree}']);
  const readbackFirstParentChain = firstParentChain(PUBLICATION.validationClone, 'FETCH_HEAD');
  const readbackCriticalBlobs = criticalBlobs(PUBLICATION.validationClone, 'FETCH_HEAD');
  const localModel = parseTreeEntry(git(root, ['ls-tree', '-l', 'HEAD', PUBLICATION.modelPath]), PUBLICATION.modelPath);
  const readbackModel = parseTreeEntry(git(PUBLICATION.validationClone, ['ls-tree', '-l', 'FETCH_HEAD', PUBLICATION.modelPath]), PUBLICATION.modelPath);
  const mainBlob = git(PUBLICATION.validationClone, ['rev-parse', `main:${PUBLICATION.modelPath}`]);
  const modelSha = execFileSync('git', ['show', `FETCH_HEAD:${PUBLICATION.modelPath}`], { cwd: PUBLICATION.validationClone, maxBuffer: 20 * 1024 * 1024 });
  const localRemoteDiff = git(PUBLICATION.validationClone, ['diff', '--name-status', localHead, 'FETCH_HEAD']);
  const modelDiff = git(PUBLICATION.validationClone, ['diff', '--name-status', `${PUBLICATION.aggregateCommit}..FETCH_HEAD`, '--', 'models']);
  const upstreamRef = git(root, ['ls-remote', '--heads', PUBLICATION.upstreamUrl, PUBLICATION.remoteRef.replace('refs/heads/', '')]);
  const baseRef = git(root, ['ls-remote', '--heads', PUBLICATION.baseUrl, PUBLICATION.remoteRef.replace('refs/heads/', '')]);
  const validationRefs = git(root, ['ls-remote', '--heads', '--tags', PUBLICATION.remoteUrl]);
  const refsAtExpectedSha = validationRefs.split('\n').filter(Boolean).filter((line) => line.split(/\s+/)[0] === localHead).map((line) => line.split(/\s+/)[1]).sort();
  const prs = ghJson(['api', `repos/${PUBLICATION.repository}/pulls?state=all&head=laphael-dong:${PUBLICATION.remoteRef.replace('refs/heads/', '')}`]);
  const releases = ghJson(['api', `repos/${PUBLICATION.repository}/releases`]);
  return {
    remoteSha, remoteRef, api, actor, canPush: repo.permissions?.push === true,
    localHead, localIdentity, previousClosureIdentity, aggregateIdentity,
    freshFetchExecuted: true, fetchHead, readbackIdentity,
    localCommitDigest, readbackCommitDigest, localTree, readbackTree, localRemoteDiff,
    localFirstParentChain, readbackFirstParentChain, previousClosureCriticalBlobs, localCriticalBlobs, readbackCriticalBlobs,
    localModel, readbackModel, mainBlob, modelSha256: sha256(modelSha), modelDiff,
    upstreamRef, baseRef, refsAtExpectedSha, pullRequestCount: prs.length, releaseCount: releases.length,
  };
}

export function validateLivePublicationFacts(facts) {
  if (!/^[0-9a-f]{40}$/.test(facts.localHead)) fail('local HEAD SHA shape');
  exact(facts.remoteRef, PUBLICATION.remoteRef, 'live remote ref');
  exact(facts.api?.ref, PUBLICATION.remoteRef, 'live GitHub API ref');
  exact(facts.api?.object?.type, 'commit', 'live GitHub API object type');
  exact(facts.remoteSha, facts.localHead, 'live remote SHA/local HEAD');
  exact(facts.api?.object?.sha, facts.localHead, 'live GitHub API object SHA/local HEAD');
  exact(facts.actor, 'laphael-dong', 'live GitHub actor');
  if (!facts.canPush) fail('live GitHub push permission missing');
  exact(facts.localIdentity.commit, facts.localHead, 'local HEAD commit object');
  exact(facts.localIdentity.parent, PUBLICATION.previousClosureCommit, 'fix commit parent');
  exact(facts.localIdentity.subject, PUBLICATION.fixSubject, 'fix commit subject');
  exact(facts.previousClosureIdentity.commit, PUBLICATION.previousClosureCommit, 'previous closure commit object');
  exact(facts.previousClosureIdentity.parent, PUBLICATION.aggregateCommit, 'previous closure parent');
  exact(facts.previousClosureIdentity.subject, PUBLICATION.previousClosureSubject, 'previous closure subject');
  exact(facts.aggregateIdentity.commit, PUBLICATION.aggregateCommit, 'aggregate commit object');
  exact(facts.aggregateIdentity.parent, PUBLICATION.aggregateParent, 'aggregate parent object');
  exact(facts.aggregateIdentity.subject, PUBLICATION.aggregateSubject, 'aggregate subject object');
  if (facts.freshFetchExecuted !== true) fail('fresh fetch was not executed');
  exact(facts.fetchHead, facts.localHead, 'fresh FETCH_HEAD/local HEAD');
  exact(facts.readbackIdentity.commit, facts.localHead, 'readback commit/local HEAD');
  exact(facts.readbackIdentity.parent, PUBLICATION.previousClosureCommit, 'readback fix parent');
  exact(facts.readbackIdentity.subject, PUBLICATION.fixSubject, 'readback fix subject');
  exact(facts.readbackCommitDigest, facts.localCommitDigest, 'commit object byte digest');
  exact(facts.readbackTree, facts.localTree, 'commit tree OID');
  exact(facts.localRemoteDiff, '', 'local HEAD/readback diff');
  const expectedChain = [[facts.localHead, PUBLICATION.previousClosureCommit], [PUBLICATION.previousClosureCommit, PUBLICATION.aggregateCommit]];
  if (JSON.stringify(facts.localFirstParentChain) !== JSON.stringify(expectedChain)) fail('local first-parent closure path drift or merge');
  if (JSON.stringify(facts.readbackFirstParentChain) !== JSON.stringify(expectedChain)) fail('readback first-parent closure path drift or merge');
  for (const path of CRITICAL_EVIDENCE_PATHS) {
    if (!/^[0-9a-f]{40}$/.test(facts.localCriticalBlobs?.[path] ?? '')) fail(`local critical evidence blob missing: ${path}`);
    exact(facts.readbackCriticalBlobs?.[path], facts.localCriticalBlobs[path], `readback critical evidence blob: ${path}`);
  }
  for (const path of IMMUTABLE_HISTORICAL_EVIDENCE_PATHS) {
    exact(facts.localCriticalBlobs[path], facts.previousClosureCriticalBlobs?.[path], `immutable historical evidence blob: ${path}`);
  }
  exact(facts.localModel?.path, PUBLICATION.modelPath, 'local model path');
  exact(facts.readbackModel?.path, PUBLICATION.modelPath, 'readback model path');
  exact(facts.localModel?.mode, '100644', 'local model mode');
  exact(facts.readbackModel?.mode, '100644', 'readback model mode');
  exact(facts.localModel?.bytes, PUBLICATION.modelBytes, 'local model bytes');
  exact(facts.readbackModel?.bytes, PUBLICATION.modelBytes, 'readback model bytes');
  exact(facts.localModel?.blob, PUBLICATION.modelBlob, 'local model blob');
  exact(facts.readbackModel?.blob, PUBLICATION.modelBlob, 'readback model blob');
  exact(facts.mainBlob, PUBLICATION.modelBlob, 'live main model blob');
  exact(facts.modelSha256, PUBLICATION.modelSha256, 'live model SHA-256');
  exact(facts.modelDiff, '', 'aggregate models diff');
  exact(facts.upstreamRef, '', 'upstream aggregate ref');
  exact(facts.baseRef, '', 'base aggregate ref');
  if (JSON.stringify(facts.refsAtExpectedSha) !== JSON.stringify([PUBLICATION.remoteRef])) fail('final SHA exists at an unauthorized branch or tag');
  exact(facts.pullRequestCount, 0, 'validation publication PR count');
  exact(facts.releaseCount, 0, 'validation release count');
  return { remoteTipKind: 'fix-closure', remoteSha: facts.remoteSha, tree: facts.localTree, commitObjectSha256: facts.localCommitDigest };
}
