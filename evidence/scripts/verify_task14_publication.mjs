import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { collectLivePublicationFacts, readAndValidatePublicationReceipt, validateLivePublicationFacts } from './task14_publication_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const aggregate = JSON.parse(await readFile(resolve(root, 'evidence/conversions/task1-4-aggregate.json'), 'utf8'));
await readAndValidatePublicationReceipt(root, aggregate);
const facts = collectLivePublicationFacts(root);
const live = validateLivePublicationFacts(facts);
console.log(JSON.stringify({ ok: true, trackedEvidenceModified: false, repository: 'laphael-dong/rimeflow-nn-validation', ref: facts.remoteRef, aggregateCommit: '19193a34f2fb2b36465538b02687a07608f7810e', remoteTipKind: live.remoteTipKind, remoteTip: live.remoteSha, modelBlob: facts.modelBlob, modelSha256: facts.modelSha256 }));
