#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateCudaEvidence, validateCudaRepositoryBoundary, validateCudaReplay } from './cuda_evidence_validation.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const value = (name, fallback) => {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
};
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const manifestPath = value('--manifest', 'evidence/conversions/cuda-ep-spike-manifest.json');
const reportPath = value('--report', 'evidence/reports/cuda-ep-spike-report.json');
const replayPath = value('--replay', 'evidence/reports/cuda-ep-replay-report.json');
const manifest = await readJson(manifestPath);
const report = await readJson(reportPath);
const replay = await readJson(replayPath);
await validateCudaEvidence(root, manifest, report);
await validateCudaReplay(root, replay);
await validateCudaRepositoryBoundary(root);
console.log(JSON.stringify({ ok: true, task: manifest.task, state: report.state, failureStage: report.failureStage, replayRounds: replay.rounds.length }));
