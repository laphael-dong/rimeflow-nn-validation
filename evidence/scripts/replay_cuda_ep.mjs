#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TRACKED_CUDA_EVIDENCE } from './cuda_evidence_validation.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const record = args.includes('--record');
const outputIndex = args.indexOf('--output');
const defaultOutput = record ? 'evidence/reports/cuda-ep-replay-report.json' : '.evidence/cuda/replay-report.json';
const output = resolve(root, outputIndex < 0 ? defaultOutput : args[outputIndex + 1]);
const recordedOutput = resolve(root, 'evidence/reports/cuda-ep-replay-report.json');
if (record ? output !== recordedOutput : !output.startsWith(resolve(root, '.evidence/cuda') + '/')) throw new Error(record ? 'CUDA record output must be the tracked replay report' : 'CUDA replay output must stay under .evidence/cuda');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const snapshot = async () => Object.fromEntries(await Promise.all(TRACKED_CUDA_EVIDENCE.map(async (path) => {
  const absolute = resolve(root, path);
  const bytes = await readFile(absolute);
  const metadata = await stat(absolute);
  return [path, { exists: true, bytes: bytes.length, sha256: sha256(bytes), mtimeNs: metadata.mtimeNs?.toString() ?? String(Math.trunc(metadata.mtimeMs * 1e6)) }];
})));
const rounds = [];
for (let round = 1; round <= 2; round++) {
  const workspace = `.evidence/cuda/replay/round-${round}`;
  const reportPath = `${workspace}/report.json`;
  const trackedBefore = await snapshot();
  const startedAt = new Date().toISOString();
  const result = spawnSync('python3', ['evidence/scripts/run_cuda_ep.py', '--workspace', workspace, '--output', reportPath], { cwd: root, encoding: 'utf8' });
  const endedAt = new Date().toISOString();
  const trackedAfter = await snapshot();
  const reportBytes = await readFile(resolve(root, reportPath));
  rounds.push({
    round,
    command: { command: ['python3', 'evidence/scripts/run_cuda_ep.py', '--workspace', workspace, '--output', reportPath], startedAt, endedAt, exitCode: result.status, stdout: result.stdout, stderr: result.stderr },
    report: JSON.parse(reportBytes),
    reportArtifact: { path: reportPath, bytes: reportBytes.length, sha256: sha256(reportBytes) },
    trackedBefore,
    trackedAfter,
  });
}
const replay = {
  schemaVersion: 2,
  task: 'T14-LNX-CUDA-01',
  mode: record ? 'recorded-blocked-host-replay' : 'blocked-host-replay',
  recorded: record,
  protectedPaths: TRACKED_CUDA_EVIDENCE,
  rounds,
  comparison: {
    stateEqual: rounds[0].report.state === rounds[1].report.state,
    failureStageEqual: rounds[0].report.failureStage === rounds[1].report.failureStage,
    failureMissingEqual: JSON.stringify(rounds[0].report.failure.missing) === JSON.stringify(rounds[1].report.failure.missing),
    exitCodeEqual: rounds[0].command.exitCode === rounds[1].command.exitCode,
  },
  trackedEvidenceUnchanged: rounds.every((item) => JSON.stringify(item.trackedBefore) === JSON.stringify(item.trackedAfter)),
};
await mkdir(dirname(output), { recursive: true });
const serialized = JSON.stringify(replay, null, 2) + '\n';
if (record) {
  const staging = `${output}.staging-${process.pid}-${Date.now()}`;
  await writeFile(staging, serialized);
  await rename(staging, output);
} else await writeFile(output, serialized);
console.log(JSON.stringify({ output, state: rounds[0].report.state, failureStage: rounds[0].report.failureStage, missing: rounds[0].report.failure.missing, exitCode: rounds[0].command.exitCode, trackedEvidenceUnchanged: replay.trackedEvidenceUnchanged }));
if (!Object.values(replay.comparison).every(Boolean) || !replay.trackedEvidenceUnchanged) process.exit(1);
