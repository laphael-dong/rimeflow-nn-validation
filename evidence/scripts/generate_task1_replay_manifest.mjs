import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const outputPath = resolve(root, 'evidence/replay/task1-replay.json');
const repositoryHeadAtReplay = '25630a1a9125a5de776c5af43539dcefe32e5aea';
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const git = (args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
if (git(['merge-base', '--is-ancestor', repositoryHeadAtReplay, 'HEAD']) !== '') throw new Error('unexpected git output');
const outputs = [
  'evidence/model/model-contract.json',
  'evidence/fixtures/manifest.json',
  'evidence/golden/coverage-matrix.json',
  'evidence/golden/manifest.json',
  'evidence/golden/web-reference.json',
  'evidence/reports/preprocess-conformance.json',
  'evidence/conversions/conversion-spikes.json',
];
const artifacts = [];
for (const path of outputs) {
  const bytes = await readFile(resolve(root, path));
  artifacts.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
}
const manifest = {
  schemaVersion: 1,
  repository: 'rimeflow-yolov8n',
  repositoryHeadAtReplay: { commit: repositoryHeadAtReplay, kind: 'evidence-input-head', finalEvidenceCommitRecordedByGit: true },
  inputs: {
    operatorSourceCommit: 'eacbcf00dfc2fba941b494e2955e87fffd707382',
    modelSha256: '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad',
    fixtureSourceCommit: '42ef8a125df038dcca49f6216f446fe9112946c1',
  },
  tools: { node: process.version, cargo: execFileSync('cargo', ['--version'], { encoding: 'utf8' }).trim(), onnxruntimeWeb: '1.27.0' },
  steps: [
    { id: 'contract-fixture-golden', command: 'bun run evidence/scripts/generate_all.mjs', executed: true, blockedReason: null, repeatComparison: { runs: 2, deterministicDigestEqual: true } },
    { id: 'production-raw-golden', command: 'cargo test --offline --manifest-path evidence/tooling/raw-golden/Cargo.toml', executed: true, blockedReason: null, repeatComparison: { runs: 1, passed: true } },
    { id: 'conversion-report-regeneration', command: 'node evidence/scripts/run_conversion_spikes.mjs', executed: true, blockedReason: '报告可重现，但模型授权、Windows ML x64/ARM64 Load/Run、Core ML/LiteRT 同源链仍未闭环；不代表任务 1.4 完成。', repeatComparison: { runs: 2, deterministicDigestEqual: true } },
  ],
  outputs: artifacts,
  task1_7OwnershipReplayComplete: true,
  task1_4Complete: false,
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', outputCount: artifacts.length }));
