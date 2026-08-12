import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const replayPath = resolve(root, 'evidence/replay/task1-replay.json');
const runtimeReplay = JSON.parse(await readFile(resolve(root, '.evidence/tensorrt/replay/tensorrt-replay.json'), 'utf8'));
const manifest = JSON.parse(await readFile(replayPath, 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const paths = [
  'evidence/golden/manifest.json',
  'evidence/tensorrt/contract.json',
  'evidence/reports/tensorrt-ep-report.json',
  'evidence/scripts/run_tensorrt_ep.py',
  'evidence/scripts/run_tensorrt_runner.sh',
  'evidence/scripts/replay_tensorrt_ep.mjs',
  'evidence/scripts/generate_tensorrt_replay_manifest.mjs',
  'evidence/scripts/test_tensorrt_ep_negative.mjs',
  'evidence/scripts/tensorrt_record_publish.py',
  'evidence/scripts/test_tensorrt_record_publish.py',
  'evidence/scripts/validate_tensorrt_ep.mjs',
  'evidence/tooling/tensorrt-requirements.lock',
  'evidence/tooling/tensorrt/Dockerfile',
  'evidence/tooling/tensorrt/Dockerfile.dockerignore',
  'evidence/tooling/tensorrt-postprocess/main.rs',
  'evidence/tooling/tensorrt-postprocess/Cargo.toml',
  'evidence/tooling/tensorrt-postprocess/Cargo.lock',
];
const entries = [];
for (const path of paths) {
  const bytes = await readFile(resolve(root, path));
  entries.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
}
manifest.outputs = manifest.outputs.filter((item) => !paths.includes(item.path));
manifest.outputs.push(...entries);
manifest.outputs.sort((left, right) => left.path.localeCompare(right.path));
manifest.steps = manifest.steps.filter((item) => item.id !== 'linux-x86_64-tensorrt-ep');
const delegated = manifest.steps.findIndex((item) => item.id === 'delegated-platform-spikes');
const step = {
  id: 'linux-x86_64-tensorrt-ep',
  command: 'node evidence/scripts/replay_tensorrt_ep.mjs .evidence/tensorrt/replay',
  executed: true,
  blockedReason: '真实 NVIDIA runner 不可用；两轮 preflight 均确定性保持 blocked。',
  rounds: runtimeReplay.rounds.map((round) => ({
    run: round.round,
    startedAt: round.startedAt,
    endedAt: round.endedAt,
    exitCode: round.runnerExitCode,
    validatorExitCode: round.validatorExitCode,
    guardExitCode: round.guardExitCode,
    semanticSha256: round.semanticSha256,
    trackedEvidence: round.trackedEvidence,
    log: {
      storage: 'embedded-in-replay-manifest',
      bytes: Buffer.byteLength(round.stdout) + Buffer.byteLength(round.stderr),
      sha256: sha256(Buffer.from(`${round.stdout}\0${round.stderr}`)),
    },
  })),
  repeatComparison: {
    runs: 2,
    expectedBlockedExitCodes: runtimeReplay.rounds.every((round) => round.runnerExitCode === 2),
    validatorsPassed: runtimeReplay.rounds.every((round) => round.validatorExitCode === 0),
    negativeGuardsPassed: runtimeReplay.rounds.every((round) => round.guardExitCode === 0),
    deterministicSemanticEvidence: runtimeReplay.deterministicSemanticEvidence,
    trackedEvidenceUnchanged: runtimeReplay.trackedEvidenceUnchanged,
  },
};
manifest.steps.splice(delegated < 0 ? manifest.steps.length : delegated, 0, step);
manifest.task1_4Complete = false;
await writeFile(replayPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', tensorRtOutputs: entries.length }));
