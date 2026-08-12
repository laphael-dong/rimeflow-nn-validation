import { copyFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const dotnet = process.env.RIMEFLOW_WINDOWS_ML_DOTNET ?? '/tmp/rimeflow-dotnet-8.0.423/dotnet';
const workspace = mkdtempSync(join(tmpdir(), 'rimeflow-winml-guards-'));
try {
  for (const [source, destination] of [
    ['evidence/tooling/windows-ml-runner-guards/Program.cs', 'Program.cs'],
    ['evidence/tooling/windows-ml-runner-guards/WindowsMlRunnerGuards.csproj', 'WindowsMlRunnerGuards.csproj'],
    ['evidence/tooling/windows-ml-runner/RunnerSupport.cs', 'RunnerSupport.cs'],
  ]) copyFileSync(resolve(root, source), join(workspace, destination));
  const result = spawnSync(dotnet, ['run', '--project', 'WindowsMlRunnerGuards.csproj', '--configuration', 'Release'], {
    cwd: workspace,
    encoding: 'utf8',
    env: { ...process.env, DOTNET_CLI_TELEMETRY_OPTOUT: '1' },
  });
  if (result.status !== 0) throw new Error(`Windows ML runner guards failed:\n${result.stdout}\n${result.stderr}`);
  process.stdout.write(result.stdout);
} finally {
  rmSync(workspace, { recursive: true, force: true });
}
