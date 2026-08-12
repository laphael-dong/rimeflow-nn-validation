import { createHash } from 'node:crypto';
import { closeSync, copyFileSync, existsSync, mkdtempSync, mkdirSync, openSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const runner = resolve(root, 'evidence/tooling/windows-ml-runner');
const dotnet = process.env.RIMEFLOW_WINDOWS_ML_DOTNET ?? '/tmp/rimeflow-dotnet-8.0.423/dotnet';
const nugetPackages = process.env.RIMEFLOW_WINDOWS_ML_NUGET_PACKAGES ?? '/tmp/rimeflow-nuget-packages';
const reportIndex = process.argv.indexOf('--report');
const reportPath = resolve(root, reportIndex >= 0 ? process.argv[reportIndex + 1] : 'evidence/reports/windows-ml-static-compile-report.json');
const sourceFiles = ['Program.cs', 'RunnerSupport.cs', 'WindowsMlSpike.csproj', 'global.json', 'packages.lock.json'];
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const iso = () => new Date().toISOString();

if (!existsSync(dotnet)) throw new Error(`pinned dotnet unavailable: ${dotnet}`);
const version = spawnSync(dotnet, ['--version'], { encoding: 'utf8', cwd: runner });
if (version.status !== 0 || version.stdout.trim() !== '8.0.423') throw new Error(`expected .NET SDK 8.0.423, got status=${version.status}, stdout=${version.stdout}, stderr=${version.stderr}`);

function runLogged(command, args, cwd, logPath) {
  const fd = openSync(logPath, 'w');
  const startedAt = iso();
  const result = spawnSync(command, args, {
    cwd,
    env: { ...process.env, DOTNET_CLI_TELEMETRY_OPTOUT: '1', NUGET_PACKAGES: nugetPackages },
    stdio: ['ignore', fd, fd],
  });
  closeSync(fd);
  return { startedAt, endedAt: iso(), exitCode: result.status, signal: result.signal };
}

function normalizedCommand(args) {
  return `DOTNET_CLI_TELEMETRY_OPTOUT=1 NUGET_PACKAGES=${nugetPackages} ${dotnet} ${args.join(' ')}`;
}

function replayTarget(rid) {
  const workspace = mkdtempSync(join(tmpdir(), `rimeflow-winml-${rid}-`));
  try {
    for (const file of sourceFiles) copyFileSync(join(runner, file), join(workspace, file));
    const cleanBefore = {
      binAbsent: !existsSync(join(workspace, 'bin')),
      objAbsent: !existsSync(join(workspace, 'obj')),
      projectAssetsAbsent: !existsSync(join(workspace, 'project.assets.json')) && !findByName(workspace, 'project.assets.json'),
    };
    if (Object.values(cleanBefore).some((value) => !value)) throw new Error(`${rid}: workspace was not clean before restore`);

    const restoreArgs = ['restore', 'WindowsMlSpike.csproj', '--locked-mode'];
    const restoreLog = join(workspace, `${rid}-restore.log`);
    const restore = runLogged(dotnet, restoreArgs, workspace, restoreLog);
    if (restore.exitCode !== 0) throw new Error(`${rid}: locked restore failed\n${readFileSync(restoreLog, 'utf8')}`);

    const compileArgs = [
      'msbuild', 'WindowsMlSpike.csproj', '-target:Compile', '-property:Configuration=Release',
      `-property:RuntimeIdentifier=${rid}`, '-property:OutputType=Library',
      '-property:WindowsAppSDKSelfContained=false', '-property:RestoreLockedMode=true',
      `-property:PathMap=${workspace}=/_/windows-ml-runner`,
      '-verbosity:diagnostic', `-binaryLogger:${rid}-compile.binlog`,
      `-fileLoggerParameters:LogFile=${rid}-compile.log;Verbosity=Diagnostic`,
    ];
    const consoleLog = join(workspace, `${rid}-compile-console.log`);
    const compile = runLogged(dotnet, compileArgs, workspace, consoleLog);
    const diagnosticLog = readFileSync(join(workspace, `${rid}-compile.log`), 'utf8');
    const assemblyPath = join(workspace, 'obj/Release/net8.0-windows10.0.17763.0', rid, 'WindowsMlSpike.dll');
    const lines = diagnosticLog.split(/\r?\n/);
    const coreCompileLine = lines.find((line) => line.includes('Target "CoreCompile:'));
    const cscLine = lines.find((line) => line.includes('Task "Csc"'));
    const programLine = lines.find((line) => line.trim() === 'Program.cs');
    const supportLine = lines.find((line) => line.trim() === 'RunnerSupport.cs');
    if (compile.exitCode !== 0 || !coreCompileLine || !cscLine || !programLine || !supportLine || !existsSync(assemblyPath)) {
      const diagnosticTail = diagnosticLog.split(/\r?\n/).filter((line) => /(^|\s)(error|warning)\s+[A-Z]+\d+|Build FAILED|CoreCompile|Task "Csc"/i.test(line)).slice(-80).join('\n');
      const consoleTail = readFileSync(consoleLog, 'utf8').split(/\r?\n/).slice(-80).join('\n');
      throw new Error(`${rid}: CoreCompile evidence incomplete; exit=${compile.exitCode}, dll=${existsSync(assemblyPath)}\n${diagnosticTail}\n${consoleTail}`);
    }
    const assembly = readFileSync(assemblyPath);
    return {
      rid,
      cleanBefore,
      workspacePolicy: 'mkdtemp workspace deleted after evidence extraction; no bin/obj/project.assets.json reused',
      restore: { command: normalizedCommand(restoreArgs), ...restore },
      compile: {
        command: normalizedCommand(compileArgs),
        ...compile,
        target: 'Compile',
        coreCompileExecuted: true,
        roslynCscExecuted: true,
        programCsCompiled: true,
        rawLogEvidence: { coreCompileLine: coreCompileLine.trim(), cscLine: cscLine.trim(), programSourceLine: programLine.trim(), supportSourceLine: supportLine.trim() },
      },
      assembly: {
        logicalPath: `obj/Release/net8.0-windows10.0.17763.0/${rid}/WindowsMlSpike.dll`,
        bytes: assembly.length,
        sha256: sha256(assembly),
        targetRidFromPath: rid,
        retained: false,
      },
    };
  } finally {
    rmSync(workspace, { recursive: true, force: true });
  }
}

function findByName(directory, name) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (findByName(path, name)) return true;
    } else if (entry.name === name) return true;
  }
  return false;
}

const report = {
  schemaVersion: 1,
  generatedAt: iso(),
  host: { os: process.platform, arch: process.arch },
  sdk: { version: version.stdout.trim(), executable: dotnet },
  nugetPackages,
  source: Object.fromEntries(sourceFiles.map((file) => [file, { bytes: statSync(join(runner, file)).size, sha256: sha256(readFileSync(join(runner, file))) }])),
  lockFileSha256: sha256(readFileSync(join(runner, 'packages.lock.json'))),
  targets: Object.fromEntries(['win-x64', 'win-arm64'].map((rid) => [rid, replayTarget(rid)])),
};
mkdirSync(dirname(reportPath), { recursive: true });
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ ok: true, report: reportPath, targets: Object.keys(report.targets) }));
