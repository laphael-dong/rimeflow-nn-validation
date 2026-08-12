import { access, chmod, copyFile, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

import { buildTrustedRustRunner, validateOpenvinoEvidence, verifyRustSourceIdentity } from './openvino_evidence_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const sources = [
  'evidence/tooling/raw-golden/Cargo.toml',
  'evidence/tooling/raw-golden/Cargo.lock',
  'evidence/tooling/raw-golden/src/main.rs',
  'evidence/tooling/raw-golden/src/lib.rs',
  'src/postprocess.rs',
];
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const [manifest, report, frozen, fixtures] = await Promise.all([
  readJson('evidence/conversions/openvino-ep-manifest.json'),
  readJson('evidence/reports/openvino-ep-report.json'),
  readJson('evidence/golden/web-reference.json'),
  readJson('evidence/fixtures/manifest.json'),
]);

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${command} failed: ${result.stdout}\n${result.stderr}`);
}

async function rejects(name, operation) {
  try {
    await operation();
  } catch {
    return name;
  }
  throw new Error(`OpenVINO Rust identity guard unexpectedly passed: ${name}`);
}

const cases = [];
const fakeRoot = resolve(root, 'evidence/tooling/raw-golden/target');
const fakeRunner = join(fakeRoot, 'debug/rimeflow-raw-golden');
const savedRoot = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-old-target-'));
const savedTarget = join(savedRoot, 'target');
let hadTarget = false;
try {
  try {
    await access(fakeRoot, constants.F_OK);
    await rename(fakeRoot, savedTarget);
    hadTarget = true;
  } catch {}
  await mkdir(dirname(fakeRunner), { recursive: true });
  const marker = join(savedRoot, 'fake-executed');
  const impostors = [
    ['pre-existing Node executable is ignored', `#!/usr/bin/env node\nrequire('node:fs').writeFileSync(${JSON.stringify(marker)}, 'node');\n`],
    ['pre-existing shell executable is ignored', `#!/bin/sh\nprintf shell > ${JSON.stringify(marker)}\n`],
    ['pre-existing decoded JSON copier is ignored', `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const report = JSON.parse(fs.readFileSync(${JSON.stringify(resolve(root, 'evidence/reports/openvino-ep-report.json'))}, 'utf8'));
const match = path.basename(process.argv[2]).match(/^(.*)-(\\d+)\\.f32le$/);
const fixture = report.rounds[0].fixtures.find((item) => item.id === match[1]);
fs.writeFileSync(process.argv[5], JSON.stringify(fixture.runs[Number(match[2]) - 1].decoded));
fs.writeFileSync(${JSON.stringify(marker)}, 'copier');
`],
  ];
  for (const [name, script] of impostors) {
    await writeFile(fakeRunner, script);
    await chmod(fakeRunner, 0o755);
    await validateOpenvinoEvidence(root, manifest, report, frozen, fixtures);
    const trusted = await buildTrustedRustRunner(root);
    try {
      if (trusted.runner === fakeRunner || trusted.runner.startsWith(fakeRoot)) throw new Error(`${name}: trusted runner used repository target`);
      const header = await readFile(trusted.runner);
      if (header[0] !== 0x7f || header.subarray(1, 4).toString() !== 'ELF') throw new Error(`${name}: temporary runner is not a built ELF executable`);
      await access(marker, constants.F_OK).then(
        () => { throw new Error(`${name}: fake executable was invoked`); },
        () => {},
      );
      cases.push(name);
    } finally {
      await trusted.cleanup();
    }
  }
} finally {
  await rm(fakeRoot, { recursive: true, force: true });
  if (hadTarget) await rename(savedTarget, fakeRoot);
  await rm(savedRoot, { recursive: true, force: true });
}

for (const relative of sources) {
  const repository = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-source-'));
  try {
    for (const source of sources) {
      const destination = join(repository, source);
      await mkdir(dirname(destination), { recursive: true });
      await copyFile(resolve(root, source), destination);
    }
    run('git', ['init', '--quiet'], repository);
    run('git', ['config', 'user.name', 'OpenVINO Guard'], repository);
    run('git', ['config', 'user.email', 'openvino-guard@example.invalid'], repository);
    run('git', ['add', ...sources], repository);
    run('git', ['commit', '--quiet', '-m', 'fixture'], repository);
    await writeFile(join(repository, relative), `${await readFile(join(repository, relative), 'utf8')}\n# identity drift\n`);
    cases.push(await rejects(`${relative} drift is rejected`, () => verifyRustSourceIdentity(repository)));
  } finally {
    await rm(repository, { recursive: true, force: true });
  }
}

cases.push(await rejects('temporary locked Cargo build failure is rejected', () => buildTrustedRustRunner(root, { cargo: '/definitely/missing/cargo' })));
console.log(JSON.stringify({ cases, ok: true, sourceCount: sources.length }));
