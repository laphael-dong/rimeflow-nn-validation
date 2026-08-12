import { access, chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

import {
  FORBIDDEN_RUST_BUILD_ENVIRONMENT,
  assertRecordedProductionPostprocess,
  buildTrustedRustRunner,
  verifyRustSourceIdentity,
} from './openvino_evidence_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const sources = [
  'evidence/tooling/raw-golden/Cargo.toml',
  'evidence/tooling/raw-golden/Cargo.lock',
  'evidence/tooling/raw-golden/src/main.rs',
  'evidence/tooling/raw-golden/src/lib.rs',
  'src/postprocess.rs',
];

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${command} failed: ${result.stdout}\n${result.stderr}`);
}

async function temporaryRepository() {
  const repository = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-rust-guard-'));
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
  return repository;
}

async function rejects(name, operation, contains = null) {
  try {
    await operation();
  } catch (error) {
    if (contains && !String(error).includes(contains)) throw new Error(`${name}: wrong rejection: ${error}`);
    return name;
  }
  throw new Error(`OpenVINO Rust identity guard unexpectedly passed: ${name}`);
}

async function missing(path) {
  await access(path, constants.F_OK).then(
    () => { throw new Error(`unexpected marker exists: ${path}`); },
    () => {},
  );
}

const cases = [];

const recorded = JSON.parse(await readFile(resolve(root, 'evidence/reports/openvino-ep-report.json'), 'utf8')).productionPostprocess;
for (const [name, mutate] of [
  ['record source artifact removal is rejected', (value) => value.sourceArtifacts.pop()],
  ['record HEAD blob drift is rejected', (value) => { value.sourceArtifacts[0].headBlobOid = '0'.repeat(40); }],
  ['record runner SHA drift is rejected', (value) => { value.runner.sha256 = '0'.repeat(64); }],
  ['record runner bytes drift is rejected', (value) => { value.runner.bytes += 1; }],
  ['record build command drift is rejected', (value) => { value.build.argv.pop(); }],
  ['record Cargo version drift is rejected', (value) => { value.build.cargoVersion = 'cargo 0'; }],
  ['record Rustc version drift is rejected', (value) => { value.build.rustcVersion = 'rustc 0'; }],
  ['record offline false is rejected', (value) => { value.build.offline = false; }],
  ['record locked false is rejected', (value) => { value.build.locked = false; }],
  ['record fresh target false is rejected', (value) => { value.build.freshTarget = false; }],
  ['record incremental true is rejected', (value) => { value.build.incremental = true; }],
]) {
  const candidate = structuredClone(recorded);
  mutate(candidate);
  cases.push(await rejects(name, async () => {
    const trusted = await buildTrustedRustRunner(root, { environment: {} });
    try { assertRecordedProductionPostprocess(candidate, trusted.provenance); } finally { await trusted.cleanup(); }
  }, 'provenance drift'));
}

const injectionRoot = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-rust-injection-'));
try {
  const marker = join(injectionRoot, 'wrapper-executed');
  const wrapper = join(injectionRoot, 'wrapper');
  await writeFile(wrapper, `#!/bin/sh\nprintf injected > ${JSON.stringify(marker)}\nexec "$@"\n`);
  await chmod(wrapper, 0o755);
  const values = new Map([
    ['RUSTC_WRAPPER', wrapper],
    ['RUSTC_WORKSPACE_WRAPPER', wrapper],
    ['RUSTFLAGS', '--cfg injected'],
    ['CARGO_ENCODED_RUSTFLAGS', '--cfg\x1finjected'],
    ['CARGO_BUILD_RUSTFLAGS', '--cfg injected'],
    ['CARGO_BUILD_RUSTC_WRAPPER', wrapper],
    ['CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER', wrapper],
    ['CARGO_TARGET_DIR', injectionRoot],
    ['CARGO_HOME', injectionRoot],
    ['CARGO_CONFIG', join(injectionRoot, 'config.toml')],
    ['RUSTC', wrapper],
    ['RUSTDOC', wrapper],
    ['RUSTUP_TOOLCHAIN', 'injected'],
    ['CARGO_BUILD_TARGET', 'injected-target'],
    ['CARGO_BUILD_JOBS', '999'],
  ]);
  for (const variable of FORBIDDEN_RUST_BUILD_ENVIRONMENT) {
    cases.push(await rejects(`${variable} injection is rejected`, () => buildTrustedRustRunner(root, { environment: { [variable]: values.get(variable) ?? 'injected' } }), 'forbidden Cargo/Rust build environment'));
  }
  for (const [variable, value] of [
    ['CARGO_ALIAS_BUILD', wrapper],
    ['CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS', '--cfg injected'],
    ['CARGO_PROFILE_RELEASE_RUSTFLAGS', '--cfg injected'],
    ['CARGO_FUTURE_CONFIG', join(injectionRoot, 'config.toml')],
    ['RUSTC_CUSTOM_WRAPPER', wrapper],
  ]) {
    cases.push(await rejects(`${variable} equivalent injection is rejected`, () => buildTrustedRustRunner(root, { environment: { [variable]: value } }), 'forbidden Cargo/Rust build environment'));
  }
  await missing(marker);
  cases.push('fake RUSTC_WRAPPER marker remains absent');
} finally {
  await rm(injectionRoot, { recursive: true, force: true });
}

const fakeRepository = await temporaryRepository();
try {
  const fakeTarget = join(fakeRepository, 'evidence/tooling/raw-golden/target/release/rimeflow-raw-golden');
  const marker = join(fakeRepository, 'fake-runner-executed');
  await mkdir(dirname(fakeTarget), { recursive: true });
  const impostors = [
    ['pre-existing Node runner is ignored', `#!/usr/bin/env node\nrequire('node:fs').writeFileSync(${JSON.stringify(marker)}, 'node');\n`],
    ['pre-existing shell runner is ignored', `#!/bin/sh\nprintf shell > ${JSON.stringify(marker)}\n`],
    ['pre-existing decoded JSON copier is ignored', `#!/bin/sh\nprintf '[]\\n' > "$5"\nprintf copier > ${JSON.stringify(marker)}\n`],
  ];
  for (const [name, script] of impostors) {
    await writeFile(fakeTarget, script);
    await chmod(fakeTarget, 0o755);
    const trusted = await buildTrustedRustRunner(fakeRepository, { environment: {} });
    try {
      if (trusted.runner.startsWith(join(fakeRepository, 'evidence/tooling/raw-golden/target'))) throw new Error(`${name}: repository runner selected`);
      const bytes = await readFile(trusted.runner);
      if (bytes.subarray(0, 4).toString('hex') !== '7f454c46') throw new Error(`${name}: fresh runner is not ELF`);
      await missing(marker);
      cases.push(name);
    } finally {
      await trusted.cleanup();
    }
  }

  await mkdir(join(fakeRepository, '.cargo'), { recursive: true });
  await writeFile(join(fakeRepository, '.cargo/config.toml'), `[build]\nrustc-wrapper = ${JSON.stringify(fakeTarget)}\n`);
  const trusted = await buildTrustedRustRunner(fakeRepository, { environment: {} });
  try {
    await missing(marker);
    if (trusted.provenance.build.repositoryRootCargoConfigParticipated !== false) throw new Error('root Cargo config participation not recorded');
    cases.push('repository root .cargo/config.toml is excluded by source mirror');
  } finally {
    await trusted.cleanup();
  }
} finally {
  await rm(fakeRepository, { recursive: true, force: true });
}

for (const relative of sources) {
  const repository = await temporaryRepository();
  try {
    await writeFile(join(repository, relative), `${await readFile(join(repository, relative), 'utf8')}\n# identity drift\n`);
    cases.push(await rejects(`${relative} bytes/SHA/blob drift is rejected`, () => verifyRustSourceIdentity(repository), 'differs from HEAD'));
  } finally {
    await rm(repository, { recursive: true, force: true });
  }
}

for (const [name, relative, contents] of [
  ['unapproved build.rs is rejected', 'evidence/tooling/raw-golden/build.rs', 'fn main() {}\n'],
  ['local .cargo/config is rejected', 'evidence/tooling/raw-golden/.cargo/config.toml', '[build]\njobs = 999\n'],
  ['undeclared Rust source is rejected', 'evidence/tooling/raw-golden/src/extra.rs', 'pub fn injected() {}\n'],
]) {
  const repository = await temporaryRepository();
  try {
    await mkdir(dirname(join(repository, relative)), { recursive: true });
    await writeFile(join(repository, relative), contents);
    cases.push(await rejects(name, () => buildTrustedRustRunner(repository, { environment: {} }), 'unapproved raw-golden'));
  } finally {
    await rm(repository, { recursive: true, force: true });
  }
}

const failedBuildRepository = await temporaryRepository();
try {
  const manifest = join(failedBuildRepository, 'evidence/tooling/raw-golden/Cargo.toml');
  await writeFile(manifest, `${await readFile(manifest, 'utf8')}\ndefinitely_missing = "999.0.0"\n`);
  run('git', ['add', 'evidence/tooling/raw-golden/Cargo.toml'], failedBuildRepository);
  run('git', ['commit', '--quiet', '-m', 'locked build failure'], failedBuildRepository);
  cases.push(await rejects('fresh offline locked build failure is rejected', () => buildTrustedRustRunner(failedBuildRepository, { environment: {} }), 'trusted production Rust build failed'));
} finally {
  await rm(failedBuildRepository, { recursive: true, force: true });
}

console.log(JSON.stringify({ cases, environmentCases: FORBIDDEN_RUST_BUILD_ENVIRONMENT.length, ok: true, sourceCount: sources.length }));
