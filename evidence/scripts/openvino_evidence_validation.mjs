import { createHash } from 'node:crypto';
import { copyFile, lstat, mkdir, mkdtemp, opendir, readFile, realpath, rm, stat, symlink, unlink, writeFile } from 'node:fs/promises';
import { tmpdir, userInfo } from 'node:os';
import { basename, dirname, isAbsolute, join, normalize, relative, resolve, sep } from 'node:path';
import { spawnSync } from 'node:child_process';

const MODEL_SHA = '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad';
const INPUT = { dtype: 'float32', elementCount: 1228800, name: 'images', shape: [1, 3, 640, 640] };
const OUTPUT = { dtype: 'float32', elementCount: 705600, name: 'output0', shape: [1, 84, 8400] };
const RUST_SOURCES = [
  'evidence/tooling/raw-golden/Cargo.toml',
  'evidence/tooling/raw-golden/Cargo.lock',
  'evidence/tooling/raw-golden/src/main.rs',
  'evidence/tooling/raw-golden/src/lib.rs',
  'src/postprocess.rs',
];
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  return value;
};
const same = (left, right) => JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
const fail = (message) => { throw new Error(`OpenVINO evidence: ${message}`); };
const nearlyEqual = (left, right) => Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= Number.EPSILON * Math.max(1, Math.abs(left), Math.abs(right)) * 8;
const OPENVINO_VENV_TOKEN = '$OPENVINO_VENV/';
const OPENVINO_CAPI_RELATIVE = 'lib/python3.12/site-packages/onnxruntime/capi';
const OPENVINO_WORKSPACE_TOKEN = '$OPENVINO_WORKSPACE/';
const OPENVINO_PROFILE_NORMALIZATION = 'openvino-profile-node-events-v1: retain only Node event name and provider in source order';

export async function resolveTrustedOpenvinoLibraryPath(root, library) {
  const expectedRelative = `${OPENVINO_CAPI_RELATIVE}/${library.name}`;
  for (const field of ['actualPath', 'path']) {
    if (typeof library[field] !== 'string' || !library[field].startsWith(OPENVINO_VENV_TOKEN)) fail(`${field} must use the trusted OpenVINO venv token: ${library.name}`);
    const relativePath = library[field].slice(OPENVINO_VENV_TOKEN.length);
    if (isAbsolute(relativePath) || relativePath.includes('\\') || normalize(relativePath) !== relativePath || relativePath !== expectedRelative) fail(`${field} has an invalid OpenVINO library relative path: ${library.name}`);
  }
  const venvRoot = await realpath(resolve(root, '.evidence/openvino/venv')).catch(() => fail('canonical OpenVINO venv is unavailable'));
  const expectedCapi = resolve(venvRoot, OPENVINO_CAPI_RELATIVE);
  const canonicalCapi = await realpath(expectedCapi).catch(() => fail('OpenVINO capi directory is unavailable'));
  if (canonicalCapi !== expectedCapi || !canonicalCapi.startsWith(`${venvRoot}${sep}`)) fail('OpenVINO capi directory escapes the trusted venv');
  const candidate = resolve(venvRoot, expectedRelative);
  if (relative(venvRoot, candidate) !== expectedRelative) fail(`OpenVINO library path containment failed: ${library.name}`);
  const canonical = await realpath(candidate).catch(() => fail(`OpenVINO library is unavailable: ${library.name}`));
  if (!canonical.startsWith(`${venvRoot}${sep}`) || dirname(canonical) !== canonicalCapi || basename(canonical) !== library.name) fail(`OpenVINO library symlink/path containment failed: ${library.name}`);
  return canonical;
}

export async function resolveTrustedOpenvinoProfilePath(root, workspace, profile, round, source = false) {
  if (typeof workspace !== 'string' || isAbsolute(workspace) || workspace.includes('\\') || normalize(workspace) !== workspace) fail('invalid OpenVINO replay workspace');
  const workspaceParts = workspace.split('/');
  if (workspaceParts.length !== 3 || workspaceParts[0] !== '.evidence' || workspaceParts[1] !== 'openvino' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(workspaceParts[2])) fail('OpenVINO replay workspace must be one direct child of .evidence/openvino');
  if (![1, 2].includes(round)) fail(`invalid OpenVINO profile round: ${round}`);
  const filename = source ? 'ort-profile.raw.json' : 'ort-profile.node-events.json';
  const expectedRelative = `round-${round}/${filename}`;
  const artifact = source ? profile.sourceArtifact : profile;
  if (artifact?.path !== `${OPENVINO_WORKSPACE_TOKEN}${expectedRelative}`) fail(`round ${round} ${source ? 'raw ' : ''}profile path token drift`);

  const repository = await realpath(root);
  const expectedWorkspace = resolve(repository, workspace);
  if (relative(repository, expectedWorkspace) !== workspace) fail('OpenVINO replay workspace containment failed');
  const canonicalWorkspace = await realpath(expectedWorkspace).catch(() => fail('canonical OpenVINO replay workspace is unavailable'));
  if (canonicalWorkspace !== expectedWorkspace || !canonicalWorkspace.startsWith(`${repository}${sep}`)) fail('OpenVINO replay workspace symlink/path escape');
  const candidate = resolve(canonicalWorkspace, expectedRelative);
  if (relative(canonicalWorkspace, candidate) !== expectedRelative) fail(`round ${round} profile containment failed`);
  const [canonicalProfile, metadata] = await Promise.all([
    realpath(candidate).catch(() => fail(`round ${round} ${source ? 'raw ' : ''}profile is unavailable`)),
    lstat(candidate).catch(() => fail(`round ${round} ${source ? 'raw ' : ''}profile is unavailable`)),
  ]);
  if (canonicalProfile !== candidate || !metadata.isFile() || metadata.isSymbolicLink() || !canonicalProfile.startsWith(`${canonicalWorkspace}${sep}`)) fail(`round ${round} profile symlink/path escape`);
  return canonicalProfile;
}

function checkedSpawn(command, args, options, label) {
  const result = spawnSync(command, args, { encoding: 'utf8', ...options });
  if (result.error || result.status !== 0) {
    fail(`${label} failed (${result.status ?? 'spawn'}): ${result.error?.message ?? ''}\n${result.stdout ?? ''}\n${result.stderr ?? ''}`);
  }
  return result;
}

export async function verifyRustSourceIdentity(root) {
  const repository = await realpath(root);
  const identities = [];
  for (const relative of RUST_SOURCES) {
    const path = resolve(repository, relative);
    const canonicalPath = await realpath(path);
    if (canonicalPath !== path) fail(`production Rust source canonical path drift: ${relative}`);
    const bytes = await readFile(path);
    const head = checkedSpawn('git', ['rev-parse', `HEAD:${relative}`], { cwd: repository }, `${relative} HEAD blob`).stdout.trim();
    const worktree = checkedSpawn('git', ['hash-object', '--', relative], { cwd: repository }, `${relative} worktree blob`).stdout.trim();
    if (!/^[0-9a-f]{40,64}$/.test(head) || worktree !== head) fail(`production Rust source differs from HEAD: ${relative}`);
    identities.push({ bytes: bytes.length, canonicalPath: relative, headBlobOid: head, path: relative, sha256: sha256(bytes) });
  }
  return identities;
}

export const FORBIDDEN_RUST_BUILD_ENVIRONMENT = [
  'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
  'CARGO_BUILD_RUSTFLAGS', 'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER',
  'CARGO_TARGET_DIR', 'CARGO_HOME', 'CARGO_CONFIG', 'RUSTC', 'RUSTDOC', 'RUSTUP_TOOLCHAIN',
  'CARGO_BUILD_TARGET', 'CARGO_BUILD_JOBS',
];

function isForbiddenBuildEnvironment(name) {
  return FORBIDDEN_RUST_BUILD_ENVIRONMENT.includes(name)
    || /^CARGO_ALIAS_/i.test(name)
    || /RUSTFLAGS/i.test(name)
    || /RUSTC.*WRAPPER/i.test(name)
    || /^CARGO.*(?:CONFIG|TARGET|WRAPPER)$/i.test(name);
}

function rejectBuildEnvironment(environment) {
  const injected = Object.keys(environment).filter(isForbiddenBuildEnvironment).sort();
  if (injected.length) fail(`forbidden Cargo/Rust build environment: ${injected.join(', ')}`);
}

function stableBuildCommand() {
  return [
    'cargo', 'build', '--offline', '--locked', '--release',
    '--manifest-path', '$SOURCE_MIRROR/evidence/tooling/raw-golden/Cargo.toml',
    '--target-dir', '$CARGO_TARGET_DIR', '--bin', 'rimeflow-raw-golden',
  ];
}

async function verifyRawGoldenInventory(repository) {
  const packageRoot = join(repository, 'evidence/tooling/raw-golden');
  const allowed = new Set(RUST_SOURCES.filter((path) => path.startsWith('evidence/tooling/raw-golden/')));
  const discovered = [];
  async function visit(directory) {
    const handle = await opendir(directory);
    for await (const entry of handle) {
      if (entry.name === 'target' && directory === packageRoot) continue;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else discovered.push(path.slice(repository.length + 1));
    }
  }
  await visit(packageRoot);
  const unexpected = discovered.filter((path) => !allowed.has(path)).sort();
  if (unexpected.length) fail(`unapproved raw-golden source/config/build input: ${unexpected.join(', ')}`);
}

async function copyTrustedRustMirror(repository, mirror, sourceArtifacts) {
  await verifyRawGoldenInventory(repository);
  const copied = [];
  for (const source of sourceArtifacts) {
    const destination = join(mirror, source.path);
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(join(repository, source.path), destination);
    const bytes = await readFile(destination);
    const identity = { bytes: bytes.length, canonicalPath: source.canonicalPath, headBlobOid: source.headBlobOid, path: source.path, sha256: sha256(bytes) };
    if (!same(identity, source)) fail(`trusted Rust mirror copy identity drift: ${source.path}`);
    copied.push(identity);
  }
  // The package imports production postprocess by a fixed relative path. A mirror
  // has no root .cargo config, build.rs, or unlisted source available to Cargo.
  for (const forbidden of ['.cargo', 'build.rs', 'evidence/tooling/raw-golden/build.rs', 'evidence/tooling/raw-golden/.cargo']) {
    if (await lstat(join(mirror, forbidden)).catch(() => null)) fail(`trusted Rust mirror unexpectedly contains ${forbidden}`);
  }
  const manifest = await readFile(join(mirror, 'evidence/tooling/raw-golden/Cargo.toml'), 'utf8');
  if (/^build\s*=/m.test(manifest)) fail('raw-golden Cargo.toml declares an unapproved build script');
  return copied;
}

async function secureCargoEnvironment(temporary, callerEnvironment) {
  rejectBuildEnvironment(callerEnvironment);
  const home = userInfo().homedir;
  const toolchain = join(home, '.rustup', 'toolchains', 'stable-x86_64-unknown-linux-gnu', 'bin');
  const cargoBin = join(toolchain, 'cargo');
  const rustcBin = join(toolchain, 'rustc');
  const [cargo, rustc] = await Promise.all([realpath(cargoBin), realpath(rustcBin)]).catch(() => fail('trusted Cargo/Rustc tools unavailable'));
  if (!cargo.startsWith(`${toolchain}/`) || !rustc.startsWith(`${toolchain}/`)) fail('trusted Cargo/Rustc path outside pinned toolchain');
  const cargoHome = join(temporary, 'cargo-home');
  await mkdir(cargoHome, { recursive: true });
  const registry = join(home, '.cargo', 'registry');
  if (!(await stat(registry).catch(() => null))?.isDirectory()) fail('offline Cargo registry unavailable');
  const handle = await opendir(cargoHome);
  const first = await handle.read();
  await handle.close();
  if (first !== null) fail('temporary CARGO_HOME was not initially empty');
  await symlink(registry, join(cargoHome, 'registry'), 'dir');
  return {
    cargo,
    rustc,
    environment: {
      CARGO_HOME: cargoHome,
      CARGO_INCREMENTAL: '0',
      CARGO_PROFILE_RELEASE_STRIP: 'symbols',
      HOME: home,
      LANG: 'C',
      LC_ALL: 'C',
      PATH: `${toolchain}:/usr/local/bin:/usr/bin:/bin`,
      RUSTUP_HOME: join(home, '.rustup'),
      SOURCE_DATE_EPOCH: '0',
    },
  };
}

function elfIdentity(bytes) {
  if (bytes.length < 20 || bytes[0] !== 0x7f || bytes.subarray(1, 4).toString() !== 'ELF') fail('trusted production Rust runner is not ELF');
  const machine = bytes.readUInt16LE(18);
  return {
    class: bytes[4] === 2 ? 'ELF64' : bytes[4] === 1 ? 'ELF32' : 'unknown',
    endianness: bytes[5] === 1 ? 'little' : bytes[5] === 2 ? 'big' : 'unknown',
    machine: machine === 62 ? 'x86_64' : `elf-machine-${machine}`,
    magic: bytes.subarray(0, 4).toString('hex'),
    osAbi: bytes[7],
  };
}

function runnerIdentity(bytes, sourceArtifacts, build) {
  return {
    build: {
      argv: stableBuildCommand(),
      cargoVersion: build.cargoVersion,
      clearedEnvironmentVariables: build.clearedEnvironmentVariables,
      controlledEnvironment: build.controlledEnvironment,
      exitCode: 0,
      freshTarget: true,
      isolatedCargoHome: true,
      isolatedSourceMirror: true,
      locked: true,
      offline: true,
      releaseStrip: 'symbols',
      rejectedEnvironmentVariables: build.rejectedEnvironmentVariables,
      repositoryRootCargoConfigParticipated: false,
      rustcVersion: build.rustcVersion,
    },
    runner: {
      bytes: bytes.length,
      elf: elfIdentity(bytes),
      logicalPath: 'fresh-target/release/rimeflow-raw-golden',
      sha256: sha256(bytes),
    },
    sourceArtifacts,
  };
}

export async function buildTrustedRustRunner(root, options = {}) {
  const repository = await realpath(root);
  const sourceIdentity = await verifyRustSourceIdentity(repository);
  const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-rust-'));
  const stableView = join(tmpdir(), 'rimeflow-openvino-rust-compiler-view-v1');
  let stableViewCreated = false;
  const cleanup = async () => {
    if (stableViewCreated && await realpath(stableView).catch(() => null) === temporary) await unlink(stableView).catch(() => {});
    await rm(temporary, { recursive: true, force: true });
  };
  try {
    await writeFile(join(temporary, 'owner-pid'), `${process.pid}\n`);
    for (let attempt = 0; !stableViewCreated && attempt < 500; attempt += 1) {
      try {
        await symlink(temporary, stableView, 'dir');
        stableViewCreated = true;
      } catch (error) {
        if (error.code !== 'EEXIST') throw error;
        const destination = await realpath(stableView).catch(() => null);
        const owner = destination ? Number.parseInt(await readFile(join(destination, 'owner-pid'), 'utf8').catch(() => ''), 10) : NaN;
        let active = false;
        if (Number.isInteger(owner)) {
          try { process.kill(owner, 0); active = true; } catch {}
        }
        if (!active && await realpath(stableView).catch(() => null) === destination) await unlink(stableView).catch(() => {});
        else await new Promise((resolveWait) => setTimeout(resolveWait, 20));
      }
    }
    if (!stableViewCreated) fail('another trusted Rust build did not release the stable compiler view');
    const target = join(stableView, 'target');
    const mirror = join(stableView, 'mirror');
    await mkdir(mirror);
    const copiedIdentity = await copyTrustedRustMirror(repository, mirror, sourceIdentity);
    const sourceIdentityAfterCopy = await verifyRustSourceIdentity(repository);
    if (!same(sourceIdentityAfterCopy, sourceIdentity)) fail('production Rust source changed while creating trusted mirror');
    const secure = await secureCargoEnvironment(stableView, options.environment ?? process.env);
    const actualArguments = [
      'build', '--offline', '--locked', '--release', '--manifest-path',
      join(mirror, 'evidence/tooling/raw-golden/Cargo.toml'), '--target-dir', target,
      '--bin', 'rimeflow-raw-golden',
    ];
    const result = spawnSync(
      secure.cargo,
      actualArguments,
      { cwd: mirror, encoding: 'utf8', env: secure.environment },
    );
    if (result.error || result.status !== 0) {
      fail(`trusted production Rust build failed (${result.status ?? 'spawn'}): ${result.error?.message ?? ''}\n${result.stdout ?? ''}\n${result.stderr ?? ''}`);
    }
    const runner = join(target, 'release', process.platform === 'win32' ? 'rimeflow-raw-golden.exe' : 'rimeflow-raw-golden');
    const runnerStat = await stat(runner).catch(() => null);
    if (!runnerStat?.isFile() || (runnerStat.mode & 0o111) === 0) fail('trusted production Rust runner missing or not executable');
    const runnerBytes = await readFile(runner);
    const cargoVersion = checkedSpawn(secure.cargo, ['--version'], { cwd: mirror, env: secure.environment }, 'trusted Cargo version').stdout.trim();
    const rustcVersion = checkedSpawn(secure.rustc, ['--version'], { cwd: mirror, env: secure.environment }, 'trusted Rustc version').stdout.trim();
    const provenance = runnerIdentity(runnerBytes, copiedIdentity, {
      cargoVersion,
      clearedEnvironmentVariables: [...FORBIDDEN_RUST_BUILD_ENVIRONMENT, 'CARGO_ALIAS_*', 'CARGO_*CONFIG/TARGET/WRAPPER*', '*RUSTFLAGS*', '*RUSTC*WRAPPER*'],
      controlledEnvironment: {
        CARGO_HOME: '$FRESH_CARGO_HOME',
        CARGO_INCREMENTAL: '0',
        CARGO_PROFILE_RELEASE_STRIP: 'symbols',
        LANG: 'C',
        LC_ALL: 'C',
        SOURCE_DATE_EPOCH: '0',
      },
      rejectedEnvironmentVariables: [...FORBIDDEN_RUST_BUILD_ENVIRONMENT, 'CARGO_ALIAS_*', 'CARGO_*CONFIG/TARGET/WRAPPER*', '*RUSTFLAGS*', '*RUSTC*WRAPPER*'],
      rustcVersion,
    });
    return {
      cleanup,
      runner,
      provenance,
      sourceIdentity,
      target,
      temporary,
    };
  } catch (error) {
    await cleanup();
    throw error;
  }
}

function verifyRecordedProductionPostprocess(recorded, trusted, label) {
  if (!recorded || typeof recorded !== 'object') fail(`${label} production postprocess provenance missing`);
  if (recorded.implementation !== 'src/postprocess.rs' || recorded.platformSpecificImplementationAdded) fail(`${label} production postprocess ownership`);
  const expected = { ...trusted.provenance, implementation: 'src/postprocess.rs', platformSpecificImplementationAdded: false };
  if (!same(recorded, expected)) fail(`${label} production postprocess provenance drift`);
}

export function assertRecordedProductionPostprocess(recorded, provenance, label = 'record') {
  verifyRecordedProductionPostprocess(recorded, { provenance }, label);
}

function verifyFixtureRunner(recorded, trusted, label) {
  if (!recorded || recorded.exitCode !== 0 || !same(recorded.runner, trusted.provenance.runner)) fail(`${label} fixture runner provenance`);
  if (!Array.isArray(recorded.command) || recorded.command[0] !== trusted.provenance.runner.logicalPath) fail(`${label} fixture runner command`);
}

export function recoverOpenvinoPublication(root, options = {}) {
  const result = spawnSync(
    options.python ?? 'python3',
    [
      'evidence/scripts/openvino_durable_publication.py',
      '--recover', resolve(root, '.evidence/openvino/transaction'),
      '--target', resolve(root, 'evidence/conversions/openvino-ep-manifest.json'),
      '--target', resolve(root, 'evidence/reports/openvino-ep-report.json'),
    ],
    { cwd: root, encoding: 'utf8' },
  );
  if (result.error || result.status !== 0) {
    fail(`durable publication recovery failed (${result.status ?? 'spawn'}): ${result.error?.message ?? ''}\n${result.stdout ?? ''}\n${result.stderr ?? ''}`);
  }
  return JSON.parse(result.stdout);
}

function readFloat32Le(bytes, label) {
  if (bytes.length !== OUTPUT.elementCount * 4) fail(`${label} byte length`);
  const values = new Float64Array(OUTPUT.elementCount);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) values[index] = view.getFloat32(index * 4, true);
  return values;
}

function rawFacts(actualBytes, referenceBytes, tolerances) {
  const actual = readFloat32Le(actualBytes, 'raw');
  const reference = readFloat32Le(referenceBytes, 'reference raw');
  let actualFiniteCount = 0;
  let referenceFiniteCount = 0;
  let allClose = true;
  let differenceSum = 0;
  let maxAbsoluteDifference = -1;
  let maxIndex = -1;
  let nearZeroElementCount = 0;
  let nearZeroMaxAbsoluteDifference = null;
  let nearZeroMismatchCount = 0;
  for (let index = 0; index < OUTPUT.elementCount; index += 1) {
    const candidate = actual[index];
    const expected = reference[index];
    if (Number.isFinite(candidate)) actualFiniteCount += 1;
    if (Number.isFinite(expected)) referenceFiniteCount += 1;
    if (!Number.isFinite(candidate) || !Number.isFinite(expected)) {
      allClose = false;
      continue;
    }
    const difference = Math.abs(candidate - expected);
    const tolerance = tolerances.rawTensorAbsolute + tolerances.rawTensorRelative * Math.abs(expected);
    differenceSum += difference;
    if (difference > maxAbsoluteDifference) {
      maxAbsoluteDifference = difference;
      maxIndex = index;
    }
    if (difference > tolerance) allClose = false;
    if (Math.abs(expected) < 1e-6) {
      nearZeroElementCount += 1;
      nearZeroMaxAbsoluteDifference = nearZeroMaxAbsoluteDifference === null ? difference : Math.max(nearZeroMaxAbsoluteDifference, difference);
      if (difference > tolerance) nearZeroMismatchCount += 1;
    }
  }
  if (actualFiniteCount !== OUTPUT.elementCount || referenceFiniteCount !== OUTPUT.elementCount) allClose = false;
  const attribute = Math.floor(maxIndex / OUTPUT.shape[2]) % OUTPUT.shape[1];
  const anchor = maxIndex % OUTPUT.shape[2];
  const maximumReference = reference[maxIndex];
  return {
    allClose,
    elementCount: actual.length,
    finiteCount: actualFiniteCount,
    maxAbsoluteDifference,
    maxAbsoluteDifferenceLocation: {
      actual: actual[maxIndex],
      anchor,
      attribute,
      flatIndex: maxIndex,
      reference: maximumReference,
      tolerance: tolerances.rawTensorAbsolute + tolerances.rawTensorRelative * Math.abs(maximumReference),
    },
    meanAbsoluteDifference: differenceSum / OUTPUT.elementCount,
    nearZero: {
      elementCount: nearZeroElementCount,
      maxAbsoluteDifference: nearZeroMaxAbsoluteDifference,
      mismatchCount: nearZeroMismatchCount,
      referenceAbsoluteThreshold: 1e-6,
    },
    referenceFiniteCount,
    referenceSha256Float32Le: sha256(referenceBytes),
    sha256Float32Le: sha256(actualBytes),
  };
}

function bboxIou(left, right) {
  const x1 = Math.max(left[0], right[0]);
  const y1 = Math.max(left[1], right[1]);
  const x2 = Math.min(left[2], right[2]);
  const y2 = Math.min(left[3], right[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function decodedFacts(actual, expected, tolerances) {
  const comparisons = [];
  for (let index = 0; index < Math.min(actual.length, expected.length); index += 1) {
    const candidate = actual[index];
    const reference = expected[index];
    const bboxIouValue = bboxIou(candidate.bbox, reference.bbox);
    const bboxMaxAbsoluteDifference = Math.max(...candidate.bbox.map((value, coordinate) => Math.abs(value - reference.bbox[coordinate])));
    const classEqual = candidate.classId === reference.classId;
    const confidenceAbsoluteDifference = Math.abs(candidate.score - reference.score);
    comparisons.push({
      bboxIou: bboxIouValue,
      bboxMaxAbsoluteDifference,
      classEqual,
      confidenceAbsoluteDifference,
      passed: classEqual && confidenceAbsoluteDifference <= tolerances.confidenceAbsolute && bboxIouValue >= tolerances.boxIouMinimum && bboxMaxAbsoluteDifference <= tolerances.decodedBoxAbsolute,
    });
  }
  return {
    actualCount: actual.length,
    comparisons,
    expectedCount: expected.length,
    passed: actual.length === expected.length && comparisons.every((item) => item.passed),
  };
}

function compareNumericStructure(actual, reported, label) {
  if (typeof actual === 'number' || typeof reported === 'number') {
    if (typeof actual !== 'number' || typeof reported !== 'number' || !nearlyEqual(actual, reported)) fail(`${label} numeric mismatch`);
    return;
  }
  if (Array.isArray(actual) || Array.isArray(reported)) {
    if (!Array.isArray(actual) || !Array.isArray(reported) || actual.length !== reported.length) fail(`${label} array mismatch`);
    actual.forEach((value, index) => compareNumericStructure(value, reported[index], `${label}[${index}]`));
    return;
  }
  if (actual && typeof actual === 'object') {
    if (!reported || typeof reported !== 'object' || !same(Object.keys(actual).sort(), Object.keys(reported).sort())) fail(`${label} fields mismatch`);
    for (const [key, value] of Object.entries(actual)) compareNumericStructure(value, reported[key], `${label}.${key}`);
    return;
  }
  if (actual !== reported) fail(`${label} mismatch`);
}

function profileFacts(events) {
  const counts = { OpenVINOExecutionProvider: 0, CPUExecutionProvider: 0, unknown: 0 };
  const unique = { OpenVINOExecutionProvider: new Set(), CPUExecutionProvider: new Set(), unknown: new Set() };
  for (const event of events.filter((item) => item.cat === 'Node')) {
    const provider = event.args?.provider;
    const key = Object.hasOwn(counts, provider) ? provider : 'unknown';
    counts[key] += 1;
    unique[key].add(event.name);
  }
  const uniqueCounts = Object.fromEntries(Object.entries(unique).map(([key, values]) => [key, values.size]));
  let executionPlan = 'unknown';
  if (uniqueCounts.OpenVINOExecutionProvider > 0) executionPlan = uniqueCounts.CPUExecutionProvider > 0 ? 'partitioned' : 'full';
  return { counts, executionPlan, uniqueCounts };
}

function normalizedProfileEvents(nodeEvents) {
  if (!Array.isArray(nodeEvents)) fail('profile nodeEvents are unavailable');
  return nodeEvents.map((event) => ({ args: { provider: event.provider }, cat: 'Node', name: event.name }));
}

function stableProfileBytes(nodeEvents) {
  return Buffer.from(`${JSON.stringify(canonical(normalizedProfileEvents(nodeEvents)), null, 2)}\n`);
}

function validateNormalizedProfile(profile, profileBytes, round) {
  if (profile.normalization !== OPENVINO_PROFILE_NORMALIZATION) fail(`round ${round} profile normalization drift`);
  const expectedBytes = stableProfileBytes(profile.nodeEvents);
  if (!profileBytes.equals(expectedBytes) || profileBytes.length !== profile.bytes || sha256(profileBytes) !== profile.sha256) fail(`round ${round} deterministic profile identity`);
  const events = JSON.parse(profileBytes);
  const facts = profileFacts(events);
  if (!same(facts.counts, profile.executionEventCounts) || !same(facts.uniqueCounts, profile.uniqueNodeCounts) || facts.executionPlan !== profile.executionPlan || facts.uniqueCounts.OpenVINOExecutionProvider < 1) fail(`round ${round} real profile provider counts`);
  return { events, facts };
}

async function resolveRecordedOpenvinoDataPath(root, path) {
  const legacyPrefix = '.evidence/openvino/record-fix03-final/';
  if (typeof path !== 'string' || !path.startsWith(legacyPrefix)) return resolve(root, path);
  const relativePath = path.slice(legacyPrefix.length);
  if (!/^round-[12]\/(?:raw\/[a-z-]+-[12]\.f32le|web-reference\/[a-z-]+\/raw\.f32le)$/.test(relativePath)) fail(`invalid recorded OpenVINO data path: ${path}`);
  const workspace = await realpath(resolve(root, '.evidence/openvino/replay-final')).catch(() => fail('canonical OpenVINO replay workspace is unavailable'));
  const candidate = resolve(workspace, relativePath);
  if (relative(workspace, candidate) !== relativePath) fail(`recorded OpenVINO data containment failed: ${path}`);
  const canonicalPath = await realpath(candidate).catch(() => fail(`replayed OpenVINO data is unavailable: ${relativePath}`));
  if (canonicalPath !== candidate || !canonicalPath.startsWith(`${workspace}${sep}`)) fail(`replayed OpenVINO data symlink/path escape: ${relativePath}`);
  return canonicalPath;
}

async function validateRaw(root, raw, fixture, frozenFixture, tolerances, temporary, rustRunner) {
  const rawPath = await resolveRecordedOpenvinoDataPath(root, raw.raw.path);
  const actualBytes = await readFile(rawPath);
  const referenceArtifact = raw.rawComparison.reference;
  const referencePath = await resolveRecordedOpenvinoDataPath(root, referenceArtifact.path);
  const referenceBytes = await readFile(referencePath);
  const expectedReferenceDigest = frozenFixture.runs[0].rawTensor.sha256Float32Le;
  const expectedMetadata = { dtype: 'float32', elementCount: OUTPUT.elementCount, shape: OUTPUT.shape };
  if (!same({ dtype: raw.raw.dtype, elementCount: raw.raw.elementCount, shape: raw.raw.shape }, expectedMetadata)) fail(`${fixture.id} raw metadata`);
  if (!same({ dtype: referenceArtifact.dtype, elementCount: referenceArtifact.elementCount, shape: referenceArtifact.shape }, expectedMetadata)) fail(`${fixture.id} reference raw metadata`);
  if (actualBytes.length !== raw.raw.bytes || sha256(actualBytes) !== raw.raw.sha256) fail(`${fixture.id} raw file identity`);
  if (referenceBytes.length !== referenceArtifact.bytes || sha256(referenceBytes) !== referenceArtifact.sha256 || referenceArtifact.sha256 !== expectedReferenceDigest) fail(`${fixture.id} reference raw identity`);
  const facts = rawFacts(actualBytes, referenceBytes, tolerances);
  if (facts.referenceFiniteCount !== OUTPUT.elementCount || facts.finiteCount !== OUTPUT.elementCount) fail(`${fixture.id} raw/reference non-finite`);
  const reportedFacts = { ...raw.rawComparison };
  delete reportedFacts.reference;
  delete facts.referenceFiniteCount;
  compareNumericStructure(facts, reportedFacts, `${fixture.id} raw comparison`);
  if (!facts.allClose || facts.nearZero.mismatchCount !== 0) fail(`${fixture.id} raw frozen comparison`);
  const decodedPath = join(temporary, `${fixture.id}-${raw.repeat}.json`);
  const rust = spawnSync(
    rustRunner,
    [rawPath, String(fixture.width), String(fixture.height), decodedPath],
    { cwd: root, encoding: 'utf8' },
  );
  if (rust.status !== 0) fail(`${fixture.id} production Rust decode failed: ${rust.stderr}`);
  const decoded = JSON.parse(await readFile(decodedPath, 'utf8'));
  if (!same(decoded, raw.decoded)) fail(`${fixture.id} report decoded output differs from real Rust harness`);
  const decodedComparison = decodedFacts(decoded, frozenFixture.runs[0].decoded, tolerances);
  compareNumericStructure(decodedComparison, raw.decodedComparison, `${fixture.id} decoded comparison`);
  if (!decodedComparison.passed || raw.passed !== (facts.allClose && decodedComparison.passed)) fail(`${fixture.id} decoded frozen comparison`);
}

export async function validateOpenvinoEvidence(root, manifest, report, frozen, fixtures) {
  const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
  if (modelBytes.length !== 12851098 || sha256(modelBytes) !== MODEL_SHA) fail('canonical ONNX identity');
  if (!manifest.artifact.noConversion || !manifest.artifact.sameCanonicalFileLoaded || manifest.artifact.path !== 'models/yolov8n.onnx' || manifest.artifact.bytes !== modelBytes.length || manifest.artifact.sha256 !== sha256(modelBytes)) fail('no-conversion artifact contract');
  if (!same(report.modelBefore, report.modelAfter) || report.modelBefore.sha256 !== MODEL_SHA || report.modelBefore.bytes !== modelBytes.length || !report.noConversion) fail('model before/after evidence');
  const lockBytes = await readFile(resolve(root, manifest.toolchain.lock.path));
  if (lockBytes.length !== manifest.toolchain.lock.bytes || sha256(lockBytes) !== manifest.toolchain.lock.sha256) fail('toolchain lock identity');
  const lockLines = lockBytes.toString().split('\n').filter((line) => line && !line.startsWith('#') && !line.startsWith('--'));
  if (lockLines.length !== 7 || lockLines.some((line) => !/^[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}$/.test(line))) fail('complete hashed dependency lock');
  if (manifest.toolchain.wheels.length !== 7 || manifest.toolchain.wheels.some((wheel) => !wheel.source.startsWith('https://files.pythonhosted.org/') || !/^[0-9a-f]{64}$/.test(wheel.sha256) || !wheel.license)) fail('wheel source/hash/license metadata');
  if (manifest.runtime.onnxruntimeOpenvino !== '1.24.1' || manifest.runtime.openvino.runtime.buildNumber !== '2025.4.1-0-test' || manifest.runtime.numpy !== '2.5.2' || manifest.runtime.python !== '3.12.3' || manifest.runtime.pip !== '24.0') fail('runtime exact versions');
  if (!manifest.runtime.buildInfo.includes('git-commit-id=b5963e82c8') || manifest.runtime.device !== 'CPU-OPENVINO_CPU' || !manifest.runtime.openvino.availableDevices.includes('CPU') || manifest.runtime.openvino.requestedDevice !== 'CPU') fail('runtime build/device introspection');
  const requiredLibraries = new Set(['libonnxruntime_providers_openvino.so', 'libonnxruntime_providers_shared.so', 'libopenvino.so.2541', 'libopenvino_c.so', 'libopenvino_intel_cpu_plugin.so', 'libopenvino_onnx_frontend.so.2541', 'onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so']);
  if (manifest.runtime.libraries.length !== requiredLibraries.size) fail('loaded library count');
  for (const library of manifest.runtime.libraries) {
    if (!requiredLibraries.delete(library.name) || !library.mappedByProcess) fail(`loaded library declaration: ${library.name}`);
    const path = await resolveTrustedOpenvinoLibraryPath(root, library);
    if (!['1.24.1', '2025.4.1'].includes(library.componentVersion)) fail(`loaded library path/version: ${library.name}`);
    const bytes = await readFile(path);
    if (bytes.length !== library.bytes || sha256(bytes) !== library.sha256) fail(`loaded library identity: ${library.name}`);
  }
  if (requiredLibraries.size) fail('missing loaded library identity');
  if (!same(manifest.ioContract.inputs, [INPUT]) || !same(manifest.ioContract.outputs, [OUTPUT]) || manifest.ioContract.outputLayout !== 'N_ATTRIBUTES_ANCHORS') fail('manifest I/O contract');
  if (manifest.provider.requested !== 'OpenVINOExecutionProvider' || manifest.provider.configured[0] !== 'OpenVINOExecutionProvider' || !manifest.provider.fallbackVisible) fail('provider request/fallback declaration');
  if (report.mode !== 'record' || report.rounds.length !== 2 || report.executionPlan !== 'full' || !same(report.tolerances, frozen.tolerances)) fail('record/tolerance/execution plan');
  if (report.status.state !== 'host-inference-verified' || !report.status.artifactVerified || !report.status.hostInferenceVerified || report.status.supported || report.status.task14Complete || report.status.adapterImplemented || report.status.packagingVerified || report.status.performanceVerified || report.status.targetPlatformClosed) fail('status closure semantics');
  const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-validator-'));
  let trustedRust;
  try {
    trustedRust = await buildTrustedRustRunner(root);
    verifyRecordedProductionPostprocess(report.productionPostprocess, trustedRust, 'record');
    for (const round of report.rounds) {
      if (!round.availableProviders.includes('OpenVINOExecutionProvider') || round.sessionProviders[0] !== 'OpenVINOExecutionProvider' || !same(round.inputs, [INPUT]) || !same(round.outputs, [OUTPUT])) fail(`round ${round.round} provider/I/O`);
      if (round.profile.sourceArtifact !== undefined) fail(`round ${round.round} tracked report must not claim a retained raw profile`);
      if (!Number.isInteger(round.profile.recordedSourceArtifact?.bytes) || round.profile.recordedSourceArtifact.bytes < 1 || !/^[0-9a-f]{64}$/.test(round.profile.recordedSourceArtifact?.sha256 ?? '') || round.profile.recordedSourceArtifact.retained !== false || round.profile.recordedSourceArtifact.path !== undefined) fail(`round ${round.round} historical raw profile identity`);
      const profilePath = await resolveTrustedOpenvinoProfilePath(root, '.evidence/openvino/replay-final', round.profile, round.round);
      const profileBytes = await readFile(profilePath);
      validateNormalizedProfile(round.profile, profileBytes, round.round);
      if (round.fixtures.length !== 5) fail(`round ${round.round} fixture count`);
      for (const item of round.fixtures) {
        const fixture = fixtures.images.find((candidate) => candidate.id === item.id);
        const frozenFixture = frozen.fixtures.find((candidate) => candidate.id === item.id);
        if (!fixture || !frozenFixture || item.runs.length !== 2 || !item.deterministic || item.runs[0].raw.sha256 !== item.runs[1].raw.sha256 || !same(item.runs[0].decoded, item.runs[1].decoded)) fail(`round ${round.round} ${item.id} determinism`);
        for (const run of item.runs) {
          verifyFixtureRunner(run.productionPostprocess, trustedRust, `round ${round.round} ${item.id}/${run.repeat}`);
          await validateRaw(root, run, fixture, frozenFixture, frozen.tolerances, temporary, trustedRust.runner);
        }
      }
    }
    for (let index = 0; index < report.rounds[0].fixtures.length; index += 1) {
      const left = report.rounds[0].fixtures[index];
      const right = report.rounds[1].fixtures[index];
      if (left.id !== right.id || left.runs[0].raw.sha256 !== right.runs[0].raw.sha256 || !same(left.runs[0].decoded, right.runs[0].decoded)) fail(`${left.id} cross-round replay determinism`);
    }
  } finally {
    await trustedRust?.cleanup();
    await rm(temporary, { recursive: true, force: true });
  }
  return { executionPlan: report.executionPlan, fixtureCount: 5, profileOpenvinoNodes: report.rounds[0].profile.uniqueNodeCounts.OpenVINOExecutionProvider, profileCpuNodes: report.rounds[0].profile.uniqueNodeCounts.CPUExecutionProvider };
}

export async function validateOpenvinoReplayEvidence(root, replay, manifest, report, workspace = '.evidence/openvino/replay-final') {
  if (replay.schemaVersion !== 1 || replay.mode !== 'replay' || replay.recordDigest !== report.recordDigest || replay.rounds.length !== 2) fail('ordinary replay identity');
  for (const [key, path] of Object.entries({ manifest: 'evidence/conversions/openvino-ep-manifest.json', report: 'evidence/reports/openvino-ep-report.json' })) {
    const absolute = resolve(root, path);
    const [current, metadata] = await Promise.all([readFile(absolute), stat(absolute, { bigint: true })]);
    const preservation = replay.trackedEvidence[key];
    const currentIdentity = { available: true, bytes: current.length, ctimeNs: Number(metadata.ctimeNs), inode: Number(metadata.ino), mtimeNs: Number(metadata.mtimeNs), path, sha256: sha256(current) };
    if (!preservation?.unchanged || !same(preservation.before, currentIdentity) || !same(preservation.after, currentIdentity)) fail(`ordinary replay changed tracked ${key}`);
  }
  const frozen = JSON.parse(await readFile(resolve(root, 'evidence/golden/web-reference.json'), 'utf8'));
  const fixtures = JSON.parse(await readFile(resolve(root, 'evidence/fixtures/manifest.json'), 'utf8'));
  if (!same(report.tolerances, frozen.tolerances)) fail('ordinary replay frozen tolerance drift');
  const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-openvino-replay-validator-'));
  let trustedRust;
  try {
    trustedRust = await buildTrustedRustRunner(root);
    verifyRecordedProductionPostprocess(report.productionPostprocess, trustedRust, 'ordinary replay record');
    if (!same(replay.productionPostprocess, report.productionPostprocess)) fail('ordinary replay production postprocess provenance drift');
    for (const round of replay.rounds) {
      const [profilePath, sourcePath] = await Promise.all([
        resolveTrustedOpenvinoProfilePath(root, workspace, round.profile, round.round),
        resolveTrustedOpenvinoProfilePath(root, workspace, round.profile, round.round, true),
      ]);
      const [profileBytes, sourceBytes] = await Promise.all([readFile(profilePath), readFile(sourcePath)]);
      const normalized = validateNormalizedProfile(round.profile, profileBytes, round.round);
      if (sourceBytes.length !== round.profile.sourceArtifact.bytes || sha256(sourceBytes) !== round.profile.sourceArtifact.sha256) fail(`ordinary replay round ${round.round} raw profile identity`);
      const sourceEvents = JSON.parse(sourceBytes);
      const sourceFacts = profileFacts(sourceEvents);
      const sourceNodeEvents = sourceEvents.filter((event) => event.cat === 'Node').map((event) => ({ name: event.name, provider: event.args?.provider }));
      if (!same(sourceFacts, normalized.facts) || !same(sourceNodeEvents, round.profile.nodeEvents)) fail(`ordinary replay round ${round.round} raw/normalized profile drift`);
      if (!round.availableProviders.includes('OpenVINOExecutionProvider') || round.sessionProviders[0] !== 'OpenVINOExecutionProvider' || !same(round.inputs, [INPUT]) || !same(round.outputs, [OUTPUT]) || round.fixtures.length !== 5) fail(`ordinary replay round ${round.round} provider/I/O`);
      for (const item of round.fixtures) {
        const fixture = fixtures.images.find((candidate) => candidate.id === item.id);
        const frozenFixture = frozen.fixtures.find((candidate) => candidate.id === item.id);
        if (!fixture || !frozenFixture || item.runs.length !== 2 || !item.deterministic) fail(`ordinary replay round ${round.round} ${item.id} structure`);
        for (const run of item.runs) {
          verifyFixtureRunner(run.productionPostprocess, trustedRust, `ordinary replay round ${round.round} ${item.id}/${run.repeat}`);
          await validateRaw(root, run, fixture, frozenFixture, frozen.tolerances, temporary, trustedRust.runner);
        }
      }
    }
  } finally {
    await trustedRust?.cleanup();
    await rm(temporary, { recursive: true, force: true });
  }
  if (manifest.status.supported || manifest.status.task14Complete) fail('ordinary replay overclaimed completion');
  return { recordDigest: replay.recordDigest, trackedEvidenceUnchanged: true };
}

export function summarizeOpenvinoForConversion(manifest, report) {
  return {
    artifact: manifest.artifact,
    artifactVerified: manifest.status.artifactVerified,
    attempt: {
      availableProviders: report.rounds[0].availableProviders,
      command: report.rounds[0].command,
      executionPlan: report.executionPlan,
      goldenPassed: report.rounds.every((round) => round.fixtures.every((fixture) => fixture.runs.every((run) => run.passed))),
      inferenceExecuted: true,
      profileNodeCounts: report.rounds[0].profile.uniqueNodeCounts,
      sessionProviders: report.rounds[0].sessionProviders,
    },
    conclusion: '官方 onnxruntime-openvino wheel 在 Linux x86_64 host 上实际执行 OpenVINO 图节点并通过五 fixture 冻结 golden；尚无 adapter、性能、打包和任务 1.4 全平台闭环。',
    format: 'onnx',
    hostInferenceVerified: manifest.status.hostInferenceVerified,
    ioChanges: 'none',
    license: manifest.usageScope.licenseAndRedistribution,
    nmsResponsibility: 'operator',
    platform: 'linux-x86_64-openvino',
    quantization: 'none / float32',
    state: manifest.status.state,
    supported: manifest.status.supported,
    task14Complete: manifest.status.task14Complete,
    tool: { name: 'onnxruntime-openvino', openvinoVersion: manifest.runtime.openvino.runtime.buildNumber, version: manifest.runtime.onnxruntimeOpenvino },
  };
}

export { profileFacts };
