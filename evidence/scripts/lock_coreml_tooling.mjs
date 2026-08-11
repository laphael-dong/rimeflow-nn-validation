import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const wheelRoot = resolve(root, '.evidence/coreml/lock-wheels');
const lockPath = resolve(root, 'evidence/tooling/coreml-requirements.lock');
const packages = [
  ['attrs', '26.1.0'],
  ['cattrs', '26.1.0'],
  ['certifi', '2026.7.22'],
  ['charset-normalizer', '3.4.9'],
  ['contourpy', '1.3.3'],
  ['coremltools', '9.0'],
  ['cycler', '0.12.1'],
  ['filelock', '3.32.2'],
  ['fonttools', '4.63.0'],
  ['fsspec', '2026.7.0'],
  ['idna', '3.18'],
  ['jinja2', '3.1.6'],
  ['kiwisolver', '1.5.0'],
  ['markupsafe', '3.0.3'],
  ['matplotlib', '3.11.1'],
  ['mpmath', '1.3.0'],
  ['networkx', '3.6.1'],
  ['numpy', '2.3.5'],
  ['nvidia-ml-py', '13.610.43'],
  ['opencv-python', '5.0.0.93'],
  ['packaging', '26.3'],
  ['pillow', '12.3.0'],
  ['polars', '1.43.2'],
  ['polars-runtime-32', '1.43.2'],
  ['protobuf', '7.35.1'],
  ['psutil', '7.2.2'],
  ['pyaml', '26.7.0'],
  ['pyparsing', '3.3.2'],
  ['python-dateutil', '2.9.0.post0'],
  ['pyyaml', '6.0.3'],
  ['requests', '2.34.2'],
  ['setuptools', '84.0.0'],
  ['six', '1.17.0'],
  ['sympy', '1.14.0'],
  ['torch', '2.7.0+cpu'],
  ['torchvision', '0.22.0+cpu'],
  ['tqdm', '4.70.0'],
  ['typing-extensions', '4.16.0'],
  ['ultralytics', '8.4.104'],
  ['ultralytics-thop', '2.1.6'],
  ['urllib3', '2.7.0'],
];

await mkdir(wheelRoot, { recursive: true });
const lines = [
  '# 由 evidence/scripts/lock_coreml_tooling.mjs 生成。',
  '# 适用平台：CPython 3.12、Linux x86_64。',
  '--index-url https://pypi.org/simple',
  '--extra-index-url https://download.pytorch.org/whl/cpu',
];
for (const [name, version] of packages) {
  const packageDir = resolve(wheelRoot, name.replaceAll('.', '-'));
  await mkdir(packageDir, { recursive: true });
  const index = name === 'torch' || name === 'torchvision'
    ? 'https://download.pytorch.org/whl/cpu'
    : 'https://pypi.org/simple';
  const result = spawnSync('python3', [
    '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
    '--index-url', index, '--dest', packageDir, `${name}==${version}`,
  ], { cwd: root, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${name}==${version}: ${result.stderr}`);
  const wheels = (await readdir(packageDir)).filter((entry) => entry.endsWith('.whl'));
  if (wheels.length !== 1) throw new Error(`${name}==${version}: 期望一个 wheel，实际 ${wheels.length}`);
  const bytes = await readFile(resolve(packageDir, wheels[0]));
  const digest = createHash('sha256').update(bytes).digest('hex');
  lines.push(`${name}==${version} --hash=sha256:${digest}`);
  console.log(`${digest}  ${wheels[0]}`);
}
await writeFile(lockPath, `${lines.join('\n')}\n`);
