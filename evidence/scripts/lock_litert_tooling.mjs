import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const wheelRoot = resolve(root, '.evidence/litert/lock-wheels');
const lockPath = resolve(root, 'evidence/tooling/litert-requirements.lock');
const packages = [
  ['absl-py', '2.5.0'],
  ['ai-edge-litert', '2.1.6'],
  ['ai-edge-quantizer', '0.8.0'],
  ['annotated-doc', '0.0.5'],
  ['anyio', '4.14.2'],
  ['backports.strenum', '1.2.8'],
  ['certifi', '2026.7.22'],
  ['charset-normalizer', '3.4.9'],
  ['click', '8.4.2'],
  ['contourpy', '1.3.3'],
  ['cycler', '0.12.1'],
  ['filelock', '3.29.0'],
  ['fire', '0.7.1'],
  ['flatbuffers', '25.12.19'],
  ['fonttools', '4.63.0'],
  ['fsspec', '2026.4.0'],
  ['h11', '0.16.0'],
  ['hf-xet', '1.6.0'],
  ['httpcore', '1.0.9'],
  ['httpx', '0.28.1'],
  ['huggingface-hub', '1.27.0'],
  ['idna', '3.18'],
  ['immutabledict', '4.2.1'],
  ['jax', '0.11.0'],
  ['jaxlib', '0.11.0'],
  ['jaxtyping', '0.3.11'],
  ['jinja2', '3.1.6'],
  ['kagglehub', '1.0.2'],
  ['kagglesdk', '0.1.37'],
  ['kiwisolver', '1.5.0'],
  ['lark', '1.3.1'],
  ['litert-converter', '0.3.1'],
  ['litert-lm-builder', '0.16.0'],
  ['litert-torch', '0.9.3'],
  ['markdown-it-py', '4.2.0'],
  ['markupsafe', '3.0.3'],
  ['matplotlib', '3.11.1'],
  ['mdurl', '0.1.2'],
  ['ml-dtypes', '0.5.4'],
  ['mpmath', '1.3.0'],
  ['multipledispatch', '1.0.0'],
  ['networkx', '3.6.1'],
  ['numpy', '2.4.4'],
  ['nvidia-ml-py', '13.610.43'],
  ['opencv-python', '5.0.0.93'],
  ['opt-einsum', '3.4.0'],
  ['ordered-set', '4.1.0'],
  ['packaging', '26.3'],
  ['pillow', '12.2.0'],
  ['polars', '1.43.2'],
  ['polars-runtime-32', '1.43.2'],
  ['protobuf', '7.35.1'],
  ['psutil', '7.2.2'],
  ['pygments', '2.20.0'],
  ['pyparsing', '3.3.2'],
  ['python-dateutil', '2.9.0.post0'],
  ['pyyaml', '6.0.3'],
  ['regex', '2026.7.19'],
  ['requests', '2.34.2'],
  ['rich', '15.0.0'],
  ['safetensors', '0.8.0'],
  ['scipy', '1.18.0'],
  ['sentencepiece', '0.2.2'],
  ['setuptools', '78.1.0'],
  ['shellingham', '1.5.4'],
  ['six', '1.17.0'],
  ['sympy', '1.14.0'],
  ['tabulate', '0.10.0'],
  ['termcolor', '3.3.0'],
  ['tokenizers', '0.22.2'],
  ['tomli', '2.4.1'],
  ['torch', '2.12.1+cpu'],
  ['torchao', '0.18.0'],
  ['torchvision', '0.27.1+cpu'],
  ['tqdm', '4.70.0'],
  ['transformers', '5.15.0'],
  ['typer', '0.27.1'],
  ['typing-extensions', '4.12.2'],
  ['ultralytics', '8.4.104'],
  ['ultralytics-thop', '2.1.6'],
  ['urllib3', '2.7.0'],
  ['wadler-lindig', '0.1.7'],
  ['xdsl', '0.28.0'],
];

await mkdir(wheelRoot, { recursive: true });
const lines = [
  '# 由 evidence/scripts/lock_litert_tooling.mjs 生成。',
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
