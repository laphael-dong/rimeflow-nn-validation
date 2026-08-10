import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const wheelDir = resolve(root, '.evidence/python-wheels');
const lockPath = resolve(root, 'evidence/tooling/requirements.lock');
const packages = [
  ['ai-edge-litert', '2.1.6'],
  ['attrs', '25.3.0'],
  ['backports.strenum', '1.3.1'],
  ['cattrs', '24.1.3'],
  ['coremltools', '9.0'],
  ['flatbuffers', '25.2.10'],
  ['ml-dtypes', '0.5.4'],
  ['mpmath', '1.3.0'],
  ['numpy', '2.2.3'],
  ['onnx', '1.22.0'],
  ['onnxruntime', '1.27.0'],
  ['packaging', '24.2'],
  ['protobuf', '5.29.3'],
  ['pyaml', '25.7.0'],
  ['pyyaml', '6.0.2'],
  ['sympy', '1.13.3'],
  ['tqdm', '4.67.1'],
  ['typing-extensions', '4.15.0'],
];

async function fetchRetry(url) {
  let last;
  for (let attempt = 1; attempt <= 10; attempt++) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      last = error;
      await new Promise((resolve) => setTimeout(resolve, attempt * 250));
    }
  }
  throw new Error(`${url}: ${last}`);
}

function compatible(filename) {
  if (!filename.endsWith('.whl') || filename.includes('musllinux')) return false;
  if (filename.endsWith('-py3-none-any.whl') || filename.endsWith('-py2.py3-none-any.whl')) return true;
  return filename.includes('x86_64') && filename.includes('manylinux') && (filename.includes('-cp312-') || filename.includes('-abi3-'));
}

await mkdir(wheelDir, { recursive: true });
const lines = [
  '# 由 evidence/scripts/lock_python_tooling.mjs 生成。',
  '# 适用平台：CPython 3.12、Linux x86_64；安装命令见 evidence/README.md。',
];
for (const [name, version] of packages) {
  const metadata = JSON.parse(await fetchRetry(`https://pypi.org/pypi/${name}/${version}/json`));
  const candidates = metadata.urls.filter((item) => compatible(item.filename)).sort((a, b) => {
    const exact = (item) => item.filename.includes('-cp312-') ? 0 : item.filename.includes('-abi3-') ? 1 : 2;
    return exact(a) - exact(b) || a.filename.localeCompare(b.filename);
  });
  if (candidates.length === 0) throw new Error(`${name}==${version}: 没有 CPython 3.12/Linux x86_64 wheel`);
  const wheel = candidates[0];
  const wheelPath = resolve(wheelDir, wheel.filename);
  let bytes = await readFile(wheelPath).catch(() => null);
  let actual = bytes && createHash('sha256').update(bytes).digest('hex');
  if (actual !== wheel.digests.sha256) {
    const mirrorUrl = wheel.url.replace('https://files.pythonhosted.org/packages/', 'https://pypi.tuna.tsinghua.edu.cn/packages/');
    const downloaded = spawnSync('curl', ['--fail', '--location', '--retry', '10', '--retry-all-errors', '--output', wheelPath, mirrorUrl], { stdio: 'inherit' });
    if (downloaded.status !== 0) throw new Error(`${wheel.filename}: curl exit ${downloaded.status}`);
    bytes = await readFile(wheelPath);
    actual = createHash('sha256').update(bytes).digest('hex');
  }
  if (actual !== wheel.digests.sha256) throw new Error(`${wheel.filename}: SHA-256 不匹配`);
  lines.push(`${name}==${version} --hash=sha256:${actual}`);
  console.log(`${actual}  ${wheel.filename}`);
}
await writeFile(lockPath, `${lines.join('\n')}\n`);
