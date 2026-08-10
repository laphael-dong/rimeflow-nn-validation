import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdir, readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const directory = resolve(root, '.evidence/mindspore');
const archive = resolve(directory, 'mindspore-lite-2.7.0-linux-x64.tar.gz');
const expected = '8bb1097100c9fec12675670ba2d4264a2cd6da3a9be093eb56631d00fc0c455b';
const url = 'https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0/MindSporeLite/lite/release/linux/x86_64/mindspore-lite-2.7.0-linux-x64.tar.gz';
const digest = async () => readFile(archive).then((bytes) => createHash('sha256').update(bytes).digest('hex'), () => null);
await mkdir(directory, { recursive: true });
if (await digest() !== expected) {
  const result = spawnSync('curl', ['--fail', '--location', '--retry', '5', '--output', archive, url], { cwd: root, stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`MindSpore Lite download exit ${result.status}`);
}
if (await digest() !== expected) throw new Error('MindSpore Lite archive SHA-256 不匹配');
const extracted = spawnSync('tar', ['-xzf', archive, '-C', directory], { cwd: root, stdio: 'inherit' });
if (extracted.status !== 0) throw new Error(`MindSpore Lite extract exit ${extracted.status}`);
console.log(JSON.stringify({ version: '2.7.0', archiveSha256: expected, officialDownloadPage: 'https://www.mindspore.cn/lite/docs/en/r2.7.0/use/downloads.html' }));
