import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sources = [
  {
    path: 'evidence/fixtures/sources/bus.jpg',
    url: 'https://raw.githubusercontent.com/ultralytics/assets/42ef8a125df038dcca49f6216f446fe9112946c1/im/bus.jpg',
    sha256: 'c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63',
  },
  {
    path: 'evidence/fixtures/sources/ultralytics-dogs.avif',
    url: 'https://raw.githubusercontent.com/ultralytics/assets/42ef8a125df038dcca49f6216f446fe9112946c1/docs/ultralytics-dogs.avif',
    sha256: '051adc223b922b391588ced5594c0868cf4e3944fbdb80a0c00f9ee14abfa15c',
  },
];

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

for (const source of sources) {
  const path = resolve(root, source.path);
  let bytes;
  try {
    bytes = await readFile(path);
  } catch {
    const response = await fetch(source.url);
    if (!response.ok) throw new Error(`${source.url}: HTTP ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, bytes);
  }
  const actual = sha256(bytes);
  if (actual !== source.sha256) throw new Error(`${source.path}: SHA-256 期望 ${source.sha256}，实际 ${actual}`);
  console.log(`${actual}  ${source.path}`);
}
