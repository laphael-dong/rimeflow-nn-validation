import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const entries = [];
for (const path of ['models/yolov8n.onnx', 'evidence/model/model-contract.json', 'evidence/fixtures/manifest.json', 'evidence/golden/web-reference.json', 'evidence/conversions/conversion-spikes.json', 'evidence/reports/web-wasm-performance.json']) {
  const bytes = await readFile(resolve(root, path)); entries.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
}
const value = { schemaVersion: 1, sourceCommit: 'eacbcf00dfc2fba941b494e2955e87fffd707382', artifacts: entries };
await writeFile(resolve(root, 'evidence/golden/manifest.json'), JSON.stringify(value, null, 2) + '\n');
console.log(sha256(Buffer.from(JSON.stringify(value, null, 2) + '\n')));
