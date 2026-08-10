import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const entries = [];
const fixtures = JSON.parse(await readFile(resolve(root, 'evidence/fixtures/manifest.json'), 'utf8'));
const paths = [
  'models/yolov8n.onnx',
  'evidence/model/model-contract.json',
  'evidence/fixtures/manifest.json',
  'evidence/fixtures/THIRD_PARTY_NOTICES.md',
  'evidence/fixtures/licenses/ultralytics-assets-AGPL-3.0.txt',
  ...new Set(fixtures.images.flatMap((item) => [item.path, typeof item.source === 'object' ? item.source.path : null]).filter(Boolean)),
  ...fixtures.rawTensorFixtures.map((item) => item.path),
  'evidence/golden/coverage-matrix.json',
  'evidence/golden/web-reference.json',
  'evidence/conversions/conversion-spikes.json',
  'evidence/reports/model-provenance.json',
  'evidence/reports/preprocess-conformance.json',
  'evidence/reports/web-wasm-performance.json',
  'evidence/tooling/requirements.lock',
  'evidence/tooling/raw-golden/src/lib.rs',
  'evidence/tooling/raw-golden/src/main.rs',
];
for (const path of paths) {
  const bytes = await readFile(resolve(root, path)); entries.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
}
const value = { schemaVersion: 1, sourceCommit: 'eacbcf00dfc2fba941b494e2955e87fffd707382', artifacts: entries };
await writeFile(resolve(root, 'evidence/golden/manifest.json'), JSON.stringify(value, null, 2) + '\n');
console.log(sha256(Buffer.from(JSON.stringify(value, null, 2) + '\n')));
