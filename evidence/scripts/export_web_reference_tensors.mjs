import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from '../tooling/web/node_modules/onnxruntime-web/dist/ort.node.min.mjs';
import { preprocessCanonical, readPpm, tensorDigest } from './preprocess_contract.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const outputRoot = resolve(root, process.argv[2] ?? '.evidence/litert/web-reference');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const manifest = JSON.parse(await readFile(resolve(root, 'evidence/fixtures/manifest.json'), 'utf8'));
const frozen = JSON.parse(await readFile(resolve(root, 'evidence/golden/web-reference.json'), 'utf8'));
const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));

if (sha256(modelBytes) !== frozen.source.modelSha256) throw new Error('规范 ONNX SHA-256 与冻结 Web reference 不一致');

await mkdir(outputRoot, { recursive: true });
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});

const fixtures = [];
for (const entry of manifest.images) {
  const expected = frozen.fixtures.find((item) => item.id === entry.id);
  if (!expected) throw new Error(`${entry.id}: 冻结 Web reference 缺失`);
  const image = readPpm(await readFile(resolve(root, entry.path)));
  const prep = preprocessCanonical(image);
  if (tensorDigest(prep.tensor) !== expected.canonicalInput.sha256Float32Le) {
    throw new Error(`${entry.id}: canonical input digest 与冻结 Web reference 不一致`);
  }
  const result = await session.run({ images: new ort.Tensor('float32', prep.tensor, [1, 3, 640, 640]) });
  const raw = result.output0.data;
  const inputBytes = Buffer.from(prep.tensor.buffer, prep.tensor.byteOffset, prep.tensor.byteLength);
  const rawBytes = Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength);
  const expectedRawSha256 = expected.runs[0].rawTensor.sha256Float32Le;
  if (sha256(rawBytes) !== expectedRawSha256) throw new Error(`${entry.id}: raw tensor digest 与冻结 Web reference 不一致`);
  const fixtureDir = resolve(outputRoot, entry.id);
  await mkdir(fixtureDir, { recursive: true });
  await writeFile(resolve(fixtureDir, 'input.f32le'), inputBytes);
  await writeFile(resolve(fixtureDir, 'raw.f32le'), rawBytes);
  fixtures.push({
    id: entry.id,
    image: { width: image.width, height: image.height },
    input: { bytes: inputBytes.length, sha256: sha256(inputBytes), shape: [1, 3, 640, 640] },
    output: { bytes: rawBytes.length, sha256: sha256(rawBytes), shape: [1, 84, 8400] },
    preprocessing: expected.preprocessing,
  });
}

await session.release();
const result = {
  fixtureCount: fixtures.length,
  fixtures,
  runtime: frozen.runtime,
  sourceReferenceSha256: sha256(await readFile(resolve(root, 'evidence/golden/web-reference.json'))),
};
await writeFile(resolve(outputRoot, 'manifest.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
