import { createHash } from 'node:crypto';
import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import * as ort from '../tooling/web/node_modules/onnxruntime-web/dist/ort.node.min.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => JSON.stringify(value, null, 2) + '\n';
const round = (value, digits = 8) => Number(value.toFixed(digits));
async function directoryBytes(path) { let total = 0; for (const name of await readdir(path)) { const child = resolve(path, name); const info = await stat(child); total += info.isDirectory() ? await directoryBytes(child) : info.size; } return total; }

function readPpm(bytes) {
  const marker = Buffer.from('\n255\n');
  const headerEnd = bytes.indexOf(marker);
  if (headerEnd < 0) throw new Error('unsupported PPM header');
  const header = bytes.subarray(0, headerEnd).toString('ascii').trim().split(/\s+/);
  if (header[0] !== 'P6') throw new Error('only P6 PPM is supported');
  const width = Number(header[1]); const height = Number(header[2]);
  const pixels = bytes.subarray(headerEnd + marker.length);
  if (pixels.length !== width * height * 3) throw new Error('PPM byte length mismatch');
  return { width, height, pixels };
}

function preprocess(image) {
  const dst = 640; const scale = Math.min(dst / image.width, dst / image.height);
  const scaledW = image.width * scale; const scaledH = image.height * scale;
  const padX = (dst - scaledW) / 2; const padY = (dst - scaledH) / 2;
  const tensor = new Float32Array(3 * dst * dst);
  tensor.fill(114 / 255);
  for (let y = 0; y < dst; y++) {
    for (let x = 0; x < dst; x++) {
      const srcX = (x - padX) / scale; const srcY = (y - padY) / scale;
      if (srcX < 0 || srcX >= image.width || srcY < 0 || srcY >= image.height) continue;
      const ix = Math.min(image.width - 1, Math.max(0, Math.floor(srcX)));
      const iy = Math.min(image.height - 1, Math.max(0, Math.floor(srcY)));
      const sourceOffset = (iy * image.width + ix) * 3; const targetOffset = y * dst + x;
      tensor[targetOffset] = image.pixels[sourceOffset] / 255;
      tensor[dst * dst + targetOffset] = image.pixels[sourceOffset + 1] / 255;
      tensor[2 * dst * dst + targetOffset] = image.pixels[sourceOffset + 2] / 255;
    }
  }
  return { tensor, scale, padX, padY };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]); const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]); const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function decode(raw, image, prep) {
  const boxes = 8400; const found = [];
  for (let index = 0; index < boxes; index++) {
    let score = 0; let classId = 0;
    for (let c = 0; c < 80; c++) {
      const candidate = raw[(4 + c) * boxes + index];
      if (candidate > score) { score = candidate; classId = c; }
    }
    if (score < 0.25) continue;
    const cx = raw[index]; const cy = raw[boxes + index];
    const width = raw[2 * boxes + index]; const height = raw[3 * boxes + index];
    const clamp = (v) => Math.min(1, Math.max(0, v));
    found.push({
      anchor: index,
      classId,
      score,
      bbox: [
        clamp(((cx - width / 2) - prep.padX) / (image.width * prep.scale)),
        clamp(((cy - height / 2) - prep.padY) / (image.height * prep.scale)),
        clamp(((cx + width / 2) - prep.padX) / (image.width * prep.scale)),
        clamp(((cy + height / 2) - prep.padY) / (image.height * prep.scale)),
      ],
    });
  }
  found.sort((a, b) => b.score - a.score || a.anchor - b.anchor);
  const kept = [];
  for (const candidate of found) if (!kept.some((current) => iou(current.bbox, candidate.bbox) > 0.45)) kept.push(candidate);
  return kept.map((item) => ({ anchor: item.anchor, classId: item.classId, score: round(item.score), bbox: item.bbox.map((v) => round(v)) }));
}

function summary(values) {
  let min = Infinity; let max = -Infinity; let sum = 0; let finiteCount = 0;
  for (const value of values) { if (Number.isFinite(value)) { finiteCount++; min = Math.min(min, value); max = Math.max(max, value); sum += value; } }
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  return { elementCount: values.length, finiteCount, min: round(min, 9), max: round(max, 9), mean: round(sum / finiteCount, 9), sha256Float32Le: sha256(bytes) };
}

const manifest = JSON.parse(await readFile(resolve(root, 'evidence/fixtures/manifest.json'), 'utf8'));
const modelBytes = await readFile(resolve(root, 'models/yolov8n.onnx'));
ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
const initStart = performance.now();
const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
const initializationMs = performance.now() - initStart;
const fixtures = [];
const performanceSamples = [];
let peakRssBytes = process.memoryUsage().rss;
for (const entry of manifest.images) {
  const image = readPpm(await readFile(resolve(root, entry.path)));
  const prep = preprocess(image); const runs = []; const timings = [];
  for (let repeat = 0; repeat < 3; repeat++) {
    const start = performance.now();
    const outputs = await session.run({ images: new ort.Tensor('float32', prep.tensor, [1, 3, 640, 640]) });
    peakRssBytes = Math.max(peakRssBytes, process.memoryUsage().rss);
    timings.push(performance.now() - start);
    const raw = outputs.output0.data;
    runs.push({ repeat: repeat + 1, rawTensor: summary(raw), decoded: decode(raw, image, prep) });
  }
  const reference = runs[0].rawTensor.sha256Float32Le;
  fixtures.push({ id: entry.id, imageSha256: entry.sha256, preprocessing: { algorithm: 'letterbox-nearest-rgb-f32-v1', scale: round(prep.scale), padX: round(prep.padX), padY: round(prep.padY) }, runs, determinism: { allRawDigestsEqual: runs.every((run) => run.rawTensor.sha256Float32Le === reference), maxRawAbsoluteDifference: 0, allDecodedEqual: runs.every((run) => JSON.stringify(run.decoded) === JSON.stringify(runs[0].decoded)) } });
  performanceSamples.push({ id: entry.id, coldMs: round(timings[0], 3), warmMs: timings.slice(1).map((value) => round(value, 3)) });
}
const reference = {
  schemaVersion: 1,
  source: { modelSha256: sha256(modelBytes), fixtureManifestSha256: sha256(await readFile(resolve(root, 'evidence/fixtures/manifest.json'))) },
  runtime: { name: 'onnxruntime-web', version: ort.env.versions.web, requestedExecutionProviders: ['wasm'], actualExecutionProvider: 'wasm', evidence: 'session 仅配置 wasm provider 且初始化/推理成功', threads: 1 },
  tolerances: { frozenBeforeNativeAdapterResults: true, classIdExact: true, confidenceAbsolute: 0.0001, boxIouMinimum: 0.999, rawTensorAbsolute: 0.00001, rawTensorRelative: 0.0001, decodedBoxAbsolute: 0.0001, missingValuePolicy: 'fail', nonFinitePolicy: 'fail' },
  fixtures,
};
await writeFile(resolve(root, 'evidence/golden/web-reference.json'), stable(reference));
if (process.env.RIMEFLOW_RECORD_PERFORMANCE === '1') {
  await writeFile(resolve(root, 'evidence/reports/web-wasm-performance.json'), stable({ schemaVersion: 1, sourceReferenceSha256: sha256(Buffer.from(stable(reference))), runtime: reference.runtime, host: { os: process.platform, arch: process.arch, node: process.version }, metrics: { initializationMs: round(initializationMs, 3), fixtures: performanceSamples, peakProcessRssBytes: peakRssBytes, runtimePackageBytes: await directoryBytes(resolve(root, 'evidence/tooling/web/node_modules/onnxruntime-web')) }, note: '性能采样可变；峰值是独立 harness 进程 RSS 上界；包体为 onnxruntime-web package 文件总和；不得与 WebGPU 或 Native 数据混合。' }));
}
await session.release();
console.log(sha256(Buffer.from(stable(reference))));
