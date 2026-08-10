import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from '../tooling/web/node_modules/onnxruntime-web/dist/ort.node.min.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const modelPath = resolve(root, 'models/yolov8n.onnx');
const shaderPath = resolve(root, 'shaders/preprocess.wgsl');
const postprocessPath = resolve(root, 'src/postprocess.rs');
const outPath = resolve(root, 'evidence/model/model-contract.json');

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => JSON.stringify(value, null, 2) + '\n';
const modelBytes = await readFile(modelPath);
const shader = await readFile(shaderPath, 'utf8');
const postprocess = await readFile(postprocessPath, 'utf8');
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
const input = session.inputMetadata[0];
const output = session.outputMetadata[0];
if (!input || !output) throw new Error('model has no input/output metadata');
const sourceSha = sha256(await readFile(postprocessPath));
const shaderRoles = {
  red: shader.includes('dst[0u * hw + idx] = pixel.r;'),
  green: shader.includes('dst[1u * hw + idx] = pixel.g;'),
  blue: shader.includes('dst[2u * hw + idx] = pixel.b;'),
  grayLetterbox: shader.includes('0.447'),
  directRange: shader.includes('pixel.r') && !shader.includes('/ 255.0'),
};
if (!Object.values(shaderRoles).every(Boolean)) throw new Error('preprocess shader contract changed');
if (!postprocess.includes('pub fn decode_yolo_output') || !postprocess.includes('pub fn nms')) {
  throw new Error('postprocess ownership markers missing');
}
const contract = {
  schemaVersion: 1,
  generatedBy: {
    name: 'rimeflow-yolov8n contract extractor',
    version: '1.0.0',
    runtime: 'onnxruntime-web',
    runtimeVersion: ort.env.versions.web,
    executionProvider: 'wasm',
  },
  source: {
    repository: 'https://github.com/caozisheng/rimeflow-yolov8n',
    commit: 'eacbcf00dfc2fba941b494e2955e87fffd707382',
    modelPath: 'models/yolov8n.onnx',
    modelSha256: sha256(modelBytes),
  },
  model: {
    logicalId: 'rimeflow-yolov8n',
    version: 'yolov8n-onnx-20260707',
    task: 'detection',
    input: {
      role: 'image',
      runtimeName: input.name,
      index: 0,
      shape: input.shape,
      dynamicDimensions: [],
      layout: 'NCHW',
      dtype: input.type,
      colorChannels: 'RGB',
      valueRange: [0, 1],
      normalize: { kind: 'none', responsibility: 'preprocess-shader' },
      preprocess: { kind: 'letterbox', size: [640, 640], fill: 0.4470588235294118, responsibility: 'preprocess-shader' },
    },
    output: {
      role: 'detections',
      runtimeName: output.name,
      index: 0,
      shape: output.shape,
      layout: 'CANDIDATE_MAJOR',
      dtype: output.type,
      semantics: { box: 'cxcywh-model-pixels', classScores: '80-coco-scores', nmsFused: false },
      postprocess: { decode: 'operator', threshold: 0.25, nms: 'operator', sourceSha256: sourceSha },
    },
  },
  responsibilities: { preprocessing: 'operator-shader', postprocessing: 'operator-rust', modelGraph: 'raw-boxes-and-class-scores' },
  runtimeMetadata: { inputMetadata: session.inputMetadata, outputMetadata: session.outputMetadata },
  verification: { inputModelSha256: sha256(modelBytes), shaderSha256: sha256(Buffer.from(shader)), postprocessSha256: sourceSha },
};
await writeFile(outPath, stable(contract));
await session.release();
console.log(outPath);
