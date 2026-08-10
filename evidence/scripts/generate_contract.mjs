import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from '../tooling/web/node_modules/onnxruntime-web/dist/ort.node.min.js';
import { PREPROCESS_CONTRACT } from './preprocess_contract.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const modelPath = resolve(root, 'models/yolov8n.onnx');
const shaderPath = resolve(root, 'shaders/preprocess.wgsl');
const postprocessPath = resolve(root, 'src/postprocess.rs');
const preprocessContractPath = resolve(root, 'evidence/scripts/preprocess_contract.mjs');
const outPath = resolve(root, 'evidence/model/model-contract.json');

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => JSON.stringify(value, null, 2) + '\n';
const modelBytes = await readFile(modelPath);
const shader = await readFile(shaderPath, 'utf8');
const postprocess = await readFile(postprocessPath, 'utf8');
const preprocessContractSource = await readFile(preprocessContractPath);
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
if (input.shape.length !== 4 || input.shape[1] !== 3) throw new Error(`unsupported input shape: ${JSON.stringify(input.shape)}`);
if (output.shape.length !== 3 || output.shape[1] !== 84) throw new Error(`unsupported YOLO output shape: ${JSON.stringify(output.shape)}`);
const boxChannelReads = ['raw[0 * num_boxes + i]', 'raw[1 * num_boxes + i]', 'raw[2 * num_boxes + i]', 'raw[3 * num_boxes + i]'];
if (!boxChannelReads.every((needle) => postprocess.includes(needle)) || !postprocess.includes('raw[(4 + c) * num_boxes + i]')) {
  throw new Error('postprocess tensor indexing no longer matches [attribute, anchor]');
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
      axes: ['batch', 'channel', 'height', 'width'],
      dynamicDimensions: [],
      layout: 'NCHW',
      dtype: input.type,
      quantization: null,
      color: PREPROCESS_CONTRACT.color,
      valueRange: PREPROCESS_CONTRACT.valueRange,
      normalize: { ...PREPROCESS_CONTRACT.normalize, responsibility: 'preprocess-shader' },
      preprocess: { ...PREPROCESS_CONTRACT, color: undefined, valueRange: undefined, normalize: undefined, responsibility: 'preprocess-shader' },
    },
    output: {
      role: 'detections',
      runtimeName: output.name,
      index: 0,
      shape: output.shape,
      axes: ['batch', 'attribute', 'anchor'],
      layout: 'N_ATTRIBUTES_ANCHORS',
      dtype: output.type,
      quantization: null,
      memoryOrder: `row-major contiguous; offset=batch*${output.shape[1] * output.shape[2]}+attribute*${output.shape[2]}+anchor`,
      semantics: {
        attributes: { box: { indices: [0, 1, 2, 3], order: ['centerX', 'centerY', 'width', 'height'] }, classScores: { startIndex: 4, count: output.shape[1] - 4, activation: 'already-applied' } },
        anchorDimension: { axis: 2, count: output.shape[2] },
        boxEncoding: { format: 'center-x-center-y-width-height', coordinateSpace: 'letterboxed-model-pixels' },
        nmsFused: false,
      },
      postprocess: { decode: 'operator', threshold: 0.25, nms: 'operator', sourceSha256: sourceSha },
    },
  },
  responsibilities: { preprocessing: 'operator-shader', postprocessing: 'operator-rust', modelGraph: 'raw-boxes-and-class-scores' },
  runtimeMetadata: { inputMetadata: session.inputMetadata, outputMetadata: session.outputMetadata },
  verification: { inputModelSha256: sha256(modelBytes), shaderSha256: sha256(Buffer.from(shader)), postprocessSha256: sourceSha, preprocessContractSha256: sha256(preprocessContractSource) },
};
await writeFile(outPath, stable(contract));
await session.release();
console.log(outPath);
