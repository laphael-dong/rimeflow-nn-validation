import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ortNative from '../tooling/web/node_modules/onnxruntime-node/dist/index.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const modelPath = resolve(root, 'models/yolov8n.onnx');
const modelBytes = await readFile(modelPath);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stable = (value) => JSON.stringify(value, null, 2) + '\n';
function command(program, args) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8', timeout: 120000 });
  return { command: [program, ...args], exitCode: result.status, signal: result.signal, stdout: (result.stdout || '').trim(), stderr: (result.stderr || result.error?.message || '').trim() };
}

const coreml = command('python3', ['-c', "import coremltools as ct; ct.convert('models/yolov8n.onnx', source='onnx')"]);
const litert = command('python3', ['-c', "from ai_edge_litert import converter; converter.convert('models/yolov8n.onnx')"]);
const windowsMl = command('pwsh', ['-NoProfile', '-Command', "$m=[Microsoft.AI.MachineLearning.LearningModel]::LoadFromFilePath((Resolve-Path 'models/yolov8n.onnx')); $m.Close()"]);
const mindspore = command('converter_lite', ['--fmk=ONNX', '--modelFile=models/yolov8n.onnx', '--outputFile=evidence/conversions/yolov8n']);
let linuxOrt;
try {
  const session = await ortNative.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
  linuxOrt = { command: ['onnxruntime-node@1.24.3', 'InferenceSession.create', '--execution-provider=cpu'], exitCode: 0, runtimeVersion: '1.24.3', requestedProvider: 'cpu', actualProvider: 'cpu', inputNames: session.inputNames, outputNames: session.outputNames, inputMetadata: session.inputMetadata, outputMetadata: session.outputMetadata, smoke: 'load-only; fixed-input inference belongs to later adapter conformance task' };
  await session.release();
} catch (error) {
  linuxOrt = { command: ['onnxruntime-node@1.24.3', 'InferenceSession.create', '--execution-provider=cpu'], exitCode: 1, error: String(error) };
}
const report = {
  schemaVersion: 1,
  source: { commit: 'eacbcf00dfc2fba941b494e2955e87fffd707382', modelPath: 'models/yolov8n.onnx', modelSha256: sha256(modelBytes) },
  spikes: [
    { platform: 'apple', format: 'coreml', state: coreml.exitCode === 0 ? 'converted-not-load-verified' : 'blocked', tool: { name: 'coremltools', version: coreml.exitCode === 0 ? 'reported-in-stdout' : 'unavailable' }, attempt: coreml, artifact: null, ioChanges: '无法检查', quantization: '未执行', nmsResponsibility: 'operator', failedOperator: null, license: 'Core ML runtime follows Apple SDK terms; model remains AGPL-3.0', redistribution: '转换器/runner 不可用，禁止发布 artifact', conclusion: '本 Linux 主机不能证明 Core ML 转换或加载。' },
    { platform: 'android', format: 'tflite/litert-compiled-model', state: litert.exitCode === 0 ? 'converted-not-load-verified' : 'blocked', tool: { name: 'LiteRT v2 converter', version: litert.exitCode === 0 ? 'reported-in-stdout' : 'unavailable' }, attempt: litert, artifact: null, ioChanges: '无法检查', quantization: '未执行', nmsResponsibility: 'operator', failedOperator: null, license: 'LiteRT Apache-2.0; model AGPL-3.0', redistribution: '未生成，禁止发布 artifact', conclusion: 'SDK/converter 缺失；Android SDK 存在但无真实设备。' },
    { platform: 'windows', format: 'onnx', state: windowsMl.exitCode === 0 ? 'load-verified' : 'blocked', conversion: '无格式转换，Windows ML 直接加载原 ONNX', tool: { name: 'Windows ML', version: 'unavailable on Linux host' }, attempt: windowsMl, artifact: { path: 'models/yolov8n.onnx', sha256: sha256(modelBytes) }, ioChanges: 'none', quantization: 'none', nmsResponsibility: 'operator', failedOperator: null, license: 'Windows ML follows Windows App SDK terms; model AGPL-3.0', redistribution: '原 ONNX 受 AGPL-3.0 约束', conclusion: '没有 Windows runner，实际加载未验证，不能标记 supported。' },
    { platform: 'harmonyos', format: 'mindir/ms', state: mindspore.exitCode === 0 ? 'converted-not-load-verified' : 'blocked', tool: { name: 'MindSpore Lite converter_lite', version: mindspore.exitCode === 0 ? 'reported-in-stdout' : 'unavailable' }, attempt: mindspore, artifact: null, ioChanges: '无法检查', quantization: '未执行', nmsResponsibility: 'operator', failedOperator: null, license: 'MindSpore Apache-2.0; model AGPL-3.0', redistribution: '未生成，禁止发布 artifact', conclusion: 'converter 与 HarmonyOS runner 缺失。' },
    { platform: 'linux-x86_64', format: 'onnx', state: linuxOrt.exitCode === 0 ? 'load-verified' : 'blocked', tool: { name: 'onnxruntime-node', version: '1.24.3' }, attempt: linuxOrt, artifact: { path: 'models/yolov8n.onnx', sha256: sha256(modelBytes) }, ioChanges: 'none', quantization: 'none', nmsResponsibility: 'operator', failedOperator: null, license: 'ONNX Runtime MIT; model AGPL-3.0', redistribution: '运行时与模型分别遵循 MIT/AGPL-3.0', conclusion: linuxOrt.exitCode === 0 ? 'Linux x86_64 CPU provider 实际加载成功；无 CUDA/TensorRT/OpenVINO，状态仅 build-verified。' : 'Linux ORT 加载失败。' },
  ],
};
await writeFile(resolve(root, 'evidence/conversions/conversion-spikes.json'), stable(report));
console.log(sha256(Buffer.from(stable(report))));
