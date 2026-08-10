import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv from '../tooling/web/node_modules/ajv/lib/ajv.js';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const readJson = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fail = (message) => { throw new Error(message); };
const contract = await readJson('evidence/model/model-contract.json');
const schema = await readJson('evidence/schemas/model-contract.schema.json');
const ajv = new Ajv({ allErrors: true, strict: false });
if (!ajv.compile(schema)(contract)) fail(`model-contract JSON Schema: ${JSON.stringify(ajv.errors)}`);
if (contract.schemaVersion !== 1) fail('contract schemaVersion');
if (contract.source.modelSha256 !== sha256(await readFile(resolve(root, contract.source.modelPath)))) fail('model sha mismatch');
if (contract.model.input.runtimeName !== contract.runtimeMetadata.inputMetadata[0].name) fail('input metadata mismatch');
if (JSON.stringify(contract.model.input.shape) !== JSON.stringify(contract.runtimeMetadata.inputMetadata[0].shape)) fail('input shape mismatch');
if (contract.model.output.runtimeName !== contract.runtimeMetadata.outputMetadata[0].name) fail('output metadata mismatch');
const fixtures = await readJson('evidence/fixtures/manifest.json');
const scenarios = new Set(fixtures.images.map((item) => item.scenario));
for (const expected of ['无检测', '单目标', '多类别', '重叠框/NMS', '极端宽高比', '边界框']) if (!scenarios.has(expected)) fail(`missing scenario ${expected}`);
for (const item of [...fixtures.images, ...fixtures.rawTensorFixtures]) {
  if (sha256(await readFile(resolve(root, item.path))) !== item.sha256) fail(`fixture sha mismatch: ${item.path}`);
}
if (fixtures.rawTensorFixtures.some((item) => item.sourceImage !== null)) fail('raw fixture must not claim image inference');
const reference = await readJson('evidence/golden/web-reference.json');
if (reference.runtime.name !== 'onnxruntime-web' || reference.runtime.version !== '1.27.0' || reference.runtime.actualExecutionProvider !== 'wasm') fail('runtime/EP mismatch');
if (reference.fixtures.length !== fixtures.images.length) fail('reference fixture count');
for (const item of reference.fixtures) {
  if (item.runs.length < 3) fail(`repeat count: ${item.id}`);
  if (!item.determinism.allRawDigestsEqual || !item.determinism.allDecodedEqual) fail(`non-deterministic: ${item.id}`);
}
for (const value of [reference.tolerances.confidenceAbsolute, reference.tolerances.boxIouMinimum, reference.tolerances.rawTensorAbsolute, reference.tolerances.rawTensorRelative]) if (!Number.isFinite(value)) fail('non-finite tolerance');
const manifest = await readJson('evidence/golden/manifest.json');
for (const item of manifest.artifacts) if (sha256(await readFile(resolve(root, item.path))) !== item.sha256) fail(`artifact sha mismatch: ${item.path}`);
const conversion = await readJson('evidence/conversions/conversion-spikes.json');
if (conversion.spikes.some((item) => item.state === 'supported')) fail('spike must not claim supported');
console.log(JSON.stringify({ ok: true, schemaVersion: 1, checkedArtifacts: manifest.artifacts.length, checkedFixtures: fixtures.images.length + fixtures.rawTensorFixtures.length }));
