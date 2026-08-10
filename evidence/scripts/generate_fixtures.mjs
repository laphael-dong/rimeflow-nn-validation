import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from '../tooling/web/node_modules/sharp/lib/index.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const imageDir = resolve(root, 'evidence/fixtures/images');
const rawDir = resolve(root, 'evidence/fixtures/raw');
await mkdir(imageDir, { recursive: true });
await mkdir(rawDir, { recursive: true });
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const json = (value) => JSON.stringify(value, null, 2) + '\n';

const upstream = {
  repository: 'https://github.com/ultralytics/assets',
  commit: '42ef8a125df038dcca49f6216f446fe9112946c1',
  license: { spdx: 'AGPL-3.0-only', url: 'https://github.com/ultralytics/assets/blob/42ef8a125df038dcca49f6216f446fe9112946c1/LICENSE' },
};
const sourceBus = { path: 'evidence/fixtures/sources/bus.jpg', upstreamPath: 'im/bus.jpg', gitBlobSha: '40eaaf5c330d0c498fbe1dcacf9bb8bf566797fe', sha256: 'c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63' };
const sourceDogs = { path: 'evidence/fixtures/sources/ultralytics-dogs.avif', upstreamPath: 'docs/ultralytics-dogs.avif', gitBlobSha: '22b83c2fe27ce174e6b8df69803adf684119986f', sha256: '051adc223b922b391588ced5594c0868cf4e3944fbdb80a0c00f9ee14abfa15c' };
for (const source of [sourceBus, sourceDogs]) {
  const actual = sha256(await readFile(resolve(root, source.path)));
  if (actual !== source.sha256) throw new Error(`${source.path}: 请先运行 fetch_fixture_sources.mjs；SHA-256 实际为 ${actual}`);
}

const rgb = async (pipeline) => {
  const { data, info } = await pipeline.removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) throw new Error(`期望 RGB 三通道，实际 ${info.channels}`);
  return { width: info.width, height: info.height, pixels: data };
};
const dogCrop = () => sharp(resolve(root, sourceDogs.path), { failOn: 'error' }).extract({ left: 650, top: 150, width: 768, height: 900 });
const images = [
  {
    id: 'no-detection', scenario: '无检测', image: { width: 640, height: 640, pixels: Buffer.alloc(640 * 640 * 3, 18) },
    license: { spdx: 'CC0-1.0', source: '由本仓库确定性脚本生成，不含人物、私人媒体或外部素材' },
    transformation: 'Buffer.alloc(640*640*3,18)',
    coverageExpectation: { detectionCount: { minimum: 0, maximum: 0 } },
  },
  {
    id: 'single-target', scenario: '单目标', image: await rgb(dogCrop()), source: sourceDogs,
    license: upstream.license, transformation: 'sharp@0.34.4 extract(left=650,top=150,width=768,height=900), removeAlpha, toColourspace(srgb), raw RGB',
    coverageExpectation: { detectionCount: { minimum: 1, maximum: 1 }, requiredClassIds: [16] },
  },
  {
    id: 'multi-class', scenario: '多类别', image: await rgb(sharp(resolve(root, sourceBus.path), { failOn: 'error' })), source: sourceBus,
    license: upstream.license, transformation: 'sharp@0.34.4 decode, removeAlpha, toColourspace(srgb), raw RGB',
    coverageExpectation: { detectionCount: { minimum: 2 }, minimumDistinctClassIds: 2, requiredClassIds: [0, 5] },
  },
  {
    id: 'boundary-box', scenario: '边界框', image: await rgb(sharp(resolve(root, sourceBus.path), { failOn: 'error' }).extract({ left: 0, top: 0, width: 809, height: 1080 })), source: sourceBus,
    license: upstream.license, transformation: 'sharp@0.34.4 extract(left=0,top=0,width=809,height=1080), removeAlpha, toColourspace(srgb), raw RGB',
    coverageExpectation: { detectionCount: { minimum: 1 }, boundaryDistanceMaximum: 0.01 },
  },
  {
    id: 'extreme-aspect', scenario: '极端宽高比', image: await rgb(dogCrop().resize({ width: 109, height: 128, fit: 'fill' }).extend({ top: 0, bottom: 0, left: 585, right: 586, background: { r: 114, g: 114, b: 114 } })), source: sourceDogs,
    license: upstream.license, transformation: 'sharp@0.34.4 extract(650,150,768,900), resize(109x128,fill), extend(left=585,right=586,rgb=114), raw RGB',
    coverageExpectation: { detectionCount: { minimum: 1 }, sourceAspectRatioMinimum: 8 },
  },
];

const manifestImages = [];
for (const fixture of images) {
  const { width, height, pixels } = fixture.image;
  const header = Buffer.from(`P6\n${width} ${height}\n255\n`, 'ascii');
  const bytes = Buffer.concat([header, pixels]);
  const path = `evidence/fixtures/images/${fixture.id}.ppm`;
  await writeFile(resolve(root, path), bytes);
  manifestImages.push({ id: fixture.id, scenario: fixture.scenario, path, width, height, sha256: sha256(bytes), license: fixture.license, source: fixture.source ? { ...upstream, ...fixture.source } : fixture.license.source, transformation: fixture.transformation, coverageExpectation: fixture.coverageExpectation });
}

const rawCases = [
  { id: 'no-detection', scenario: '无检测', sourceImage: null, geometry: { width: 640, height: 640 }, anchors: [] },
  { id: 'single-target', scenario: '单目标', sourceImage: null, geometry: { width: 640, height: 640 }, anchors: [{ anchor: 0, cx: 320, cy: 320, width: 120, height: 240, classId: 0, score: 0.91 }] },
  { id: 'multi-class', scenario: '多类别', sourceImage: null, geometry: { width: 640, height: 360 }, anchors: [{ anchor: 1, cx: 180, cy: 340, width: 220, height: 120, classId: 2, score: 0.88 }, { anchor: 2, cx: 470, cy: 280, width: 90, height: 260, classId: 0, score: 0.86 }] },
  { id: 'overlap-nms', scenario: '重叠框/NMS', sourceImage: null, geometry: { width: 640, height: 640 }, anchors: [{ anchor: 3, cx: 320, cy: 320, width: 250, height: 260, classId: 0, score: 0.92 }, { anchor: 4, cx: 330, cy: 325, width: 245, height: 250, classId: 0, score: 0.84 }] },
  { id: 'extreme-aspect', scenario: '极端宽高比', sourceImage: null, geometry: { width: 1280, height: 128 }, anchors: [{ anchor: 5, cx: 510, cy: 320, width: 180, height: 52, classId: 7, score: 0.79 }] },
  { id: 'boundary-box', scenario: '边界框', sourceImage: null, geometry: { width: 640, height: 640 }, anchors: [{ anchor: 6, cx: 20, cy: 360, width: 100, height: 300, classId: 0, score: 0.81 }] },
];
const iou = (a, b) => { const x1 = Math.max(a[0], b[0]); const y1 = Math.max(a[1], b[1]); const x2 = Math.min(a[2], b[2]); const y2 = Math.min(a[3], b[3]); const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter; return union <= 0 ? 0 : inter / union; };
const expected = (item) => { const scale = Math.min(640 / item.geometry.width, 640 / item.geometry.height); const padX = (640 - item.geometry.width * scale) / 2; const padY = (640 - item.geometry.height * scale) / 2; const clamp = (v) => Math.min(1, Math.max(0, v)); const decoded = item.anchors.map((a) => ({ anchor: a.anchor, classId: a.classId, score: a.score, bbox: [clamp((a.cx - a.width / 2 - padX) / (item.geometry.width * scale)), clamp((a.cy - a.height / 2 - padY) / (item.geometry.height * scale)), clamp((a.cx + a.width / 2 - padX) / (item.geometry.width * scale)), clamp((a.cy + a.height / 2 - padY) / (item.geometry.height * scale))].map((v) => Number(v.toFixed(8))) })).sort((a, b) => b.score - a.score); const kept = []; for (const candidate of decoded) if (!kept.some((current) => iou(current.bbox, candidate.bbox) > 0.45)) kept.push(candidate); return { decodeThreshold: 0.25, nmsIouThreshold: 0.45, decoded, nms: kept }; };
const rawManifest = [];
for (const item of rawCases) {
  const value = { schemaVersion: 1, kind: 'manually-constructed-raw-tensor', modelShape: [1, 84, 8400], layout: 'channels-first-over-anchors', sourceImage: item.sourceImage, scenario: item.scenario, geometry: item.geometry, anchors: item.anchors, expectedPostprocess: expected(item), note: '此 fixture 只验证 decode/NMS，不声明来自图片推理。' };
  const bytes = Buffer.from(json(value));
  const path = `evidence/fixtures/raw/${item.id}.json`;
  await writeFile(resolve(root, path), bytes);
  rawManifest.push({ id: item.id, scenario: item.scenario, path, sha256: sha256(bytes), kind: 'raw-tensor', sourceImage: null, license: 'CC0-1.0' });
}
const manifest = {
  schemaVersion: 1,
  generatedBy: { name: 'generate_fixtures.mjs', version: '2.0.0', imageLibrary: 'sharp@0.34.4', lockfile: 'evidence/tooling/web/bun.lock' },
  license: { policy: '逐文件记录；仓库 MIT 不覆盖外部 fixture', privacy: '仅使用固定公开仓库素材；不使用素材/ 或私人媒体' },
  images: manifestImages,
  rawTensorFixtures: rawManifest,
};
await writeFile(resolve(root, 'evidence/fixtures/manifest.json'), json(manifest));
const coverageMatrix = {
  schemaVersion: 1,
  generatedBy: { name: 'generate_fixtures.mjs', version: '2.0.0' },
  layers: ['modelInference', 'preprocessing', 'decode', 'nms'],
  cases: [
    ...manifestImages.map((item) => ({ id: item.id, sourceKind: 'real-image-inference', modelInference: { covered: true, evidence: 'evidence/golden/web-reference.json' }, preprocessing: { covered: true, evidence: 'evidence/reports/preprocess-conformance.json' }, decode: { covered: true, evidence: 'evidence/golden/web-reference.json' }, nms: { covered: false, reason: '该图片不用于证明重叠框抑制；NMS 由独立 raw tensor + 生产 Rust 测试覆盖' } })),
    { id: 'overlap-nms', sourceKind: 'manually-constructed-raw-tensor', modelInference: { covered: false, reason: '人工 raw tensor 不来自图片推理' }, preprocessing: { covered: false, reason: '人工 raw tensor 绕过图片预处理' }, decode: { covered: true, evidence: 'tests/task1_raw_golden.rs' }, nms: { covered: true, evidence: 'tests/task1_raw_golden.rs' } },
  ],
};
await writeFile(resolve(root, 'evidence/golden/coverage-matrix.json'), json(coverageMatrix));
console.log(sha256(await readFile(resolve(root, 'evidence/fixtures/manifest.json'))));
