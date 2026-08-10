import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const imageDir = resolve(root, 'evidence/fixtures/images');
const rawDir = resolve(root, 'evidence/fixtures/raw');
await mkdir(imageDir, { recursive: true });
await mkdir(rawDir, { recursive: true });
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const json = (value) => JSON.stringify(value, null, 2) + '\n';

const fixtures = [
  { id: 'no-detection', scenario: '无检测', width: 640, height: 640, draw: () => [18, 18, 18] },
  { id: 'single-target', scenario: '单目标', width: 640, height: 640, draw: (x, y) => x > 270 && x < 370 && y > 90 && y < 590 ? [220, 220, 220] : [40, 80, 110] },
  { id: 'multi-class', scenario: '多类别', width: 640, height: 360, draw: (x, y) => y > 210 && x > 50 && x < 350 ? [210, 45, 45] : x > 450 && y > 80 ? [230, 230, 210] : [70, 120, 80] },
  { id: 'overlap-nms', scenario: '重叠框/NMS', width: 640, height: 640, draw: (x, y) => x > 180 && x < 470 && y > 150 && y < 520 ? [205, 205, 205] : x > 290 && x < 560 && y > 230 && y < 600 ? [160, 160, 210] : [35, 45, 60] },
  { id: 'extreme-aspect', scenario: '极端宽高比', width: 1280, height: 128, draw: (x, y) => x > 900 && x < 1170 && y > 15 && y < 115 ? [245, 190, 40] : [30, 95, 130] },
  { id: 'boundary-box', scenario: '边界框', width: 640, height: 640, draw: (x, y) => x < 150 && y > 170 && y < 610 ? [235, 235, 225] : [80, 45, 95] },
];

const manifestImages = [];
for (const fixture of fixtures) {
  const header = Buffer.from(`P6\n${fixture.width} ${fixture.height}\n255\n`, 'ascii');
  const pixels = Buffer.alloc(fixture.width * fixture.height * 3);
  for (let y = 0; y < fixture.height; y++) {
    for (let x = 0; x < fixture.width; x++) {
      const color = fixture.draw(x, y);
      const offset = (y * fixture.width + x) * 3;
      pixels[offset] = color[0]; pixels[offset + 1] = color[1]; pixels[offset + 2] = color[2];
    }
  }
  const bytes = Buffer.concat([header, pixels]);
  const path = `evidence/fixtures/images/${fixture.id}.ppm`;
  await writeFile(resolve(root, path), bytes);
  manifestImages.push({ id: fixture.id, scenario: fixture.scenario, path, width: fixture.width, height: fixture.height, sha256: sha256(bytes), license: 'CC0-1.0', source: '由本仓库确定性脚本生成，不含人物或私人媒体' });
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
  generatedBy: { name: 'generate_fixtures.mjs', version: '1.0.0' },
  license: { spdx: 'CC0-1.0', privacy: '无人物、无私人媒体、无外部素材' },
  images: manifestImages,
  rawTensorFixtures: rawManifest,
};
await writeFile(resolve(root, 'evidence/fixtures/manifest.json'), json(manifest));
console.log(sha256(await readFile(resolve(root, 'evidence/fixtures/manifest.json'))));
