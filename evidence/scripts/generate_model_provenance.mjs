import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const git = (args) => {
  const result = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr);
  return result.stdout.trim();
};
const modelPath = resolve(root, 'models/yolov8n.onnx');
const python = resolve(root, '.evidence/python-tools/bin/python');
const probe = spawnSync(python, [resolve(root, 'evidence/scripts/probe_python_converters.py'), modelPath], { cwd: root, encoding: 'utf8' });
if (probe.status !== 0) throw new Error(`Python metadata probe failed:\n${probe.stdout}\n${probe.stderr}`);
const toolProbe = JSON.parse(probe.stdout.trim().split('\n').at(-1));
const modelBytes = await readFile(modelPath);
const licenseBytes = await readFile(resolve(root, 'LICENSE'));
const [firstRepositoryCommit, firstRepositoryCommitDate] = git(['log', '--diff-filter=A', '--follow', '-1', '--format=%H|%aI', '--', 'models/yolov8n.onnx']).split('|');
const originalReadme = git(['show', `${firstRepositoryCommit}:README.md`]);
const exportBlock = originalReadme.match(/```bash\n([\s\S]*?)\n```/)?.[1].trim().split('\n').join(' && ');
if (!exportBlock) throw new Error('首次模型提交的 README 没有导出命令');
const ptHistory = git(['log', '--all', '--format=%H', '--', '*.pt']);
const report = {
  schemaVersion: 1,
  model: {
    path: 'models/yolov8n.onnx',
    sha256: sha256(modelBytes),
    firstRepositoryCommit,
    firstRepositoryCommitDate,
    embeddedMetadata: toolProbe.onnxMetadata,
  },
  declaredExportRecipe: {
    source: 'README.md in firstRepositoryCommit',
    command: exportBlock,
    reproducible: false,
    reason: '命令没有锁定 ultralytics 或 yolov8n.pt digest；仓库历史没有提交 .pt、训练配置或导出日志。',
  },
  originalTrainingArtifact: {
    path: null,
    version: null,
    sha256: null,
    sourceUrl: null,
    state: 'unverifiable',
    gitPtArtifactCommits: ptHistory ? ptHistory.split('\n') : [],
  },
  licensing: {
    repositoryCode: { declaredLicense: 'MIT', licensePath: 'LICENSE', licenseSha256: sha256(licenseBytes) },
    fixtureFiles: { state: 'per-file', manifestPath: 'evidence/fixtures/manifest.json', note: '外部 Ultralytics assets 与脚本生成 CC0 fixture 逐文件记录；不由仓库 MIT 统一覆盖。' },
    originalWeights: { declaredByOnnxMetadata: toolProbe.onnxMetadata.license ?? null, enterpriseLicenseEvidence: null, authorizationState: 'unverified' },
    onnxWeights: { licenseState: 'embedded metadata declares AGPL-3.0; exact original artifact and authorization are unresolved' },
    conversionArtifacts: { licenseState: 'inherit unresolved model-weight obligations plus converter/runtime terms', redistributionAllowed: false },
    rimecutPackageRedistribution: { allowed: false, reason: '没有可验证的原始权重来源、准确 .pt SHA 或 Ultralytics Enterprise 授权依据。' },
  },
  decision: {
    task14: 'blocked',
    task1: 'blocked',
    publication: 'prohibited',
    reason: '仓库 MIT 声明不能覆盖模型权重；在授权链闭环前禁止发布或分发 ONNX 及其转换产物。',
  },
  tooling: { python: toolProbe.python, onnx: toolProbe.onnxVersion },
};
await writeFile(resolve(root, 'evidence/reports/model-provenance.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(sha256(Buffer.from(`${JSON.stringify(report, null, 2)}\n`)));
