import { createHash } from 'node:crypto';
import { access, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const FIXTURE_LICENSE_SHA256 = '0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0';
const UPSTREAM_ASSETS_COMMIT = '42ef8a125df038dcca49f6216f446fe9112946c1';
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

export async function isGitTracked(root, path) {
  const result = spawnSync('git', ['ls-files', '--error-unmatch', '--', path], {
    cwd: root,
    encoding: 'utf8',
  });
  return result.status === 0;
}

export async function validateCoverageEvidence(root, matrix) {
  for (const item of matrix.cases) {
    for (const layer of matrix.layers) {
      const evidence = item[layer];
      if (!evidence?.covered) continue;
      if (!evidence.path) throw new Error(`coverage evidence path missing: ${item.id}/${layer}`);
      await access(resolve(root, evidence.path));
      if (evidence.kind === 'test') {
        if (!evidence.testId || !evidence.command) throw new Error(`test evidence metadata missing: ${item.id}/${layer}`);
        if (!(await isGitTracked(root, evidence.path))) throw new Error(`test evidence is not Git tracked: ${evidence.path}`);
      }
    }
  }
}

export async function validateThirdPartyFixtureLicenses(root, fixtures) {
  if (fixtures.license.productPackaging !== 'excluded') throw new Error('fixture product packaging policy must be excluded');
  for (const item of fixtures.images) {
    if (typeof item.source !== 'object') continue;
    const licensePath = item.source.license?.localPath;
    const noticePath = item.source.license?.noticePath;
    if (!licensePath || !noticePath) throw new Error(`local third-party license missing: ${item.id}`);
    for (const path of [licensePath, noticePath]) {
      const bytes = await readFile(resolve(root, path));
      if (bytes.length === 0) throw new Error(`empty third-party license evidence: ${path}`);
      if (!(await isGitTracked(root, path))) throw new Error(`third-party license evidence is not Git tracked: ${path}`);
    }
    const licenseBytes = await readFile(resolve(root, licensePath));
    const licenseText = licenseBytes.toString('utf8');
    if (sha256(licenseBytes) !== FIXTURE_LICENSE_SHA256 || !licenseText.includes('GNU AFFERO GENERAL PUBLIC LICENSE') || !licenseText.includes('Version 3, 19 November 2007')) throw new Error(`third-party license text drift: ${licensePath}`);
    const notice = await readFile(resolve(root, noticePath), 'utf8');
    if (!notice.includes(UPSTREAM_ASSETS_COMMIT) || !notice.includes(item.source.gitBlobSha) || !notice.includes(item.source.sha256)) throw new Error(`third-party NOTICE provenance drift: ${item.id}`);
    if (item.distribution?.scope !== 'test-and-evidence-only' || item.distribution?.rimecutProductPackage !== 'prohibited') {
      throw new Error(`unsafe external fixture distribution policy: ${item.id}`);
    }
  }
}
