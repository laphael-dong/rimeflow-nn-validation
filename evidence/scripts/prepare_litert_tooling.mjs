import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const environment = resolve(root, process.env.RIMEFLOW_LITERT_VENV ?? '.evidence/litert/venv');
const python = resolve(environment, 'bin/python');
const pip = resolve(environment, 'bin/pip');
const expectedPython = 'Python 3.12.3';
const expectedPip = 'pip 24.0';
function run(program, args) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8', stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${program} exit ${result.status}`);
}
function output(program, args) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${program} exit ${result.status}: ${result.stderr}`);
  return result.stdout.trim();
}
if (output('python3', ['--version']) !== expectedPython) throw new Error(`需要 ${expectedPython}`);
run('python3', ['-m', 'venv', environment]);
if (output(python, ['--version']) !== expectedPython) throw new Error(`venv 需要 ${expectedPython}`);
if (!output(pip, ['--version']).startsWith(expectedPip)) throw new Error(`venv 需要 ${expectedPip}`);
run(pip, [
  'install', '--require-hashes', '--only-binary=:all:',
  '-r', resolve(root, 'evidence/tooling/litert-requirements.lock'),
]);
run(pip, ['check']);
if (!output(pip, ['--version']).startsWith(expectedPip)) throw new Error(`安装后 pip 必须保持 ${expectedPip}`);
run(python, ['-c', [
  'import ai_edge_litert, litert_torch, torch, torchvision, ultralytics',
  'print(torch.__version__, torchvision.__version__, ultralytics.__version__, litert_torch.__version__)',
].join('; ')]);
