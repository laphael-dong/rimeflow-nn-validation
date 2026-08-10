import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const python = resolve(root, '.evidence/python-tools/bin/python');
function run(program, args) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8', stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${program} exit ${result.status}`);
}
run('python3', ['-m', 'venv', resolve(root, '.evidence/python-tools')]);
run(resolve(root, '.evidence/python-tools/bin/pip'), [
  'install', '--no-index', '--find-links', resolve(root, '.evidence/python-wheels'),
  '--require-hashes', '--no-deps', '--ignore-requires-python',
  '-r', resolve(root, 'evidence/tooling/requirements.lock'),
]);
run(resolve(root, '.evidence/python-tools/bin/pip'), ['check']);
run(python, ['-c', 'import ai_edge_litert, coremltools, onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__, coremltools.__version__)']);
