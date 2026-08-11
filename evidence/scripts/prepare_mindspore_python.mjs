import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const environment = resolve(root, process.env.RIMEFLOW_MINDSPORE_VENV ?? '.evidence/mindspore/python-venv');
const python = resolve(environment, 'bin/python');
const pip = resolve(environment, 'bin/pip');

function run(program, args, env = process.env) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8', env, stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${program} exit ${result.status}`);
}

run('node', ['evidence/scripts/prepare_coreml_tooling.mjs'], {
  ...process.env,
  RIMEFLOW_COREML_VENV: environment,
});
run(pip, [
  'install', '--require-hashes', '--only-binary=:all:', '--no-deps',
  '-r', resolve(root, 'evidence/tooling/mindspore-python-addons.lock'),
]);
run(pip, ['check']);
run(python, ['-c', [
  'import numpy, onnx, onnxruntime, torch, torchvision, ultralytics',
  'print(numpy.__version__, onnx.__version__, onnxruntime.__version__, torch.__version__, torchvision.__version__, ultralytics.__version__)',
].join('; ')]);
