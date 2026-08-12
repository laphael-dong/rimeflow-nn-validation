import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const environment = resolve(root, process.env.RIMEFLOW_OPENVINO_VENV ?? '.evidence/openvino/venv');
const python = resolve(environment, 'bin/python');
const pip = resolve(environment, 'bin/pip');
const lock = resolve(root, 'evidence/tooling/openvino-requirements.lock');
const wheels = resolve(root, process.env.RIMEFLOW_OPENVINO_WHEELS ?? '.evidence/openvino/wheels');
const expectedPython = 'Python 3.12.3';
const expectedPip = 'pip 24.0';

function output(program, args) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${program} exit ${result.status}: ${result.stderr}`);
  return result.stdout.trim();
}

function run(program, args) {
  const result = spawnSync(program, args, { cwd: root, encoding: 'utf8', stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${program} exit ${result.status}`);
}

if (output('python3', ['--version']) !== expectedPython) throw new Error(`需要 ${expectedPython}`);
run('python3', ['-m', 'venv', environment]);
if (output(python, ['--version']) !== expectedPython) throw new Error(`venv 需要 ${expectedPython}`);
if (!output(pip, ['--version']).startsWith(expectedPip)) throw new Error(`venv 需要 ${expectedPip}`);
run(pip, ['install', '--no-index', '--find-links', wheels, '--require-hashes', '--only-binary=:all:', '-r', lock]);
run(pip, ['check']);
run(python, ['-c', [
  'import numpy, onnxruntime as ort',
  "assert ort.__version__ == '1.24.1'",
  "assert numpy.__version__ == '2.5.2'",
  "assert 'OpenVINOExecutionProvider' in ort.get_available_providers()",
  "print(ort.__version__, numpy.__version__, ort.get_available_providers(), ort.get_device())",
].join('; ')]);
