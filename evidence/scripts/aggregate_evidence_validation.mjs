import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

export const REQUIRED_SPIKES = [
  ['apple', 'coreml'],
  ['android', 'litert-v2'],
  ['windows', 'windows-ml'],
  ['harmonyos', 'mindspore-lite'],
  ['linux-x86_64-cpu', 'cpuexecutionprovider'],
  ['linux-x86_64-openvino', 'openvinoexecutionprovider'],
  ['linux-x86_64-cuda', 'cudaexecutionprovider'],
  ['linux-x86_64-tensorrt', 'tensorrtexecutionprovider'],
];

const REQUIRED_COMMITS = {
  openvino: '7eba039ef1c55408216f0f54d543f0fdcbf1693b',
  cuda: 'f0a4e7fc9e412d040c2a1bbc6db75def39d1acd7',
  tensorrt: '228956ea9992baa279bf71a57164b383e4823878',
};
const MODEL_SHA = '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad';
const key = (item) => `${item.platform}:${item.provider}`;
const fail = (message) => { throw new Error(message); };

export async function validateTask14Aggregate(root, aggregate, conversion, replay) {
  const expectedKeys = REQUIRED_SPIKES.map(([platform, provider]) => `${platform}:${provider}`).sort();
  const actualKeys = aggregate.spikes.map(key).sort();
  if (new Set(actualKeys).size !== actualKeys.length || JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) fail('required spike provider set is missing, duplicated, or incorrect');
  if (conversion.spikes.length !== REQUIRED_SPIKES.length || new Set(conversion.spikes.map(key)).size !== REQUIRED_SPIKES.length || JSON.stringify(conversion.spikes.map(key).sort()) !== JSON.stringify(expectedKeys)) fail('conversion aggregate provider set drift');
  if (JSON.stringify(aggregate.source.providerCommits) !== JSON.stringify(REQUIRED_COMMITS)) fail('provider source provenance drift');
  if (aggregate.source.canonicalOnnx.path !== 'models/yolov8n.onnx' || aggregate.source.canonicalOnnx.sha256 !== MODEL_SHA) fail('aggregate model identity drift');
  const model = await readFile(resolve(root, aggregate.source.canonicalOnnx.path));
  if (model.length !== aggregate.source.canonicalOnnx.bytes || createHash('sha256').update(model).digest('hex') !== MODEL_SHA) fail('aggregate model byte identity drift');

  const closure = aggregate.closure;
  if (!aggregate.spikes.every((item) => item.spikeClosure === true) && closure.technicalSpikeClosure) fail('technicalSpikeClosure requires every required spike record');
  if (closure.allRequiredSpikesRecorded !== true || closure.technicalSpikeClosure !== true) fail('technical spike closure must be recorded');
  if (closure.publicationVerified !== false || closure.task14Complete !== false || closure.openspecTask1_4Checked !== false || closure.allPlatformsSupported !== false) fail('publication/task/platform closure overclaim');
  if ((aggregate.publication.pushed !== true || aggregate.publication.remoteRefVerified !== true) && closure.task14Complete) fail('task14Complete requires pushed and remote-ref-verified evidence');
  if (aggregate.publication.pushed !== true && closure.openspecTask1_4Checked) fail('OpenSpec task 1.4 cannot be checked before push');
  if (closure.publicationVerified !== true && closure.task14Complete) fail('task14Complete requires publicationVerified');
  if (closure.technicalSpikeClosure && closure.allPlatformsSupported) fail('technical spike closure cannot claim all platforms supported');
  if (aggregate.spikes.some((item) => item.supported !== false)) fail('aggregate must not promote an unsupported spike');

  const byPlatform = Object.fromEntries(aggregate.spikes.map((item) => [item.platform, item]));
  const requireBlockedTruth = (platform) => {
    const item = byPlatform[platform];
    if (item.state !== 'blocked' || item.runtimeExecuted || item.hostInferenceVerified || item.goldenExecuted) fail(`${platform} blocked evidence overclaim`);
  };
  requireBlockedTruth('windows');
  requireBlockedTruth('linux-x86_64-cuda');
  requireBlockedTruth('linux-x86_64-tensorrt');
  if (byPlatform['linux-x86_64-openvino'].state !== 'host-inference-verified' || !byPlatform['linux-x86_64-openvino'].runtimeExecuted || !byPlatform['linux-x86_64-openvino'].hostInferenceVerified || !byPlatform['linux-x86_64-openvino'].goldenExecuted) fail('OpenVINO execution evidence missing');
  if (byPlatform['linux-x86_64-cpu'].state !== 'inference-verified' || !byPlatform['linux-x86_64-cpu'].runtimeExecuted) fail('Linux CPU ORT baseline missing');
  if (!['artifact-spec-verified', 'host-inference-verified', 'inference-verified', 'blocked'].every((state) => aggregate.spikes.some((item) => item.state === state))) fail('aggregate state coverage drift');

  if (JSON.stringify(conversion.closure) !== JSON.stringify(closure) || JSON.stringify(replay.closure) !== JSON.stringify(closure)) fail('shared aggregate closure drift');
  if (replay.task1_4Complete !== false || replay.closure.openspecTask1_4Checked !== false) fail('replay task completion overclaim');
  const replayProviders = replay.steps.filter((item) => item.provider).map((item) => `${item.platform}:${item.provider}`).sort();
  if (new Set(replayProviders).size !== replayProviders.length || !expectedKeys.every((item) => replayProviders.includes(item))) fail('shared replay dropped or duplicated provider evidence');
  if (replay.steps.filter((item) => item.id === 'publication-and-platform-closure').length !== 1) fail('publication closure replay step duplicated or missing');
  return { requiredSpikes: expectedKeys.length, providerReplaySteps: replayProviders.length };
}
