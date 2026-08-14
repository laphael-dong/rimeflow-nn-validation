import { spawnSync } from 'node:child_process';

await import('./fetch_fixture_sources.mjs');
await import('./generate_contract.mjs');
await import('./generate_fixtures.mjs');
await import('./run_web_golden.mjs');
await import('./generate_model_provenance.mjs');
await import('./run_conversion_spikes.mjs');
const conformance = spawnSync('cargo', ['run', '--release', '--offline', '--manifest-path', 'evidence/tooling/preprocess-conformance/Cargo.toml', '--', '.'], { stdio: 'inherit' });
if (conformance.status !== 0) throw new Error(`CPU/WGSL conformance exit ${conformance.status}`);
// Task 1.4 owns the final provider union. It writes the complete conversion closure,
// then finalizes the golden manifest after that closure is present.
await import('./generate_task14_aggregate.mjs');
