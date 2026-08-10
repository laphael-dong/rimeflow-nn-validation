import { spawnSync } from 'node:child_process';

await import('./fetch_fixture_sources.mjs');
await import('./generate_contract.mjs');
await import('./generate_fixtures.mjs');
await import('./run_web_golden.mjs');
await import('./generate_model_provenance.mjs');
await import('./run_conversion_spikes.mjs');
const conformance = spawnSync('cargo', ['run', '--release', '--offline', '--manifest-path', 'evidence/tooling/preprocess-conformance/Cargo.toml', '--', '.'], { stdio: 'inherit' });
if (conformance.status !== 0) throw new Error(`CPU/WGSL conformance exit ${conformance.status}`);
await import('./finalize_manifest.mjs');
