#!/usr/bin/env node

import { resolve } from 'node:path';

import { buildTrustedRustRunner } from './openvino_evidence_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
let trusted;
try {
  trusted = await buildTrustedRustRunner(root);
  process.stdout.write(`${JSON.stringify({ provenance: trusted.provenance, runner: trusted.runner })}\n`);
  await new Promise((resolveInput, rejectInput) => {
    process.stdin.once('end', resolveInput);
    process.stdin.once('error', rejectInput);
    process.stdin.resume();
  });
} finally {
  await trusted?.cleanup();
}
