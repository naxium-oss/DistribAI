'use strict';
/**
 * "Boss gate": phase-contract pytest + Ruff on first-party Python trees.
 * Run after substantive changes until green (e.g. `npm run verify:boss`).
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ROOT = legacy/ (this surface's own root since the 2026-09-24 restructure).
const ROOT = path.resolve(__dirname, '..', '..');

function resolvePython() {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }
  const candidates = [
    path.join(ROOT, '.venv312', 'Scripts', 'python.exe'),
    path.join(ROOT, 'venv', 'Scripts', 'python.exe'),
    path.join(ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(ROOT, 'venv', 'bin', 'python3'),
    path.join(ROOT, 'venv', 'bin', 'python'),
    path.join(ROOT, '.venv312', 'bin', 'python'),
    path.join(ROOT, '.venv', 'bin', 'python'),
    // The shared venv stayed at the REPO root after the 2026-09-24 restructure.
    path.join(ROOT, '..', '.venv', 'bin', 'python'),
    path.join(ROOT, '..', 'venv', 'bin', 'python'),
  ];
  for (const exe of candidates) {
    if (fs.existsSync(exe)) return exe;
  }
  return 'python';
}

const py = resolvePython();

const ruffArgs = [
  '-m',
  'ruff',
  'check',
  // 2026-09-24 restructure: this whole surface now lives under legacy/ (its
  // own root). The C++ port under tools/cpp_port is the primary codebase and
  // has its own gates (make port-check).
  'services_python',
  'worker',
  'tests',
  'tools',
  'scripts',
  '--exclude',
  'worker/src/distribai_proto',
];

const ruff = spawnSync(py, ruffArgs, { stdio: 'inherit', cwd: ROOT, env: process.env });
if (ruff.status !== 0) {
  console.error('[boss_gate] ruff failed');
  process.exit(ruff.status === null ? 1 : ruff.status);
}

// The shared venv stayed at the REPO root after the 2026-09-24 restructure.
const fast = spawnSync(process.execPath, [path.join(__dirname, 'run_pytest_fast.cjs'), '--unit-only'], {
  stdio: 'inherit',
  cwd: ROOT,
  env: {
    ...process.env,
    DISTRIBAI_FAST_TEST: process.env.DISTRIBAI_FAST_TEST || '1',
  },
});

process.exit(fast.status === null ? 1 : fast.status);
