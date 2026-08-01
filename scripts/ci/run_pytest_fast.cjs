'use strict';
/**
 * Fast pytest gate: one pytest with xdist, no skips (tests/conftest.py), DISTRIBAI_FAST_TEST=1.
 * Wall-time target 20s on a typical dev machine (warn if slower).
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const unitOnly = process.argv.includes('--unit-only');
const TEST_DIRS = unitOnly
  ? ['tests/unit', 'tests/security']
  : ['tests/unit', 'tests/security', 'tests/integration', 'tests/e2e'];

function resolvePython() {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  for (const exe of [
    path.join(ROOT, '.venv312', 'Scripts', 'python.exe'),
    path.join(ROOT, 'venv', 'Scripts', 'python.exe'),
    path.join(ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(ROOT, 'venv', 'bin', 'python3'),
    path.join(ROOT, 'venv', 'bin', 'python'),
    path.join(ROOT, '.venv312', 'bin', 'python'),
    path.join(ROOT, '.venv', 'bin', 'python'),
  ]) {
    if (fs.existsSync(exe)) return exe;
  }
  return 'python';
}

const py = resolvePython();
const hasXdist =
  spawnSync(py, ['-c', 'import xdist'], { cwd: ROOT, encoding: 'utf8' }).status === 0;

function defaultWorkerCount() {
  if (process.env.DISTRIBAI_PYTEST_WORKERS) return process.env.DISTRIBAI_PYTEST_WORKERS;
  // WinError 10055 (socket buffer exhaustion) under heavy xdist on Windows hosts.
  if (process.platform === 'win32') return '2';
  return '4';
}

const args = ['-m', 'pytest', ...TEST_DIRS, '-q', '--tb=line'];
if (hasXdist) {
  args.push('-n', defaultWorkerCount(), '--dist', 'loadfile');
} else {
  console.warn('[run_pytest_fast] pip install pytest-xdist for parallel runs');
}

const env = { ...process.env, DISTRIBAI_FAST_TEST: process.env.DISTRIBAI_FAST_TEST || '1' };
const started = Date.now();
const proc = spawnSync(py, args, { stdio: 'inherit', cwd: ROOT, env });
const elapsed = ((Date.now() - started) / 1000).toFixed(1);
console.log(`[run_pytest_fast] finished in ${elapsed}s (0 skipped enforced)`);
if (proc.status !== 0) process.exit(proc.status === null ? 1 : proc.status);
const strict = process.env.DISTRIBAI_STRICT_TIME === '1';
if (Number(elapsed) > 20) {
  const msg = `[run_pytest_fast] exceeded 20s target (${elapsed}s)`;
  if (strict) {
    console.error(msg);
    process.exit(1);
  }
  console.warn(msg);
}
process.exit(0);
