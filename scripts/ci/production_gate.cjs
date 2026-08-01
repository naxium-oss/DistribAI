'use strict';
/**
 * Full production gate: boss contracts + pytest slices + Playwright UI.
 * Run: npm run verify:production
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

function resolvePython() {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  const candidates = [
    path.join(ROOT, '.venv312', 'Scripts', 'python.exe'),
    path.join(ROOT, 'venv', 'Scripts', 'python.exe'),
    path.join(ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(ROOT, 'venv', 'bin', 'python3'),
    path.join(ROOT, 'venv', 'bin', 'python'),
    path.join(ROOT, '.venv312', 'bin', 'python'),
    path.join(ROOT, '.venv', 'bin', 'python'),
  ];
  for (const exe of candidates) {
    if (fs.existsSync(exe)) return exe;
  }
  return 'python';
}

function run(label, command, args, options = {}) {
  console.log(`\n[production_gate] ${label}`);
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    cwd: ROOT,
    shell: options.shell === true,
    env: process.env,
  });
  if (result.status !== 0) {
    console.error(`[production_gate] FAILED: ${label}`);
    process.exit(result.status === null ? 1 : result.status);
  }
}

run('boss gate (ruff + unit/security pytest)', process.execPath, [path.join(__dirname, 'boss_gate.cjs')]);
run('full pytest', process.execPath, [path.join(__dirname, 'run_pytest_fast.cjs')]);

if (process.platform === 'win32') {
  run('playwright ui', 'npm', ['run', 'test:ui'], { shell: true });
} else {
  run('playwright ui', 'npm', ['run', 'test:ui']);
}

console.log('\n[production_gate] All production checks passed.');
