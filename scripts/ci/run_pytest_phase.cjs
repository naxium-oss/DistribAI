'use strict';
/**
 * Run the fast phase-contract pytest subset using the repo venv when present.
 * Respects PYTHON_BIN (full path or command on PATH).
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const PHASE_TESTS = [
  'tests/unit/test_phase1_runtime_contracts.py',
  'tests/unit/test_phase2_live_path_contracts.py',
  'tests/unit/test_phase3_api_parity.py',
  'tests/unit/test_phase4_harness_and_gui.py',
];

function resolvePython() {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }
  const root = path.resolve(__dirname, '..', '..');
  const candidates = [
    path.join(root, '.venv312', 'Scripts', 'python.exe'),
    path.join(root, 'venv', 'Scripts', 'python.exe'),
    path.join(root, '.venv', 'Scripts', 'python.exe'),
    path.join(root, 'venv', 'bin', 'python3'),
    path.join(root, 'venv', 'bin', 'python'),
    path.join(root, '.venv312', 'bin', 'python'),
    path.join(root, '.venv', 'bin', 'python'),
  ];
  for (const exe of candidates) {
    if (fs.existsSync(exe)) return exe;
  }
  return 'python';
}

const py = resolvePython();
const proc = spawnSync(
  py,
  ['-m', 'pytest', ...PHASE_TESTS, '-q', '--tb=line'],
  { stdio: 'inherit', env: process.env }
);
process.exit(proc.status === null ? 1 : proc.status);
