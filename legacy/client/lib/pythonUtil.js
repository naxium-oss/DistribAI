/** Locate a usable Python interpreter for spawning worker/bench scripts. */
'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

function candidatePaths() {
    const root = path.join(__dirname, '..', '..');
    const isWin = process.platform === 'win32';
    const locals = [
        path.join(root, '.venv312', isWin ? 'Scripts\\python.exe' : 'bin/python'),
        path.join(root, 'venv', isWin ? 'Scripts\\python.exe' : 'bin/python'),
        path.join(root, '.venv', isWin ? 'Scripts\\python.exe' : 'bin/python'),
    ];
    return locals;
}

/** Prefer PYTHON_BIN, then repo venvs, else probe common launcher names. */
function findPython() {
    if (process.env.PYTHON_BIN) {
        return process.env.PYTHON_BIN;
    }

    for (const local of candidatePaths()) {
        if (fs.existsSync(local)) {
            return local;
        }
    }

    const candidates = process.platform === 'win32'
        ? ['py', 'python', 'python3']
        : ['python3', 'python'];

    for (const candidate of candidates) {
        try {
            const probe = spawnSync(candidate, ['--version'], {
                encoding: 'utf8',
                timeout: 2500,
                windowsHide: true,
            });
            if (!probe.error && probe.status === 0) {
                return candidate;
            }
        } catch (_) {
            /* next */
        }
    }

    return 'python';
}

module.exports = { findPython };
