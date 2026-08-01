/** Path helpers for the Express client (config + dashboard static roots). */
'use strict';

const os = require('os');
const path = require('path');

/**
 * Build absolute paths rooted at the client package directory.
 * @param {string} clientDir Absolute path to `client/`
 */
function createPaths(clientDir) {
    const CONFIG_DIR = path.join(os.homedir(), '.distribai');
    return {
        CONFIG_DIR,
        DESKTOP_CONFIG: path.join(CONFIG_DIR, 'desktop.json'),
        NODE_DATA_FILE: path.join(CONFIG_DIR, 'node_data.json'),
        ORCH_SYNC_FILE: path.join(CONFIG_DIR, 'orch_sync.json'),
        NODE_DATA_SECRET_FILE: path.join(CONFIG_DIR, '.node-data-secret'),
        BENCHMARK_SCRIPT: process.env.BENCHMARK_SCRIPT || path.join(
            clientDir, '..', 'worker', 'src', 'benchmark', 'bench_runner.py'
        ),
        RESULTS_FILE: process.env.RESULTS_FILE || path.join(os.tmpdir(), 'distribai_benchmark_results.json'),
        DASHBOARD_STATIC: process.env.DASHBOARD_STATIC || path.join(clientDir, '../worker/src/dashboard/static'),
        REPO_ROOT: path.join(clientDir, '..'),
    };
}

module.exports = { createPaths };
