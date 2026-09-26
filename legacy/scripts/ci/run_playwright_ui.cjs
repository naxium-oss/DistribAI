'use strict';
/**
 * Run Playwright UI tests, then safe teardown (port PLAYWRIGHT_PORT + test webServer only).
 */
const { spawnSync } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const killScript = path.join(__dirname, 'kill_playwright_servers.cjs');

function runPlaywright() {
  const useShell = process.platform === 'win32';
  return spawnSync('npx', ['playwright', 'test'], {
    stdio: 'inherit',
    cwd: ROOT,
    shell: useShell,
    env: process.env,
  });
}

function safeCleanup() {
  return spawnSync(process.execPath, [killScript], {
    stdio: 'inherit',
    cwd: ROOT,
    env: process.env,
  });
}

const pw = runPlaywright();
const exitCode = pw.status === null ? 1 : pw.status;
safeCleanup();
process.exit(exitCode);
