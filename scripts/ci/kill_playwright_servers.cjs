'use strict';
/**
 * Safe Playwright teardown — never kills user Chrome/Edge/Chromium by name.
 *
 * Stops only:
 *   1) processes LISTENING on PLAYWRIGHT_PORT (default 3210)
 *   2) node.exe running client/server.js (Playwright webServer)
 *
 * Usage: node scripts/ci/kill_playwright_servers.cjs [--dry-run]
 */
const { execSync } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const port = Number(process.env.PLAYWRIGHT_PORT || 3210);
const dryRun = process.argv.includes('--dry-run');

function log(msg) {
  if (!process.env.DISTRIBAI_QUIET_KILL) {
    console.log(`[kill_playwright_servers] ${msg}`);
  }
}

function unique(nums) {
  return [...new Set(nums.filter((n) => Number.isInteger(n) && n > 0))];
}

function pidsListeningOnPort(portNum) {
  if (process.platform === 'win32') {
    try {
      const out = execSync('netstat -ano -p tcp', { encoding: 'utf8', cwd: ROOT });
      const pids = [];
      const re = new RegExp(`:${portNum}\\s+.*LISTENING\\s+(\\d+)\\s*$`, 'i');
      for (const line of out.split(/\r?\n/)) {
        const m = line.match(re);
        if (m) pids.push(Number(m[1]));
      }
      return unique(pids);
    } catch {
      return [];
    }
  }
  try {
    const out = execSync(`lsof -nP -iTCP:${portNum} -sTCP:LISTEN -t`, { encoding: 'utf8', cwd: ROOT });
    return unique(
      out
        .split(/\r?\n/)
        .map((s) => Number(s.trim()))
        .filter(Boolean)
    );
  } catch {
    return [];
  }
}

function nodeDashboardServerPids() {
  const needle = 'client/server.js';
  if (process.platform === 'win32') {
    try {
      const script = [
        'Get-CimInstance Win32_Process |',
        "Where-Object { $_.Name -eq 'node.exe' -and $_.CommandLine -and ($_.CommandLine -match 'client[\\\\/]server\\.js') } |",
        'Select-Object -ExpandProperty ProcessId',
      ].join(' ');
      const out = execSync(`powershell -NoProfile -Command "${script}"`, {
        encoding: 'utf8',
        cwd: ROOT,
      });
      return unique(
        out
          .split(/\r?\n/)
          .map((s) => Number(s.trim()))
          .filter(Boolean)
      );
    } catch {
      return [];
    }
  }
  try {
    const out = execSync('pgrep -f "node.*client/server\\.js"', { encoding: 'utf8', cwd: ROOT });
    return unique(
      out
        .split(/\r?\n/)
        .map((s) => Number(s.trim()))
        .filter(Boolean)
    );
  } catch {
    return [];
  }
}

function killPid(pid) {
  if (dryRun) {
    log(`would stop pid ${pid}`);
    return;
  }
  try {
    if (process.platform === 'win32') {
      execSync(`taskkill /PID ${pid} /F`, { stdio: 'ignore', cwd: ROOT });
    } else {
      process.kill(pid, 'SIGTERM');
    }
    log(`stopped pid ${pid}`);
  } catch {
    /* already exited */
  }
}

function main() {
  const portPids = pidsListeningOnPort(port);
  const nodePids = nodeDashboardServerPids();
  const targets = unique([...portPids, ...nodePids]);

  if (targets.length === 0) {
    log(`no listeners on port ${port} and no test node client/server.js processes`);
    return;
  }

  for (const pid of targets) {
    const tags = [];
    if (portPids.includes(pid)) tags.push(`port:${port}`);
    if (nodePids.includes(pid)) tags.push('client/server.js');
    log(`target pid ${pid} (${tags.join(', ')})`);
    killPid(pid);
  }
}

main();
