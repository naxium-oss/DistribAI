// Playwright browser coverage for the local dashboard shell.
const { defineConfig, devices } = require('@playwright/test');

const port = Number(process.env.PLAYWRIGHT_PORT || 3210);

module.exports = defineConfig({
  testDir: './tests/playwright',
  // Single worker: one shared webServer (client/server.js) is not safe under parallel load on Windows.
  workers: 1,
  timeout: 30_000,
  retries: 1,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure'
  },
  webServer: {
    command: `node client/server.js`,
    url: `http://127.0.0.1:${port}/api/status`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: {
      ...process.env,
      PORT: String(port),
      AUTO_START_ORCH: '0',
      PYTHON_BIN: process.env.PYTHON_BIN || 'python'
    }
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ]
});
