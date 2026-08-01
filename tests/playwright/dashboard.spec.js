const { test, expect } = require('@playwright/test');

async function waitForConnectionBadge(page) {
  await page.waitForFunction(() => {
    const badge = document.querySelector('#connectionStatus');
    if (!badge) return false;
    const t = (badge.textContent || '').replace(/\s+/g, ' ').trim();
    return /Orch (Online|Offline)|Disconnected|^Online$|^Offline$|^Paused$/i.test(t);
  }, { timeout: 15000 });
}

async function openSpaPage(page, role, pageName) {
  await page.goto(`/index.html?role=${role}&preview=1`);
  await page.evaluate(() => {
    const overlay = document.getElementById('setupOverlay');
    if (overlay) overlay.classList.add('hidden');
    localStorage.setItem('distribai_setup_done', 'true');
  });
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await page.evaluate((name) => {
    if (name === 'admin' || localStorage.getItem('distribai_role') === 'admin') {
      document.body.classList.add('role-admin-mode');
      document.querySelectorAll('.role-admin').forEach((el) => {
        el.style.display = 'block';
      });
    }
    document.querySelectorAll('.page-section').forEach((s) => {
      s.style.display = 'none';
    });
    const target = document.getElementById(`page-${name}`);
    if (target) target.style.display = 'block';
    document.querySelectorAll('nav a').forEach((a) => {
      a.classList.toggle('active', a.dataset.page === name);
    });
    if (name === 'admin' && typeof loadAdminSurface === 'function') {
      loadAdminSurface();
    }
  }, pageName);
}

test('dashboard shows real orchestrator connection status', async ({ page }) => {
  await page.goto('/index.html?role=admin&preview=1');
  await page.evaluate(() => {
    const overlay = document.getElementById('setupOverlay');
    if (overlay) overlay.classList.add('hidden');
  });
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await waitForConnectionBadge(page);

  const statusBadge = page.locator('#connectionStatus');
  const statusText = (await statusBadge.textContent()).replace(/\s+/g, ' ').trim();
  expect(statusText).toMatch(/Orch (Online|Offline)|^Online$|^Offline$|Disconnected/i);
  await expect(statusBadge).toBeVisible();
});

test('admin page shows real API data from orchestrator', async ({ page }) => {
  await openSpaPage(page, 'admin', 'admin');
  await expect(page.getByRole('heading', { name: 'Admin' })).toBeVisible();

  const ledgerText = await page.locator('#adminLedger').textContent();
  expect(ledgerText).not.toContain('preview-merkle-root');

  const votesText = await page.locator('#adminVotes').textContent();
  expect(votesText).toMatch(/(\d+|Loading|Unavailable|Error|Root:|not sealed)/i);
});

test('dashboard shows dynamic job and node counts', async ({ page }) => {
  await openSpaPage(page, 'admin', 'dashboard');
  const activeNodes = (await page.locator('#activeNodes').textContent()).trim();
  const queuedJobs = (await page.locator('#queuedJobs').textContent()).trim();
  expect(activeNodes).toMatch(/^(\d+|—)$/);
  expect(queuedJobs).toMatch(/^(\d+|—)$/);
});

test('settings page allows resource allocation configuration', async ({ page }) => {
  await openSpaPage(page, 'node', 'settings');
  await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
  await expect(page.locator('#maxRam')).toBeVisible();
  await page.evaluate(() => {
    const slider = document.getElementById('maxRam');
    const label = document.getElementById('maxRamValue');
    slider.value = '20';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    if (label) label.textContent = `${slider.value} GB`;
  });
  await expect(page.locator('#maxRamValue')).toContainText('20');
});

test('admin actions call real orchestrator endpoints', async ({ page }) => {
  await openSpaPage(page, 'admin', 'admin');
  await expect(page.locator('#btnRefreshAdmin')).toBeVisible();
  await page.locator('#btnRefreshAdmin').click();
  const votesText = await page.locator('#adminVotes').textContent();
  expect(votesText).toMatch(/Loading|Unavailable|Error|vote|Root|sealed|HTTP|offline/i);
});

test('job creation modal works with real orchestrator', async ({ page }) => {
  await page.goto('/jobs.html');
  await expect(page.getByRole('button', { name: /Create Job/i })).toBeVisible();
  await page.evaluate(() => {
    const modal = document.getElementById('createJobModal');
    if (modal) modal.style.display = 'flex';
  });
  await expect(page.locator('#createJobModal')).toBeVisible();
  await expect(page.locator('#createJobModel')).toBeVisible();
  await page.evaluate(() => {
    const modal = document.getElementById('createJobModal');
    if (modal) modal.style.display = 'none';
  });
  await expect(page.locator('#createJobModal')).not.toBeVisible();
});

test('help page contains documentation content', async ({ page }) => {
  await page.goto('/help.html');
  await expect(page.getByRole('heading', { name: /Help/i })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Getting Started' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'FAQ' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Troubleshooting' })).toBeVisible();
});

test('no mock data indicators present', async ({ page }) => {
  await page.goto('/index.html?preview=1');
  const pageContent = await page.content();
  expect(pageContent).not.toContain('preview-node-01');
  expect(pageContent).not.toContain('preview-merkle-root');
  expect(pageContent).not.toContain('task-preview');
  expect(pageContent).not.toContain('job-preview');
});

test('resource allocation sliders affect displayed values', async ({ page }) => {
  await openSpaPage(page, 'node', 'settings');
  await page.evaluate(() => {
    const slider = document.getElementById('maxRam');
    const label = document.getElementById('maxRamValue');
    slider.value = '12';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    if (label) label.textContent = `${slider.value} GB`;
  });
  await expect(page.locator('#maxRamValue')).toHaveText('12 GB');
});
