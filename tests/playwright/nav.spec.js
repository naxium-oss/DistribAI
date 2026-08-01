const { test, expect } = require('@playwright/test');

const NODE_NAV_LABELS = [
  'Dashboard',
  'Jobs',
  'Credits',
  'Settings',
  'Admin',
  'Benchmark',
  'Help',
  'Thanks',
  'Dev',
];

async function expectNodeNav(page) {
  const nav = page.locator('nav#main-nav[role="navigation"]');
  await expect(nav).toBeVisible();
  await expect(nav).toHaveAttribute('aria-label', 'Primary');
  await expect(nav.locator('a')).toHaveCount(NODE_NAV_LABELS.length);
  for (const label of NODE_NAV_LABELS) {
    await expect(nav.getByRole('link', { name: label, exact: true })).toBeVisible();
  }
}

test('contributor pages share the same primary nav', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('distribai_role', 'admin');
  });
  for (const path of ['/dashboard.html', '/admin.html', '/jobs.html']) {
    await page.goto(path);
    await expectNodeNav(page);
  }
});

test('contributor node-prefixed dashboard paths resolve', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('distribai_role', 'admin');
  });
  for (const path of ['/node/dashboard.html', '/node/admin.html']) {
    const response = await page.goto(path);
    expect(response && response.status()).toBe(200);
    await expectNodeNav(page);
  }
});

test('legacy nodes.html redirects to dashboard', async ({ page }) => {
  const response = await page.goto('/nodes.html');
  expect(response && response.status()).toBe(200);
  await expect(page).toHaveURL(/\/dashboard\.html$/);
});

test('orchestrator subpages include node view link in shared nav', async ({ page }) => {
  await page.goto('/orchestrator-jobs.html');
  const nav = page.locator('nav#main-nav[role="navigation"]');
  await expect(nav).toBeVisible();
  await expect(nav.getByRole('link', { name: 'Node View' })).toBeVisible();
  await expect(nav.getByRole('link', { name: 'Help' })).toBeVisible();
});
