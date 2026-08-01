const { test, expect } = require('@playwright/test');

/** Node overview shell uses `/dashboard.html`; full SPA uses `/` (see client/server.js). */
test('node dashboard exposes primary landmarks for assistive tech', async ({ page }) => {
  await page.goto('/dashboard.html');

  const main = page.locator('main[role="main"]');
  await expect(main).toBeVisible();

  const nav = page.locator('nav#main-nav[role="navigation"]');
  await expect(nav).toBeVisible();
  await expect(nav).toHaveAttribute('aria-label', 'Primary');

  await expect(main).toHaveAttribute('aria-label', 'Application content');
});

test('orchestrator dashboard exposes primary landmarks for assistive tech', async ({ page }) => {
  await page.goto('/orchestrator.html');

  const main = page.locator('main[role="main"]');
  await expect(main).toBeVisible();

  const nav = page.locator('nav#main-nav[role="navigation"]');
  await expect(nav).toBeVisible();
  await expect(nav).toHaveAttribute('aria-label', 'Primary');

  await expect(main).toHaveAttribute('aria-label', 'Application content');
});
