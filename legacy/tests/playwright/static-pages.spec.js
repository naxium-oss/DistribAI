const { test, expect } = require('@playwright/test');

test('credits page loads primary landmarks', async ({ page }) => {
  await page.goto('/credits.html');
  await expect(page.getByRole('heading', { name: 'Credits' })).toBeVisible();
  await expect(page.locator('#totalBalance')).toBeVisible();
});

test('thanks page presents restrained acknowledgements', async ({ page }) => {
  await page.goto('/thanks.html');

  await expect(page.getByRole('heading', { name: 'Special Thanks' })).toBeVisible();
  await expect(page.getByText('Folding@Home', { exact: true })).toBeVisible();
  await expect(page.getByText('Hivemind', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'To the Community' })).toBeVisible();

  const bodyText = await page.locator('body').innerText();
  expect(bodyText).not.toContain('🧬');
  expect(bodyText).not.toContain('🐝');
  expect(bodyText).not.toContain('💚');
  expect(bodyText).not.toContain('Alone we can do so little');
  expect(bodyText).not.toContain('heartbeat of this network');
});

test('job detail page handles missing id', async ({ page }) => {
  await page.goto('/job.html');
  await expect(page.getByText(/Missing job id/i)).toBeVisible();
});

test('job detail page renders shell with query id', async ({ page }) => {
  await page.goto('/job.html?id=job-smoke-test');
  await expect(page.getByRole('heading', { name: 'Job detail' })).toBeVisible();
  await expect(page.locator('#jobSubtitle')).toContainText('job-smoke-test');
});
