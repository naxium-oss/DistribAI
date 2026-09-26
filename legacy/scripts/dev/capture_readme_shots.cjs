/**
 * Capture contributor UI shots for README.md gallery.
 *
 * Usage (UI on :3000):
 *   node scripts/dev/capture_readme_shots.cjs
 *
 * Writes PNGs, then compresses to JPEG via Pillow when available:
 *   .venv312/Scripts/python.exe -c "..."  (see compress block at end of run)
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const outDir = path.join('docs', 'assets', 'screenshots');
fs.mkdirSync(outDir, { recursive: true });

const base = process.env.README_SHOT_BASE || 'http://127.0.0.1:3000';

const shots = [
  { name: 'dashboard.png', url: '/dashboard.html', width: 1440, height: 900 },
  { name: 'jobs.png', url: '/jobs.html', width: 1440, height: 900 },
  { name: 'settings.png', url: '/settings.html', width: 1440, height: 900 },
  { name: 'benchmark.png', url: '/benchmark.html', width: 1440, height: 900 },
  { name: 'thanks.png', url: '/thanks.html', width: 1440, height: 900 },
  { name: 'help.png', url: '/help.html', width: 1440, height: 900 },
  { name: 'dashboard-mobile.png', url: '/dashboard.html', width: 390, height: 844 },
  { name: 'nav-mobile.png', url: '/jobs.html', width: 390, height: 844, openNav: true },
];

(async () => {
  const browser = await chromium.launch({ headless: true });
  for (const shot of shots) {
    const context = await browser.newContext({
      viewport: { width: shot.width, height: shot.height },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();
    await page.goto(base + shot.url + '?readme=1', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(800);
    if (shot.openNav) {
      const toggle = page.locator('#navToggle');
      if (await toggle.count()) {
        await toggle.click();
        await page.waitForTimeout(400);
      }
    }
    const dest = path.join(outDir, shot.name);
    await page.screenshot({ path: dest, fullPage: false });
    console.log('wrote', dest, fs.statSync(dest).size);
    await context.close();
  }
  await browser.close();

  const pyCandidates = [
    path.join('.venv312', 'Scripts', 'python.exe'),
    path.join('venv', 'Scripts', 'python.exe'),
    'python',
  ];
  const py = pyCandidates.find((candidate) => {
    if (candidate === 'python') {
      return true;
    }
    return fs.existsSync(candidate);
  });
  const compress = `
from PIL import Image
from pathlib import Path
root = Path(${JSON.stringify(outDir.replace(/\\/g, '/'))})
for p in sorted(root.glob('*.png')):
    im = Image.open(p).convert('RGB')
    max_w = 780 if 'mobile' in p.name or 'nav' in p.name else 1600
    if im.width > max_w:
        h = int(im.height * (max_w / im.width))
        im = im.resize((max_w, h), Image.Resampling.LANCZOS)
    out = p.with_suffix('.jpg')
    im.save(out, 'JPEG', quality=82, optimize=True, progressive=True)
    print(out.name, out.stat().st_size)
    p.unlink()
`;
  const result = spawnSync(py, ['-c', compress], { encoding: 'utf8' });
  if (result.status !== 0) {
    console.warn('JPEG compress skipped (install Pillow in venv):', result.stderr || result.error);
  } else {
    process.stdout.write(result.stdout || '');
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
