import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);

function getArg(flag, fallback) {
  const idx = args.indexOf(flag);
  if (idx === -1 || idx === args.length - 1) return fallback;
  return args[idx + 1];
}

const baseUrl = getArg('--base', 'http://127.0.0.1:4325');
const outDir = getArg('--out', 'analysis/mobile-check');
const label = getArg('--label', 'run');

const viewports = [
  { name: '375x812', width: 375, height: 812 },
  { name: '390x844', width: 390, height: 844 },
  { name: '430x932', width: 430, height: 932 },
  { name: '768x1024', width: 768, height: 1024 },
];

const routes = [
  '/',
  '/archive/',
  '/posts/2025/03-29-llm-usage/',
  '/tags/data-engineer/',
  '/categories/llm/',
];

mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const results = [];

for (const viewport of viewports) {
  const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height } });
  const page = await context.newPage();

  for (const route of routes) {
    const url = new URL(route, baseUrl).toString();
    await page.goto(url, { waitUntil: 'networkidle' });

    const diagnostics = await page.evaluate(() => {
      const overflowX = document.documentElement.scrollWidth > window.innerWidth + 1;
      const hasSearchButton = Boolean(document.querySelector('button[data-open-modal]'));
      const headerLinkLabels = Array.from(document.querySelectorAll('nav.header-links a'))
        .map((el) => el.textContent?.trim())
        .filter(Boolean);
      const footerLinks = document.querySelectorAll('footer a').length;

      return {
        overflowX,
        hasSearchButton,
        headerLinkLabels,
        footerLinks,
      };
    });

    const safeRoute = route.replaceAll('/', '_').replace(/^_+|_+$/g, '') || 'root';
    const screenshotPath = join(outDir, `${label}-${viewport.name}-${safeRoute}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });

    results.push({
      viewport: viewport.name,
      route,
      url,
      screenshotPath,
      ...diagnostics,
    });
  }

  await context.close();
}

await browser.close();

const summary = {
  label,
  baseUrl,
  generatedAt: new Date().toISOString(),
  totalChecks: results.length,
  overflowCount: results.filter((r) => r.overflowX).length,
  results,
};

writeFileSync(join(outDir, `${label}-summary.json`), JSON.stringify(summary, null, 2));
console.log(`Saved ${summary.totalChecks} checks to ${outDir}`);
console.log(`Horizontal overflow cases: ${summary.overflowCount}`);
