import { test, expect } from '@playwright/test';

const pages = [
  '/posts/2026/03-07-prompt-in-llm/',
  '/posts/2026/08-27-rag-engineering-01-inspectable-rag-baseline/',
  '/posts/2026/09-23-rag-engineering-10-security-governance/',
  '/posts/2025/09-29-langchain/',
  '/collections/rag-engineering/',
];

for (const width of [375, 1440]) for (const theme of ['light', 'dark']) {
  test(`blog release at ${width}px in ${theme}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const [index, path] of pages.entries()) {
      await page.goto(path);
      await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
      await expect(page.locator('main').first()).toBeVisible();
      const widthCheck = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
      expect(widthCheck.page <= widthCheck.viewport, path).toBe(true);
      if (index < 3) await expect(page.locator('article.post')).toBeVisible();
    }
  });
}

test('inference walkthrough anchors and PDF reader are usable', async ({ page }) => {
  await page.goto('/posts/2026/03-07-prompt-in-llm/');
  const links = page.locator('.reading-map a');
  await expect(links).toHaveCount(7);
  for (const link of await links.all()) {
    const hash = await link.getAttribute('href');
    expect(await page.locator(hash!).count()).toBe(1);
  }
  await page.locator('.pdf-disclosure summary').click();
  const frame = page.frameLocator('.pdf-embed iframe');
  await expect(frame.locator('#viewerContainer')).toBeVisible();
  await expect(frame.locator('#status')).toBeHidden({ timeout: 20_000 });
});

test('representative blog figures load and stay readable', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 900 });
  await page.goto('/posts/2026/08-27-rag-engineering-01-inspectable-rag-baseline/');
  const diagram = page.locator('.rag-diagram').first();
  await diagram.scrollIntoViewIfNeeded();
  await expect(diagram.locator('.rag-figure-light')).toHaveJSProperty('complete', true);
  expect(await diagram.locator('.rag-figure-light').evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
  await expect(page.locator('.rag-table').first()).toBeVisible();
  await page.goto('/posts/2026/03-07-prompt-in-llm/');
  await expect(page.locator('.reading-map')).toBeVisible();
});

test('RAG article navigation follows the ten-part series', async ({ page }) => {
  await page.goto('/posts/2026/08-27-rag-engineering-01-inspectable-rag-baseline/');
  await expect(page.locator('.post-nav__link--next')).toHaveAttribute('href', '/posts/2026/08-30-rag-engineering-02-chunking-is-a-retrieval-decision/');
  await page.goto('/posts/2026/09-23-rag-engineering-10-security-governance/');
  await expect(page.locator('.post-nav__link').first()).toHaveAttribute('href', '/posts/2026/09-20-rag-engineering-09-rag-evaluation-regression-gates/');
});
