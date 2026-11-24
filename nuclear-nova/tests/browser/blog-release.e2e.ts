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
      await page.evaluate(value => { window.theme?.setTheme(value as 'light' | 'dark'); }, theme);
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

test('article contents group deep headings and navigate with the keyboard', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/posts/2020/batch-normalization/');
  const outline = page.getByRole('navigation', { name: 'On this page' });
  const branch = outline.getByRole('treeitem', { name: 'Independent and identically distributed (IID)', exact: true });
  const leaf = outline.getByRole('treeitem', { name: 'Whitening', exact: true });
  await expect(branch).toHaveAttribute('aria-level', '2');
  await expect(leaf).toHaveAttribute('aria-level', '3');
  await expect(branch.getByRole('treeitem', { name: 'Whitening' })).toBeVisible();
  await branch.press('ArrowLeft');
  await expect(branch).toHaveAttribute('aria-expanded', 'false');
  await expect(leaf).toBeHidden();
  await branch.press('ArrowRight');
  await leaf.press('Enter');
  await expect(page).toHaveURL(/#whitening$/);
  await expect(leaf.getByRole('link')).toHaveAttribute('aria-current', 'location');

  await page.setViewportSize({ width: 390, height: 900 });
  const disclosure = page.locator('.post-mobile-contents details');
  await disclosure.locator('summary').click();
  await expect(disclosure.getByRole('treeitem', { name: 'Whitening', exact: true })).toBeVisible();
});

test('left sidebar collapses and navigates posts by year', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/posts/2026/03-07-prompt-in-llm/');
  const sidebar = page.locator('.blog-primary');
  const collapse = sidebar.getByRole('button', { name: 'Collapse posts sidebar' });
  const expand = sidebar.getByRole('button', { name: 'Expand posts sidebar' });
  const nav = sidebar.getByRole('navigation', { name: 'Posts by year' });
  const article = page.locator('article.post');
  await expect(nav).toBeVisible();
  const expanded = (await sidebar.boundingBox())!;
  const articleStart = (await article.boundingBox())!.x;
  expect(expanded.x).toBeLessThan(articleStart);
  await expect(nav.locator('[aria-current="page"]')).toHaveAttribute('href', '/posts/2026/03-07-prompt-in-llm/');

  await expect(collapse).toHaveAttribute('aria-expanded', 'true');
  await collapse.press('Enter');
  await expect(expand).toBeFocused();
  await expect(expand).toHaveAttribute('aria-expanded', 'false');
  await expect(expand).toHaveText('Posts');
  await expect(nav).toBeHidden();
  await expect.poll(async () => {
    const { width, height } = (await sidebar.boundingBox())!;
    return [width, height];
  }).toEqual([84, 40]);
  const collapsed = (await sidebar.boundingBox())!;
  expect(collapsed.x).toBe(expanded.x);
  expect((await article.boundingBox())!.x).toBeLessThan(articleStart);
  await page.reload();
  await expect(nav).toBeHidden();
  await expand.press('Enter');
  await expect(collapse).toBeFocused();
  const older = nav.locator('.post-year').filter({ has: page.getByText('2025', { exact: true }) });
  await expect(older).not.toHaveAttribute('open', '');
  await older.locator('summary').press('Enter');
  const search = sidebar.getByRole('searchbox', { name: 'Find a post' });
  await search.fill('2024 autogen');
  await expect(nav.getByRole('link', { name: 'AutoGen HttpClient' })).toBeVisible();
  await expect(older).toBeHidden();
  await search.fill('no-such-post-42');
  await expect(sidebar.locator('.blog-posts-search').getByRole('status')).toHaveText('No matching posts.');
  await search.clear();
  await expect(older).toHaveAttribute('open', '');
  await older.locator('a[href="/posts/2025/09-29-langchain/"]').click();
  await expect(page).toHaveURL(/\/posts\/2025\/09-29-langchain\/$/);
  await expect(nav.locator('[aria-current="page"]')).toBeVisible();

  await page.setViewportSize({ width: 390, height: 900 });
  await expect(nav).toBeHidden();
  await expect(collapse).toBeHidden();
  await sidebar.locator('.blog-posts > summary').press('Enter');
  await expect(nav.locator('[aria-current="page"]')).toBeVisible();
});

test('sidebar post previews reveal truncated titles on hover and keyboard focus', async ({ page }) => {
  await page.goto('/posts/2026/03-07-prompt-in-llm/');
  const nav = page.getByRole('navigation', { name: 'Posts by year' });
  const link = nav.locator('[aria-current="page"]');
  const title = (await link.textContent())!.trim();
  const card = page.locator('.sidebar-post-preview');
  const hydratedSidebar = page.locator('astro-island:not([ssr]):has(#blog-posts-nav)');

  await expect(hydratedSidebar).toHaveCount(1);
  expect(await link.locator('span').evaluate(span => span.scrollWidth > span.clientWidth)).toBe(true);
  await expect(link).toHaveAccessibleName(title);
  await link.hover();
  await expect(card.locator('strong')).toHaveText(title);
  await expect(card.locator('time')).toHaveAttribute('datetime', '2026-03-07');
  await card.hover();
  await expect(card).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(card).toBeHidden();

  await page.mouse.move(0, 0);
  await page.reload();
  await expect(hydratedSidebar).toHaveCount(1);
  const search = page.getByRole('searchbox', { name: 'Find a post' });
  await search.fill('From Prompt to Response');
  await search.press('Tab');
  await page.keyboard.press('Tab');
  await expect(link).toBeFocused();
  await expect(card).toBeVisible();
});

test('code tokens have distinct syntax colors in both Butter themes', async ({ page }) => {
  await page.goto('/posts/2026/08-27-rag-engineering-01-inspectable-rag-baseline/');
  const json = page.locator('pre[data-language="json"]').first();
  const key = json.locator('code span[style*="--shiki-light:"]').filter({ hasText: /chunk_id/ }).first();
  const value = json.locator('code span[style*="--shiki-light:"]').filter({ hasText: /support-v2:p1:0/ }).first();
  const python = page.locator('pre[data-language="python"]').first();
  const keyword = python.locator('code span[style*="--shiki-light:"]').filter({ hasText: /^assert$/ }).first();

  await expect(key).toHaveCSS('color', 'rgb(22, 90, 158)');
  await expect(value).toHaveCSS('color', 'rgb(7, 106, 67)');
  await expect(keyword).toHaveCSS('color', 'rgb(116, 49, 165)');
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(key).toHaveCSS('color', 'rgb(142, 197, 255)');
  await expect(value).toHaveCSS('color', 'rgb(155, 212, 156)');
  await expect(keyword).toHaveCSS('color', 'rgb(216, 173, 255)');
});

test('inline code stands out in both Butter themes', async ({ page }) => {
  await page.goto('/posts/2024/how-to-execute-python-module/');
  const inlineCode = page.locator('.post-content p > code').filter({ hasText: /^runpy$/ });
  await expect(inlineCode).toHaveCSS('color', 'rgb(68, 60, 0)');
  await expect(inlineCode).toHaveCSS('box-shadow', /inset/);
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(inlineCode).toHaveCSS('color', 'rgb(253, 238, 140)');
  await expect(inlineCode).toHaveCSS('box-shadow', /inset/);
});

test('paragraph emphasis uses distinct Butter accents in each theme', async ({ page }) => {
  await page.goto('/posts/2026/09-14-rag-engineering-07-query-planning-safe-structured-retrieval/');
  const prose = page.locator('.post-content');
  const defaultBold = prose.locator('p strong').filter({ hasText: /^illustrative plan$/ });
  const process = prose.locator('p strong').filter({ hasText: /^illustrative intermediate representation$/ });
  const data = prose.locator('p strong').filter({ hasText: /^SQL example$/ });
  await process.evaluate((term) => term.setAttribute('data-emphasis', 'process'));
  await data.evaluate((term) => term.setAttribute('data-emphasis', 'data'));

  await expect(defaultBold).toHaveCSS('color', 'rgb(82, 35, 123)');
  await expect(process).toHaveCSS('color', 'rgb(0, 72, 45)');
  await expect(data).toHaveCSS('color', 'rgb(32, 58, 108)');
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(defaultBold).toHaveCSS('color', 'rgb(221, 185, 246)');
  await expect(process).toHaveCSS('color', 'rgb(148, 211, 187)');
  await expect(data).toHaveCSS('color', 'rgb(189, 197, 235)');
});

test('Butter theme follows the homepage toggle and persists across navigation', async ({ page }) => {
  await page.goto('/');
  const primary = page.getByRole('link', { name: 'Browse articles' });
  const secondary = page.getByRole('link', { name: 'Reading collections' });
  const title = page.getByRole('heading', { level: 1 });
  const header = page.getByRole('banner');
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(253, 251, 228)');
  await expect(header).toHaveCSS('background-color', 'rgb(253, 251, 228)');
  await expect(title).toHaveCSS('color', 'rgb(34, 91, 255)');
  await expect(title).toHaveCSS('font-family', /^Outfit,/);
  await expect.poll(() => page.evaluate(() => [...document.fonts].some(font => font.family === 'Outfit' && font.status === 'loaded'))).toBe(true);
  await expect(primary).toHaveCSS('background-color', 'rgb(34, 91, 255)');
  await expect(primary).toHaveCSS('color', 'rgb(255, 255, 255)');
  await expect(primary).toHaveCSS('border-radius', '8px');
  await expect(primary).toHaveCSS('height', '40px');
  await expect(secondary).toHaveCSS('border-top-color', 'rgb(34, 91, 255)');
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(38, 26, 19)');
  await expect(header).toHaveCSS('background-color', 'rgb(38, 26, 19)');
  await expect(title).toHaveCSS('color', 'rgb(253, 238, 140)');
  await expect(primary).toHaveCSS('background-color', 'rgb(253, 238, 140)');
  await expect(primary).toHaveCSS('color', 'rgb(29, 28, 17)');
  await expect(secondary).toHaveCSS('border-top-color', 'rgb(253, 238, 140)');
  await primary.click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
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

test('article lightbox opens by keyboard and returns focus', async ({ page }) => {
  await page.goto('/posts/2025/07-16-fastmcp-mcp-server/');
  const image = page.locator('.article-image-link:visible').first();
  await image.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(image).toBeFocused();
});

test('RAG article navigation follows the ten-part series', async ({ page }) => {
  await page.goto('/posts/2026/08-27-rag-engineering-01-inspectable-rag-baseline/');
  await expect(page.locator('.post-nav__link--next')).toHaveAttribute('href', '/posts/2026/08-30-rag-engineering-02-chunking-is-a-retrieval-decision/');
  await page.goto('/posts/2026/09-23-rag-engineering-10-security-governance/');
  await expect(page.locator('.post-nav__link').first()).toHaveAttribute('href', '/posts/2026/09-20-rag-engineering-09-rag-evaluation-regression-gates/');
});
