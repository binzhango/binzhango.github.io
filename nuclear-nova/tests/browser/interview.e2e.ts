import { test, expect } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
const path = '/posts/2025/10-30-llm-interview/';
const key = 'bz:interview:progress:v1';
test.beforeEach(async ({ page }) => { await page.goto(path); await expect(page.locator('[data-controls]')).toBeVisible(); });

test('read, combined filtering, empty state, expanded details, and deep links', async ({ page }) => {
  await expect(page.locator('[data-interview-bank]')).toHaveAttribute('data-mode', 'read');
  await expect(page.locator('.interview-brief').first()).toBeVisible();
  await page.locator('[data-topic-filter]').selectOption('inference');
  await page.locator('[data-search]').fill('  KV cache ');
  const matches = page.locator('[data-question]:visible');
  expect(await matches.count()).toBeGreaterThan(0);
  expect(await matches.evaluateAll(cards => cards.every(card => card.getAttribute('data-topic') === 'inference' && /kv/i.test(card.querySelector('.interview-prompt')!.textContent!) && /cache/i.test(card.querySelector('.interview-prompt')!.textContent!)))).toBe(true);
  await page.locator('[data-expand]').click();
  await expect.poll(() => matches.evaluateAll(cards => cards.every(card => card.querySelector<HTMLDetailsElement>('.interview-detail')!.open))).toBe(true);
  await page.locator('[data-search]').fill('no-such-interview-question-xyz');
  await expect(page.locator('[data-empty]')).toBeVisible();
  await page.locator('[data-empty] [data-clear]').click();
  await page.goto(`${path}#question-llm-001`);
  await expect(page.locator('#question-llm-001')).toBeInViewport();
  await page.goto(`${path}#parameter-efficient-fine-tuning-peft`);
  await expect.poll(() => page.locator('#parameter-efficient-fine-tuning-peft').evaluate(el => { const top = el.getBoundingClientRect().top; return top >= 0 && top < innerHeight; })).toBe(true);
});

test('practice reveal preserves focus; ratings stay on card; navigation hides answers; resume and reset', async ({ page }) => {
  await page.locator('[data-mode-button="practice"]').click();
  const card = page.locator('[data-question]:visible');
  const first = await card.getAttribute('data-question');
  await expect(card.locator('.interview-brief')).toBeHidden();
  const reveal = page.locator('[data-reveal]');
  await reveal.focus(); await page.keyboard.press('Enter'); await page.keyboard.press('Enter');
  await expect(card.locator('.interview-brief')).toBeVisible();
  await expect(reveal).toBeFocused();
  await page.locator('[data-rate="again"]').click();
  await expect(card).toHaveAttribute('data-question', first!);
  await page.locator('[data-next]').click();
  const second = await card.getAttribute('data-question');
  expect(second).not.toBe(first);
  await expect(card.locator('.interview-prompt')).toBeFocused();
  await expect(card.locator('.interview-brief')).toBeHidden();
  expect(await page.evaluate((key) => Object.keys(JSON.parse(localStorage.getItem(key)!).ratings).length, key)).toBe(1);
  await page.reload();
  await expect(page.locator('[data-interview-bank]')).toHaveAttribute('data-mode', 'read');
  await page.locator('[data-mode-button="practice"]').click();
  await expect(card).toHaveAttribute('data-question', second!);
  await page.locator('[data-reset]').click();
  await page.locator('[data-reset-yes]').click();
  expect(await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!).ratings, key)).toEqual({});
});

test('finish, review Again, rating changes, and restart within the filtered set', async ({ page }) => {
  const question = await page.locator('#question-llm-001 .interview-prompt').innerText();
  await page.locator('[data-search]').fill(question);
  await expect(page.locator('[data-question]:visible')).toHaveCount(1);
  await page.locator('[data-mode-button="practice"]').click();
  await page.locator('[data-reveal]').click();
  await page.locator('[data-rate="again"]').click();
  await page.locator('[data-next]').click();
  await expect(page.locator('[data-summary]')).toHaveText('1 reviewed: 0 Got it, 1 Again. 0 unreviewed.');
  await page.locator('[data-review-again]').click();
  await expect(page.locator('[data-question]:visible .interview-brief')).toBeHidden();
  await page.locator('[data-reveal]').click();
  await page.locator('[data-rate="got-it"]').click();
  await expect(page.locator('[data-question]:visible')).toHaveCount(1);
  await page.locator('[data-next]').click();
  await expect(page.locator('[data-summary]')).toContainText('1 Got it, 0 Again');
  await page.locator('[data-restart]').click();
  await expect(page.locator('[data-question]:visible')).toHaveCount(1);
});

test('corrupt and unavailable storage recover, with reduced motion and mode changes mid-flip', async ({ page }) => {
  await page.evaluate((key) => localStorage.setItem(key, '{bad'), key); await page.reload();
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.locator('[data-mode-button="practice"]').click();
  await page.locator('[data-reveal]').click();
  await expect(page.locator('[data-question]:visible .interview-brief')).toBeVisible();
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.locator('[data-next]').click();
  await page.locator('[data-reveal]').click();
  await page.locator('[data-mode-button="read"]').click();
  await expect(page.locator('[data-question]:visible .interview-brief').first()).toBeVisible();
  await page.addInitScript(() => { Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('denied', 'SecurityError'); } }); });
  await page.reload(); await page.locator('[data-mode-button="practice"]').click();
  await expect(page.locator('[data-storage-note]')).toContainText('unavailable');
  await page.locator('[data-reveal]').click(); await page.locator('[data-rate="again"]').click();
  await expect(page.locator('[data-progress]')).toContainText('1 Again');
});

test('code copy, wrapping and expanded view preserve the original source', async ({ page, context }) => {
  await page.goto(`${path}#question-llm-058`);
  await page.locator('#question-llm-058 .interview-detail summary').click();
  const wrapper = page.locator('#question-llm-058 .code-block-wrapper');
  await wrapper.locator('.code-wrap-button').click();
  await expect(wrapper).toHaveAttribute('data-wrap', 'false');
  await wrapper.locator('.code-expand-button').click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(wrapper.locator('.code-expand-button')).toBeFocused();
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await wrapper.locator('.copy-button').click();
  await expect(wrapper.locator('.code-status')).toHaveText('Code copied');
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toContain('import torch');
  expect(copied).toContain('assert output.shape == (2, 3, 32)');
  expect(copied).not.toContain('Wrap');
});

test('changing filters closes a previously expanded explanation on the newly selected card', async ({ page }) => {
  await page.locator('[data-expand]').click();
  await page.locator('[data-mode-button="practice"]').click();
  const secondPrompt = await page.locator('#question-llm-002 .interview-prompt').textContent();
  await page.locator('[data-search]').fill(secondPrompt!);
  await page.locator('[data-reveal]').click();
  await expect(page.locator('#question-llm-002 .interview-detail')).not.toHaveAttribute('open', '');
});

test('expand state follows filtered answers and a repeated permalink restores Read', async ({ page }) => {
  await page.locator('[data-topic-filter]').selectOption('ml-foundations');
  await page.locator('[data-expand]').click();
  await page.locator('[data-topic-filter]').selectOption('training');
  await expect(page.locator('[data-expand]')).toHaveText('Expand all answers');
  await page.goto(`${path}#question-llm-001`);
  await page.locator('[data-mode-button="practice"]').click();
  await page.locator('#question-llm-001 .interview-question-meta a').click();
  await expect(page.locator('[data-interview-bank]')).toHaveAttribute('data-mode', 'read');
  await expect(page.locator('#question-llm-001')).toBeInViewport();
});

test('unique generated IDs, valid local references, code helpers and math', async ({ page }) => {
  await expect(page.locator('.interview-sources')).toHaveCount(0);
  await expect(page.locator('[data-question] a[href^="http"]')).toHaveCount(0);
  const duplicates = await page.locator('[id]').evaluateAll((nodes) => { const ids = nodes.map(n => n.id); return ids.filter((id, i) => ids.indexOf(id) !== i); });
  expect([...new Set(duplicates)]).toEqual([]);
  const broken = await page.locator('.interview-bank a[href^="#"]').evaluateAll((links) => links.map(a => a.getAttribute('href')!).filter(h => !document.getElementById(decodeURIComponent(h.slice(1)))));
  expect(broken).toEqual([]);
  expect(await page.locator('.interview-bank mjx-container').count()).toBeGreaterThan(0);
  await page.locator('[data-expand]').click();
  const block = page.locator('.interview-bank pre').first(); await expect(block).toBeVisible();
  // Helpers are installed once, even when Astro's page-load event is repeated.
  const before = await page.locator('.interview-bank button').count();
  await page.evaluate(() => document.dispatchEvent(new Event('astro:page-load')));
  expect(await page.locator('.interview-bank button').count()).toBe(before);
});

test('print exposes all answers from filtered practice', async ({ page }) => {
  await page.locator('[data-search]').fill('KV cache');
  await page.locator('[data-mode-button="practice"]').click();
  await page.emulateMedia({ media: 'print' });
  await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
  await expect(page.locator('[data-controls]')).toBeHidden();
  const cards = page.locator('[data-question]');
  expect(await page.locator('[data-question]:visible').count()).toBe(await cards.count());
  await expect(page.locator('#question-llm-001 .interview-explanation')).toBeVisible();
  await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
});

test('server reading remains usable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false }); const page = await context.newPage();
  await page.goto(`${test.info().project.use.baseURL}${path}`);
  await expect(page.locator('[data-controls]')).toBeHidden();
  await expect(page.locator('.interview-brief').first()).toBeVisible();
  await page.locator('.interview-detail summary').first().click();
  await expect(page.locator('.interview-explanation').first()).toBeVisible();
  await page.emulateMedia({ media: 'print' });
  await expect(page.locator('#question-llm-002 .interview-explanation')).toBeVisible();
  await context.close();
});

for (const width of [375, 768, 1024, 1440]) for (const theme of ['light', 'dark']) {
  test(`responsive ${width}px ${theme}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `test-results/interview/${width}-${theme}-header.png` });
    await page.locator('[data-mode-button="practice"]').click();
    await page.locator('[data-reveal]').click();
    await page.locator('[data-question]:visible .interview-detail summary').click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const shortTargets = await page.locator('.interview-controls button, .interview-controls input, .interview-controls select, .interview-practice-actions button').evaluateAll(nodes => nodes.filter(n => n.getBoundingClientRect().height > 0 && n.getBoundingClientRect().height < 44).map(n => n.textContent));
    expect(shortTargets).toEqual([]);
    await page.locator('[data-controls]').scrollIntoViewIfNeeded();
    await page.screenshot({ path: `test-results/interview/${width}-${theme}.png`, fullPage: false });
    // Check varied content in the same responsive/theme configuration.
    for (const id of ['llm-058', 'llm-046', 'ai-239', 'ai-263', 'ai-098']) {
      const prompt = await page.locator(`#question-${id} .interview-prompt`).textContent();
      await page.locator('[data-search]').fill(prompt!);
      await page.locator('[data-reveal]').click();
      const card = page.locator(`#question-${id}`);
      await card.locator('.interview-detail summary').click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      if (id === 'llm-058') expect(await card.locator('.code-controls button').evaluateAll(nodes => nodes.every(node => node.getBoundingClientRect().height >= 44))).toBe(true);
      if (id === 'ai-263') await expect(card.locator('.interview-answer-kind')).toHaveText('Response framework');
      if (id === 'ai-098') await expect(card.locator('.interview-answer-kind')).toHaveText('Suggested approach');
      if (width === 375 || width === 1440) await card.screenshot({ path: `test-results/interview/${width}-${theme}-${id}.png`, style: '.site-header, .skip-link { visibility: hidden !important; }' });
    }
    const longPrompt = await page.locator('.interview-prompt').evaluateAll(nodes => nodes.map(n => n.textContent!).sort((a,b) => b.length-a.length)[0]);
    await page.locator('[data-search]').fill(longPrompt);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}

test('complete-bank interaction measurements and local fonts', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.reload();
  await expect(page.locator('[data-controls]')).toBeVisible();
  const measurements = await page.evaluate(async () => {
    await document.fonts.ready;
    const input = document.querySelector<HTMLInputElement>('[data-search]')!;
    const samples = [];
    for (const query of ['KV cache', 'transformer', 'system design', 'zz-no-result-zz', '']) {
      const start = performance.now();
      input.value = query;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      const handlerMs = performance.now() - start;
      await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      samples.push({ query, handlerMs, twoFramesMs: performance.now() - start });
    }
    return {
      questions: document.querySelectorAll('[data-question]').length,
      briefAnswers: document.querySelectorAll('.interview-brief').length,
      geistLoaded: document.fonts.check('16px "Interview Geist"'),
      monoLoaded: document.fonts.check('16px "Interview Geist Mono"'),
      samples,
    };
  });
  expect(errors).toEqual([]);
  expect(measurements.questions).toBe(762);
  expect(measurements.briefAnswers).toBe(measurements.questions);
  expect(measurements.geistLoaded && measurements.monoLoaded).toBe(true);
  await writeFile('test-results/interview/interaction-measurements.json', JSON.stringify(measurements, null, 2));
});

test('an unrelated article and the blog listing retain their normal layout', async ({ page }) => {
  await page.goto('/posts/2025/10-29-epoch/');
  await expect(page.locator('.post')).toBeVisible();
  await expect(page.locator('.post--interview')).toHaveCount(0);
  await expect(page.locator('[data-controls]')).toHaveCount(0);
  await page.screenshot({ path: 'test-results/interview/unrelated-article.png' });
  await page.goto('/posts/');
  await expect(page.getByRole('main')).toBeVisible();
  await expect(page.locator('.interview-bank')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/interview/blog-listing.png' });
});
