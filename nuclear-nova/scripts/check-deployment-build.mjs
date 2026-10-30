import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import config from '../astro.config.mjs';

const output = new URL('../dist/', import.meta.url);
const readOutput = (path) => readFile(new URL(path.replace(/^\//, ''), output), 'utf8');

// These articles previously shipped without their diagrams despite a green CI run.
const diagramArticles = [
  '2024/12-15-gradio-with-llm',
  '2025/07-16-fastmcp-mcp-server',
  '2025/09-29-langchain',
  '2025/10-02-langchain',
];
for (const slug of diagramArticles) {
  const html = await readOutput(`posts/${slug}/index.html`);
  assert.match(html, /<img\b[^>]*src="data:image\/svg\+xml[;,]/, `Missing rendered Mermaid diagram: ${slug}`);
}

const site = config.site.replace(/\/$/, '');
assert((await readOutput('sitemap-index.xml')).includes(`${site}/sitemap-0.xml`), 'Sitemap must use the deployment origin');
const redirects = Object.entries(config.redirects);
if (process.env.DEPLOY_TARGET === 'github-pages') {
  for (const [source, target] of redirects) {
    const destination = typeof target === 'string' ? target : target.destination;
    const html = await readOutput(`${source.replace(/\/$/, '')}/index.html`);
    assert.match(html, /http-equiv="refresh"/, `Missing HTML redirect: ${source}`);
    assert(html.includes(destination), `Wrong redirect destination: ${source}`);
  }
} else {
  const manifest = JSON.parse(await readFile(new URL('../.vercel/output/config.json', import.meta.url), 'utf8'));
  assert.equal(manifest.version, 3, 'Expected Vercel Build Output API v3');
  for (const [source, target] of redirects) {
    const destination = typeof target === 'string' ? target : target.destination;
    assert(manifest.routes.some(route => route.src && new RegExp(route.src).test(source)
      && route.headers?.Location === destination && route.status >= 300 && route.status < 400),
    `Missing Vercel redirect: ${source}`);
  }
}
console.log(`Deployment output verified: ${diagramArticles.length} Mermaid articles, ${redirects.length} redirects, and sitemap origin ${site}.`);
