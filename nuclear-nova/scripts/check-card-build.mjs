import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
const home = await readFile(path.join(root, 'index.html'), 'utf8');
const studio = await readFile(path.join(root, 'card-studio/index.html'), 'utf8');
assert(home.includes('data-home-card') && home.includes('href="/card-studio/"'), 'Homepage must link to the integrated editor');
assert(studio.includes('<base href="/card-studio/"'), 'Editor assets need a stable base URL');
const origin = 'https://blog.example';
const documentUrl = new URL('/card-studio/', origin);
const required = new Set(['card-studio/card-config.json', 'card-studio/home-card.bundle.js']);
for (const match of studio.matchAll(/(?:src|href)="([^"]+)"/g)) {
  const url = new URL(match[1], documentUrl);
  assert(!['localhost', '127.0.0.1'].includes(url.hostname), 'Editor must not reference a local development server');
  if (url.origin !== origin) continue;
  required.add(url.pathname.replace(/^\//, '') + (url.pathname.endsWith('/') ? 'index.html' : ''));
}
const config = JSON.parse(await readFile(path.join(root, 'card-studio/card-config.json'), 'utf8'));
for (const asset of Object.values(config.assets)) {
  const url = new URL(asset, documentUrl);
  assert.equal(url.origin, origin, 'Card artwork must ship with the blog');
  required.add(url.pathname.slice(1));
}
for (const file of required) {
  assert((await stat(path.join(root, file))).size > 0, `Missing or empty card asset: ${file}`);
}
console.log(`Card integration verified: homepage, editor route, and ${required.size} local resources.`);
