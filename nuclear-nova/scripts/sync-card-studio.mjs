// Publish only the standalone viewer and its referenced assets, never Blender or node_modules.
import { mkdir, copyFile, readFile, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import sharp from 'sharp';
import { build } from 'esbuild';

const root = fileURLToPath(new URL('../', import.meta.url));
const source = path.join(root, 'tools/favicon-card/web');
const destination = path.join(root, 'public/card-studio');
await build({
  entryPoints: [path.join(source, 'app.js'), path.join(source, 'home-card.js')],
  outdir: source,
  entryNames: '[name].bundle',
  bundle: true,
  format: 'esm',
  target: 'es2020',
  minify: true,
});
await mkdir(path.join(destination, 'assets'), { recursive: true });
// The editor document is now owned by src/pages/card-studio.astro.
await rm(path.join(destination, 'index.html'), { force: true });
const config = JSON.parse(await readFile(path.join(root, 'tools/favicon-card/card-config.json'), 'utf8'));
await copyFile(path.join(root, 'tools/favicon-card/card-config.json'), path.join(source, 'card-config.json'));
// Prepared artwork is authoritative; keep the isolated viewer and blog in sync.
for (const [kind, asset] of Object.entries(config.assets)) {
  if (kind !== 'model') await copyFile(path.join(root, 'tools/favicon-card', asset), path.join(source, asset));
}
await copyFile(path.join(root, 'tools/favicon-card/assets/favicon.png'), path.join(source, 'assets/favicon.png'));
for (const file of ['style.css', 'app.bundle.js', 'home-card.bundle.js', 'card-config.json', 'assets/favicon.png', ...Object.values(config.assets)]) {
  await copyFile(path.join(source, file), path.join(destination, file));
}
await sharp(path.join(root, 'tools/favicon-card/renders/hero.png'))
  .resize(600).webp({ quality: 85 }).toFile(path.join(destination, 'assets/preview.webp'));
await copyFile(path.join(destination, 'assets/preview.webp'), path.join(source, 'assets/preview.webp'));
console.log('Card studio synced to /card-studio/');
