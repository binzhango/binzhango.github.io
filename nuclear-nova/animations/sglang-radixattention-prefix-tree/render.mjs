import { chromium } from 'playwright';
import { mkdir, readFile, readdir, access, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const cwd = process.cwd();
const scene = JSON.parse(await readFile(path.join(cwd, 'scene.json'), 'utf8'));
const { width, height, fps, duration } = scene.canvas;
const formats = scene.exports?.formats || ['mp4', 'gif'];
const out = path.join(cwd, 'output');
const frames = path.join(out, 'frames');
await rm(frames, { recursive: true, force: true });
await mkdir(frames, { recursive: true });
await Promise.all(['diagram.mp4', 'diagram.webp', 'diagram.gif', 'palette.png', 'contact-sheet.png'].map(file => rm(path.join(out, file), { force: true })));

async function cachedChromium() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  const root = path.join(os.homedir(), 'Library', 'Caches', 'ms-playwright');
  try {
    const dirs = (await readdir(root)).filter(x => x.startsWith('chromium_headless_shell-')).sort().reverse();
    for (const dir of dirs) {
      const candidate = path.join(root, dir, 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell');
      try { await access(candidate); return candidate; } catch {}
    }
  } catch {}
  return undefined;
}

let browser;
try {
  browser = await chromium.launch({ headless: true });
} catch (error) {
  const executablePath = await cachedChromium();
  if (!executablePath) throw error;
  console.warn(`Using cached Chromium: ${executablePath}`);
  browser = await chromium.launch({ headless: true, executablePath });
}

const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
await page.addInitScript(value => { window.__SCENE__ = value; }, scene);
const url = pathToFileURL(path.join(cwd, 'index.html')).href;
for (let i = 0; i < fps * duration; i++) {
  await page.goto(`${url}?frame=${i}`);
  await page.waitForFunction(() => window.__READY__);
  await page.screenshot({ path: path.join(frames, `frame-${String(i).padStart(4, '0')}.png`) });
}
await browser.close();

function ff(args) {
  const result = spawnSync('ffmpeg', ['-y', ...args], { stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status || 1);
}

const input = ['-framerate', String(fps), '-i', path.join(frames, 'frame-%04d.png')];
const encoders = spawnSync('ffmpeg', ['-hide_banner', '-encoders'], { encoding: 'utf8' });

if (formats.includes('mp4')) {
  ff([...input, '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', path.join(out, 'diagram.mp4')]);
}

if (formats.includes('webp')) {
  if (/\blibwebp\b/.test(encoders.stdout || '')) {
    ff([...input, '-loop', '0', '-c:v', 'libwebp', '-lossless', '0', '-compression_level', '6', '-q:v', '78', '-an', path.join(out, 'diagram.webp')]);
  } else {
    console.warn('Requested WebP, but FFmpeg lacks libwebp; use MP4 or install a libwebp-enabled FFmpeg build.');
  }
}

if (formats.includes('gif')) {
  const palette = path.join(out, 'palette.png');
  ff([...input, '-vf', 'palettegen=stats_mode=diff', '-frames:v', '1', palette]);
  ff([...input, '-i', palette, '-lavfi', 'paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle', '-loop', '0', path.join(out, 'diagram.gif')]);
}

const contactSource = formats.includes('mp4') ? path.join(out, 'diagram.mp4') : path.join(frames, 'frame-%04d.png');
const contactInput = formats.includes('mp4') ? ['-i', contactSource] : input;
ff([...contactInput, '-vf', `fps=${6 / duration},scale=640:-1,tile=3x2`, '-frames:v', '1', path.join(out, 'contact-sheet.png')]);
console.log(`Rendered ${fps * duration} frames; requested formats: ${formats.join(', ')}`);
