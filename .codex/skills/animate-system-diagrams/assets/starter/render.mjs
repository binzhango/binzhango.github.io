import { chromium } from 'playwright';
import { mkdir, readFile, readdir, access } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const cwd=process.cwd(),scene=JSON.parse(await readFile(path.join(cwd,'scene.json'),'utf8'));const {width,height,fps,duration}=scene.canvas;const out=path.join(cwd,'output'),frames=path.join(out,'frames');await mkdir(frames,{recursive:true});
async function cachedChromium(){if(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE)return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;const root=path.join(os.homedir(),'Library','Caches','ms-playwright');try{const dirs=(await readdir(root)).filter(x=>x.startsWith('chromium_headless_shell-')).sort().reverse();for(const dir of dirs){const candidate=path.join(root,dir,'chrome-headless-shell-mac-arm64','chrome-headless-shell');try{await access(candidate);return candidate}catch{}}}catch{}return undefined}
let browser;try{browser=await chromium.launch({headless:true})}catch(error){const executablePath=await cachedChromium();if(!executablePath)throw error;console.warn(`Using cached Chromium: ${executablePath}`);browser=await chromium.launch({headless:true,executablePath})}const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});await page.addInitScript(value=>{window.__SCENE__=value},scene);const url=pathToFileURL(path.join(cwd,'index.html')).href;
for(let i=0;i<fps*duration;i++){await page.goto(`${url}?frame=${i}`);await page.waitForFunction(()=>window.__READY__);await page.screenshot({path:path.join(frames,`frame-${String(i).padStart(4,'0')}.png`)});}await browser.close();
function ff(args){const r=spawnSync('ffmpeg',['-y',...args],{stdio:'inherit'});if(r.status!==0)process.exit(r.status||1)}
const input=['-framerate',String(fps),'-i',path.join(frames,'frame-%04d.png')];ff([...input,'-c:v','libx264','-pix_fmt','yuv420p','-crf','19',path.join(out,'diagram.mp4')]);
const encoders=spawnSync('ffmpeg',['-hide_banner','-encoders'],{encoding:'utf8'});if(/\blibwebp\b/.test(encoders.stdout||''))ff([...input,'-loop','0','-c:v','libwebp','-lossless','0','-compression_level','6','-q:v','78','-an',path.join(out,'diagram.webp')]);else console.warn('FFmpeg lacks libwebp; skipping animated WebP export');
const palette=path.join(out,'palette.png');ff([...input,'-vf','palettegen=stats_mode=diff','-frames:v','1',palette]);ff([...input,'-i',palette,'-lavfi','paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle','-loop','0',path.join(out,'diagram.gif')]);
ff(['-i',path.join(out,'diagram.mp4'),'-vf',`fps=1,scale=640:-1,tile=3x2`,'-frames:v','1',path.join(out,'contact-sheet.png')]);console.log(`Rendered ${fps*duration} frames to ${out}`);
