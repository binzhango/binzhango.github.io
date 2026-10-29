import { chromium } from '../../../node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const out=path.resolve(root,'../../../public/diagrams/jev-decisions');
const runtime=process.argv[2];
if(!runtime) throw new Error('Pass a local copy of https://viewer.diagrams.net/js/viewer-static.min.js');
const manifest=JSON.parse(await fs.readFile(path.join(root,'manifest.json'),'utf8'));
const review='/private/tmp/jev-diagram-review';await fs.mkdir(review,{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage({viewport:{width:760,height:900}});
await page.setContent('<html><body><div id="graph" style="width:760px;height:900px"></div></body></html>');
await page.addScriptTag({path:runtime});
const receipt={renderer:'Official draw.io GraphViewer / Graph.getSvg',runtimeSha256:crypto.createHash('sha256').update(await fs.readFile(runtime)).digest('hex'),diagrams:[]};
for(const d of manifest){
 for(const theme of ['light','dark']){
  const stem=d.slug+(theme==='dark'?'.dark':'');
  const xml=await fs.readFile(path.join(root,stem+'.drawio'),'utf8');
  const result=await page.evaluate(({xml,theme,title})=>{
   document.getElementById('graph').replaceChildren();
   const viewer=new GraphViewer(document.getElementById('graph'),mxUtils.parseXml(xml).documentElement,{'auto-fit':false,toolbar:'',lightbox:false,'dark-mode':false});
   const bg=theme==='dark'?'#101b2b':'#f8fafc';
   const svg=viewer.graph.getSvg(bg,1,24);
   svg.setAttribute('content',xml);svg.setAttribute('role','img');svg.setAttribute('aria-label',title);
   svg.setAttribute('style','background:'+bg+';color-scheme:only light');
   const titleNode=document.createElementNS('http://www.w3.org/2000/svg','title');titleNode.textContent=title;svg.prepend(titleNode);
   return {svg:mxUtils.getXml(svg),width:svg.getAttribute('width'),height:svg.getAttribute('height'),vertices:viewer.graph.getChildVertices(viewer.graph.getDefaultParent()).length,edges:viewer.graph.getChildEdges(viewer.graph.getDefaultParent()).length};
  },{xml,theme,title:d.title});
  if(result.vertices!==d.nodes.length||result.edges!==d.edges.length)throw new Error('Lost native cells: '+stem);
  await fs.writeFile(path.join(out,stem+'.svg'),result.svg);
  receipt.diagrams.push({name:stem,vertices:result.vertices,edges:result.edges,width:result.width,height:result.height,sha256:crypto.createHash('sha256').update(result.svg).digest('hex')});
 }
 console.log(d.slug+' exported');
}
const shot=await browser.newPage();
for(const d of manifest){
 const svgPath=path.join(out,d.slug+'.svg');
 await shot.goto('file://'+svgPath);await shot.locator('svg').screenshot({path:path.join(review,d.slug+'.png')});
}
await fs.writeFile(path.join(root,'export-receipt.json'),JSON.stringify(receipt,null,2)+'\n');
await browser.close();
