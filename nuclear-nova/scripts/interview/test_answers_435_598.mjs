// Integration-test the published streaming proxy with local fake TTS.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import { once } from 'node:events';
const text = await fs.readFile(new URL('../../src/content/interview/ai-536.md', import.meta.url), 'utf8');
const code = text.match(/```javascript\n([\s\S]*?)\n```/)[1];
const { createTtsServer } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
let mode = 'normal', cancelled = false;
const upstream = http.createServer(async (req,res) => {
  for await (const _ of req) {}
  assert.equal(req.headers.authorization, 'Bearer fixture-token');
  if (mode === 'error') { res.writeHead(503).end(); return; }
  if (mode === 'wrong-type') { res.writeHead(200, {'content-type':'text/plain'}).end('bad'); return; }
  res.writeHead(200, {'content-type':'audio/mpeg'});
  if (mode === 'normal') { res.end(Buffer.from('fixture-audio')); return; }
  res.write(Buffer.alloc(64,1));
  const interval = setInterval(() => res.write(Buffer.alloc(64,2)), 10);
  res.on('close', () => {clearInterval(interval); cancelled=true;});
});
upstream.listen(0,'127.0.0.1'); await once(upstream,'listening');
const proxy=createTtsServer(`http://127.0.0.1:${upstream.address().port}`, 'fixture-token');
proxy.listen(0,'127.0.0.1'); await once(proxy,'listening');
const url=`http://127.0.0.1:${proxy.address().port}/tts`;
try {
  let response=await fetch(url,{method:'POST',body:'{}'});
  assert.equal(response.status,200); assert.equal(await response.text(),'fixture-audio');
  mode='error'; response=await fetch(url,{method:'POST',body:'{}'}); assert.equal(response.status,502); await response.text();
  mode='wrong-type'; response=await fetch(url,{method:'POST',body:'{}'}); assert.equal(response.status,502); await response.text();
  mode='stream'; const controller=new AbortController();
  response=await fetch(url,{method:'POST',body:'{}',signal:controller.signal});
  const reader=response.body.getReader(); assert.equal((await reader.read()).done,false);
  controller.abort();
  for(let i=0;i<100&&!cancelled;i++) await new Promise(resolve=>setTimeout(resolve,10));
  assert.equal(cancelled,true,'browser cancellation must close upstream synthesis');
  console.log('Streaming TTS: normal response, upstream error, media validation, and cancellation passed');
} finally {
  proxy.closeAllConnections(); upstream.closeAllConnections();
  await Promise.all([new Promise(resolve=>proxy.close(resolve)),new Promise(resolve=>upstream.close(resolve))]);
}
