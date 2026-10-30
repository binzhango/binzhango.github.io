"""Execute authored snippets and substantive boundary/property checks."""
import asyncio
from collections import Counter, deque
from ipaddress import ip_address, ip_network
from pathlib import Path
import random
import runpy
import threading
import time

ROOT = Path(__file__).parent
ns = {int(p.name.split('-')[1]): runpy.run_path(str(p))
      for p in sorted(ROOT.glob('ai-*.py'))}
rng = random.Random(271434)

# Router weights remain normalized and differentiable; invalid k fails.
import torch
x = torch.randn(9, 5, requires_grad=True)
w = torch.randn(5, 7, requires_grad=True)
ids, gates, aux = ns[286]['route'](x, w, 3)
assert ids.shape == (9, 3) and torch.allclose(gates.sum(-1), torch.ones(9))
(gates.square().sum() + aux).backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
for bad_k in [0, 8]:
    try: ns[286]['route'](x, w, bad_k)
    except ValueError: pass
    else: raise AssertionError('invalid router k accepted')

# FIM must omit held-out text, include neighbor, reject reserved markers.
fim = ns[295]['repo_fim']
prompt, target = fim('repo', {'a': 'abcDEFghi', 'b': 'neighbor'}, 'a', 3, 6)
assert 'DEF' not in prompt and 'neighbor' in prompt and target.startswith('DEF')
try: fim('repo', {'a': '<|fim_middle|>'}, 'a', 0, 1)
except ValueError: pass
else: raise AssertionError('marker accepted')

# Division truncates toward zero over signs and boundaries.
for a in range(-100, 101):
    for b in range(-17, 18):
        if b:
            expected = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            assert ns[328]['divide'](a, b) == expected
try: ns[328]['divide'](1, 0)
except ZeroDivisionError: pass
else: raise AssertionError('zero divisor accepted')
assert ns[328]['components'](6, [(0,1),(1,2),(2,0),(4,5),(4,5)]) == 3

# Space-Saving error intervals enclose exact counts on varied streams.
for capacity in [1, 3, 10]:
    summary = ns[327]['SpaceSaving'](capacity)
    counts = Counter()
    for _ in range(1000):
        key = rng.randrange(17)
        summary.add(key); counts[key] += 1
        assert len(summary.counts) <= capacity
        for key, (estimate, error) in summary.counts.items():
            assert estimate-error <= counts[key] <= estimate

# Many copy-on-write forks preserve independent sequence values.
m = ns[362]['BlockManager'](100, 3)
expected = {'base': []}; m.create('base')
for step in range(100):
    source = rng.choice(list(expected))
    if step % 7 == 0:
        name = f'fork-{step}'; m.fork(source, name); expected[name] = expected[source].copy()
    else:
        m.append(source, step); expected[source].append(step)
    for name, values in expected.items(): assert m.values(name) == values
for name in list(expected): m.release(name)
assert len(m.free_ids) == 100 and not m.refs

# Vectorized NMS agrees with an independent scalar greedy reference.
import numpy as np

def scalar_nms(boxes, scores, threshold):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i]); kept = []
    def iou(a,b):
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
        return intersection/union if union else 0
    while order:
        i = order.pop(0); kept.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= threshold]
    return kept

for _ in range(50):
    boxes = []
    for i in range(30):
        x,y = rng.random(),rng.random(); dx,dy = rng.random(),rng.random()
        boxes.append([x,y,x+dx,y+dy])
    scores = [rng.random() for _ in boxes]
    assert ns[374]['nms'](boxes,scores) == scalar_nms(boxes,scores,0.5)

# Ring wrap and overflow agree with a bounded deque.
ring = ns[375]['RingBuffer'](7); reference = deque(maxlen=7)
for i in range(1000):
    if reference and rng.random() < 0.3:
        assert ring.pop() == reference.popleft()
    else: ring.append(i); reference.append(float(i))

# Heavy hitters retain every item above the proven threshold.
stream = [rng.randrange(10) for _ in range(500)] + ['heavy'] * 300
rng.shuffle(stream)
h = ns[385]['HeavyHitters'](5)
for key in stream: h.add(key)
for key,count in Counter(stream).items():
    if count > len(stream)/6: assert key in h.counters
window = ns[385]['WindowCounter'](5)
assert window.count(0,10) == 10 and window.count(4,1) == 11
assert window.count(5) == 1 and window.count(9) == 0
try: window.count(8)
except ValueError: pass
else: raise AssertionError('out-of-order time accepted')

# Logger timer, close rejection and sink failure propagation.
flushed = threading.Event(); received = []
def sink(batch): received.extend(batch); flushed.set()
logger = ns[399]['BatchingLogger'](sink, interval=0.03)
logger.log('timed')
assert flushed.wait(1), 'timer did not flush partial batch'
logger.close(); assert received == ['timed']
try: logger.log('closed')
except RuntimeError: pass
else: raise AssertionError('closed logger accepted message')
def fail(batch): raise OSError('test sink failure')
logger = ns[399]['BatchingLogger'](fail, batch_size=1)
logger.log('fail')
try: logger.close()
except RuntimeError: pass
else: raise AssertionError('sink failure hidden')
assert logger.failed_batch == ['fail']

# Compiled address intervals agree with direct CIDR membership.
allow = ['10.0.0.0/8','192.168.0.0/16','2001:db8::/32']
deny = ['10.20.0.0/16','10.20.1.0/24','192.168.1.0/24','2001:db8:1::/48']
policy = ns[401]['IPPolicy'](allow, deny)
an,dn = list(map(ip_network,allow)), list(map(ip_network,deny))
for _ in range(1000):
    address = ip_address(rng.getrandbits(32))
    expected = any(address in net for net in an) and not any(address in net for net in dn)
    assert policy.is_allowed(str(address)) == expected
for text in ['10.0.0.0','10.255.255.255','10.20.0.0','10.20.255.255','2001:db8:1::1']:
    address = ip_address(text)
    assert policy.is_allowed(text) == (any(address in net for net in an) and not any(address in net for net in dn))

async def async_checks():
    # Runtime timeout invokes cleanup before chip-group reuse.
    class Slow(ns[410]['FakeDevice']):
        async def collect(self): await asyncio.Event().wait()
    device = Slow(); runtime = ns[410]['Runtime'](device)
    try: await runtime.run('double-v1',[1],timeout=0.01)
    except asyncio.TimeoutError: pass
    else: raise AssertionError('expected timeout')
    assert not device.active and device.cleared == 1
    # Normal SSE completion, send error, and early disconnect all close generator.
    for mode in ['normal','send-error','early-disconnect']:
        queue = asyncio.Queue(); closed = []; sent = []
        async def receive(): return await queue.get()
        async def generator():
            try:
                if mode == 'early-disconnect': await asyncio.Event().wait()
                yield 'a'; yield 'b'
            finally: closed.append(True)
        async def send(event):
            sent.append(event)
            if mode == 'send-error' and event.get('body'): raise OSError('disconnected')
        if mode == 'early-disconnect': await queue.put({'type':'http.disconnect'})
        try: await asyncio.wait_for(ns[420]['stream_tokens'](receive,send,generator),1)
        except OSError:
            assert mode == 'send-error'
        assert closed == [True]
        if mode == 'normal': assert sent[-1]['more_body'] is False

asyncio.run(async_checks())
assert 2*28*4096*8*128*2 == 448*1024**2
assert (70000 + 230 - 1)//230 == 305
print('PASS: 16 snippets plus router, FIM, division, heavy-hitter, KV-sharing, NMS, ring, window, logger, CIDR, runtime, SSE and numerical checks')
