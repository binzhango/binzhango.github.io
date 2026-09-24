"""Run published examples, then independent invariants for the 221–270 batch."""
from pathlib import Path
import re
import asyncio
import torch

root = Path(__file__).resolve().parents[2] / 'src/content/interview'
examples = {}
for n in (236, 237, 246, 254, 267):
    namespace = {}
    for code in re.findall(r'^```python\n(.*?)^```', (root/f'ai-{n:03}.md').read_text(), re.M|re.S):
        exec(compile(code, f'ai-{n:03}.md', 'exec'), namespace)
    examples[n] = namespace

store = examples[236]['KVStore']()
store.begin(); store.set('x', None); store.begin(); store.set('x', 7)
store.commit(); assert store.get('x') == 7
store.commit(); assert store.get('x') == 7
store.begin(); store.delete('x'); store.commit()
try:
    store.get('x'); raise AssertionError('deletion did not commit')
except KeyError:
    pass
try:
    store.commit(); raise AssertionError('missing transaction accepted')
except RuntimeError:
    pass
iterator = examples[237]['FlattenIterator']([[], 0, [-2]])
assert next(iterator) == 0 and next(iterator) == -2
for _ in range(2):
    try:
        next(iterator); raise AssertionError('iterator not exhausted')
    except StopIteration:
        pass
now = [0.0]
limiter = examples[254]['TokenLimiter'](10, 2, lambda: now[0])
assert limiter.reserve('x', 'a', 10)
assert not limiter.reserve('x', 'b', 1)
now[0] = .5
assert limiter.reserve('x', 'b', 1)
limiter.settle('x', 'a', 9)
try:
    limiter.settle('x', 'a', 9); raise AssertionError('double refund accepted')
except ValueError:
    pass

# Independently gather all expert outputs, then mask the selected expert IDs.
for k in (1, 2, 4):
    model = examples[267]['SharedMoE'](4, 4, k)
    x = torch.randn(7, 4)
    logits, ids = model.router(x).topk(k, dim=-1)
    weights = logits.softmax(-1)
    all_outputs = torch.stack([expert(x) for expert in model.experts], dim=1)
    chosen = all_outputs[torch.arange(7)[:, None], ids]
    expected = model.shared(x) + (chosen * weights[:, :, None]).sum(1)
    assert torch.allclose(model(x), expected, atol=1e-6)
    if k == 1:
        model(x).square().sum().backward()
        assert torch.count_nonzero(model.router.weight.grad) == 0

async def check_service():
    ns = examples[246]
    async def empty(_): return []
    async def wrong(*_): return {'answer': 'invented', 'source_ids': ['missing']}
    for retrieve, generate in [(empty, ns['fixture_generate']), (ns['fixture_retrieve'], wrong)]:
        try:
            await ns['answer_question']('q', retrieve, generate)
            raise AssertionError('invalid evidence/citation accepted')
        except ValueError:
            pass
    canceled = asyncio.Event()
    async def slow(_):
        try: await asyncio.sleep(10)
        finally: canceled.set()
    task = asyncio.create_task(ns['answer_question']('q', slow, ns['fixture_generate']))
    await asyncio.sleep(.01)
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    assert canceled.is_set()
    try:
        await ns['answer_question']('q', slow, ns['fixture_generate'])
        raise AssertionError('timeout missing')
    except TimeoutError:
        pass
asyncio.run(check_service())

assert 2*80*8*128*2*131072 == 40*1024**3
assert 2*32*8*128*2*4096 == 512*1024**2
assert 100_000_000*1024*4/1e9 == 409.6
assert 100_000_000*64/1e9 == 6.4
assert 671e9*2/1e9 == 1342
print('5 published Python examples, independent edge/reference checks, and 5 calculations passed.')
