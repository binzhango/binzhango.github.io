"""Execute published Python/SQL examples and scenario-specific regression checks."""
from pathlib import Path
import asyncio, contextlib, math, random, re, sqlite3, tempfile
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src/content/interview"
spaces = {}
for number in range(435, 599):
    text = (DATA / f"ai-{number:03}.md").read_text()
    assert text.count("## Question\n") == 1
    assert text.count("## Brief answer\n") == 1
    assert text.count("## Detailed answer\n") == 1
    brief = text.split("## Brief answer\n\n")[1].split("\n\n## Detailed answer")[0]
    detail = text.split("## Detailed answer\n\n")[1]
    assert len(brief.split()) >= 25, number
    assert len(detail.split()) >= 95, number
    for code in re.findall(r"```python\n(.*?)\n```", text, re.S):
        scope = {"__name__": f"question_{number}"}
        exec(compile(code, f"ai-{number}.md", "exec"), scope)
        spaces[number] = scope
print(f"Executed {len(spaces)} Python examples and inline assertions")

def rejects(fn, error=ValueError):
    try: fn()
    except error: return
    raise AssertionError("expected rejection")

# Annotation lifecycle: expiration, re-review, conflict, priority, consensus.
Q = spaces[438]["AnnotationQueue"]
q=Q(lease=5); q.add("low"); q.add("high", priority=4)
t=q.claim("a",0); assert t[0]=="high"
rejects(lambda:q.submit(t,"yes",5))
t2=q.claim("a",5); assert t2 != t
rejects(lambda:q.submit(t,"yes",6))
q.rereview("high",9); rejects(lambda:q.submit(t2,"yes",6))
q=Q(); q.add("t",2)
a,b=q.claim("a",0),q.claim("b",0)
q.submit(a,"yes",1); assert q.submit(b,"no",1)=="adjudication"
rejects(lambda:q.submit(a,"no",2))
# Sweep-line examples against an integer-time oracle.
f=spaces[439]["peak_sessions"]
rejects(lambda:f([(3,1)]))
rng=random.Random(17)
for _ in range(100):
    sessions=[sorted([rng.randrange(8),rng.randrange(8)]) for _ in range(10)]
    peak,spans=f(sessions)
    counts=[sum(a<=t<b for a,b in sessions) for t in range(8)]
    assert peak==max(counts)
    assert {t for a,b in spans for t in range(a,b)}=={t for t,c in enumerate(counts) if c==peak and c>0}
# Provider failure classes, cancellation and shared deadline.
async def provider_tests():
    ns=spaces[450]; complete=ns['complete']; transient=ns['TransientError']
    class Provider:
        def __init__(self, mode): self.mode=mode; self.cancelled=False
        async def complete(self,prompt):
            if self.mode=='transient': raise transient('busy')
            if self.mode=='permanent': raise ValueError('invalid')
            if self.mode=='slow':
                try: await asyncio.sleep(10)
                finally: self.cancelled=True
            return 'ok'
    assert await complete([Provider('transient'),Provider('ok')],'x')=='ok'
    slow=Provider('slow')
    assert await complete([slow,Provider('ok')],'x',attempt_timeout=.005)=='ok'
    assert slow.cancelled
    try: await complete([Provider('permanent'),Provider('ok')],'x')
    except ValueError: pass
    else: raise AssertionError('permanent error masked')
    try: await complete([Provider('transient')],'x')
    except RuntimeError: pass
    else: raise AssertionError('missing exhaustion')
    slow=Provider('slow'); task=asyncio.create_task(complete([slow],'x'))
    await asyncio.sleep(.001); task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    else: raise AssertionError('cancellation swallowed')
asyncio.run(provider_tests())
# Shingle and batch edge cases.
ns=spaces[451]; assert ns['signature'](ns['shingles']('a b c d e'))==ns['signature'](ns['shingles']('a b c d e'))
assert len(ns['band_keys'](ns['signature']({('a',)})))==8
assert ns['signature'](set()) is None
batch=spaces[452]['batches']
rejects(lambda:list(batch([('x','x',4)],2,3)))
rejects(lambda:list(batch([],0,3)))
assert list(batch([('x','',0)],1,1))==[[('x','',0)]]
beam=spaces[453]['beam_search']
assert beam(lambda p:[('x',-.1)],width=1,max_new=3)==('x','x','x')
assert beam(lambda p:[],max_new=0)==()
rejects(lambda:beam(lambda p:[],max_new=1))
# Tree identity and incremental diff.
m=spaces[465]['merkle']; rejects(lambda:m({'../a':b'x'})); rejects(lambda:m({'a':b'x','a/b':b'y'}))
ns=spaces[466]; build,diff=ns['build_tree'],ns['changed']
assert diff(build({'a':b'x'}),build({'a/b':b'y'}))==['a','a/b']
assert diff(build({'a':b'x'}),build({}))==['a']
assert diff(build({'d/a':b'x'}),build({'d/a':b'x'}))==[]
# Exact file deduplication.
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    for name,data in [('a',b'x'),('b',b'x'),('c',b'y'),('e',b''),('f',b'')]: (root/name).write_bytes(data)
    groups=spaces[468]['duplicate_files'](root.iterdir())
    assert {frozenset(p.name for p in group) for group in groups}=={frozenset(['a','b']),frozenset(['e','f'])}
# Rope randomized edits checked against a plain string oracle.
ns=spaces[469]; tree=None; plain=''
for _ in range(300):
    if rng.random()<.7:
        at=rng.randrange(len(plain)+1); text=rng.choice(['x','ab','\n','\nq'])
        tree=ns['insert'](tree,at,text); plain=plain[:at]+text+plain[at:]
    else:
        a,b=sorted([rng.randrange(len(plain)+1),rng.randrange(len(plain)+1)])
        tree=ns['delete'](tree,a,b); plain=plain[:a]+plain[b:]
    assert ns['flatten'](tree)==plain
    starts=[0]+[i+1 for i,ch in enumerate(plain) if ch=='\n']
    assert [ns['line_start'](tree,i) for i in range(len(starts))]==starts
rejects(lambda:ns['line_start'](tree,len(starts)),IndexError)
# Clause context limits and missing definition rejection.
f=spaces[503]['clause_chunks']; count=lambda s:len(s.split())
rejects(lambda:list(f([{'id':'1','text':'x','terms':['X']}],{},count,20)))
rejects(lambda:list(f([{'id':'1','text':'a b c','terms':[]}],{},count,1)))
assert list(f([],{},count,10))==[]
f=spaces[514]['authorized_topk']; assert f([[(1,'a')]],1,lambda d:False)==[]
assert f([[(9,'a'),(2,'b')],[(8,'a'),(3,'c')]],3,lambda d:True)==[(9,'a'),(3,'c'),(2,'b')]
# Prompt mandatory budget and stable prefix.
f=spaces[524]['build_prompt']; count=lambda m:sum(len(x.split()) for x in m)
rejects(lambda:f(['a b'],[],'c',2,0,count))
assert f(['a'],[['old'],['new']],'q',4,1,count)==['a','new','q']
# Trajectory independent minima, invalid lengths, masked NaNs.
import numpy as np
f=spaces[567]['min_errors']; pred=np.array([[[[0.],[4.]],[[3.],[2.]]]])
a,b=f(pred,np.zeros((1,2,1)),[[1,1]])
assert np.allclose(a,[2]) and np.allclose(b,[2])
a,b=f(np.array([[[[1.],[np.nan],[3.]]]]),np.zeros((1,3,1)),[[1,0,1]])
assert np.allclose(a,[2]) and np.allclose(b,[3])
rejects(lambda:f(np.zeros((1,1,1,1)),np.zeros((1,1,1)),[[0]]))
for n in [float('nan'),float('inf'),-1]: rejects(lambda n=n:spaces[579]['Circle'](n))
# SQL grain: two issues cannot duplicate an order total.
sql=re.search(r'```sql\n(.*?)\n```',(DATA/'ai-580.md').read_text(),re.S)[1]
con=sqlite3.connect(':memory:')
con.executescript("""CREATE TABLE orders(id INTEGER, customer_id TEXT, status TEXT);
CREATE TABLE order_lines(order_id INTEGER, supplier_id INTEGER, amount INTEGER);
CREATE TABLE supplier_issues(supplier_id INTEGER, resolved_at TEXT);
INSERT INTO orders VALUES(1,'a','open'),(2,'a','closed'),(3,'b','open');
INSERT INTO order_lines VALUES(1,8,10),(1,8,20),(2,8,50),(3,9,40);
INSERT INTO supplier_issues VALUES(8,NULL),(8,NULL),(9,'2026-01-01');""")
assert con.execute(sql).fetchall()==[('a',1,30)]
f=spaces[581]['fetch_all']; assert f(lambda **kw:{'items':[],'total_pages':0})==[]
rejects(lambda:f(lambda **kw:{'items':[1],'total_pages':0}))
rejects(lambda:f(lambda **kw:{'items':[], 'total_pages': True}))
f=spaces[582]['tally']; rejects(lambda:f([('a','x',1),('a','x',2)]))
f=spaces[583]['spread']; assert f({'a':['a']},{'a'},2)==[{'a'},{'a'},{'a'}]
f=spaces[585]['deduplicate']; assert len(f([{'source':'a','id':'1'},{'source':'b','id':'1'}]))==2
rejects(lambda:f([{'source':'a','id':'1'},{'source':'a','id':'1'}]))
print('All range 435–598 semantic regression checks and SQL fixture passed')
