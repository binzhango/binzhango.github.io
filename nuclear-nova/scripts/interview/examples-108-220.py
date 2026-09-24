"""Execute published interview examples and independent edge-case checks.

Run with a Python environment containing numpy, torch and pandas. Go and Node
are used for the language-specific examples. No network or shared app server.
"""
from pathlib import Path
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "src/content/interview"

CHECKS = {
108: r'''
rng = np.random.default_rng(4)
q, k, v = (rng.normal(size=(7, 4)) for _ in range(3))
full = attention(q, k, v)
for i in range(7):
    assert np.allclose(full[i], attention(q[i:i+1], k[:i+1], v[:i+1], offset=i)[0])
v[6] += 1000
assert np.allclose(full[:6], attention(q, k, v)[:6])
try: attention(np.ones((1,2)), np.ones((1,2)), np.ones((1,2)), offset=-1)
except ValueError: pass
else: raise AssertionError('fully masked row accepted')
''',
109: r'''
torch.manual_seed(4)
x = torch.randn(2,5,16,requires_grad=True)
m = GroupedAttention(16,4,2)
y = m(x)
q=m.q(x).view(2,5,4,4).transpose(1,2)
k=m.k(x).view(2,5,2,4).transpose(1,2)
v=m.v(x).view(2,5,2,4).transpose(1,2)
reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
reference=m.out(reference.transpose(1,2).contiguous().view(2,5,16))
assert torch.allclose(y,reference,atol=1e-6)
changed = x.detach().clone(); changed[:,4] += 10
assert torch.allclose(y[:,:4],m(changed)[:,:4],atol=1e-6)
y.square().sum().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
try: GroupedAttention(16,4,3)
except ValueError: pass
else: raise AssertionError('nondividing groups accepted')
''',
110: r'''
rng = np.random.default_rng(5)
q,k,v = (rng.normal(size=(6,3)) for _ in range(3))
c = KVCache(6,3)
for i in range(6):
    scores = k[:i+1] @ q[i] / 3**.5
    weights = np.exp(scores-scores.max()); weights /= weights.sum()
    assert np.allclose(c.step(q[i],k[i],v[i]),weights @ v[:i+1])
try: c.step(q[0],k[0],v[0])
except OverflowError: pass
else: raise AssertionError('overflow accepted')
''',
111: r'''
assert merge_word(list('aaaaa'),('a','a')) == ['aa','aa','a']
assert train_bpe([],10) == []
assert train_bpe(['ab','ac'],1) == [('a','b')]
assert ''.join(encode_bpe('λunknown',merges)) == 'λunknown'
''',
112: r'''
for seed in range(100):
    assert sample([0,10,-100],k=1,seed=seed) == 1
    assert sample([0,10,-100],p=.5,seed=seed) == 1
assert sample([1,2],temperature=1e-300) == 1
for kwargs in ({'p':0},{'temperature':-1},{'k':0}):
    try: sample([1,2],**kwargs)
    except ValueError: pass
    else: raise AssertionError('invalid sampling parameter accepted')
''',
113: r'''
clock = [0.]
c = LRU(1, lambda:clock[0]); c.put('a',1,2); c.put('a',2,4)
clock[0] = 2; assert c.get('a') == 2
clock[0] = 4; assert c.get('a','missing') == 'missing'
c = LRU(0); c.put('a',1); assert c.get('a') is None
''',
114: r'''
clock=[0.]; b=Bucket(5,2,lambda:clock[0])
assert b.allow(5) and not b.allow()
clock[0]=.5; assert b.allow(1)
clock[0]=100; assert not b.allow(6) and b.allow(5)
from concurrent.futures import ThreadPoolExecutor
b=Bucket(10,0)
with ThreadPoolExecutor(max_workers=20) as pool:
    assert sum(pool.map(lambda _: b.allow(), range(100))) == 10
''',
115: r'''
async def check():
    active=peak=0
    async def call(x):
        nonlocal active,peak
        active+=1; peak=max(peak,active)
        try:
            await asyncio.sleep(.001)
            if x==3: raise ValueError('permanent')
            return x
        finally: active-=1
    result=await process(range(20),call,workers=3)
    assert peak <= 3 and len(result)==20 and result[3][0]=='error'
    assert result[19]==('ok',19)
asyncio.run(check())
''',
116: r'''
payload = b':comment\rdata: {\rdata: "x": 1}\r\rdata: [DONE]\n\n'
for cut in range(len(payload)+1):
    p=SSEJSON(); assert p.feed(payload[:cut])+p.feed(payload[cut:],True)==[{'x':1},None]
p=SSEJSON(); assert p.feed(b'data: {"x":1}',True)==[]
p=SSEJSON(); assert p.feed(b'\xef\xbb\xbfdata: {}\n\n',True)==[{}]
''',
117: r'''
units=['a b','c d','e f','g h']
for budget in range(2,9):
    for overlap in range(budget):
        chunks=chunk_units(units,budget,overlap)
        assert all(sum(len(u.split()) for u in c)<=budget for c in chunks)
        assert all(u in units for c in chunks for u in c)
        assert all(any(u in c for c in chunks) for u in units)
try: chunk_units(['too many words'],2)
except ValueError: pass
else: raise AssertionError('oversize unit split or accepted')
''',
118: r'''
assert cosine_search([[1,0],[2,0]], [1,0], 5) == [(0,1.),(1,1.)]
assert cosine_search([[1,0]], [1,0], 0) == []
try: cosine_search([[0,0]],[1,0],1)
except ValueError: pass
else: raise AssertionError('zero vector accepted')
''',
119: r'''
calls=[]
responses=iter([{'tool':'send','args':{}},{'type':'final','text':'denied'}])
assert agent('send',lambda _:next(responses),{'send':lambda:calls.append(1)},lambda n,a:False)=='denied'
assert not calls
try: agent('loop',lambda _:{'tool':'missing','args':{}},{},lambda n,a:True,steps=2)
except RuntimeError: pass
else: raise AssertionError('budget ignored')
''',
120: r'''
b=Bank(); b.create('a'); b.create('b'); b.deposit('a',100)
assert not b.create('a') and not b.transfer('a','a',10)
assert not b.transfer('a','missing',1) and not b.deposit('a',-1)
before=sum(b.balance.values()); b.transfer('a','b',40)
assert sum(b.balance.values())==before and b.top_spenders(1)==['a']
''',
121: r'''
import tempfile
s=Store(); s.set('a','one',0,2); s.set('b','two'); s.set('c','three'); s.delete('c')
assert s.scan('a',1)==[('a','one')]
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'snapshot.json'; s.compact(p,2)
    assert json.loads(p.read_text())=={'b':['two',None]}
assert not s.delete('a',2)
''',
122: r'''
s=Scheduler(); s.schedule(1,lambda:1/0); s.schedule(1,lambda:'ok')
assert s.run_due(0)==[]
assert s.run_due(1)==[(0,None,'ZeroDivisionError'),(1,'ok',None)]
''',
123: r'''
s=School(); s.students['u']=Student('u','A'); s.courses['c']=Course('c','ML')
assert s.average('u') is None
s.enroll('u','c')
try:s.enroll('u','c')
except ValueError:pass
else:raise AssertionError('duplicate enrollment')
try:s.grade('u','c',101)
except ValueError:pass
else:raise AssertionError('invalid grade')
''',
124: r'''
pages={'https://example.test/':['/a','#fragment','https://other.test/'],
       'https://example.test/a':['/b','/'], 'https://example.test/b':[]}
async def fetch(u): return pages[u]
assert asyncio.run(crawl_async('https://example.test/',fetch)) == crawl_sync('https://example.test/',pages.__getitem__)
assert len(asyncio.run(crawl_async('https://example.test/',fetch,limit=2)))==2
''',
125: r'''
assert stack_events([(0,['a','a']), (1,['a']), (2,[])]) == [(0,'start','a'),(0,'start','a'),(1,'end','a'),(2,'end','a')]
assert stack_events([])==[]
try:stack_events([(2,[]),(1,[])])
except ValueError:pass
else:raise AssertionError('unordered samples accepted')
''',
126: r'''
l=SlidingLimiter({'x':1},10)
assert l.allow('x',0) and not l.allow('x',9.999) and l.allow('x',10)
try:l.allow('x',9)
except ValueError:pass
else:raise AssertionError('backward clock accepted')
''',
127: r'''
async def check():
    calls={}; results={}; active=peak=0
    random.uniform=lambda a,b:0
    async def call(i,text):
        nonlocal active,peak
        calls[i]=calls.get(i,0)+1
        if i==1 and calls[i]==1:raise Retryable()
        if i==2:raise ValueError()
        active+=1; peak=max(peak,active)
        await asyncio.sleep(.001)
        active-=1;return text.upper()
    async def save(i,result):results[i]=result
    await run_documents(((i,str(i)) for i in range(20)),call,save,workers=3)
    assert len(results)==20 and results[2]['status']=='failed'
    assert calls[1]==2 and peak<=3
    async def broken_save(i,result):raise OSError('storage unavailable')
    try:await run_documents([(0,'x')],call,broken_save,workers=1)
    except ExceptionGroup:pass
    else:raise AssertionError('persistence failure swallowed')
asyncio.run(check())
''',
157: r'''
s=TransactionalStore();s.begin();s.commit();assert s.get('x') is None
try:s.abort()
except RuntimeError:pass
else:raise AssertionError('abort without transaction')
s.begin()
try:s.begin()
except RuntimeError:pass
else:raise AssertionError('nested transaction accepted')
''',
158: r'''
try:
    with connection:
        repo.save(User(2,'temporary'))
        raise RuntimeError()
except RuntimeError:pass
assert repo.get(2) is None
with connection:repo.save(User(1,'changed'))
assert repo.get(1).name=='changed'
''',
161: r'''
calls=[]
try:checkout([(-1,2)],calls.append)
except ValueError:pass
else:raise AssertionError('invalid input')
assert calls==[]
''',
162: r'''
assert fibonacci(10)[-1]==34
for n in [-1,True,1.5]:
    try:fibonacci(n)
    except ValueError:pass
    else:raise AssertionError('invalid n')
''',
163: r'''
assert simulate({'a':['b'],'b':['a']},{'a'},3)[-1]=={'a','b'}
assert simulate({},set(),3)==[set(),set(),set(),set()]
''',
164: r'''
assert kl([0,1],[.5,.5])==math.log(2)
assert not math.isclose(kl([.8,.2],[.5,.5]),kl([.5,.5],[.8,.2]))
try:kl([.2,.2],[.5,.5])
except ValueError:pass
else:raise AssertionError('unnormalized distribution accepted')
''',
167: r'''
assert test['age'].isna().tolist()==[True,False]
assert clean(test)['city'].tolist()==['Unknown','C']
''',
188: r'''
import random
from collections import Counter
rng=random.Random(2)
for _ in range(100):
    stream=[rng.randrange(8) for _ in range(100)]
    h=HeavyHitters(3)
    for x in stream:h.add(x)
    exact=Counter(stream)
    assert all(h.counts.get(x,0)<=n and n-h.counts.get(x,0)<=25 for x,n in exact.items())
''',
189: r'''
assert closest_key({'a':['b'],'b':['a']},'a',{'z'}) is None
assert closest_key({},'a',{'a'})==('a',0)
''',
190: r'''
assert rmse([1e200],[0])==1e200
for a,b in [([],[]),([1],[1,2]),([float('nan')],[0])]:
    try:rmse(a,b)
    except ValueError:pass
    else:raise AssertionError('invalid RMSE inputs')
''',
191: r'''
assert bigrams('')==Counter()
assert bigrams("Don't stop. STOP now")==Counter({("don't","stop"):1,('stop','now'):1})
''',
210: r'''
import itertools,math
for n in range(2,6):
    for xs in itertools.product([-1,0,2],repeat=n):
        expected=[math.prod(xs[:i]+xs[i+1:]) for i in range(n)]
        assert product_except_self(xs)==expected
''',
211: r'''
import itertools
def brute(s,t):
    if not t:return ''
    required=Counter(t)
    for length in range(1,len(s)+1):
        for start in range(len(s)-length+1):
            part=s[start:start+length]
            if not required-Counter(part):return part
    return ''
for n in range(5):
    for chars in itertools.product('ab',repeat=n):
        s=''.join(chars)
        for t in ['','a','b','aa','ab','bb']:
            assert minimum_window(s,t)==brute(s,t)
''',
212: r'''
for text in ['[]','[1,null]','[null,null]']:
    try:deserialize(text)
    except ValueError:pass
    else:raise AssertionError('malformed tree accepted')
assert deserialize(serialize(Node(1,Node(1),Node(1)))).left.value==1
''',
213: r'''
one=circular_list(Node(4));assert one.left is one and one.right is one
head=circular_list(Node(1,None,Node(2,None,Node(3))))
assert head.left.value==3 and head.left.left.value==2 and head.left.left.left is head
''',
214: r'''
assert alien_order([])==''
assert alien_order(['abc','abc'])=='abc'
assert alien_order(['z','x','a'])=='zxa'
''',
215: r'''
assert closest_points([],3)==[] and frequent([],3)==[] and rooms([])==0
assert rooms([(0,1),(1,2),(2,3)])==1
assert rooms([(0,3),(0,3),(0,3)])==3
''',
216: r'''
import itertools,re
patterns=['','a','b','.', 'a*','.*','a*b*','a*b','ab*','a.*b']
for n in range(5):
    for chars in itertools.product('ab',repeat=n):
        text=''.join(chars)
        for pattern in patterns:
            assert regex_match(text,pattern)==bool(re.fullmatch(pattern,text))
''',
217: r'''
assert top_engaged([],3)==[] and top_engaged(['a'],0)==[]
assert top_engaged(['b','a'],2)==[('a',1),('b',1)]
''',
}


def code(number, language):
    body = (CONTENT / f"ai-{number:03}.md").read_text()
    return "\n\n".join(re.findall(rf"```{language}\n(.*?)\n```", body, re.S))


def run():
    results = []
    with tempfile.TemporaryDirectory(prefix="interview-108-220-") as directory:
        folder = Path(directory)
        for number in range(108, 221):
            source = code(number, "python")
            if not source:
                continue
            path = folder / f"example_{number}.py"
            path.write_text(source + "\n" + CHECKS.get(number, ""))
            result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise AssertionError(f"ai-{number}: {result.stderr}\n{result.stdout}")
            results.append({"question": f"ai-{number:03}", "language": "python", "status": "passed"})

        db = sqlite3.connect(":memory:")
        db.executescript("""
            CREATE TABLE order_items(order_id INTEGER,product_id TEXT,quantity INTEGER);
            INSERT INTO order_items VALUES(1,'a',2),(1,'a',1),(1,'b',1),(2,'a',1),(2,'b',1),(2,'c',1),(3,'c',1);
            CREATE TABLE subscriptions(subscription_id INTEGER,user_id INTEGER,start_at INTEGER,end_at INTEGER);
            INSERT INTO subscriptions VALUES(1,1,0,3),(2,1,3,5),(3,2,0,9),(4,2,2,3),(5,3,1,NULL),(6,3,5,8);
            CREATE TABLE salaries(employee_id INTEGER,salary INTEGER,effective_at INTEGER,salary_row_id INTEGER);
            INSERT INTO salaries VALUES(1,100,2020,1),(1,90,2021,2),(1,500,2030,3),(2,80,2020,4),(2,85,2020,5);
        """)
        assert db.execute(code(129,"sql")).fetchall() == [('a','b',2),('a','c',1),('b','c',1)]
        assert db.execute(code(130,"sql")).fetchall() == [(2,),(3,)]
        assert db.execute(code(131,"sql"),{"as_of":2022}).fetchall() == [(1,90,2021),(2,85,2020)]
        results.extend({"question":f"ai-{n}","language":"sql","status":"passed"} for n in [129,130,131])

        go_path = folder / "crawler.go"
        go_path.write_text(code(159,"go"))
        go_result = subprocess.run(["go","run",str(go_path)], capture_output=True,
                                   env={**os.environ, "GOCACHE":str(folder/'go-cache')},
                                   text=True, timeout=120)
        if go_result.returncode:
            raise AssertionError(f"ai-159: {go_result.stderr}")
        results.append({"question":"ai-159","language":"go","status":"passed"})

        html = code(160,"html")
        js = re.search(r'<script type="module">(.*?)</script>',html,re.S).group(1)
        node_path=folder / 'ui-test.cjs'
        node_path.write_text(r'''
const assert=require('node:assert/strict');
const elements={
  '#reload':{disabled:false,addEventListener(type,fn){this.click=fn}},
  '#status':{textContent:''},
  '#items':{children:[],replaceChildren(...nodes){this.children=nodes}}
};
global.document={querySelector:s=>elements[s],createElement:()=>({textContent:''})};
let response={ok:true,json:async()=>[{id:1,title:'<script>not HTML</script>'}]};
global.fetch=async()=>response;
''' + js + r'''
setImmediate(async()=>{
  assert.equal(elements['#items'].children[0].textContent,'<script>not HTML</script>');
  assert.equal(elements['#status'].textContent,'1 items');
  response={ok:true,json:async()=>[]};await elements['#reload'].click();
  assert.equal(elements['#status'].textContent,'No items yet.');
  response={ok:false};await elements['#reload'].click();
  assert.equal(elements['#status'].textContent,'Could not load items. Try again.');
  assert.equal(elements['#reload'].disabled,false);
});
''')
        subprocess.run(['node',str(node_path)],check=True,capture_output=True,text=True,timeout=30)
        results.append({"question":"ai-160","language":"html/javascript","status":"passed"})

    # Independently checked arithmetic used in the conceptual explanations.
    import math
    assert math.isclose(math.exp(math.log(4)),4)
    assert 1 + .5*6 + .5*4 == 6 and 1 + .5*6 == 4
    assert 70_000_000_000*16 == 1_120_000_000_000
    assert 1_000_000*768*4 == 3_072_000_000
    import numpy as np
    z=np.array([1.,-2.,.4]); target=np.array([0.,1.,0.])
    p=np.exp(z-z.max()); p/=p.sum()
    def loss(logits):
        maximum=logits.max()
        return maximum+np.log(np.exp(logits-maximum).sum())-target@logits
    numeric=[]
    for i in range(3):
        delta=np.zeros(3);delta[i]=1e-6
        numeric.append((loss(z+delta)-loss(z-delta))/(2e-6))
    assert np.allclose(numeric,p-target,atol=1e-7)
    print(json.dumps({"examples":results,"total":len(results),"arithmetic_checks":5},indent=2))


if __name__ == '__main__':
    run()
