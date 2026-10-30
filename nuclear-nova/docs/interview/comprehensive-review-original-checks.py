from pathlib import Path
import json,math,re,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2]
checks=[]
def close(name,actual,expected,tol=1e-4):
    assert math.isclose(actual,expected,abs_tol=tol), (name,actual,expected)
    checks.append({'check':name,'actual':actual,'expected':expected})
close('spam precision',8/12,2/3)
close('spam recall',8/10,.8)
close('dropout expected activation scale',.8*1.25,1)
close('sequence NLL',-math.log(.2*.5),2.302585)
close('masked target NLL',-math.log(.5),.693147)
close('mean target NLL',(-math.log(.5)-math.log(.25))/2,1.039721)
close('perplexity',math.exp(math.log(4)),4)
close('label smoothing target',.9+.1/3,.933333)
close('RMSNorm first coordinate',1/math.sqrt(5),.447214)
close('RMSNorm second coordinate',3/math.sqrt(5),1.341641)
close('DeepNorm alpha 32 layers',64**.25,2.828427)
close('DeepNorm beta 32 layers',256**(-.25),.25)
phi=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
close('GELU at 1',phi(1),.841345)
close('GELU at -1',-phi(-1),-.158655)
sigmoid=lambda x:1/(1+math.exp(-x))
close('SiLU at 1',sigmoid(1),.731059)
close('SiLU at -1',-sigmoid(-1),-.268941)
close('GLU gate +2',sigmoid(2),.880797)
close('GLU gate -2',sigmoid(-2),.119203)
close('GEGLU coordinate positive',2*phi(1),1.682689)
close('GEGLU coordinate negative',-2*phi(-1),-.317311)
close('SwiGLU coordinate positive',2*sigmoid(1),1.462117)
close('SwiGLU coordinate negative',-2*sigmoid(-1),-.537883)
close('8192 attention single matrix MiB',8192**2*2/2**20,128)
close('32 attention matrices GiB',8192**2*2*32/2**30,4)
close('single KV head bytes per cached token',2*32*128*2,16384)
close('single KV head 8192 token MiB',2*32*8192*128*2/2**20,128)
close('MHA example GiB',2*32*8192*128*2*32/2**30,4)
close('forward KL',.5*math.log(.5/.9)+.5*math.log(.5/.1),.510826)
close('reverse KL',.9*math.log(.9/.5)+.1*math.log(.1/.5),.368064)
close('target entropy',-.75*math.log(.75)-.25*math.log(.25),.562335)
close('BCE wrong confident gradient',.001-1,-.999)
close('MSE wrong confident gradient',(.001-1)*.001*.999,-.000998001,1e-9)
close('binary child entropy bits',-.75*math.log2(.75)-.25*math.log2(.25),.811278)
close('split information gain bits',1+.75*math.log2(.75)+.25*math.log2(.25),.188722)
close('stable CE target second',1+math.log1p(math.exp(-1)),1.313262)
close('7B full tuning GiB',7e9*16/2**30,104.308128)
close('training FLOPs',6*7e9*1e12,4.2e22)
close('illustrative training days',4.2e22/1e15/86400,486.111111)
close('table growth percent',(12.4-10)/10*100,24)
close('RRF example',1/61+1/63,.032266)
close('LoRA full projection params',4096**2,16777216)
close('LoRA rank8 params',8*(4096+4096),65536)
close('LoRA percent',65536/16777216*100,.390625)
close('adapter weights',2*768*64,98304)
close('adapter biases',64+768,832)

def load(number):
    body=(root/f'src/content/interview/llm-{number:03}.md').read_text()
    code=re.search(r'^```python\n(.*?)^```',body,re.M|re.S).group(1)
    ns={'__name__':'example_test'}
    exec(compile(code,f'llm-{number:03}.md','exec'),ns)
    return code,ns
code,ns=load(141)
chunk=ns['chunk_paragraphs']
assert chunk('   \n\n  ')==[]
assert chunk('a b c',1,0)==['a','b','c']
assert chunk('a b c d',4,3)==['a b c d']
for args in [(0,0),(3,3),(3,-1)]:
    try: chunk('a b',*args)
    except ValueError: pass
    else: raise AssertionError(args)
for maximum in range(1,10):
 for overlap in range(maximum):
    words=[f'w{i}' for i in range(30)]
    output=chunk(' '.join(words),maximum,overlap)
    assert all(len(c.split())<=maximum for c in output)
    assert set(' '.join(output).split())==set(words)
checks.append({'check':'llm-141 executable example, invalid bounds, empty input, 45 window/overlap combinations','result':'passed'})

code,ns=load(139)
from reportlab.pdfgen import canvas
with tempfile.TemporaryDirectory(prefix='original-pdf-check-') as temp:
    path=Path(temp)/'fixture.pdf'
    c=canvas.Canvas(str(path))
    c.drawString(72,750,'Example report')
    for x in (72,200,320): c.line(x,640,x,710)
    for y in (640,675,710): c.line(72,y,320,y)
    c.drawString(80,690,'Item'); c.drawString(210,690,'Value')
    c.drawString(80,655,'Revenue'); c.drawString(210,655,'12.4')
    c.showPage(); c.showPage(); c.save()
    records=ns['extract_pdf'](path)
    assert len(records)==2
    assert 'Example report' in records[0]['text']
    assert records[0]['tables']==[[['Item','Value'],['Revenue','12.4']]],records
    assert records[1]['text']=='' and records[1]['tables']==[]
    script=Path(temp)/'extract_pdf.py';script.write_text(code)
    good=subprocess.run([sys.executable,str(script),str(path)],capture_output=True,text=True,check=True)
    assert json.loads(good.stdout)==records
    assert subprocess.run([sys.executable,str(script)],capture_output=True).returncode!=0
    assert subprocess.run([sys.executable,str(script),str(path)+'missing'],capture_output=True).returncode!=0
checks.append({'check':'llm-139 executable PDF CLI, synthetic native text/ruled table/blank page, missing args/file','result':'passed'})

code,ns=load(58)
import torch
with torch.no_grad():
    changed=ns['source'].clone()
    changed[ns['padding']]=10000
    out,_=ns['attention'](ns['query'],changed,changed,key_padding_mask=ns['padding'],need_weights=False)
    assert torch.allclose(out,ns['output'],atol=1e-6)
checks.append({'check':'llm-058 padded-source perturbation leaves output unchanged','result':'passed'})
Path(__file__).with_name('comprehensive-review-original-results.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'{len(checks)} independent numerical and code checks passed')
