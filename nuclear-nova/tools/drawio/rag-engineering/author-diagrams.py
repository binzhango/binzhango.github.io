"""Author native draw.io cells; export them with render-diagrams.mjs."""
from pathlib import Path
from xml.etree import ElementTree as E
import json
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[2]/'public'/'diagrams'/'rag-engineering'
OUT.mkdir(parents=True,exist_ok=True)
DIAGRAMS=[]
def node(id,title,detail,x,y,w=260,h=92,kind='process',tone='blue'):
 return dict(id=id,title=title,detail=detail,x=x,y=y-(max(h,120)-h)/2 if kind in ('document','database') else y,w=w,h=max(h,120) if kind in ('document','database') else h,kind=kind,tone=tone)
def edge(a,b,label='',points=None,dashed=False):
 return dict(a=a,b=b,label=label,points=points or [],dashed=dashed)
def add(slug,title,nodes,edges,caption):
 DIAGRAMS.append(dict(slug=slug,title=title,nodes=nodes,edges=edges,caption=caption))
add('01-request','From documents to an answer',[
 node('docs','Source documents','API guides · policies',20,55,200,90,'document'),node('chunk','Parse + chunk','Text with source locations',260,55,200,90),node('index','Search index','Representations + source IDs',500,55,200,90,'database','violet'),
 node('question','Question','What does the policy say?',20,255,200,90,'process'),node('retrieve','Retrieve','Rank passages for the query',260,255,200,90),node('prompt','Build prompt','Question + selected passages',500,255,200,90,'process','violet'),node('llm','LLM','Generate using the context',500,425,200,90,'process','green'),node('answer','Answer + sources','Resolve IDs to source passages',260,425,200,90,'document','green')],[edge('docs','chunk'),edge('chunk','index'),edge('question','retrieve'),edge('index','retrieve','search',[(600,205),(360,205)]),edge('retrieve','prompt'),edge('prompt','llm'),edge('llm','answer')],'Preparation happens when sources change. Retrieval and generation run for each question.')
add('02-parsing','Parsing and chunking change different things',[
 node('source','Original document','Reading order + visual layout',30,45,275,92,'document'),node('page','A policy table','Severity | Target\nP1 | 15 minutes',390,45,275,92,'document','slate'),
 node('parse','Parse structure','Recover text, headers and rows',30,215,275,92),node('elements','Normalized elements','Table header + typed rows\nSource: page 2, table 1',390,215,275,92,'document','violet'),
 node('split','Choose boundaries','Group related evidence',30,385,275,92),node('chunks','Searchable chunk','P1 target: 15 minutes\nSource: page 2, table 1',390,385,275,92,'document','green')],[edge('source','parse'),edge('parse','split'),edge('source','page',dashed=True),edge('parse','elements',dashed=True),edge('split','chunks',dashed=True)],'A splitter cannot restore a table header that parsing discarded.')
add('03-ingestion','Publish a complete version',[
 node('accept','Accept version','Stable source + revision ID',30,35),node('stage','Stage derived data','Parse · chunk · embed',30,200),node('check','Validate completeness','All expected artifacts exist?',30,365,260,100,'gate','violet'),node('publish','Publish version','Atomically switch active pointer',390,365,280,100,'process','green'),node('retry','Record failure + retry','Keep the current version active',390,200,280,92,'process','amber')],[edge('accept','stage'),edge('stage','check'),edge('check','publish','pass'),edge('check','retry','fail',[(340,415),(340,246)]),edge('retry','stage','retry')],'Staging can fail without exposing a partially built index to readers.')
add('04-compatibility','One vector space, one search contract',[
 node('query','Query encoder','Model · revision · normalization',25,35,295,100),node('index','Index profile','Document encoder + metric',380,35,295,100,'database','violet'),node('gate','Profiles compatible?','Validate before similarity search',215,235,275,105,'gate','violet'),node('search','Search','Compare compatible vectors',25,445,295,92,'process','green'),node('reject','Reject or reroute','Use the matching index',380,445,295,92,'process','amber')],[edge('query','gate'),edge('index','gate'),edge('gate','search','yes'),edge('gate','reject','no')],'Matching vector dimensions alone does not establish compatibility.')
add('05-expansion','Search a sentence; send a useful source region',[
 node('hit','Sentence hit','"Acknowledge P1 within 15 min."',30,35,310,100,'document'),node('parent','Parent section','Deadline + scope + exception',390,35,280,100,'document','violet'),node('dedup','Deduplicate regions','Three hits → one parent section',390,225,280,100),node('context','Final context','780 / 900 tokens\nPolicy + escalation owner',30,415,310,100,'document','green'),node('budget','Pack within budget','520 policy + 260 escalation',390,415,280,100)],[edge('hit','parent','expand'),edge('parent','dedup'),edge('dedup','budget'),edge('budget','context')],'Keep the triggering sentence ID when expanding; count the parent only once.')
add('06-hybrid','Two retrieval signals, one candidate list',[
 node('query','Question','"API key rotation endpoint"',225,25,280,92),node('dense','Dense search','Semantic similarity',25,200,285,92),node('keyword','Keyword search','Exact identifiers and terms',415,200,285,92,'process','violet'),node('fusion','Rank fusion','Combine ranks by stable chunk ID',225,385,280,92,'process','green'),node('rerank','Rerank + pack','Score relevance, then fit context',225,555,280,92,'process','green')],[edge('query','dense'),edge('query','keyword'),edge('dense','fusion'),edge('keyword','fusion'),edge('fusion','rerank')],'Fusion combines ranked lists. A reranker separately evaluates query–passage relevance.')
add('06-correction','A bounded retrieval retry',[
 node('first','Retrieve','Inspect evidence coverage',30,30,270,92),node('gate','Enough support?','Can evidence answer the question?',30,205,270,100,'gate','violet'),node('answer','Generate','Use the supported evidence',405,205,270,100,'process','green'),node('retry','Retry once','Revise query or retrieval strategy',30,400,270,100,'process','amber'),node('last','Check support again','Enough → generate\nStill missing → abstain',405,400,270,100,'process','amber')],[edge('first','gate'),edge('gate','answer','yes'),edge('gate','retry','no'),edge('retry','last'),edge('last','answer','supported')],'A retry has a budget and an explicit stop. Similarity alone is not an evidence-sufficiency test.')
add('07-routing','Choose the operation the question requires',[
 node('question','Question','Explain a rule, or calculate a value?',200,25,300,100),node('route','Route by operation','Preserve filters and requested units',200,215,300,100,'gate','violet'),node('passage','Retrieve passages','Explain the P1 policy',25,410,300,100),node('rows','Query eligible rows','AVG(response_minutes) for P1',380,410,300,100,'database','violet'),node('response','Compose the answer','Cite passages or the query result',200,610,300,100,'document','green')],[edge('question','route'),edge('route','passage','explain'),edge('route','rows','aggregate'),edge('passage','response'),edge('rows','response')],'Nearest passages are not a complete population for an aggregate calculation.')
add('08-citations','A citation must pass three different checks',[
 node('claim','Generated claim','"The target is 30 minutes. [C1]"',195,30,330,105,'document'),node('shape','1. Structure','Expected fields and types?',25,240,210,120,'gate'),node('identity','2. Source identity','Does C1 exist in the registry?',255,240,210,120,'gate','violet'),node('support','3. Evidence support','Does C1 actually say 30 min?',485,240,210,120,'gate','amber'),node('source','Supplied source C1','"The target is 15 minutes."',410,460,285,100,'document','amber')],[edge('claim','shape'),edge('claim','identity'),edge('claim','support'),edge('source','support','compare')],'Well-formed output and a real citation ID can still carry an unsupported claim.')
add('09-evaluation','Compare systems on the same cases',[
 node('dataset','Fixed evaluation set','Questions + evidence labels',200,25,310,100,'document'),node('baseline','Baseline','Frozen source and config',25,220,300,100),node('candidate','Candidate','One controlled change',395,220,300,100,'process','violet'),node('compare','Compare each case','Retrieval · context · answer',200,415,310,100,'process','green'),node('inspect','Inspect changed failures','Explain gains and regressions',200,595,310,100,'document','green')],[edge('dataset','baseline'),edge('dataset','candidate'),edge('baseline','compare'),edge('candidate','compare'),edge('compare','inspect')],'A higher average is useful only when the changed cases are understood.')
add('10-authorization','Access scope travels with the request',[
 node('request','Request identity','Caller + requested collection',25,25,300,100),node('access','Access allowed?','Apply the current policy',25,205,300,100,'gate','violet'),node('deny','Deny','Return no protected context',405,205,275,100,'process','amber'),node('retrieve','Scoped retrieval','Enforce allowed source IDs',25,395,300,100),node('context','Authorized context','Check expansion + cache scope',405,395,275,100,'document','violet'),node('generate','Generate','Use only the admitted context',405,585,275,100,'process','green')],[edge('request','access'),edge('access','deny','no'),edge('access','retrieve','yes'),edge('retrieve','context'),edge('context','generate')],'Apply authorization during retrieval, expansion, and cache reuse—not only in the prompt.')

# Separate the retry route from the publish route and return edge.
ingestion=next(d for d in DIAGRAMS if d['slug']=='03-ingestion')
ingestion['edges'][3]['points']=[(340,385),(340,330),(530,330)]
ingestion['edges'][3]['style']='exitX=1;exitY=0.2;entryX=0.5;entryY=1;'
ingestion['edges'][3]['offset']=(23,0)
ingestion['edges'][2]['style']='exitX=1;exitY=0.75;entryX=0;entryY=0.75;'
ingestion['edges'][2]['offset']=(0,-16)
ingestion['edges'][4]['offset']=(0,-16)
next(d for d in DIAGRAMS if d['slug']=='05-expansion')['edges'][0]['offset']=(0,-18)

# Article-specific compositions: layout and visual grammar follow the mechanism.
def redesign(slug,layout,positions,extra=None):
 d=next(d for d in DIAGRAMS if d['slug']==slug);d['layout']=layout
 for n in d['nodes']:
  if n['id'] in positions:
   n['x'],n['y'],n['w'],n['h']=positions[n['id']]
 for e in d['edges']:
  e['points']=[];e.pop('style',None);e.pop('offset',None)
 if extra:extra(d)

def parsing(d):
 for n in d['nodes']:
  if n['id'] in ('source','parse','split'):n['kind']='rail'
redesign('02-parsing','editorial-rail',{'source':(25,40,260,100),'parse':(25,225,260,100),'split':(25,410,260,100),'page':(395,30,300,130),'elements':(395,210,300,135),'chunks':(395,395,300,135)},parsing)

def ingestion(d):
 for n in d['nodes']:
  if n['id']=='check':n['kind']='diamond';n['title']='Check build';n['detail']='All artifacts ready?'
  if n['id']=='accept':n['kind']='terminal'
 d['edges']=[edge('accept','stage'),edge('stage','check'),edge('check','publish','pass'),edge('check','retry','fail'),edge('retry','stage','retry',[(142,195),(280,195),(280,97)])]
 d['edges'][-1]['style']='exitX=0.5;exitY=0;entryX=0;entryY=0.5;'
 d['edges'][-1]['offset']=(0,-15)
redesign('03-ingestion','release-lifecycle',{'accept':(25,50,220,95),'stage':(300,50,270,95),'check':(320,220,230,170),'publish':(645,255,275,100),'retry':(25,255,235,100)},ingestion)

def compatibility(d):
 for n in d['nodes']:
  if n['id'] in ('query','index'):n['kind']='profile'
  if n['id']=='gate':n['kind']='rail'
redesign('04-compatibility','contract-comparison',{'query':(30,40,305,135),'index':(425,40,305,135),'gate':(225,270,310,110),'search':(30,485,305,100),'reject':(425,485,305,100)},compatibility)

def hybrid(d):
 for n in d['nodes']:
  n['kind']='rail' if n['id'] in ('dense','keyword') else 'process'
 d['edges']=[edge('query','dense'),edge('query','keyword'),edge('dense','fusion'),edge('keyword','fusion',points=[(665,397),(665,280)]),edge('fusion','rerank')]
 d['edges'][3]['style']='exitX=1;exitY=0.5;entryX=0;entryY=0.75;'
redesign('06-hybrid','parallel-lanes',{'query':(25,195,220,115),'dense':(330,50,285,115),'keyword':(330,340,285,115),'fusion':(715,195,285,115),'rerank':(715,445,285,100)},hybrid)

def correction(d):
 for n in d['nodes']:
  if n['id']=='gate':n['kind']='diamond';n['detail']='Coverage check'
  if n['id']=='answer':n['kind']='terminal'
 d['edges']=[edge('first','gate'),edge('gate','answer','yes'),edge('gate','retry','no'),edge('retry','last'),edge('last','answer','supported')]
redesign('06-correction','bounded-loop',{'first':(25,45,250,100),'gate':(350,15,235,165),'answer':(690,45,270,100),'retry':(335,310,270,105),'last':(690,310,270,110)},correction)

def routing(d):
 for n in d['nodes']:
  if n['id']=='route':n['kind']='diamond';n['detail']='Preserve constraints'
  if n['id']=='passage':n['kind']='document'
redesign('07-routing','decision-tree',{'question':(240,25,330,100),'route':(280,200,250,180),'passage':(25,475,325,130),'rows':(460,475,325,130),'response':(240,700,330,120)},routing)

def citations(d):
 for n in d['nodes']:
  if n['id'] in ('shape','identity','support'):n['kind']='checkcard'
  if n['id']=='claim':n['kind']='rail'
 d['edges']=[edge('claim','shape'),edge('claim','identity'),edge('claim','support'),edge('source','support','compare')]
redesign('08-citations','validation-cards',{'claim':(270,25,340,105),'shape':(25,250,245,165),'identity':(325,250,245,165),'support':(625,250,245,165),'source':(590,530,305,125)},citations)

def authorization(d):
 # Native containers are behind the nodes and express the conceptual trust boundary.
 panels=[node('outside','UNTRUSTED REQUEST','',10,10,330,590,'panel','slate'),node('boundary','APPLICATION ACCESS BOUNDARY','',385,10,680,590,'panel','violet')]
 d['nodes']=panels+d['nodes']
 for n in d['nodes']:
  if n['id']=='access':n['kind']='diamond';n['detail']='Current permission'
  if n['id']=='deny':n['kind']='terminal'
 d['edges']=[edge('request','access','authenticate'),edge('access','deny','no'),edge('access','retrieve','yes'),edge('retrieve','context'),edge('context','generate')]
redesign('10-authorization','trust-boundaries',{'request':(45,210,260,115),'access':(420,170,240,185),'deny':(425,440,255,105),'retrieve':(745,90,285,105),'context':(745,265,285,120),'generate':(745,460,285,105)},authorization)

PALETTES={'light':{'bg':'#f8fafc','ink':'#15263a','muted':'#42536a','edge':'#586d86','blue':('#e6f3ff','#287cb5'),'violet':('#efedff','#7463c4'),'green':('#e2f5ed','#278162'),'amber':('#fff2df','#ad741b'),'slate':('#eef2f6','#74849a')},'dark':{'bg':'#101b2b','ink':'#f1f6fc','muted':'#c1ccdb','edge':'#9bb0cb','blue':('#15364c','#55b5df'),'violet':('#2d294c','#ac9cee'),'green':('#153d34','#66cba6'),'amber':('#45331e','#e1b36b'),'slate':('#263447','#9aaec8')}}
def xml(d,theme):
 c=PALETTES[theme];mf=E.Element('mxfile',host='app.diagrams.net',type='device'); page=E.SubElement(mf,'diagram',id=d['slug'],name=d['title']); model=E.SubElement(page,'mxGraphModel',grid='1',gridSize='10',page='0',background=c['bg']);root=E.SubElement(model,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
 for n in d['nodes']:
  fill,stroke=c[n['tone']]; shape={'document':'shape=document;boundedLbl=1;','database':'shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=12;','gate':'rounded=1;arcSize=30;dashed=1;','process':'rounded=1;arcSize=12;','rail':'rounded=0;align=left;spacingLeft=20;','terminal':'rounded=1;arcSize=50;','diamond':'shape=rhombus;perimeter=rhombusPerimeter;','profile':'shape=swimlane;startSize=42;rounded=0;','checkcard':'rounded=0;align=left;spacingLeft=18;','panel':'rounded=1;arcSize=5;dashed=1;verticalAlign=top;spacingTop=16;'}[n['kind']]
  label='<b>'+n['title']+'</b><br><font style="font-size:16px" color="'+c['muted']+'">'+n['detail'].replace('\n','<br>')+'</font>'
  if n['kind']=='panel':
   label=n['title'];fill=c['bg']
  if n['kind']=='profile':
   shape='rounded=0;align=left;spacingLeft=18;'
  if n['kind']=='rail':
   fill=stroke;stroke=fill
   label='<b>'+n['title']+'</b><br><font style="font-size:16px">'+n['detail'].replace('\n','<br>')+'</font>'
  extra=''
  if n['kind']=='rail':extra='fontColor='+('#ffffff' if theme=='light' else '#101b2b')+';'
  if n['kind']=='panel':extra='verticalAlign=top;fontSize=16;fontStyle=1;'
  cell=E.SubElement(root,'mxCell',id=n['id'],value=label,style=f'{shape}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.7;fontColor={c["ink"]};fontSize=19;fontFamily=Helvetica;spacing=12;verticalAlign=middle;{extra}',vertex='1',parent='1');E.SubElement(cell,'mxGeometry',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),attrib={'as':'geometry'})
 for i,e in enumerate(d['edges']):
  style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=20;html=1;strokeColor={c["edge"]};strokeWidth=2;endArrow=block;endFill=1;fontColor={c["ink"]};fontSize=16;fontFamily=Helvetica;labelBackgroundColor={c["bg"]};'
  style+=e.get('style','')
  if e['dashed']:style+='dashed=1;endArrow=open;'
  cell=E.SubElement(root,'mxCell',id=f'edge-{i}',value=e['label'],style=style,edge='1',parent='1',source=e['a'],target=e['b']);geo=E.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
  if e.get('offset'):
   x,y=e['offset'];E.SubElement(geo,'mxPoint',x=str(x),y=str(y),attrib={'as':'offset'})
  if e['points']:
   arr=E.SubElement(geo,'Array',attrib={'as':'points'})
   for x,y in e['points']:E.SubElement(arr,'mxPoint',x=str(x),y=str(y))
 return mf
combined=E.Element('mxfile',host='app.diagrams.net',type='device')
for d in DIAGRAMS:
 for theme in PALETTES:
  tree=xml(d,theme); dest=ROOT/(d['slug']+('.dark' if theme=='dark' else '')+'.drawio');E.indent(tree); E.ElementTree(tree).write(dest,encoding='unicode');(OUT/dest.name).write_bytes(dest.read_bytes())
  if theme=='light':combined.append(tree.find('diagram'))
E.indent(combined);E.ElementTree(combined).write(OUT/'rag-engineering.drawio',encoding='unicode')
(ROOT/'manifest.json').write_text(json.dumps(DIAGRAMS,indent=2)+'\n')
print(f'Authored {len(DIAGRAMS)} native diagrams in light/dark themes')
