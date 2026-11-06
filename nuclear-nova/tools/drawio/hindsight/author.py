"""Native draw.io diagrams, with compact mobile compositions of the same graph."""
from pathlib import Path
from xml.etree import ElementTree as E
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / 'public/diagrams/hindsight'
OUT.mkdir(parents=True, exist_ok=True)
graphs = [
    dict(slug='decision-journey', title='A decision survives the conversation',
         nodes=[('source', '1 · Record decision', 'Planner records Alice’s constraints', 'process'),
                ('bank', '2 · Memory bank', 'Retain extracts and stores facts', 'store'),
                ('coder', '3 · New coder', 'Recall → evidence → model answer', 'process')],
         edges=[('source', 'bank'), ('bank', 'coder')],
         desktop=[(0, 0, 220, 140), (280, 0, 220, 140), (560, 0, 220, 140)],
         mobile=[(0, 0, 300, 120), (0, 170, 300, 120), (0, 340, 300, 120)]),
    dict(slug='recall-pipeline', title='From a question to selected evidence',
         nodes=[('query', 'Question + bank', 'Scope the search', 'process'),
                ('semantic', 'Semantic + keyword', 'Meaning and exact terms', 'process'),
                ('graph', 'Graph + temporal', 'Relationships and time', 'process'),
                ('fusion', 'Fuse ranked lists', 'Reciprocal rank fusion', 'process'),
                ('rank', 'Rerank + diversify', 'Relevance and coverage', 'process'),
                ('evidence', 'Return evidence', 'Fit the fact-token budget', 'process')],
         edges=[('query', 'semantic'), ('query', 'graph'), ('semantic', 'fusion'), ('graph', 'fusion'), ('fusion', 'rank'), ('rank', 'evidence')],
         desktop=[(280,0,220,90),(0,150,220,90),(560,150,220,90),(280,300,220,90),(280,450,220,90),(280,600,220,110)],
         mobile=[(40,0,260,90),(0,150,160,140),(180,150,160,140),(40,350,260,90),(40,500,260,90),(40,650,260,100)])
]
manifest=[]
for graph in graphs:
    for layout in ('desktop', 'mobile'):
        slug=graph['slug'] + ('.mobile' if layout == 'mobile' else '')
        manifest.append(dict(slug=slug,title=graph['title'],nodes=graph['nodes'],edges=graph['edges']))
        for theme in ('light', 'dark'):
            dark=theme=='dark'
            bg,ink,muted,fill,stroke=('#101b2b','#f1f6fc','#c1ccdb','#153d34','#66cba6') if dark else ('#f8fafc','#15263a','#42536a','#e2f5ed','#278162')
            mx=E.Element('mxfile',host='app.diagrams.net',type='device')
            diagram=E.SubElement(mx,'diagram',id=slug,name=graph['title'])
            model=E.SubElement(diagram,'mxGraphModel',grid='1',page='0',background=bg)
            root=E.SubElement(model,'root')
            E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
            for (id,title,detail,kind),(x,y,w,h) in zip(graph['nodes'],graph[layout]):
                shape='shape=cylinder3;boundedLbl=1;size=12;' if kind=='store' else 'rounded=0;'
                label=f'<b>{title}</b><br><font color="{muted}" style="font-size:17px">{detail}</font>'
                cell=E.SubElement(root,'mxCell',id=id,value=label,vertex='1',parent='1',style=f'{shape}html=1;whiteSpace=wrap;fillColor={fill};strokeColor={stroke};fontColor={ink};fontFamily=Helvetica;fontSize=20;spacing=12;strokeWidth=1.5;')
                E.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
            for i,(a,b) in enumerate(graph['edges']):
                cell=E.SubElement(root,'mxCell',id=f'e{i}',edge='1',parent='1',source=a,target=b,style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;endFill=1;strokeWidth=2;strokeColor={stroke};')
                E.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
            E.indent(mx)
            (OUT/(slug+('.dark' if dark else '')+'.drawio')).write_text(E.tostring(mx,encoding='unicode'))
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Authored two native graphs in desktop/mobile and light/dark variants')
