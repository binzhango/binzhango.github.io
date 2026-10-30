"""One-time, lossless source extraction. Never overwrite authored question files."""
from pathlib import Path
import re, json, hashlib

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'docs/interview/sources'
OUT = ROOT / 'src/content/interview'
TOPICS = {
    'ML and DL Fundamentals': 'ml-foundations', 'Machine Learning': 'ml-foundations',
    'LLM Fundamentals': 'ml-foundations',
    'LLM Internals and Architecture': 'llm-architecture',
    'Fundamentals of Large Language Models (LLMs)': 'llm-architecture',
    'Advanced Topics in LLMs': 'llm-architecture',
    'Fine-Tuning Large Models': 'training', 'Parameter-Efficient Fine-Tuning (PEFT)': 'training',
    'Fine-Tuning, Post-Training and Alignment': 'training',
    'Inference, Serving and GPU Performance': 'inference',
    'Retrieval-Augmented Generation (RAG)': 'rag', 'RAG and Retrieval': 'rag',
    'LangChain and Agent-Based Systems': 'agents', 'Agents and Tool Use': 'agents',
    'Evaluation and Observability': 'evaluation', 'Safety, Security and Responsible AI': 'safety',
    'Multimodal, Speech and Voice AI': 'multimodal', 'AI System Design': 'system-design',
    'Coding and Data Structures': 'coding', 'Applied and Forward-Deployed Scenarios': 'applied',
    'Behavioral and Culture': 'behavioral',
}

def original():
    s=(SOURCES/'original-llm-interview.md').read_text()
    matches=list(re.finditer(r':::note\[Question: (.*?)\]\{\.question-note\}\s*:::',s))
    result=[]
    for i,m in enumerate(matches):
        section=re.findall(r'^## (.+)$',s[:m.start()],re.M)[-1]
        subs=re.findall(r'<summary>(.*?)</summary>',s[:m.start()])
        sub=next((x for x in reversed(subs) if x!='Detailed answer'),section)
        start=s.index('**Brief answer.**',m.end())+len('**Brief answer.**')
        detailed=s.index('<summary>Detailed answer</summary>',start)
        brief=s[start:s.rindex('<details>',start,detailed)].strip()
        content_start=detailed+len('<summary>Detailed answer</summary>')
        # Balance native details rather than truncating a nested explanation.
        depth=1;end=None
        for tag in re.finditer(r'</?details>',s[content_start:]):
            depth+=-1 if tag.group().startswith('</') else 1
            if depth==0: end=content_start+tag.start();break
        assert end is not None
        q=m[1]
        if q=='How does it differ from full fine-tuning?':q='How does classic adapter tuning differ from full fine-tuning?'
        result.append(dict(sourceId=f'original-{i+1:03}',questionId=f'llm-{i+1:03}',question=q,originalQuestion=m[1],topic=TOPICS[section],subtopic=sub,brief=brief,detail=s[content_start:end].strip(),kind='factual',source='original',references=[]))
    assert len(result)==167
    return result

def additional():
    s=(SOURCES/'additional-questions.md').read_text().split('## Common Questions Asked Across Companies\n',1)[1].split('### License')[0]
    result=[];topic='';sub=''
    for line in s.splitlines():
        if line.startswith(('### ','#### ')):
            heading=line.lstrip('# ').strip()
            if heading in TOPICS:topic=TOPICS[heading];sub=heading
        if line.startswith('- '):
            q=line[2:];i=len(result)+1
            result.append(dict(sourceId=f'additional-{i:03}',questionId=f'ai-{i:03}',question=q,originalQuestion=q,topic=topic,subtopic=sub,brief='',detail='',kind='behavioral' if topic=='behavioral' else 'coding' if topic=='coding' else 'design' if topic in ['system-design','applied'] or q.lower().startswith('design ') else 'factual',source='additional',references=[]))
        elif line.startswith('  - Answer:'):
            result[-1]['references'] += [dict(label=a,url=b) for a,b in re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)',line)]
    assert len(result)==598
    return result

def normalize(q):return re.sub(r'[^\w]+',' ',q.casefold()).strip()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    seen={};manifest=[];rows=original()+additional()
    for order,row in enumerate(rows):
        key=normalize(row['question']);target=seen.get(key)
        if target:
            manifest.append(dict(sourceId=row['sourceId'],originalQuestion=row['originalQuestion'],questionId=target,action='duplicate',reason='Identical question after case/punctuation normalization'))
            continue
        seen[key]=row['questionId']
        manifest.append(dict(sourceId=row['sourceId'],originalQuestion=row['originalQuestion'],questionId=row['questionId'],action='retained',reason='Distinct prompt; source constraints preserved'))
        metadata=dict(questionId=row['questionId'],topic=row['topic'],subtopic=row['subtopic'],order=order+1,kind=row['kind'],status='reviewed' if row['source']=='original' else 'draft',reviewedOn='2026-09-19' if row['source']=='original' else None,sources=[dict(label='Original LLM interview article' if row['source']=='original' else 'Outcome School question collection',url='https://binzhango.net/posts/2025/10-30-llm-interview/' if row['source']=='original' else 'https://github.com/pallavi-shekhar/ai-engineering-interview-questions-company-wise'),*row['references']])
        body='---\n'+'\n'.join(k+': '+json.dumps(v,ensure_ascii=False) for k,v in metadata.items())+'\n---\n\n## Question\n\n'+row['question']+'\n\n## Brief answer\n\n'+row['brief']+'\n\n## Detailed answer\n\n'+row['detail']+'\n'
        path=OUT/(row['questionId']+'.md')
        if not path.exists():path.write_text(body)
    report=dict(sourceCounts={'original':167,'additional':598},sourceHashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.glob('*.md')},entries=manifest)
    (ROOT/'docs/interview/reconciliation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'docs/interview/extracted.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(rows)} source entries; {len(seen)} canonical questions; {len(rows)-len(seen)} exact duplicates')

if __name__=='__main__':main()
