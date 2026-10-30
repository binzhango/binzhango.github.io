"""Apply an explicitly authored answer batch from stdin to canonical Markdown."""
from pathlib import Path
import sys,re
root=Path(__file__).resolve().parents[2]/'src/content/interview'
count=0
for part in re.split(r'^@@',sys.stdin.read(),flags=re.M)[1:]:
    number,body=part.split('\n',1)
    brief,detail=body.strip().split('\n\n',1)
    assert brief.strip() and detail.strip(),number
    path=root/f'ai-{int(number.strip()):03}.md'
    old=path.read_text()
    header=old.split('## Brief answer')[0]
    path.write_text(header+'## Brief answer\n\n'+brief.strip()+'\n\n## Detailed answer\n\n'+detail.strip()+'\n')
    count+=1
print(f'Wrote {count} authored answers; review status unchanged.')
