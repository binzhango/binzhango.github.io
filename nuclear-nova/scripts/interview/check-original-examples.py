"""Execute the original article's Python examples in their documented order.

Requires torch and transformers. The BERT example downloads its named public
checkpoint on first run; set HF_HOME to an isolated cache for reproducibility.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2] / 'src/content/interview'
shared = {}
for question in ['llm-058', 'llm-061', 'llm-062', 'llm-063', 'llm-070', 'llm-160']:
    text = (root / f'{question}.md').read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    namespace = shared if question in {'llm-061', 'llm-062', 'llm-063'} else {}
    for code in blocks:
        exec(compile(code, f'{question}.md', 'exec'), namespace)
    if question == 'llm-070':
        assert round(namespace['ce'].item(), 4) == 0.1698
        assert round(namespace['bce'].item(), 4) == 0.2201
    print(f'{question}: passed ({len(blocks)} code block)', flush=True)
print('All 6 original Python examples passed.')
