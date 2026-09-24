import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

// AST text extraction keeps inequalities in equations from swallowing following prose.
export function interviewReadingTime(markdown: string): string {
  const tree = unified().use(remarkParse).use(remarkMath).parse(markdown);
  const words: string[] = [];
  function visit(node: { type: string; value?: string; children?: any[] }) {
    if (node.type === 'text' || node.type === 'inlineCode') words.push(node.value ?? '');
    node.children?.forEach(visit);
  }
  visit(tree);
  const count = words.join(' ').trim().split(/\s+/).filter(Boolean).length;
  return `${Math.max(1, Math.ceil(count / 220))} min read`;
}
