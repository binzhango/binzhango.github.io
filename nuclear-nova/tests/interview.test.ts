import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import { emptyProgress, filterQuestions, parseProgress, summarize } from '../src/utils/interview-state';
import { validateInterviewBank } from '../src/utils/interview-validation';
import { interviewReadingTime } from '../src/utils/interview-reading-time';
import remarkInterview from '../src/plugins/remark-interview.mjs';
import rehypeInterviewIds from '../src/plugins/rehype-interview-ids.mjs';

const questions = [
  { id: 'a', topic: 'inference', text: 'What is a KV cache?' },
  { id: 'b', topic: 'rag', text: 'When should retrieval results be cached?' },
  { id: 'c', topic: 'inference', text: 'How does prefix caching affect TTFT?' },
];
describe('interview filtering and saved self-assessment', () => {
  it('combines topic and all query terms, with whitespace/case normalization', () => {
    expect(filterQuestions(questions, 'inference', ' CACHE  kv ')).toEqual([questions[0]]);
    expect(filterQuestions(questions, 'rag', 'KV')).toEqual([]);
    expect(filterQuestions(questions, '', '  ')).toEqual(questions);
  });
  it('recovers from invalid storage and drops unknown IDs and rating values', () => {
    const ids = new Set(['a', 'b']);
    for (const raw of [null, '{bad', 'null', '{"version":2}', '[]']) expect(parseProgress(raw, ids)).toEqual(emptyProgress());
    expect(parseProgress(JSON.stringify({ version: 1, ratings: { a: 'again', b: 'mastered', z: 'got-it' }, lastQuestion: 'z' }), ids)).toEqual({ version: 1, ratings: { a: 'again' }, lastQuestion: null });
    expect(parseProgress(JSON.stringify({ version: 1, ratings: { a: 'got-it' }, lastQuestion: 'a' }), ids).lastQuestion).toBe('a');
  });
  it('counts explicit ratings only and restricts summary to the current set', () => {
    expect(summarize(['a', 'b', 'c'], { a: 'again', b: 'got-it', z: 'again' })).toEqual({ reviewed: 2, again: 1, gotIt: 1, unreviewed: 1 });
  });
});

const paragraph = (value: string) => ({ type: 'paragraph', children: [{ type: 'text', value }] });
const heading = (value: string) => ({ type: 'heading', depth: 2, children: [{ type: 'text', value }] });
describe('interview Markdown integration', () => {
  it('counts prose after math inequalities without counting hidden markup or code', () => {
    expect(interviewReadingTime('$1<G<H$\n\n' + 'word '.repeat(440) + '\n\n$x > 0$')).toBe('2 min read');
  });
  it('preserves code and math nodes inside native details, without touching unrelated articles', () => {
    const code = { type: 'code', lang: 'python', value: 'print(1)' };
    const math = { type: 'math', value: 'x^2' };
    const tree: any = { children: [heading('Question'), paragraph('Explain this.'), heading('Brief answer'), paragraph('Brief.'), heading('Detailed answer'), code, math] };
    const other = structuredClone(tree);
    remarkInterview()(other, { path: '/src/content/docs/posts/other.md' });
    expect(other).toEqual(tree);
    remarkInterview()(tree, { path: '/src/content/interview/a.md', data: { astro: { frontmatter: { kind: 'design' } } } });
    expect(tree.children[0].data.hProperties.className).toEqual(['interview-surface']);
    expect(tree.children[0].children[0].data.hProperties.id).toBe('prompt-a');
    expect(tree.children[1].data.hName).toBe('details');
    expect(tree.children[1].children[1].children).toEqual([code, math]);
    expect(JSON.stringify(tree)).toContain('Suggested approach');
  });
  it('namespaces IDs and local references for each independently rendered entry', () => {
    const tree = { children: [
      { properties: { id: 'prompt-a' } },
      { properties: { id: 'eq-1' } },
      { properties: { href: '#eq-1', ariaLabelledBy: 'eq-1 prompt-a' } },
    ] };
    rehypeInterviewIds()(tree, { path: '/src/content/interview/a.md' });
    expect(tree.children[0].properties.id).toBe('prompt-a');
    expect(tree.children[1].properties.id).toBe('a-eq-1');
    expect(tree.children[2].properties).toMatchObject({ href: '#a-eq-1', ariaLabelledBy: 'a-eq-1 prompt-a' });
  });
  it('rejects empty answers, ID mismatches, duplicate order and missing review', () => {
    const good = { id: 'a', body: '## Question\nQ\n## Brief answer\nB\n## Detailed answer\nD', data: { questionId: 'a', topic: 'rag', order: 1, status: 'reviewed', reviewedOn: '2026-09-20' } };
    expect(() => validateInterviewBank([good])).not.toThrow();
    expect(() => validateInterviewBank([{ ...good, body: good.body.replace('answer\nD', 'answer\n') }])).toThrow('empty Detailed answer');
    expect(() => validateInterviewBank([{ ...good, data: { ...good.data, status: 'draft' } }])).toThrow('not reviewed');
    expect(() => validateInterviewBank([good, { ...good, id: 'b' }])).toThrow('duplicate');
  });
});

describe('source reconciliation', () => {
  const root = new URL('../', import.meta.url);
  const report = JSON.parse(readFileSync(new URL('docs/interview/reconciliation.json', root), 'utf8'));
  const extracted = JSON.parse(readFileSync(new URL('docs/interview/extracted.json', root), 'utf8'));
  const records = readdirSync(new URL('src/content/interview/', root)).filter((file) => file.endsWith('.md')).map((file) => {
    const { data, content } = matter(readFileSync(new URL(`src/content/interview/${file}`, root), 'utf8'));
    return { id: file.slice(0, -3), data, body: content };
  });
  it('accounts for every source entry with a unique source identifier and a real target', () => {
    expect(report.entries).toHaveLength(765);
    expect(new Set(report.entries.map((entry: any) => entry.sourceId)).size).toBe(765);
    const ids = new Set(records.map((record) => record.id));
    for (const entry of report.entries) expect(ids.has(entry.questionId), entry.sourceId).toBe(true);
    expect(report.publishedCount).toBe(records.length);
  });
  it('requires reviewed complete content and preserves the snapshotted source hashes', () => {
    expect(() => validateInterviewBank(records as any)).not.toThrow();
    for (const [name, hash] of Object.entries(report.sourceHashes)) {
      expect(createHash('sha256').update(readFileSync(new URL(`docs/interview/sources/${name}`, root))).digest('hex')).toBe(hash);
    }
  });
  it('retains original code and equations through the authorized answer rewrite', () => {
    const parser = unified().use(remarkParse).use(remarkMath);
    const technicalNodes = (markdown: string) => {
      const result: { type: string; value: string }[] = [];
      const walk = (node: any) => {
        if (['code', 'math', 'inlineMath'].includes(node.type)) result.push({ type: node.type, value: node.value });
        node.children?.forEach(walk);
      };
      walk(parser.parse(markdown));
      return result;
    };
    const byId = new Map(records.map((record) => [record.id, record]));
    for (const source of extracted.filter((entry: any) => entry.source === 'original')) {
      const target = report.entries.find((entry: any) => entry.sourceId === source.sourceId).questionId;
      const retained = technicalNodes(byId.get(target)!.body);
      for (const node of technicalNodes(`${source.brief}\n\n${source.detail}`)) {
        expect(retained, `${source.sourceId}: ${node.type}`).toContainEqual(node);
      }
    }
  });
  it('keeps research links out of published question and answer prose', () => {
    const parser = unified().use(remarkParse);
    for (const record of records) {
      const walk = (node: any) => {
        if (node.type === 'link' || node.type === 'definition') {
          expect(node.url, record.id).not.toMatch(/^https?:/);
        }
        node.children?.forEach(walk);
      };
      walk(parser.parse(record.body));
    }
  });
});
