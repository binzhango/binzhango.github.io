import { describe, expect, it } from 'vitest';
import rehypeReadableContent from '../src/plugins/rehype-readable-content.mjs';
import remarkLegacyEmoji from '../src/plugins/remark-legacy-emoji.mjs';

const element = (tagName: string, children: any[] = [], properties = {}) => ({ type: 'element', tagName, children, properties });
const text = (value: string) => ({ type: 'text', value });
const table = () => element('table', [element('thead', [element('tr', [element('th', [text('A')]), element('th', [text('B')])])])]);

describe('article rendering accessibility', () => {
  it('keeps table semantics inside a labelled keyboard scroll region, including blockquotes', () => {
    const original = table();
    const tree = element('root', [element('h2', [text('Comparison')]), element('blockquote', [original])]);
    rehypeReadableContent()(tree);
    const wrapper = tree.children[1].children[0];
    expect(wrapper.properties).toMatchObject({ role: 'region', tabIndex: 0, ariaLabel: 'Comparison table — scroll horizontally' });
    expect(wrapper.children[0]).toBe(original);
    rehypeReadableContent()(tree);
    expect(wrapper.children[0]).toBe(original);
  });
  it('preserves the authored RAG table scroll region without nesting another', () => {
    const original = table();
    const tree = element('root', [{ type: 'raw', value: '<div class="rag-table rag-table--header" role="region" tabindex="0">' }, text('\n'), original, { type: 'raw', value: '</div>' }]);
    rehypeReadableContent()(tree);
    expect(tree.children[2]).toBe(original);
  });
  it('separates a step label without changing heading text, anchors, or inline emphasis', () => {
    const heading = element('h2', [text('Step 2: Serving with '), element('em', [text('batching')])], { id: 'step-2-serving-with-batching' });
    const tree = element('root', [heading]);
    const readText = (node: any): string => node.value ?? node.children?.map(readText).join('') ?? '';
    rehypeReadableContent()(tree);
    expect(heading.properties).toMatchObject({ id: 'step-2-serving-with-batching' });
    expect(readText(heading)).toBe('Step 2: Serving with batching');
    expect(heading.children[2].tagName).toBe('em');
    rehypeReadableContent()(tree);
    expect(readText(heading)).toBe('Step 2: Serving with batching');
    expect(heading.children).toHaveLength(3);
  });
  it('makes overflowed code and math reachable without JavaScript', () => {
    const tree = element('root', [element('pre'), element('mjx-container')]);
    rehypeReadableContent()(tree);
    expect(tree.children.map((node) => node.properties.tabIndex)).toEqual([0, 0]);
  });
  it('renders migrated emoji in prose while preserving literal code examples', () => {
    const tree = { children: [text(':bulb: Tip'), { type: 'code', value: ':bulb:' }, { type: 'inlineCode', value: ':warning:' }] };
    remarkLegacyEmoji()(tree);
    expect(tree.children.map((node) => node.value)).toEqual(['💡 Tip', ':bulb:', ':warning:']);
  });
});
