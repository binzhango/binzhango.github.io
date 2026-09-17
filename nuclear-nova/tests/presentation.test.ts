import { describe, expect, it } from 'vitest';
import { getTopicTone, getDisplayTitle } from '../src/utils/presentation';
import { getCodeText } from '../src/scripts/code-view';

describe('consistent discovery presentation', () => {
  it('normalizes legacy categories and keeps unknown topics on the default accent', () => {
    expect(getTopicTone('LLM')).toBe(getTopicTone('LARGE LANGUAGE MODELS'));
    expect(getTopicTone('AI ENGINEERING')).toBe('teal');
    expect(getTopicTone('DATA SCIENCE')).toBe('amber');
    expect(getTopicTone('ML')).toBe('rose');
    expect(getTopicTone()).toBe('blue');
    expect(getTopicTone('New subject')).toBe('blue');
  });
  it('separates the known series prefix without changing other titles', () => {
    const title = 'RAG Engineering 02: Chunking, Parsing, and the Evidence You Lose';
    expect(getDisplayTitle(title)).toEqual({ series: 'RAG Engineering / 02', title: 'Chunking, Parsing, and the Evidence You Lose' });
    for (const other of ['RAG Engineering', 'RAG Engineering 02:', 'From Prompt to Response: A Walkthrough']) {
      expect(getDisplayTitle(other)).toEqual({ title: other, series: undefined });
    }
  });
});

describe('code reader copy fidelity', () => {
  // Only the read-only DOM text interface is needed; wrapping and gutters are CSS.
  const source = (lines: string[], legacy = false) => ({
    textContent: 'visual text must not replace logical lines',
    querySelectorAll: () => lines.map(textContent => ({
      textContent: legacy ? `gutter ${textContent}` : textContent,
      querySelector: () => legacy ? { textContent } : null,
    })),
  }) as unknown as Element;
  it('preserves indentation, blank lines, literal escapes, and long lines', () => {
    const lines = ['def parse():', '    value = "\\n"', '', `    return "${'x'.repeat(500)}"`, ''];
    expect(getCodeText(source(lines))).toBe(lines.join('\n'));
  });
  it('excludes legacy gutter text and supports unhighlighted snippets', () => {
    expect(getCodeText(source(['  first', '', 'last'], true))).toBe('  first\n\nlast');
    expect(getCodeText({ textContent: 'a\n  b\n', querySelectorAll: () => [] } as unknown as Element)).toBe('a\n  b\n');
  });
});
