import { describe, expect, it } from 'vitest';

import { normalizeCategory, slugifyCategory, slugifyTag } from '../src/utils/taxonomy';

describe('taxonomy helpers', () => {
  it('normalizes known categories to canonical labels', () => {
    expect(normalizeCategory('ai engineering')).toBe('AI ENGINEERING');
    expect(normalizeCategory('data engineering')).toBe('DATA ENGINEERING');
    expect(normalizeCategory('large language models')).toBe('LARGE LANGUAGE MODELS');
    expect(normalizeCategory('machine learning')).toBe('MACHINE LEARNING');
  });

  it('maps legacy tool categories to broader shelves', () => {
    expect(normalizeCategory('llm')).toBe('LARGE LANGUAGE MODELS');
    expect(normalizeCategory('spark')).toBe('DATA ENGINEERING');
    expect(normalizeCategory('python')).toBe('SOFTWARE ENGINEERING');
  });

  it('slugifies category links consistently', () => {
    expect(slugifyCategory('AI ENGINEERING')).toBe('ai-engineering');
    expect(slugifyCategory('LARGE LANGUAGE MODELS')).toBe('large-language-models');
    expect(slugifyCategory('DATA SCIENCE')).toBe('data-science');
    expect(slugifyCategory('  Data Platform  ')).toBe('data-platform');
  });

  it('slugifies tags with spacing/case normalization', () => {
    expect(slugifyTag('Data Engineer')).toBe('data-engineer');
    expect(slugifyTag('Data   Science')).toBe('data-science');
    expect(slugifyTag('MCP & AI')).toBe('mcp-and-ai');
  });
});
