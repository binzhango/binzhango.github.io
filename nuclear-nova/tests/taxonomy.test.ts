import { describe, expect, it } from 'vitest';

import { normalizeCategory, slugifyCategory, slugifyTag } from '../src/utils/taxonomy';

describe('taxonomy helpers', () => {
  it('normalizes known categories to canonical labels', () => {
    expect(normalizeCategory('ml')).toBe('ML');
    expect(normalizeCategory('azure')).toBe('Azure');
    expect(normalizeCategory('scala')).toBe('Scala');
  });

  it('slugifies category links consistently', () => {
    expect(slugifyCategory('ML')).toBe('ml');
    expect(slugifyCategory('Snowflake')).toBe('snowflake');
    expect(slugifyCategory('  Data Platform  ')).toBe('data-platform');
  });

  it('slugifies tags with spacing/case normalization', () => {
    expect(slugifyTag('Data Engineer')).toBe('data-engineer');
    expect(slugifyTag('Data   Science')).toBe('data-science');
    expect(slugifyTag('MCP & AI')).toBe('mcp-and-ai');
  });
});
