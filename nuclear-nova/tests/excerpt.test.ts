import { describe, expect, it } from 'vitest';

import { getPostExcerpt } from '../src/utils/excerpt';

describe('excerpt utility', () => {
  it('removes comments, markdown, and html tags', () => {
    const excerpt = getPostExcerpt(
      {
        data: {},
        body: 'Hello **World** <!-- more --> [Docs](https://example.com) <b>tag</b>.',
      },
      200
    );

    expect(excerpt).toBe('Hello World Docs tag.');
  });

  it('prefers frontmatter excerpt when provided', () => {
    const excerpt = getPostExcerpt(
      {
        data: { excerpt: 'Frontmatter `excerpt` with [link](https://example.com)' },
        body: 'Body content',
      },
      200
    );

    expect(excerpt).toBe('Frontmatter excerpt with link');
  });

  it('truncates long excerpts with ellipsis', () => {
    const excerpt = getPostExcerpt(
      {
        data: {},
        body: 'A '.repeat(200),
      },
      50
    );

    expect(excerpt.endsWith('...')).toBe(true);
    expect(excerpt.length).toBeLessThanOrEqual(53);
  });
});
