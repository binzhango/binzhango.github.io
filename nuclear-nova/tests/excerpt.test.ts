import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import matter from 'gray-matter';

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

  it.each([
    '| Feature | Transport |\n| --- | --- |\n| Streaming | Supported |',
    'Feature | Transport\n:--- | ---:\nStreaming | Supported',
    '> | Feature | Transport |\n> | :---: | --- |\n> | Streaming | Supported |',
  ])('skips a leading table and selects the following prose: %s', (table) => {
    const paragraph = 'The server and client communicate using a configured transport.';
    expect(getPostExcerpt({ data: {}, body: `# Overview\n\n${table}\n\n${paragraph}` }))
      .toBe(paragraph);
  });

  it('does not return table markup when the article contains only a table', () => {
    expect(getPostExcerpt({ data: {}, body: '| Feature | Transport |\n| --- | --- |\n| Streaming | Supported |' }))
      .toBe('');
  });

  it('keeps short fallback text without reintroducing a table', () => {
    expect(getPostExcerpt({ data: {}, body: 'Overview\n\n| Feature | Transport |\n| --- | --- |\n| Streaming | Supported |' }))
      .toBe('Overview');
  });

  it('preserves prose next to a table without a blank line', () => {
    const paragraph = 'The server and client communicate using a configured transport.';
    expect(getPostExcerpt({ data: {}, body: `${paragraph}\n| Feature | Transport |\n| --- | --- |\n| Streaming | Supported |` }))
      .toBe(paragraph);
  });

  it('preserves literal pipes in prose', () => {
    const paragraph = 'Use the shell pipe | to connect one command to another.';
    expect(getPostExcerpt({ data: {}, body: paragraph })).toBe(paragraph);
  });

  it.each([
    '```python linenums="1"\nimport airflow\n\nspark = SparkSession.builder.getOrCreate()\n```',
    '```mermaid\n\n%%{init: { look: "handDrawn" } }%%\n\nflowchart LR\nA --> B\n```',
    '```sh\nExecution logs:\n\nOut: None\n[Step 1: Duration 146.87 seconds]\n```',
    '~~~python\nimport airflow\n\nprint("hello")\n~~~',
    '````markdown\n```python\n\nprint("nested fence")\n```\n````',
    '> ```sh\n> Execution logs:\n>\n> Out: None\n> ```',
    '    import airflow\n\n    print("indented code")',
    '<pre>Execution logs:\n\nOut: None</pre>',
    '<!-- hidden comment\n\nmore hidden text -->',
    '$$\nx = y\n\nz = x + y\n$$',
  ])('excludes complete non-prose blocks, including internal blank lines: %s', (block) => {
    const paragraph = 'This paragraph explains how the application works.';
    expect(getPostExcerpt({ data: {}, body: `${block}\n\n${paragraph}` })).toBe(paragraph);
    expect(getPostExcerpt({ data: {}, body: block })).toBe('');
  });

  it('does not flatten an unclosed code fence as a fallback', () => {
    expect(getPostExcerpt({ data: {}, body: '# Example\n\n```python\n\nimport airflow' })).toBe('');
  });

  it('finds prose directly after a heading and ignores image alt text', () => {
    const paragraph = 'This introduction explains the application architecture.';
    expect(getPostExcerpt({ data: {}, body: `# Overview\n${paragraph}\n\n![type:video](video.mp4)` }))
      .toBe(paragraph);
  });

  it('skips a table of contents before the introduction', () => {
    const paragraph = 'The goal is to implement and train a small language model.';
    expect(getPostExcerpt({ data: {}, body: `1. [Objective](#objective)\n2. [Setup](#setup)\n\n# Objective\n${paragraph}` }))
      .toBe(paragraph);
  });

  it('keeps inline code identifiers, reference links, and decoded entities', () => {
    expect(getPostExcerpt({ data: {}, body: 'Use `my_client` with **[Docs][ref]** &amp; <b>examples</b>.\n\n[ref]: https://example.com' }))
      .toBe('Use my_client with Docs & examples.');
  });

  it('uses callout prose without leaking directive titles or code', () => {
    const paragraph = 'The graph manages state across multiple workflow steps.';
    expect(getPostExcerpt({ data: {}, body: `:::tip[Answer]\n${paragraph}\n\n\`\`\`python\n\nprint("log")\n\`\`\`\n:::` }))
      .toBe(paragraph);
  });

  it('prefers article prose over a decorative quotation', () => {
    const paragraph = 'This article explains how to build a useful application.';
    expect(getPostExcerpt({ data: {}, body: `> A decorative quotation that is long enough for a preview.\n\n${paragraph}` }))
      .toBe(paragraph);
  });

  it('prefers excerpt, then description, then article prose', () => {
    const data = { excerpt: 'An authored excerpt.', description: 'An authored description.' };
    const body = 'This is the article body.';
    expect(getPostExcerpt({ data, body })).toBe(data.excerpt);
    expect(getPostExcerpt({ data: { description: data.description }, body })).toBe(data.description);
    expect(getPostExcerpt({ data: {}, body })).toBe(body);
  });

  it.each([
    ['2024/11-15-Pyspark Usage.md', 'Migrating a history Scala project to Python'],
    ['2024/12-15-Gradio with LLM.md', 'We have a lot of pdf files'],
    ['2025/05-04-text-to-sql.md', 'This is implemented from the guide Code'],
  ])('extracts actual prose from %s even without an authored summary', (file, beginning) => {
    const post = matter(readFileSync(new URL(`../src/content/docs/posts/${file}`, import.meta.url), 'utf8'));
    expect(getPostExcerpt({ data: {}, body: post.content })).toMatch(new RegExp(`^${beginning}`));
  });

  it('keeps every existing post preview free of code, diagrams, logs, and table markup', () => {
    const root = new URL('../src/content/docs/posts/', import.meta.url);
    const placeholders = new Set(['2012/repos.md', '2020/Whitening-transformation.md']);
    for (const file of readdirSync(root, { recursive: true, encoding: 'utf8' }).filter((file) => /\.mdx?$/.test(file))) {
      const post = matter(readFileSync(new URL(file, root), 'utf8'));
      const preview = getPostExcerpt({ data: post.data, body: post.content }, 180);
      expect(preview, file).not.toMatch(/```|~~~|%%\{|\|\s*:?-{3,}|Out: None|Input tokens:|type:video|linenums=|:::/);
      expect(preview.length, file).toBeLessThanOrEqual(183);
      if (!placeholders.has(file)) expect(preview.length, file).toBeGreaterThan(35);
    }
  });
});
