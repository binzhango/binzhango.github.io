/** Add native, keyboard-accessible scrolling without depending on client JavaScript. */
export default function rehypeReadableContent() {
  return (tree) => {
    let heading = 'Article';
    const text = (node) => node.value ?? node.children?.map(text).join('') ?? '';
    function walk(parent) {
      if (!parent.children) return;
      for (let i = 0; i < parent.children.length; i++) {
        const node = parent.children[i];
        if (/^h[1-6]$/.test(node.tagName ?? '')) heading = text(node);
        if (node.tagName === 'h2' && node.children?.[0]?.type === 'text') {
          const first = node.children[0];
          const step = first.value.match(/^(Step \d+:)(\s+)(.+)$/);
          if (step) {
            node.properties ??= {};
            node.properties.className = [...(node.properties.className ?? []), 'chapter-heading'];
            node.children.splice(0, 1,
              { type: 'element', tagName: 'span', properties: { className: ['chapter-heading__label'] }, children: [{ type: 'text', value: step[1] }] },
              { type: 'text', value: step[2] + step[3] }
            );
          }
        }
        if (node.tagName === 'pre' || node.tagName === 'mjx-container') {
          node.properties ??= {};
          node.properties.tabIndex = 0;
          if (node.tagName === 'pre') node.properties.ariaLabel = 'Code example';
        }
        if (node.tagName === 'table') {
          // Authored RAG tables already have labelled scroll regions. Raw HTML
          // wrappers remain sibling nodes in Astro's Markdown HAST.
          const previous = parent.children.slice(0, i).findLast(
            (item) => item.type !== 'text' || item.value.trim()
          );
          const authoredWrapper = previous?.type === 'raw' &&
            /<div\b[^>]*class=["'][^"']*\brag-table\b/.test(previous.value);
          if (!authoredWrapper && !parent.properties?.className?.includes('table-scroll')) {
            const header = node.children.find((child) => child.tagName === 'thead');
            const row = header?.children.find((child) => child.tagName === 'tr');
            const columns = row?.children.filter((child) => child.tagName === 'th').length ?? 3;
            const width = Math.min(60, Math.max(30, columns * 12));
            parent.children[i] = {
              type: 'element', tagName: 'div',
              properties: {
                className: ['table-scroll'], role: 'region', tabIndex: 0,
                ariaLabel: `${heading} table — scroll horizontally`,
                style: `--table-min-width: ${width}rem`,
              },
              children: [node],
            };
          }
          // Nested authored HTML remains intact; only the outer table scrolls.
          continue;
        }
        walk(node);
      }
    }
    walk(tree);
  };
}
