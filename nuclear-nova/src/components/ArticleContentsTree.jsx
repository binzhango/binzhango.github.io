import { TreeList } from '@astryxdesign/core/TreeList';

export default function ArticleContentsTree({ headings }) {
  const items = [];
  const ancestors = [];

  for (const { depth, slug, text } of headings) {
    while (ancestors.length && ancestors.at(-1).depth >= depth) ancestors.pop();

    const item = { id: slug, label: text, href: `#${encodeURIComponent(slug)}`, children: [], isExpanded: true };
    const parent = ancestors.at(-1)?.item;
    (parent?.children ?? items).push(item);
    ancestors.push({ depth, item });
  }

  return <TreeList aria-label="Article headings" items={items} density="compact" variant="lineGuides" />;
}
