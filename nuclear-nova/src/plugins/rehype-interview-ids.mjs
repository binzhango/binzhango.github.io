import { interviewId } from './remark-interview.mjs';

/** Independently compiled Markdown entries can contain colliding MathJax/heading IDs. */
export default function rehypeInterviewIds() {
  return (tree, file) => {
    const id = interviewId(file);
    if (!id) return;
    const ids = new Map();
    const walk = (node, visit) => { visit(node); node.children?.forEach((child) => walk(child, visit)); };
    walk(tree, (node) => {
      const old = node.properties?.id;
      if (old && old !== `prompt-${id}`) {
        const next = `${id}-${old}`;
        ids.set(String(old), next); node.properties.id = next;
      }
    });
    walk(tree, (node) => {
      if (!node.properties) return;
      for (const [key, value] of Object.entries(node.properties)) {
        if (['href', 'xLinkHref'].includes(key) && typeof value === 'string' && ids.has(value.slice(1)) && value.startsWith('#')) node.properties[key] = `#${ids.get(value.slice(1))}`;
        if (['ariaLabelledBy', 'ariaDescribedBy', 'headers'].includes(key)) {
          const tokens = Array.isArray(value) ? value : String(value).split(/\s+/);
          node.properties[key] = tokens.map((token) => ids.get(token) ?? token).join(' ');
        }
        if (typeof value === 'string' && value.includes('url(#')) node.properties[key] = value.replace(/url\(#([^)]+)\)/g, (match, target) => ids.has(target) ? `url(#${ids.get(target)})` : match);
      }
    });
  };
}
