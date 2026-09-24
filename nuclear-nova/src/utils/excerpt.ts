import type { Nodes } from 'mdast';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';

interface ExcerptSource {
  data: {
    excerpt?: string;
    description?: string;
  };
  body?: string;
}

const ELLIPSIS = '...';
const parser = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkDirective)
  .use(remarkMath);

function inlineText(node: Nodes): string {
  switch (node.type) {
    case 'text':
    case 'inlineCode':
      return node.value;
    case 'break':
      return ' ';
    case 'html':
      return /^<br\s*\/?\s*>$/i.test(node.value) ? ' ' : '';
    case 'paragraph':
    case 'emphasis':
    case 'strong':
    case 'delete':
    case 'link':
    case 'linkReference':
      return node.children.map(inlineText).join('');
    default:
      return '';
  }
}

function extractExcerptSource(input: string): string {
  // Parse the whole document so blank lines inside code, diagrams, and logs
  // cannot turn their contents into candidate paragraphs.
  const tree = parser.parse(input);
  const nestedParagraphs = (node: Nodes): Nodes[] => {
    if (node.type === 'paragraph') return node.data?.directiveLabel ? [] : [node];
    if (node.type === 'blockquote' || node.type === 'containerDirective') {
      return node.children.flatMap(nestedParagraphs);
    }
    return [];
  };
  const paragraphs = [
    ...tree.children.filter((node) => node.type === 'paragraph'),
    ...tree.children.filter((node) => node.type !== 'paragraph').flatMap(nestedParagraphs),
  ]
    .map((node) => inlineText(node)
      .replace(/:(bulb|question_mark|exclamation|warning|smile):/g, '')
      .replace(/\s+/g, ' ')
      .replace(/\s+([.,!?;:])/g, '$1')
      .trim())
    .filter(Boolean);

  // Only fall back to another paragraph, never to flattened raw Markdown.
  return paragraphs.find((text) => text.length >= 35 || /[.!?]/.test(text))
    ?? paragraphs[0]
    ?? '';
}

function truncate(input: string, maxLength: number): string {
  if (input.length <= maxLength) return input;

  const clipped = input.slice(0, maxLength + 1);
  const lastWordBoundary = clipped.lastIndexOf(' ');
  const safeCutoff =
    lastWordBoundary > Math.floor(maxLength * 0.6) ? lastWordBoundary : maxLength;

  return `${clipped.slice(0, safeCutoff).trim()}${ELLIPSIS}`;
}

export function getPostExcerpt(post: ExcerptSource, maxLength = 150): string {
  const source = post.data.excerpt ?? post.data.description ?? post.body ?? '';
  const cleaned = extractExcerptSource(source);
  return truncate(cleaned, maxLength);
}
