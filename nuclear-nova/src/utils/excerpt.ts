interface ExcerptSource {
  data: {
    excerpt?: string;
  };
  body?: string;
}

const ELLIPSIS = '...';

function toPlainText(input: string): string {
  return input
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/^[>#-]+\s?/gm, '')
    .replace(/[*_~]+/g, '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([.,!?;:])/g, '$1')
    .trim();
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
  const source = post.data.excerpt ?? post.body ?? '';
  const cleaned = toPlainText(source);
  return truncate(cleaned, maxLength);
}
