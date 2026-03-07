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
    .replace(/<summary>[\s\S]*?<\/summary>/gi, ' ')
    .replace(/<\/?details>/gi, ' ')
    .replace(/^:{3,}\S*(?:\[[^\]]*\])?(?:\{[^}]*\})?\s*$/gm, ' ')
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

function isUsableExcerpt(rawBlock: string, cleanedBlock: string): boolean {
  if (!cleanedBlock) return false;

  const normalizedRaw = rawBlock.trim();

  if (!normalizedRaw) return false;
  if (/^#{1,6}\s+/.test(normalizedRaw)) return false;
  if (/^:{3,}/.test(normalizedRaw)) return false;
  if (/^<\/?details>/i.test(normalizedRaw)) return false;
  if (/^<summary>/i.test(normalizedRaw)) return false;
  if (/^(Question|Questions|Answer|Sample Code)\b/i.test(cleanedBlock)) return false;
  if (cleanedBlock.length < 35 && !/[.!?]/.test(cleanedBlock)) return false;

  return true;
}

function extractExcerptSource(input: string): string {
  const blocks = input
    .split(/\n\s*\n+/)
    .map((block) => ({
      raw: block,
      cleaned: toPlainText(block),
    }));

  for (const block of blocks) {
    if (isUsableExcerpt(block.raw, block.cleaned)) {
      return block.cleaned;
    }
  }

  return toPlainText(input);
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
  const cleaned = extractExcerptSource(source);
  return truncate(cleaned, maxLength);
}
