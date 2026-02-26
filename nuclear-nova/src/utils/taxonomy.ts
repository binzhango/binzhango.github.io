const KNOWN_CATEGORIES = [
  'python',
  'k8s',
  'spark',
  'ML',
  'airflow',
  'LLM',
  'Azure',
  'Snowflake',
  'Scala',
  'rust',
] as const;

const CATEGORY_LOOKUP = new Map(
  KNOWN_CATEGORIES.map((category) => [category.toLowerCase(), category])
);

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/["']/g, '')
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function normalizeCategory(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return trimmed;
  return CATEGORY_LOOKUP.get(trimmed.toLowerCase()) ?? trimmed;
}

export function slugifyCategory(name: string): string {
  return slugify(normalizeCategory(name));
}

export function slugifyTag(name: string): string {
  return slugify(name);
}
