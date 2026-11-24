const CATEGORY_LABELS = new Map([
  ['AI ENGINEERING', 'AI engineering'],
  ['DATA ENGINEERING', 'Data engineering'],
  ['DATA SCIENCE', 'Data science'],
  ['DEVOPS', 'DevOps'],
  ['LARGE LANGUAGE MODELS', 'Large language models'],
  ['MACHINE LEARNING', 'Machine learning'],
  ['SOFTWARE ENGINEERING', 'Software engineering'],
]);

const CATEGORY_LOOKUP = new Map(
  Array.from(CATEGORY_LABELS.keys(), (category) => [category.toLowerCase(), category])
);

const LEGACY_CATEGORY_LOOKUP = new Map([
  ['airflow', 'DATA ENGINEERING'],
  ['azure', 'DATA ENGINEERING'],
  ['k8s', 'DEVOPS'],
  ['llm', 'LARGE LANGUAGE MODELS'],
  ['ml', 'MACHINE LEARNING'],
  ['python', 'SOFTWARE ENGINEERING'],
  ['rust', 'SOFTWARE ENGINEERING'],
  ['scala', 'SOFTWARE ENGINEERING'],
  ['snowflake', 'DATA SCIENCE'],
  ['spark', 'DATA ENGINEERING'],
]);

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
  const lower = trimmed.toLowerCase();
  return CATEGORY_LOOKUP.get(lower) ?? LEGACY_CATEGORY_LOOKUP.get(lower) ?? trimmed;
}

export function slugifyCategory(name: string): string {
  return slugify(normalizeCategory(name));
}

export function getCategoryLabel(name: string): string {
  const category = normalizeCategory(name);
  return CATEGORY_LABELS.get(category) ?? category;
}

export function slugifyTag(name: string): string {
  return slugify(name);
}
