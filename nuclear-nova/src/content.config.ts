import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const allowedCategories = [
  'AI ENGINEERING',
  'DATA ENGINEERING',
  'DATA SCIENCE',
  'DEVOPS',
  'LARGE LANGUAGE MODELS',
  'MACHINE LEARNING',
  'SOFTWARE ENGINEERING',
] as const;

const legacyCategoryMap = new Map([
  ['airflow', 'DATA ENGINEERING'],
  ['ai engineering', 'AI ENGINEERING'],
  ['azure', 'DATA ENGINEERING'],
  ['data engineering', 'DATA ENGINEERING'],
  ['data science', 'DATA SCIENCE'],
  ['devops', 'DEVOPS'],
  ['k8s', 'DEVOPS'],
  ['large language models', 'LARGE LANGUAGE MODELS'],
  ['llm', 'LARGE LANGUAGE MODELS'],
  ['ml', 'MACHINE LEARNING'],
  ['machine learning', 'MACHINE LEARNING'],
  ['python', 'SOFTWARE ENGINEERING'],
  ['rust', 'SOFTWARE ENGINEERING'],
  ['scala', 'SOFTWARE ENGINEERING'],
  ['snowflake', 'DATA SCIENCE'],
  ['software engineering', 'SOFTWARE ENGINEERING'],
  ['spark', 'DATA ENGINEERING'],
]);

function uniqueValues(values: string[]) {
  return [...new Set(values)];
}

function cleanValue(value: string) {
  return value.trim().replace(/\s+/g, ' ');
}

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/docs/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    excerpt: z.string().optional(),
    date: z
      .union([z.date(), z.string(), z.record(z.any())])
      .optional()
      .transform((val) => {
        if (!val) return undefined;
        if (val instanceof Date) return val;
        if (typeof val === 'string') return new Date(val);
        if (typeof val === 'object' && val.created) return new Date(val.created);
        return undefined;
      }),
    categories: z.array(z.string()).optional().transform((val) => {
      if (!val) return undefined;
      return uniqueValues(val.map((category) => {
        const cleanCategory = cleanValue(category);
        const lower = cleanCategory.toLowerCase();
        const match = allowedCategories.find((candidate) => candidate.toLowerCase() === lower);
        return match ?? legacyCategoryMap.get(lower) ?? cleanCategory.toUpperCase();
      }).filter(Boolean));
    }),
    tags: z.array(z.string()).optional().transform((val) => {
      if (!val) return undefined;
      return uniqueValues(val.map((tag) => cleanValue(tag).toLowerCase()).filter(Boolean));
    }),
    author: z.string().optional(),
    authors: z.union([z.array(z.string()), z.string()]).optional(),
    pin: z.boolean().optional(),
  }),
});

export const collections = { blog };
