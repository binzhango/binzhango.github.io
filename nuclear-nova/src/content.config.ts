import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const allowedCategories = [
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
      return val.map((category) => {
        const lower = category.toLowerCase();
        const match = allowedCategories.find((candidate) => candidate.toLowerCase() === lower);
        return match ?? category;
      });
    }),
    tags: z.array(z.string()).optional(),
    author: z.string().optional(),
    authors: z.union([z.array(z.string()), z.string()]).optional(),
    pin: z.boolean().optional(),
  }),
});

export const collections = { blog };
