import { defineCollection, z } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Define allowed categories as per requirements
const allowedCategories = ['python', 'k8s', 'spark', 'ML', 'airflow', 'LLM', 'Azure', 'Snowflake', 'Scala', 'rust'] as const;

export const collections = {
	docs: defineCollection({ 
		loader: docsLoader(), 
		schema: docsSchema({
			extend: z.object({
				// Date field with parsing and transformation logic
				date: z.union([z.date(), z.string(), z.record(z.any())]).optional().transform((val) => {
					if (!val) return undefined;
					if (val instanceof Date) return val;
					if (typeof val === 'string') return new Date(val);
					if (typeof val === 'object' && val.created) return new Date(val.created);
					return undefined;
				}),
				// Categories with enum validation for allowed categories
				// Transform to handle case variations (e.g., 'Spark' -> 'spark')
				categories: z.array(z.string()).optional().transform((val) => {
					if (!val) return undefined;
					return val.map(cat => {
						// Handle case variations
						const lowerCat = cat.toLowerCase();
						// Find matching category (case-insensitive)
						const match = allowedCategories.find(allowed => allowed.toLowerCase() === lowerCat);
						return match || cat; // Return matched category or original if no match
					});
				}),
				// Tags as array of strings
				tags: z.array(z.string()).optional(),
				// Author field (singular)
				author: z.string().optional(),
				// Authors field (plural, for compatibility)
				authors: z.union([z.array(z.string()), z.string()]).optional(),
				// Excerpt field for SEO and previews
				excerpt: z.string().optional(),
			}),
		})
	}),
};
