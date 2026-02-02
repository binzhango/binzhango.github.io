import { getCollection } from 'astro:content';
import { OGImageRoute } from 'astro-og-canvas';

// Get all entries from the `docs` content collection
const entries = await getCollection('docs');

// Map the entry array to an object with the page ID as key and the
// frontmatter data as value
const pages = Object.fromEntries(entries.map(({ data, id }) => [id, { data }]));

export const { getStaticPaths, GET } = await OGImageRoute({
    // Pass down the documentation pages
    pages,
    
    // Define the name of the parameter used in the endpoint path
    param: 'slug',
    
    // Define a function called for each page to customize the generated image
    getImageOptions: (_id, page: (typeof pages)[number]) => {
        return {
            // Use the page title and description/excerpt as the image title and description
            title: page.data.title,
            description: page.data.excerpt || page.data.description || '',
            
            // Custom background color (light blue from requirements)
            bgGradient: [[3, 169, 244]], // Light blue (#03a9f4)
            
            // Orange accent border (from requirements)
            border: { color: [255, 152, 0], width: 20 }, // Orange (#ff9800)
            
            // Padding for better visual spacing
            padding: 120,
            
            // Font configuration
            font: {
                title: {
                    size: 72,
                    lineHeight: 1.2,
                    weight: 'Bold',
                },
                description: {
                    size: 36,
                    lineHeight: 1.4,
                    weight: 'Normal',
                },
            },
            
            // Image quality
            quality: 90,
        };
    },
});
