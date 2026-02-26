import { getCollection } from 'astro:content';
import { OGImageRoute } from 'astro-og-canvas';

const entries = await getCollection('docs');
const pages = Object.fromEntries(entries.map(({ data, id }) => [id, { data }]));
const localFonts = [
    './src/assets/fonts/inter-latin-400-normal.woff2',
    './src/assets/fonts/inter-latin-700-normal.woff2',
];

export const { getStaticPaths, GET } = await OGImageRoute({
    pages,
    param: 'slug',
    getImageOptions: (_id, page: (typeof pages)[string]) => {
        return {
            title: page.data.title || 'B~~~~~Z',
            description: page.data.excerpt || page.data.description || '',
            bgGradient: [[3, 169, 244]], // Light blue (#03a9f4)
            border: { color: [255, 152, 0], width: 20 }, // Orange (#ff9800)
            padding: 120,
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
            fonts: localFonts,
            quality: 90,
        };
    },
});
