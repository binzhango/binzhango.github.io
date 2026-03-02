// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import vercel from '@astrojs/vercel';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeMathJax from 'rehype-mathjax';
import rehypeMermaid from 'rehype-mermaid';
import { transformerNotationDiff, transformerNotationHighlight } from '@shikijs/transformers';

// https://astro.build/config
export default defineConfig({
    site: 'https://binzhango.com', // Update with actual custom domain
    output: 'static',
    adapter: vercel(),
    integrations: [
        mdx(),
        sitemap(),
    ],
    markdown: {
        remarkPlugins: [remarkMath, remarkGfm],
        rehypePlugins: [
            [
                rehypeMathJax,
                {
                    tex: {
                        inlineMath: [['$', '$']],
                        displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    },
                    svg: {
                        fontCache: 'global',
                    },
                },
            ],
            [rehypeMermaid, { strategy: 'img-svg' }],
        ],
        shikiConfig: {
            // Choose from Shiki's built-in themes (or add your own)
            // https://shiki.style/themes
            theme: 'github-dark',
            // Alternatively, provide multiple themes
            // See note below for using dual light/dark themes
            themes: {
                light: 'github-light',
                dark: 'github-dark',
            },
            // Add custom languages
            // Note: Shiki has countless langs built-in, including .astro!
            // https://shiki.style/languages
            langs: [],
            // Enable word wrap to prevent horizontal scrolling
            wrap: true,
            // Add custom transformers: https://shiki.style/guide/transformers
            // Find common transformers: https://shiki.style/packages/transformers
            transformers: [
                transformerNotationDiff(),
                transformerNotationHighlight(),
            ],
            // Enable line numbers
            defaultColor: false,
        },
    },
});
