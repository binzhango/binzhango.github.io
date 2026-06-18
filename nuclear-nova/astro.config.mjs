// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import vercel from '@astrojs/vercel';
import sitemap from '@astrojs/sitemap';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeMathJax from 'rehype-mathjax/chtml';
import rehypeMermaid from 'rehype-mermaid';
import { transformerNotationDiff, transformerNotationHighlight } from '@shikijs/transformers';
import remarkStarlightAdmonitions from './src/plugins/remark-starlight-admonitions.mjs';
import rehypeMermaidShikiAdapter from './src/plugins/rehype-mermaid-shiki-adapter.mjs';

// https://astro.build/config
export default defineConfig({
    site: process.env.SITE_URL ?? 'https://binzhango.net',
    output: 'static',
    devToolbar: {
        enabled: false,
    },
    adapter: vercel(),
    integrations: [
        mdx(),
        sitemap(),
    ],
    markdown: {
        remarkPlugins: [remarkDirective, remarkMath, remarkGfm, remarkStarlightAdmonitions],
        rehypePlugins: [
            [
                rehypeMathJax,
                {
                    tex: {
                        inlineMath: [['$', '$']],
                        displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    },
                    chtml: {
                        fontURL: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2',
                    },
                },
            ],
            rehypeMermaidShikiAdapter,
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
            // Preserve code formatting and use horizontal scrolling for long lines.
            wrap: false,
            // Add custom transformers: https://shiki.style/guide/transformers
            // Find common transformers: https://shiki.style/packages/transformers
            transformers: [
                transformerNotationDiff(),
                transformerNotationHighlight(),
            ],
            // Expose dual-theme CSS variables; global.css applies them per theme.
            defaultColor: false,
        },
    },
});
