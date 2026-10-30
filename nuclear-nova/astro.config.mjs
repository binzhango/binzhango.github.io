// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
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
import rehypeReadableContent from './src/plugins/rehype-readable-content.mjs';
import remarkLegacyEmoji from './src/plugins/remark-legacy-emoji.mjs';
import remarkInterview from './src/plugins/remark-interview.mjs';
import rehypeInterviewIds from './src/plugins/rehype-interview-ids.mjs';

// https://astro.build/config
export default defineConfig({
    site: process.env.SITE_URL ?? 'https://binzhango.net',
    output: 'static',
    // Preserve HTML-aware spacing between inline elements after the Astro 7 upgrade.
    compressHTML: true,
    redirects: {
        '/posts/2026/08-27-local-lke-01-inspectable-rag-baseline':
            '/posts/2026/08-27-rag-engineering-01-inspectable-rag-baseline',
        '/posts/2026/08-30-local-lke-02-chunking-is-a-retrieval-decision':
            '/posts/2026/08-30-rag-engineering-02-chunking-is-a-retrieval-decision',
        '/posts/2026/09-02-local-lke-03-safe-versioned-ingestion':
            '/posts/2026/09-02-rag-engineering-03-safe-versioned-ingestion',
        '/posts/2026/09-05-local-lke-04-embedding-profile-index-contract':
            '/posts/2026/09-05-rag-engineering-04-embedding-profile-index-contract',
        '/posts/2026/09-08-local-lke-05-context-expansion-multimodal-search':
            '/posts/2026/09-08-rag-engineering-05-context-expansion-multimodal-search',
        '/posts/2026/09-11-local-lke-06-hybrid-corrective-retrieval':
            '/posts/2026/09-11-rag-engineering-06-hybrid-corrective-retrieval',
        '/posts/2026/09-14-local-lke-07-query-planning-safe-structured-retrieval':
            '/posts/2026/09-14-rag-engineering-07-query-planning-safe-structured-retrieval',
        '/posts/2026/09-17-local-lke-08-validated-cited-generation':
            '/posts/2026/09-17-rag-engineering-08-validated-cited-generation',
        '/posts/2026/09-20-local-lke-09-rag-evaluation-regression-gates':
            '/posts/2026/09-20-rag-engineering-09-rag-evaluation-regression-gates',
        '/posts/2026/09-23-local-lke-10-security-governance':
            '/posts/2026/09-23-rag-engineering-10-security-governance',
        '/collections/local-lke-rag-engineering': '/collections/rag-engineering',
        '/tags/local-lke': '/tags/rag-engineering',
        '/tools/local-lke-baseline-flow.html': '/tools/rag-baseline-flow.html',
        '/posts/2026/08-27-local-lke-02-chunking-is-a-retrieval-decision':
            '/posts/2026/08-30-rag-engineering-02-chunking-is-a-retrieval-decision',
        '/posts/2026/08-27-local-lke-03-safe-versioned-ingestion':
            '/posts/2026/09-02-rag-engineering-03-safe-versioned-ingestion',
        '/posts/2026/08-27-local-lke-04-embedding-profile-index-contract':
            '/posts/2026/09-05-rag-engineering-04-embedding-profile-index-contract',
        '/posts/2026/08-27-local-lke-05-context-expansion-multimodal-search':
            '/posts/2026/09-08-rag-engineering-05-context-expansion-multimodal-search',
        '/posts/2026/08-27-local-lke-06-hybrid-corrective-retrieval':
            '/posts/2026/09-11-rag-engineering-06-hybrid-corrective-retrieval',
        '/posts/2026/08-27-local-lke-07-query-planning-safe-structured-retrieval':
            '/posts/2026/09-14-rag-engineering-07-query-planning-safe-structured-retrieval',
        '/posts/2026/08-27-local-lke-08-validated-cited-generation':
            '/posts/2026/09-17-rag-engineering-08-validated-cited-generation',
        '/posts/2026/08-27-local-lke-09-rag-evaluation-regression-gates':
            '/posts/2026/09-20-rag-engineering-09-rag-evaluation-regression-gates',
        '/posts/2026/08-27-local-lke-10-security-governance':
            '/posts/2026/09-23-rag-engineering-10-security-governance',
    },
    devToolbar: {
        enabled: false,
    },
    // GitHub Pages needs HTML redirects; Vercel emits redirects in its routing config.
    adapter: process.env.DEPLOY_TARGET === 'github-pages' ? undefined : vercel(),
    integrations: [
        mdx(),
        sitemap(),
    ],
    markdown: {
        // Keep the remark/rehype pipeline used by math, diagrams, and interview content.
        processor: unified({
            remarkPlugins: [remarkDirective, remarkMath, remarkGfm, remarkLegacyEmoji, remarkStarlightAdmonitions, remarkInterview],
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
                rehypeReadableContent,
                rehypeInterviewIds,
            ],
        }),
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
