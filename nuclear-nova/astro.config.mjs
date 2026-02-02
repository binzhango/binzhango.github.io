// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { getSortedPosts } from './src/sidebar-utils.mjs';
import mdx from '@astrojs/mdx';
import vercel from '@astrojs/vercel';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeMathJax from 'rehype-mathjax';
import rehypeMermaid from 'rehype-mermaid';
import { transformerNotationDiff, transformerNotationHighlight } from '@shikijs/transformers';
import { pluginLineNumbers } from '@expressive-code/plugin-line-numbers';

// https://astro.build/config
export default defineConfig({
    site: 'https://binzhango.com', // Update with actual custom domain
    output: 'static',
    adapter: vercel(),
    integrations: [
        starlight({
            title: 'B~~~~~Z',
            defaultLocale: 'root',
            locales: {
                root: {
                    label: 'English',
                    lang: 'en',
                },
            },
            customCss: ['./src/styles/global.css'],
            expressiveCode: {
                themes: ['github-dark', 'github-light'],
                plugins: [pluginLineNumbers()],
                defaultProps: {
                    // Enable line numbers by default
                    showLineNumbers: true,
                    // Start line numbers at 1
                    startLineNumber: 1,
                    // Enable word wrap
                    wrap: true,
                },
                // Disable copy button
                frames: {
                    showCopyToClipboardButton: false,
                },
                styleOverrides: {
                    // Ensure line numbers are visible
                    codePaddingInline: '1rem',
                },
            },
            social: [
                {
                    icon: 'github',
                    label: 'GitHub',
                    href: 'https://github.com/binzhango',
                },
                {
                    icon: 'discord',
                    label: 'Discord',
                    href: 'https://discord.com/invite/binzhango',
                },
                {
                    icon: 'threads',
                    label: 'Threads',
                    href: 'https://www.threads.net/@binzhango',
                },
            ],
            customCss: ['./src/styles/global.css'],
            sidebar: getSortedPosts(),
            components: {
                Header: './src/components/Header.astro',
                Footer: './src/components/Footer.astro',
                Head: './src/components/Head.astro',
                ContentPanel: './src/components/ContentPanel.astro',
                LinkCard: './src/components/LinkCard.astro',
                Card: './src/components/Card.astro',
            },
        }),
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