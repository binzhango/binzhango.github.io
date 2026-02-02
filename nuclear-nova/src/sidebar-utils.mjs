import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';

export function getSortedPosts() {
    const postsDir = path.join(process.cwd(), 'src/content/docs/posts');
    if (!fs.existsSync(postsDir)) return [];

    const items = [];

    // Recursive function to get all md files
    function getFiles(dir) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const res = path.resolve(dir, entry.name);
            if (entry.isDirectory()) {
                getFiles(res);
            } else if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.mdx'))) {
                const content = fs.readFileSync(res, 'utf-8');
                const { data } = matter(content);

                // Construct slug relative to content/docs
                let relPath = path.relative(path.join(process.cwd(), 'src/content/docs'), res);
                relPath = relPath.replace(/\.(md|mdx)$/, '');

                // Clean slug
                const segments = relPath.split(path.sep);
                const cleanSlug = segments.map(s => s.toLowerCase().trim().replace(/ /g, '-')).join('/');

                let date = data.date ? new Date(data.date) : new Date(0);

                items.push({
                    label: data.title || entry.name,
                    link: cleanSlug,
                    date: date
                });
            }
        }
    }

    try {
        getFiles(postsDir);
    } catch (e) {
        console.error("Error reading posts:", e);
        return [];
    }

    // Sort all items by date descending first
    items.sort((a, b) => b.date.getTime() - a.date.getTime());

    // Group by Year
    const grouped = {};
    items.forEach(item => {
        const year = item.date.getFullYear().toString();
        if (year === '1970') return; // Skip invalid
        if (!grouped[year]) {
            grouped[year] = [];
        }
        grouped[year].push({ label: item.label, link: item.link });
    });

    // Create sidebar structure sorted by year descending
    const years = Object.keys(grouped).sort((a, b) => b - a);
    const sidebar = years.map((year, index) => {
        return {
            label: year,
            items: grouped[year],
            collapsed: index >= 2 // Collapse all years except the first 2 (latest 2 years)
        };
    });

    return sidebar;
}
