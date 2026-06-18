function getText(node) {
    if (node.type === 'text') return node.value;
    if (!Array.isArray(node.children)) return '';
    return node.children.map(getText).join('');
}

function restoreMermaidBlocks(node) {
    if (!Array.isArray(node.children)) return;

    for (const child of node.children) {
        if (
            child.type === 'element'
            && child.tagName === 'pre'
            && child.properties?.dataLanguage === 'mermaid'
        ) {
            child.properties = {};
            child.children = [{
                type: 'element',
                tagName: 'code',
                properties: { className: ['language-mermaid'] },
                children: [{ type: 'text', value: getText(child) }],
            }];
            continue;
        }

        restoreMermaidBlocks(child);
    }
}

/** Restore the HAST shape rehype-mermaid expects after Astro/Shiki runs. */
export default function rehypeMermaidShikiAdapter() {
    return (tree) => restoreMermaidBlocks(tree);
}
