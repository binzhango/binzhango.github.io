import { toHast } from 'mdast-util-to-hast';
import { toString } from 'mdast-util-to-string';

const ADMONITION_VARIANTS = {
    caution: 'caution',
    danger: 'danger',
    note: 'note',
    tip: 'tip',
    warning: 'caution',
};

const DEFAULT_TITLES = {
    caution: 'Caution',
    danger: 'Danger',
    note: 'Note',
    tip: 'Tip',
    warning: 'Warning',
};

function getAttributeClasses(attributes = {}) {
    const values = [];

    for (const key of ['class', 'className']) {
        const value = attributes[key];

        if (Array.isArray(value)) {
            values.push(...value);
            continue;
        }

        if (typeof value === 'string') {
            values.push(...value.split(/\s+/));
        }
    }

    return [...new Set(values.filter(Boolean))];
}

function getHtmlProperties(attributes = {}) {
    const properties = {};

    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'class' || key === 'className') continue;
        properties[key] = value;
    }

    return properties;
}

function isDirective(node) {
    return node?.type === 'containerDirective' || node?.type === 'leafDirective';
}

function transformNode(node) {
    if (Array.isArray(node.children)) {
        node.children.forEach(transformNode);
    }

    if (!isDirective(node)) return;

    const variant = ADMONITION_VARIANTS[node.name];
    if (!variant) return;

    const children = Array.isArray(node.children) ? [...node.children] : [];
    let title = DEFAULT_TITLES[node.name] ?? node.name;

    if (children[0]?.type === 'paragraph' && children[0].data?.directiveLabel) {
        title = toString(children[0]).trim() || title;
        children.shift();
    }

    const className = [
        'starlight-aside',
        `starlight-aside--${variant}`,
        ...getAttributeClasses(node.attributes),
    ];

    if (node.name === 'warning') {
        className.push('starlight-aside--warning');
    }

    const contentRoot = toHast(
        { type: 'root', children },
        { allowDangerousHtml: true },
    );

    node.children = [];
    node.data ??= {};
    node.data.hName = 'aside';
    node.data.hProperties = {
        ...getHtmlProperties(node.attributes),
        className,
    };
    node.data.hChildren = [
        {
            type: 'element',
            tagName: 'p',
            properties: { className: ['starlight-aside__title'] },
            children: [{ type: 'text', value: title }],
        },
        {
            type: 'element',
            tagName: 'div',
            properties: { className: ['starlight-aside__content'] },
            children: contentRoot.children ?? [],
        },
    ];
}

export default function remarkStarlightAdmonitions() {
    return (tree) => {
        transformNode(tree);
    };
}
