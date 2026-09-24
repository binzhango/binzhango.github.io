/** Structural Markdown sections stay in the normal math/code pipeline. */
export function interviewId(file) {
  const path = String(file.path ?? file.history?.[0] ?? '').replaceAll('\\', '/');
  return /\/content\/interview\/([a-z0-9-]+)\.md$/.exec(path)?.[1];
}
const text = (node) => node.value ?? node.children?.map(text).join('') ?? '';
const wrapper = (tag, className, children, properties = {}) => ({
  type: 'interviewPart', data: { hName: tag, hProperties: { className, ...properties } }, children,
});

export default function remarkInterview() {
  return (tree, file) => {
    const id = interviewId(file);
    if (!id) return;
    const kind = file.data?.astro?.frontmatter?.kind ?? 'factual';
    const label = ({ factual: 'Brief answer', coding: 'Solution outline', design: 'Suggested approach', behavioral: 'Response framework' })[kind] ?? 'Brief answer';
    const sections = new Map();
    let section;
    for (const node of tree.children) {
      if (node.type === 'heading' && node.depth === 2 && ['Question', 'Brief answer', 'Detailed answer'].includes(text(node))) {
        const name = text(node);
        if (sections.has(name)) throw new Error(`${id}: duplicate ${name} section`);
        section = []; sections.set(name, section);
      } else if (section) section.push(node);
    }
    for (const name of ['Question', 'Brief answer', 'Detailed answer']) {
      if (!sections.has(name)) throw new Error(`${id}: missing ${name} section`);
    }
    const question = sections.get('Question');
    if (question.length !== 1 || question[0].type !== 'paragraph') throw new Error(`${id}: question must be one Markdown paragraph`);
    question[0] = { type: 'heading', depth: 3, data: { hProperties: { id: `prompt-${id}`, tabIndex: -1, className: ['interview-prompt'] } }, children: question[0].children };
    tree.children = [
      wrapper('div', ['interview-surface'], [
        ...question,
        wrapper('div', ['interview-brief'], [wrapper('p', ['interview-answer-kind'], [{ type: 'text', value: label }]), ...sections.get('Brief answer')]),
      ]),
      wrapper('details', ['interview-detail'], [
        wrapper('summary', [], [{ type: 'text', value: 'Detailed explanation' }]),
        wrapper('div', ['interview-explanation'], sections.get('Detailed answer')),
      ]),
    ];
  };
}
