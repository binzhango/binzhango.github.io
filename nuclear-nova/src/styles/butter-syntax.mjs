import { readFileSync } from 'node:fs';

const css = readFileSync(new URL(import.meta.resolve('@astryxdesign/theme-butter/theme.css')), 'utf8');
const colors = new Map(
  Array.from(css.matchAll(/--(color-[\w-]+): light-dark\((#[\da-f]+), (#[\da-f]+)\);/gi),
    ([, name, light, dark]) => [name, [light, dark]])
);

const scopes = {
  keyword: ['keyword', 'storage', 'variable.language'],
  string: ['string', 'markup.inline.raw', 'punctuation.definition.string'],
  comment: ['comment', 'punctuation.definition.comment'],
  number: ['constant.numeric'],
  function: ['entity.name.function', 'support.function', 'meta.function-call.generic.python'],
  type: ['entity.name.type', 'entity.name.class', 'support.type', 'support.class'],
  variable: ['variable.parameter', 'variable.other.normal.shell'],
  operator: ['keyword.operator'],
  constant: ['constant.language', 'constant.character', 'variable.other.constant', 'constant.other.option'],
  tag: ['entity.name.tag'],
  attribute: ['entity.other.attribute-name'],
  property: ['variable.other.property', 'support.type.property-name', 'punctuation.support.type.property-name', 'meta.object-literal.key'],
  punctuation: ['punctuation'],
};

// Separate adjacent token roles on the light and dark Butter code surfaces.
const codeAccents = {
  keyword: ['#7431a5', '#d8adff'],
  string: ['#076a43', '#9bd49c'],
  number: ['#9b470e', '#ffc186'],
  function: ['#1458aa', '#90c1ff'],
  type: ['#4e44a6', '#b7b0ff'],
  variable: ['#795017', '#edd081'],
  constant: ['#a02e56', '#ff9eb8'],
  property: ['#165a9e', '#8ec5ff'],
};

/** Use Butter for the base colors with stronger differentiation between code tokens. */
function theme(mode, index) {
  const color = (name) => {
    const value = colors.get(name)?.[index];
    if (!value) throw new Error(`Missing Butter color token: ${name}`);
    return value;
  };
  const foreground = color('color-text-primary');
  const background = color('color-syntax-background');
  return {
    name: `butter-${mode}`,
    type: mode,
    colors: { 'editor.foreground': foreground, 'editor.background': background },
    tokenColors: Object.entries(scopes).map(([role, scope]) => ({
      scope,
      settings: { foreground: codeAccents[role]?.[index] ?? color(`color-syntax-${role}`) },
    })),
  };
}

export const butterSyntax = { light: theme('light', 0), dark: theme('dark', 1) };
