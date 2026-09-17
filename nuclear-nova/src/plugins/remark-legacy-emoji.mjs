// Preserve the small set of emoji shortcodes used by migrated articles.
const emoji = { bulb: '💡', question_mark: '❓', exclamation: '❗', warning: '⚠️', smile: '😄' };
export default function remarkLegacyEmoji() {
  return (tree) => {
    function walk(node) {
      if (node.type === 'text') {
        node.value = node.value.replace(/:(bulb|question_mark|exclamation|warning|smile):/g,
          (_, name) => emoji[name]);
      }
      node.children?.forEach(walk);
    }
    walk(tree);
  };
}
