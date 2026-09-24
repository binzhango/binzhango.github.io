import { slugifyCategory } from './taxonomy';

export type TopicTone = 'blue' | 'teal' | 'violet' | 'rose' | 'amber';
const categoryTones: Record<string, TopicTone> = {
  'ai-engineering': 'teal',
  'large-language-models': 'violet',
  'machine-learning': 'rose',
  'data-science': 'amber',
  'software-engineering': 'violet',
  devops: 'amber',
};

export function getTopicTone(category?: string): TopicTone {
  return categoryTones[slugifyCategory(category ?? '')] ?? 'blue';
}

/** Presentation only: canonical titles and content metadata remain untouched. */
export function getDisplayTitle(title: string) {
  const match = /^RAG Engineering (\d{2}):\s+(.+)$/.exec(title);
  return match
    ? { title: match[2], series: `RAG Engineering / ${match[1]}` }
    : { title, series: undefined };
}
