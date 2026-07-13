import { getAllPosts, slugifyId, type Post } from './post-utils';
import { getPostExcerpt } from './excerpt';

export interface CuratedCollection {
  slug: string;
  title: string;
  description: string;
  posts: Post[];
}

interface CollectionBlueprint extends Omit<CuratedCollection, 'posts'> {
  postIds: string[];
}

const collectionBlueprints: CollectionBlueprint[] = [
  {
    slug: 'llm-inference-systems',
    title: 'LLM Inference & Serving',
    description: 'Understand how prompts become tokens, how attention and KV cache work, and how serving engines scale inference.',
    postIds: [
      '2026/03-07-prompt-in-llm',
      '2026/03-15-attention-dilution',
      '2026/07-05-vllm-sglang-attention',
      '2026/07-03-reproduce-vllm-lmcache-cpu-macbook',
    ],
  },
  {
    slug: 'agents-tools-mcp',
    title: 'AI Agents & MCP',
    description: 'Learn how agents use tools, how MCP connects systems, and how multi-agent workflows coordinate.',
    postIds: [
      '2026/03-08-ai-terminology',
      '2025/07-11-llm-tools',
      '2025/06-23-mcp-transports',
      '2026/06-18-git-native-message-channel-for-local-coding-agents',
    ],
  },
  {
    slug: 'useful-llm-applications',
    title: 'Practical LLM Applications',
    description: 'Build local chat applications, RAG pipelines, reranking workflows, and natural-language data tools.',
    postIds: [
      '2024/12-15-gradio-with-llm',
      '2025/02-08-autogen',
      '2025/04-22-reranking-in-rag',
      '2025/05-04-text-to-sql',
    ],
  },
  {
    slug: 'spark-data-engineering',
    title: 'Spark & Data Engineering',
    description: 'Learn DataFrame operations, SQL tuning, Spark optimization, and structured streaming.',
    postIds: [
      '2020/spark-dataframe-window-function',
      '2020/sparksql_tuning',
      '2020/spark-optimization',
      '2020/spark-structured-streaming',
    ],
  },
];

const curatedExcerpts = new Map([
  ['2024/12-15-gradio-with-llm', 'Build a lightweight chat interface for a local Ollama model with Gradio.'],
  ['2025/06-23-mcp-transports', 'Compare MCP transport options, from local standard I/O to streaming HTTP connections.'],
  ['2025/05-04-text-to-sql', 'Turn natural-language questions into executable SQL with a tool-using agent.'],
  ['2020/sparksql_tuning', 'Configure Spark SQL and tune distributed queries for more efficient execution.'],
]);

export function getCollectionExcerpt(post: Post, maxLength = 190): string {
  return curatedExcerpts.get(slugifyId(post.id)) ?? getPostExcerpt(post, maxLength);
}

export async function getCuratedCollections(): Promise<CuratedCollection[]> {
  const posts = await getAllPosts();
  const postsById = new Map(posts.map((post) => [slugifyId(post.id), post]));

  return collectionBlueprints
    .map(({ postIds, ...collection }) => ({
      ...collection,
      posts: postIds
        .map((postId) => postsById.get(postId))
        .filter((post): post is Post => Boolean(post)),
    }))
    .filter((collection) => collection.posts.length > 0);
}
