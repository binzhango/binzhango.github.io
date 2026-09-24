export const interviewTopics = [
  ['ml-foundations', 'ML Foundations'],
  ['llm-architecture', 'LLM Architecture'],
  ['training', 'Training, Fine-Tuning and Alignment'],
  ['inference', 'Inference and Serving'],
  ['rag', 'RAG and Retrieval'],
  ['agents', 'Agents and Tools'],
  ['evaluation', 'Evaluation and Observability'],
  ['safety', 'Safety and Security'],
  ['multimodal', 'Multimodal, Speech and Robotics'],
  ['system-design', 'AI System Design'],
  ['coding', 'Coding and Data Structures'],
  ['applied', 'Applied Engineering and Delivery'],
  ['behavioral', 'Behavioral and Communication'],
] as const;

export const interviewTopicIds = interviewTopics.map(([id]) => id) as [string, ...string[]];
export const legacyTopicAnchors: Record<string, string[]> = {
  'ml-foundations': ['machine-learning', 'llm-fundamentals'],
  'llm-architecture': ['fundamentals-of-large-language-models-llms', 'advanced-topics-in-llms'],
  training: ['fine-tuning-large-models', 'parameter-efficient-fine-tuning-peft'],
  agents: ['langchain-and-agent-based-systems'],
  rag: ['retrieval-augmented-generation-rag'],
};
