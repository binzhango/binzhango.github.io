---
title: Autogen Intro and RAG Workflow
authors:
  - BZ
date: 2025-02-08
categories: 
  - LLM
---

# Introduction to Autogen
<!-- more -->

[Autogen](https://github.com/microsoft/autogen)

AutoGen is a framework for creating multi-agent AI applications that can act autonomously or work alongside humans.

🔹 Agent Types in AutoGen

| Agent Type | Description | Use Case|
|--------------|-------------|-------------|
| BaseChatAgent | The foundation for all other agents, handling basic messaging and conversation logic. | Custom agent development, extending its capabilities.|
| AssistantAgent | A chatbot-like agent designed to assist with general problem-solving and respond to queries. | AI assistants, research helpers, customer support.|
| UserProxyAgent | Simulates a human user by sending human-like inputs to other agents. | Automating user interactions, debugging multi-agent systems.|
| CodeExecutorAgent | Specialized agent that executes Python code, typically used for coding tasks. | Automating programming tasks, AI-assisted development.|
| SocietyOfMindAgent | Manages multiple agents as a hierarchical system to collaborate on complex tasks. | Coordinating multiple AI agents for team-based problem-solving. |

> Tips :bulb:
>
> | Need	| Agent Type|
> |-------------|--------------|
> | Basic AI conversation |	AssistantAgent
> | Simulating human input |	UserProxyAgent
> | Executing Python code |	CodeExecutorAgent
> | Coordinating multiple agents |	SocietyOfMindAgent
> | Custom agent development |	BaseChatAgent

# Example of Autogen RAG workflow

```python linenums="1"
doc_extractor_topic_type = "DocExtractorAgent"
retrieval_topic_type = "RetrievalAgent"
user_topic_type = "User"


@type_subscription(topic_type=doc_extractor_topic_type)
class DocExtractorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, filename:str) -> None:
        super().__init__("A concept extractor agent.")
        self.filename = filename
        self._model_client = model_client

    @message_handler
    async def handle_user_description(self, message: Message, ctx: MessageContext) -> None:
        response = f"Complete extracting content from File {message}"
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(retrieval_topic_type, source=self.id.key))



# Define the RetrievalAgent by extending BaseChatAgent
@type_subscription(topic_type=retrieval_topic_type)
class RetrievalAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, v:PGVector) -> None:
        super().__init__("Custom Retrieval Agent")
        self.vector_db = v
        self._model_client = model_client


    @message_handler
    async def handle_intermediate_text(self, message: Message, ctx: MessageContext) -> None:
        response = self.vector_db.similarity_search(message.content)
        string_output_comprehension = "\n".join([doc.page_content for doc in response])
        assert isinstance(string_output_comprehension, str)
        print(f"{'-'*80}\n{self.id.type}:\n{string_output_comprehension}")
        await self.publish_message(Message(string_output_comprehension), topic_id=TopicId(user_topic_type, source=self.id.key))



@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A user agent that outputs")
        self._system_message = SystemMessage(
            content=(
                "Your are a data analyst who is responsible for retrieving and summarizing data from provided content "
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_final_copy(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Below is the info about the report:\n\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        print(f"\n{'-'*80}\n{self.id.type} received final copy:\n{response}")


async def main():
    config = {}
    model_client = ChatCompletionClient.load_component(config)
    runtime = SingleThreadedAgentRuntime()
    await DocExtractorAgent.register(runtime, type=doc_extractor_topic_type,
                                     factory=lambda : DocExtractorAgent(model_client=model_client))
    await RetrievalAgent.register(runtime, type=retrieval_topic_type,
                                  factory=lambda : RetrievalAgent(model_client=model_client))
    
    await UserAgent.register(runtime, type=user_topic_type,
                             factory=lambda : UserAgent(model_client=model_client))
    
    

if __name__ == '__main__':
    asyncio.run(main())
```