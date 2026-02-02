---
title: RAG-Reranking
authors:
  - BZ
date: 2025-04-22
categories: 
  - LLM
---

<!-- more -->

# Reranking in Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a powerful approach that combines retrieval and generation to produce high-quality responses. 
However, the quality of the final response can be significantly influenced by the effectiveness of the retrieval process.

`Reranking` can improve the quality of the final response by reordering the retrieved documents based on their relevance to the query. 
In this blog, we will discuss how reranking can be integrated into RAG and its benefits.

This [Blog](https://www.galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model) is good for you to choose correct `reranker` model.

# `langchain` implementation

```python linenums="1" title="reranking.py"

from langchain_community.document_loaders import TextLoader
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Text split
# chunking
documents = TextLoader("state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2/")
connection = "postgresql+psycopg://"  # Uses psycopg3!
collection_name = "reranking_test"


# indexing
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

vector_store.add_documents(texts, ids=[i for i, _ in enumerate(texts, start=1)])

# regular retrieval
retriever = vector_store.as_retriever(search_kwargs={"k": 20})

query = "What is the plan for the economy?"
docs = retriever.invoke(query)
print("\nRetrieved Documents:\n")
pretty_print_docs(docs)

# reranking with CrossEncoder
model = HuggingFaceCrossEncoder(model_name="/Users/binzhang/models/BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever #reordering after retrieval
)
compressed_docs = compression_retriever.invoke(query)
print("\nReranked Documents:\n")
pretty_print_docs(compressed_docs)

```

# Calculate Score of reranking pairs

```python linenums="1" title="calculate_score.py"
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize the cross encoder
cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3",
    model_kwargs={'device': 'cpu'}
)

# Create text pairs to score
text_pairs = [
    ("How do I bake bread?", "This is a recipe for sourdough bread"),
    ("How do I bake bread?", "The weather is nice today")
]

# Get similarity scores
scores = cross_encoder.score(text_pairs)

print(scores)
```



