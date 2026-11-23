# Module 2: Retrieval Augmented Generation with ChromaDB

## Ingestion pipeline
RAG starts with ingestion: load documents, split them into chunks, embed every chunk and store the vectors in a vector database. The Sprints curriculum pipeline uses chunks of 500 tokens with an overlap of 50 tokens, so that a sentence cut at a chunk boundary still appears completely in the next chunk. RecursiveCharacterTextSplitter with a tiktoken length function enforces the token limit.

## ChromaDB
ChromaDB stores embeddings together with metadata such as the source file name. A persistent client keeps the collection on disk so ingestion runs once. Querying with similarity_search returns the k nearest chunks. For a small curriculum collection a query should return well under 1000 milliseconds; the retriever logs its latency so slow queries are visible.

## Embeddings
An embedding model maps text to a vector so that semantically similar texts are close together. Use the same embedding model for ingestion and for queries, otherwise distances are meaningless. A collection created with one embedding model must not be queried with another.

## Retrieval quality
Retrieval fails when the question uses different words than the documents. Query reformulation, metadata filtering and a larger k are common fixes. Always show the source of every retrieved chunk so the learner can verify the answer.
