"""Document ingestion, vectorization and retrieval (ChromaDB) for the Advisor Agent.

Curriculum mapping: Sprint 1 - "Document ingestion + vectorization pipeline". Pipeline:
load Sprints curriculum markdown -> split (500-token chunks, 50-token overlap) -> embed ->
persist in ChromaDB. Retrieval latency is measured on every call against the client's
1000 ms budget.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from src.config import (CHROMA_DIR, CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, CURRICULUM_DIR,
                        RETRIEVAL_BUDGET_MS, RETRIEVAL_TOP_K)
from src.fallback import unable_to_verify
from src.llm import get_embeddings, llm_mode

_STORE = None


def load_curriculum_documents(directory: Path = CURRICULUM_DIR):
    """Load every curriculum markdown file as a LangChain Document (metadata.source = file name)."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(str(directory), glob="**/*.md", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = Path(d.metadata["source"]).name
    return docs


def split_documents(docs):
    """Chunk documents: 500 tokens with 50 tokens overlap (client requirement).

    Uses a tiktoken (cl100k_base) length function. If the tiktoken vocabulary cannot be
    downloaded (offline machine) it falls back to ~4 characters per token.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
    except Exception:  # noqa: BLE001
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_TOKENS * 4,
                                                  chunk_overlap=CHUNK_OVERLAP_TOKENS * 4)
    return splitter.split_documents(docs)


def _collection_name() -> str:
    # One collection per embedding model: vectors from different models must never be mixed.
    return f"sprints_curriculum_{llm_mode()}"


def build_vector_store(rebuild: bool = False):
    """Create (or reuse) the persistent ChromaDB collection and ingest the curriculum if empty."""
    global _STORE
    from langchain_chroma import Chroma

    def _open():
        return Chroma(collection_name=_collection_name(), embedding_function=get_embeddings(),
                      persist_directory=str(CHROMA_DIR))

    store = _open()
    if rebuild:
        store.delete_collection()
        store = _open()
    if not store.get(limit=1)["ids"]:
        chunks = split_documents(load_curriculum_documents())
        store.add_documents(chunks)
    _STORE = store
    return store


def get_store():
    """Return the cached vector store, building it on first use."""
    return _STORE or build_vector_store()


def retrieve(query: str, k: int = RETRIEVAL_TOP_K) -> dict[str, Any]:
    """Semantic search over the curriculum with latency tracking.

    Returns ``{'status': 'ok', 'context', 'sources', 'latency_ms', 'within_budget'}`` or a
    structured 'Unable to verify' payload if ChromaDB fails.
    """
    try:
        store = get_store()
        t0 = time.perf_counter()
        docs = store.similarity_search(query, k=k)
        latency = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        return unable_to_verify("chromadb", f"vector store error: {type(exc).__name__}")
    context = "\n\n".join(f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)
    return {"status": "ok", "context": context,
            "sources": sorted({d.metadata.get("source", "?") for d in docs}),
            "latency_ms": round(latency, 1), "within_budget": latency <= RETRIEVAL_BUDGET_MS}


def benchmark_retrieval(queries: list[str] | None = None, runs: int = 20) -> dict[str, float]:
    """Measure retrieval latency (ms) to evidence the < 1000 ms acceptance criterion."""
    queries = queries or ["How does agentic RAG reformulate a query?", "What is a MemorySaver checkpointer?",
                          "How many refinement iterations are allowed?", "How is learner code sandboxed?"]
    retrieve(queries[0])  # warm-up (model/collection load is not part of steady-state latency)
    times = []
    for i in range(runs):
        r = retrieve(queries[i % len(queries)])
        times.append(r["latency_ms"])
    times.sort()
    return {"runs": runs, "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1), "max_ms": round(times[-1], 1),
            "budget_ms": RETRIEVAL_BUDGET_MS, "all_within_budget": times[-1] <= RETRIEVAL_BUDGET_MS}


def get_retriever_tool():
    """Return the LangChain ``@tool`` exposing curriculum search to agents."""
    from langchain_core.tools import tool

    @tool
    def search_curriculum(query: str) -> dict:
        """Search the Sprints curriculum (ChromaDB) and return relevant chunks with sources.

        Curriculum mapping: Sprint 1 RAG ingestion pipeline, used by the Advisor Agent.
        """
        return retrieve(query)

    return search_curriculum
