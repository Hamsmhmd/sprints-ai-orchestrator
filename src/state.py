"""Global LangGraph state shared by the Orchestrator and its worker agents.

Curriculum mapping: Sprint 2 - "StateGraph with a global TypedDict state". The state is the
single source of truth passed between nodes; the checkpointer persists it per thread_id so each
learner keeps an isolated, durable context.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    """Global state. ``total=False`` lets each node return only the keys it changes."""

    # --- conversation (reducer: messages are appended, never overwritten)
    messages: Annotated[list[AnyMessage], add_messages]
    learner_id: str                 # long-term fact, persists across turns of a thread
    intent: str                     # curriculum_question | profile_question | career_match | code_grading
    # --- Advisor / agentic RAG
    current_query: str
    retrieved_context: str
    retrieved_sources: list[str]
    rag_attempts: int
    context_sufficient: bool
    retrieval_failed: bool          # ChromaDB error -> skip straight to 'Unable to verify'
    # --- Assessor / evaluator-optimizer
    code_submission: str
    expected_output: str
    execution_result: dict[str, Any]
    assessment: dict[str, Any]
    judge_verdict: dict[str, Any]
    refinement_count: int
    # --- Placement + human-in-the-loop
    recommendation: dict[str, Any]
    approval_status: str            # not_required | approved | rejected
    # --- output
    final_answer: str


def last_human_text(messages: list) -> str:
    """Return the content of the most recent HumanMessage ('' if none)."""
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "human":
            return str(m.content)
    return ""


def history_text(messages: list, n: int = 6) -> str:
    """Short-term memory window: last ``n`` human/AI turns as plain text (tool traffic excluded).

    The full history stays in the checkpoint; only this window is put in prompts to bound
    token cost (memory strategy documented in the technical report).
    """
    turns = [m for m in (messages or []) if getattr(m, "type", "") in ("human", "ai") and m.content]
    return "\n".join(f"{m.type.upper()}: {str(m.content)[:300]}" for m in turns[-n:]) or "(no previous turns)"
