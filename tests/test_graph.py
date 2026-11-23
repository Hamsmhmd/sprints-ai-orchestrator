"""End-to-end graph tests (mock mode). Run: python -m pytest -q  (needs requirements installed)."""
import os

os.environ["LLM_MODE"] = "mock"

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator
from src.llm import get_llm, stubborn_judge
from src.main import auto_approver, pending_interrupt, run_turn
from src.tools import db_queries
from src.tools.rag_retriever import build_vector_store

GOOD = 'Grade me. Expected output: 55\n```python\ndef f(n):\n    """Sum of squares."""\n    return sum(i*i for i in range(1, n+1))\nprint(f(5))\n```'


@pytest.fixture(scope="module", autouse=True)
def _setup():
    db_queries.build_database()
    build_vector_store(rebuild=True)


@pytest.fixture()
def graph():
    return build_orchestrator()


def test_curriculum_answer_has_sources(graph):
    out = run_turn(graph, "t1", "How does agentic RAG decide to search again?", "SPR-1001")
    assert "Sources:" in out["answer"] and "Unable to verify" not in out["answer"]


def test_unanswerable_question_is_unable_to_verify(graph):
    out = run_turn(graph, "t2", "What is the capital of France?", "SPR-1001")
    assert out["answer"].startswith("Unable to verify")
    assert out["state"]["rag_attempts"] == 3          # reformulated and retried, then gave up


def test_grading_pass_and_loop_is_bounded(graph):
    out = run_turn(graph, "t3", GOOD, "SPR-1001")
    assert "PASS" in out["answer"] and out["state"]["refinement_count"] <= 3


def test_stubborn_judge_stops_at_three():
    g = build_orchestrator(judge_llm=stubborn_judge(get_llm()))
    out = run_turn(g, "t4", GOOD, "SPR-1001")
    assert out["state"]["refinement_count"] == 3 and "human grader" in out["answer"]


def test_malicious_code_blocked(graph):
    out = run_turn(graph, "t5", "Grade\n```python\nimport os\nprint(os.getcwd())\n```", "SPR-1001")
    assert "0/100" in out["answer"] and "Security policy" in out["answer"]


def test_hitl_pauses_then_resumes_on_approval(graph):
    cfg = {"configurable": {"thread_id": "t6"}}
    out = run_turn(graph, "t6", "Which partner opening fits me?", "SPR-1001", approver=None)
    assert out["answer"] is None and out["paused"]["employer"] == "Fawry"
    graph.invoke(Command(resume={"approved": True, "note": "ok"}), cfg)
    assert "APPROVED" in graph.get_state(cfg).values["final_answer"]


def test_hitl_rejection_withholds_recommendation(graph):
    out = run_turn(graph, "t7", "Which partner opening fits me?", "SPR-1001",
                   approver=lambda p: {"approved": False, "note": "needs interview"})
    assert "held back" in out["answer"] and out["state"]["approval_status"] == "rejected"


def test_not_ready_learner_gets_no_recommendation(graph):
    out = run_turn(graph, "t8", "Which partner opening fits me?", "SPR-1007", auto_approver)
    assert "No placement recommendation" in out["answer"] and out["paused"] is None


def test_memory_isolation_and_persistence(graph):
    run_turn(graph, "A", "What is my profile?", "SPR-1002")
    run_turn(graph, "B", "What is my profile?", "SPR-1003")
    a2 = run_turn(graph, "A", "What is my target salary?")        # no learner id -> restored from checkpoint
    assert "Mona Ali" in a2["answer"] and "Mona Ali" not in run_turn(graph, "B", "What is my profile?")["answer"]


def test_protocol_messages_are_typed(graph):
    out = run_turn(graph, "t9", "What is my profile?", "SPR-1001")
    tools = [m for m in out["state"]["messages"] if isinstance(m, ToolMessage)]
    assert tools and tools[0].tool_call_id


def test_db_failure_returns_unable_to_verify(graph, monkeypatch):
    monkeypatch.setattr(db_queries, "DB_PATH", db_queries.DB_PATH.parent / "missing.db")
    out = run_turn(graph, "t10", "Which partner opening fits me?", "SPR-1001")
    assert out["answer"].startswith("Unable to verify")
