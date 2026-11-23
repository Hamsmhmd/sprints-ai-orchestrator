"""CLI entry point: ingest, demo, chat, ask-db, bench.

Run from the repository root:  python -m src.main demo

Curriculum mapping: Sprint 2 - running the compiled StateGraph with thread-scoped checkpoints
and resuming an ``interrupt`` with ``Command(resume=...)``.
"""
from __future__ import annotations

import argparse
import uuid
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.config import CSV_PATH, DB_PATH
from src.llm import get_llm, llm_mode
from src.logging_utils import setup_logging

Approver = Callable[[dict], dict]


def console_approver(payload: dict) -> dict:
    """Ask a human reviewer in the terminal to approve / reject a tier-1 placement."""
    print("\n" + "=" * 70)
    print("HUMAN APPROVAL REQUIRED (tier-1 partner placement)")
    for k in ("learner", "learner_id", "employer", "title", "opening_id", "salary_band_egp"):
        print(f"  {k:16}: {payload.get(k)}")
    answer = input("Approve this recommendation? [y/n]: ").strip().lower()
    note = input("Reviewer note (optional): ").strip()
    print("=" * 70)
    return {"approved": answer.startswith("y"), "note": note, "reviewer": "human"}


def auto_approver(payload: dict) -> dict:
    """Non-interactive approver for tests / CI."""
    return {"approved": True, "note": "auto-approved (test)", "reviewer": "auto"}


def pending_interrupt(snapshot) -> dict | None:
    """Return the payload of a pending ``interrupt`` in a graph snapshot, if any."""
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def run_turn(graph, thread_id: str, text: str, learner_id: str | None = None,
             approver: Approver | None = None) -> dict:
    """Send one learner message through the graph on ``thread_id`` and return the outcome.

    If the graph pauses at the HITL gate and ``approver`` is given, the decision is collected and
    the graph resumed. Otherwise the pending payload is returned under ``paused``.
    """
    config = {"configurable": {"thread_id": thread_id}}
    payload: dict = {"messages": [HumanMessage(content=text)]}
    if learner_id:
        payload["learner_id"] = learner_id
    graph.invoke(payload, config)
    snap = graph.get_state(config)
    while snap.next:
        pending = pending_interrupt(snap)
        if pending is None or approver is None:
            return {"answer": None, "paused": pending, "thread_id": thread_id}
        graph.invoke(Command(resume=approver(pending)), config)
        snap = graph.get_state(config)
    return {"answer": snap.values.get("final_answer"), "paused": None, "thread_id": thread_id,
            "state": snap.values}


DEMO_CODE = '''Please grade my submission. Expected output: 55
```python
def sum_of_squares(n):
    """Return the sum of squares from 1 to n."""
    return sum(i * i for i in range(1, n + 1))

print(sum_of_squares(5))
```'''
BUGGY_CODE = '''Grade my code:
```python
def average(xs):
    return sum(xs) / len(xs)
print(average([]))
```'''
EVIL_CODE = '''Check my code please:
```python
import os
print(os.listdir("/"))
```'''


def _banner(text: str) -> None:
    print(f"\n{'#' * 70}\n# {text}\n{'#' * 70}")


def cmd_demo(args) -> None:
    """End-to-end demo: curriculum query -> secure grading -> routing + HITL -> memory isolation."""
    from src.tools.rag_retriever import build_vector_store
    from src.agents.orchestrator import build_orchestrator

    setup_logging(not args.quiet)
    print(f"LLM mode: {llm_mode().upper()}")
    build_vector_store()
    graph = build_orchestrator()
    approver = auto_approver if args.auto_approve else console_approver
    ahmed = "session-SPR-1001"

    _banner("1) Curriculum query (agentic RAG, learner SPR-1001)")
    out = run_turn(graph, ahmed, "How does agentic RAG decide to search again? (SPR-1001)", "SPR-1001", approver)
    print("\nANSWER:\n" + out["answer"])

    _banner("2) Secure grading of a coding submission (sandbox + evaluator-optimizer)")
    print("ANSWER:\n" + run_turn(graph, ahmed, DEMO_CODE, approver=approver)["answer"])

    _banner("3) Sandbox blocks a malicious submission")
    print("ANSWER:\n" + run_turn(graph, ahmed, EVIL_CODE, approver=approver)["answer"])

    _banner("4) Career routing from live DB records -> HITL for tier-1 partner")
    out = run_turn(graph, ahmed, "Which partner opening fits me?", approver=approver)
    print("\nANSWER:\n" + out["answer"])

    _banner("5) Memory isolation: another learner on another thread (needs review status)")
    out = run_turn(graph, "session-SPR-1007", "Which partner opening fits me? (SPR-1007)", "SPR-1007", approver)
    print("ANSWER:\n" + out["answer"])
    for tid in (ahmed, "session-SPR-1007"):
        st = graph.get_state({"configurable": {"thread_id": tid}}).values
        print(f"thread {tid}: learner_id={st['learner_id']}, stored messages={len(st['messages'])}")


def cmd_chat(args) -> None:
    """Interactive chat for one learner session (state persists for the whole process)."""
    from src.tools.rag_retriever import build_vector_store
    from src.agents.orchestrator import build_orchestrator

    setup_logging(not args.quiet)
    build_vector_store()
    graph = build_orchestrator()
    thread = f"session-{args.learner}-{uuid.uuid4().hex[:6]}"
    print(f"[{llm_mode()} mode] chatting as {args.learner} on thread {thread}. Type 'exit' to quit.\n"
          "Tip: paste code inside a ```python block to have it graded (end with 'Expected output: ...').")
    first = True
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        out = run_turn(graph, thread, text, args.learner if first else None, console_approver)
        first = False
        print("\nSprints AI: " + str(out["answer"]))


def cmd_ask_db(args) -> None:
    """Ask the SQL ReAct agent a free-form question about the learner database (needs OpenAI)."""
    if llm_mode() != "openai":
        raise SystemExit("ask-db uses a tool-calling model: set OPENAI_API_KEY (offline mock mode has no ReAct agent).")
    from src.tools.db_queries import build_sql_react_agent

    setup_logging(not args.quiet)
    agent = build_sql_react_agent(get_llm())
    out = agent.invoke({"messages": [HumanMessage(content=args.question)]})
    for m in out["messages"]:
        print(f"{type(m).__name__}: {str(m.content)[:300]}")


def cmd_ingest(args) -> None:
    """(Re)build the SQLite database and the ChromaDB collection."""
    from src.tools.db_queries import build_database
    from src.tools.rag_retriever import build_vector_store

    n = build_database(CSV_PATH, DB_PATH)
    print(f"SQLite: {n} learners loaded into {DB_PATH.name}")
    store = build_vector_store(rebuild=True)
    print(f"ChromaDB: {len(store.get()['ids'])} chunks indexed ({llm_mode()} embeddings)")


def cmd_bench(args) -> None:
    """Benchmark retrieval latency against the 1000 ms acceptance criterion."""
    from src.tools.rag_retriever import benchmark_retrieval, build_vector_store

    build_vector_store()
    print(benchmark_retrieval(runs=args.runs))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.main", description="Sprints AI Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="run the end-to-end demo scenario"); p.add_argument("--auto-approve", action="store_true"); p.add_argument("--quiet", action="store_true"); p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("chat", help="interactive chat"); p.add_argument("--learner", required=True); p.add_argument("--quiet", action="store_true"); p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("ask-db", help="ReAct SQL agent (OpenAI only)"); p.add_argument("question"); p.add_argument("--quiet", action="store_true"); p.set_defaults(fn=cmd_ask_db)
    p = sub.add_parser("ingest", help="rebuild SQLite + ChromaDB"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("bench", help="retrieval latency benchmark"); p.add_argument("--runs", type=int, default=20); p.set_defaults(fn=cmd_bench)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
