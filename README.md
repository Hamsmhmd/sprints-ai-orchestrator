# Sprints Multi-Agent AI Career & Assessment Orchestrator

Graduation project - *Agentic AI Engineering with LangGraph*. A stateful orchestrator-worker system:

| Agent | Job | Key patterns |
|---|---|---|
| **Orchestrator** | classifies intent, routes, enforces human approval | `StateGraph` + `TypedDict`, `MemorySaver`, `interrupt` (HITL) |
| **Advisor** | curriculum Q&A, learner profile, partner matching | Agentic RAG with self-reflection loop, ChromaDB, SQL tool / ReAct agent |
| **Assessor** | grades Python submissions | sandboxed REPL tool, Pydantic output, Evaluator-Optimizer (max 3 refinements) |

Client constraints (discovery call): unified LangGraph state + per-learner checkpointing; strictly
sandboxed grading; **no hallucinated career matches** (matching is pure SQL, LLM never writes job facts);
live end-to-end demo.

## Quick start (no API key needed)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.main ingest          # builds SQLite + ChromaDB
python -m src.main demo            # full scenario (asks for human approval on Fawry)
python -m src.main demo --auto-approve    # non-interactive
python -m pytest -q                # unit tests (sandbox + database)
python -m src.main bench           # retrieval latency vs the 1000 ms budget
```

**Modes.** With no `OPENAI_API_KEY` the system runs in deterministic **MOCK mode** (rule-based stand-in
for the LLM + offline hashing embeddings) so everything is runnable and testable for free. With a key
(copy `.env.example` to `.env`) it uses `gpt-4o-mini` + `text-embedding-3-small`. The graph, tools, loops
and HITL are identical in both modes. Mock mode is *not* intelligent: it only proves the plumbing.

## Layout
```
data/            learner CSV, generated SQLite DB (learners + job_openings), curriculum_docs/
src/state.py     global TypedDict state          src/schemas.py   Pydantic schemas
src/agents/      orchestrator, advisor_agent, assessor_agent
src/tools/       rag_retriever (ChromaDB), code_repl (sandbox), db_queries (SQL + ReAct)
src/main.py      CLI (ingest / demo / chat / ask-db / bench)
notebooks/       evaluator_optimizer_loop.ipynb  (loop evidence + 3-iteration halt proof)
docs/            charter, changelog, technical report, final presentation
```

## Requirement -> where it lives
| Requirement | File |
|---|---|
| Ingestion, 500/50-token chunks, ChromaDB | `src/tools/rag_retriever.py` |
| Pydantic templates + parsers (score, feedback_points, pass_status) | `src/schemas.py`, `src/agents/common.py` |
| ReAct agent + SQL tool | `src/tools/db_queries.py` (`build_sql_react_agent`) |
| Sandboxed Python REPL tool | `src/tools/code_repl.py` |
| StateGraph + TypedDict routing | `src/state.py`, `src/agents/orchestrator.py` |
| MemorySaver + thread isolation | `build_orchestrator`, `run_turn` in `src/main.py` |
| Agentic RAG self-reflection | `src/agents/advisor_agent.py` (`rag_*` nodes) |
| Evaluator-Optimizer, max 3 | `src/agents/assessor_agent.py` |
| HITL `interrupt` for tier-1 | `hitl_gate` in `src/agents/orchestrator.py` |
| Message protocol + verbose logs | `src/logging_utils.py` |
| "Unable to verify" fallbacks | `src/fallback.py`, tools, `tests/test_core.py` |

## Known limitations / assumptions
- The CSV has **25** learners (the kickoff deck says 500). `job_openings` is **mock demo data** (the CSV has no
  openings); tier-1 = Fawry only (configurable via `TIER1_PARTNERS`).
- The sandbox is defence-in-depth (AST screen + isolated subprocess + limits). Production should add container isolation.
- `MemorySaver` is in-process memory; use a SQLite/Postgres checkpointer to survive restarts.
- Curriculum documents are authored mock content, written for this project.
