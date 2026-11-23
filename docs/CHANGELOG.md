# Decision log
- **D1 Mock mode.** No API key available -> every LLM call goes through one Runnable interface; a rule-based mock
  and offline hashing embeddings stand in. Same graph in both modes.
- **D2 Deterministic matching.** Client's top fear is hallucination -> job matching is SQL only; the LLM never writes job facts.
- **D3 Custom sandbox instead of stock PythonREPL.** The stock utility executes in-process (unsafe); client demanded strict sandboxing.
- **D4 Mock `job_openings` table** because the CSV has no openings but the client wants routing "to a partner opening".
- **D5 HITL gate is a separate node** so the node re-run on resume has no side effects before `interrupt`.
- **D6 Per-turn state reset in router** because the checkpointer persists state across turns of one thread.
- **D7 Caps:** 3 RAG attempts, 3 refinements; after the cap the system says "Unable to verify" / flags for a human grader.
- **D8 Not executed here:** the build sandbox had no LangGraph/Chroma/network; sandbox + DB were tested, the graph tests are in `tests/test_graph.py` for the user to run.
