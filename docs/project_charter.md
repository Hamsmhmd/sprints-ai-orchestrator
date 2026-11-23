# Project Charter - Sprints Multi-Agent AI Career & Assessment Orchestrator

**Problem.** Sprints' student-success team is overwhelmed by repetitive curriculum questions, manual Python
grading and hand-matching graduates to partners (Fawry, e& Egypt, Swvl, Vodafone Egypt). Target: cut learner wait times by 80%.

**Scope.** Orchestrator + Advisor (agentic RAG, SQL/ReAct, career matching) + Assessor (sandboxed grading,
evaluator-optimizer), per-learner memory, human approval for tier-1 placements, live end-to-end demo.
Out of scope: web UI, authentication, production deployment, real partner job feeds.

**Success criteria.** All 4 Sprint-1 and 6 Sprint-2 requirements working; retrieval < 1000 ms; <= 3 refinements;
"Unable to verify" on tool failure; no learner code escapes the sandbox; zero invented career facts.

**Risks.** (1) Hallucinated matches -> matching is pure SQL. (2) Unsafe learner code -> AST screen + isolated process.
(3) Runaway LLM loops -> hard caps (3 RAG attempts, 3 refinements). (4) No API key available -> deterministic mock mode.

**Assumptions.** 25-row CSV used (deck says 500); mock `job_openings` table; tier-1 = Fawry (configurable);
curriculum documents are authored mock content; MemorySaver is acceptable for the prototype.
