"""Central configuration for the Sprints AI Orchestrator.

Curriculum mapping: Sprint 1 (LangChain foundations) - every tunable value that the
client's quality bar mentions (chunk size, latency budget, loop limits) lives here so
that it can be audited in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRICULUM_DIR = DATA_DIR / "curriculum_docs"
CSV_PATH = DATA_DIR / "sprints_learner_db.csv"
DB_PATH = DATA_DIR / "sprints_learner.db"
CHROMA_DIR = ROOT / "chroma_db"

# --- RAG (client requirement: 500-token chunks, 50-token overlap, <1000 ms retrieval)
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
RETRIEVAL_BUDGET_MS = 1000
RETRIEVAL_TOP_K = 4
MAX_RAG_ATTEMPTS = 3          # initial search + up to 2 reformulations

# --- Assessor (client requirement: evaluator-optimizer halts after 3 refinements)
MAX_REFINEMENTS = 3
PASS_SCORE = 60

# --- Sandbox (client non-negotiable: strictly sandboxed grading)
SANDBOX_TIMEOUT_S = 5
SANDBOX_MAX_OUTPUT_CHARS = 4000

# --- Human-in-the-loop: placements to these partners must be approved by a human.
# The brief only names Fawry as an example of a tier-1 partner; it is configurable.
TIER1_PARTNERS = {p.strip() for p in os.getenv("TIER1_PARTNERS", "Fawry").split(",") if p.strip()}

VERBOSE = os.getenv("VERBOSE", "1") == "1"
