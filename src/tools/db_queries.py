"""SQLite tools for the Advisor Agent (learner records + partner openings).

Curriculum mapping: Sprint 1 - "SQL tool + ReAct agent". The ``sprints_learner_db`` CSV is
loaded into SQLite; the agent may only read it (read-only connection, SELECT-only guard).

Anti-hallucination design (client's biggest worry): every career statement the agents make
is built from rows returned here. If the DB fails or the learner does not exist, tools return
a structured 'Unable to verify' payload instead of letting the LLM guess.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from typing import Any

from src.config import CSV_PATH, DB_PATH
from src.fallback import unable_to_verify

SCHEMA_DESCRIPTION = """Tables:
learners(learner_id TEXT PK, name TEXT, bootcamp_track TEXT, completed_modules INT,
         python_assignment_score INT, target_salary_egp INT, preferred_employers TEXT,
         readiness_status TEXT)   -- readiness_status in: Ready for Placement, In Progress, Needs Review, Dropped
job_openings(opening_id TEXT PK, employer TEXT, title TEXT, track TEXT,
             salary_min_egp INT, salary_max_egp INT)
"""

# --- MOCK partner openings (the CSV has no openings table; the client asked for routing to
# "a partner opening based on live database records", so we seed an explicit demo table).
_TITLES = {
    "Advanced Agentic AI": "AI Agents Engineer",
    "Backend": "Backend Engineer",
    "Frontend": "Frontend Engineer",
    "Data Science": "Data Scientist",
}
_SALARY = {  # (min, max) EGP / month - illustrative demo numbers
    "Advanced Agentic AI": (28000, 42000),
    "Backend": (16000, 26000),
    "Frontend": (12000, 22000),
    "Data Science": (22000, 34000),
}
_EMPLOYERS = ["Fawry", "Swvl", "Vodafone Egypt", "e& Egypt"]


def build_database(csv_path=CSV_PATH, db_path=DB_PATH) -> int:
    """(Re)build the SQLite database from the starter CSV. Returns the number of learners."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
        CREATE TABLE learners(
            learner_id TEXT PRIMARY KEY, name TEXT NOT NULL, bootcamp_track TEXT NOT NULL,
            completed_modules INTEGER, python_assignment_score INTEGER,
            target_salary_egp INTEGER, preferred_employers TEXT, readiness_status TEXT);
        CREATE TABLE job_openings(
            opening_id TEXT PRIMARY KEY, employer TEXT NOT NULL, title TEXT NOT NULL,
            track TEXT NOT NULL, salary_min_egp INTEGER, salary_max_egp INTEGER);
        """)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = [(r["learner_id"].strip(), r["name"].strip(), r["bootcamp_track"].strip(),
                     int(r["completed_modules"]), int(r["python_assignment_score"]),
                     int(r["target_salary_egp"]), r["preferred_employers"].strip(),
                     r["readiness_status"].strip()) for r in csv.DictReader(fh)]
        conn.executemany("INSERT INTO learners VALUES (?,?,?,?,?,?,?,?)", rows)
        openings, n = [], 1
        for employer in _EMPLOYERS:
            for track, title in _TITLES.items():
                lo, hi = _SALARY[track]
                openings.append((f"JOB-{n:03d}", employer, title, track, lo, hi))
                n += 1
        conn.executemany("INSERT INTO job_openings VALUES (?,?,?,?,?,?)", openings)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _connect_ro() -> sqlite3.Connection:
    """Open the database read-only so agents can never modify learner records."""
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_learner_profile(learner_id: str) -> dict[str, Any] | None:
    """Fetch completed modules, EGP target salary and preferred employers for a learner.

    Raises sqlite3.Error if the DB is unavailable. Returns None if the learner is unknown.
    """
    conn = _connect_ro()
    try:
        row = conn.execute("SELECT * FROM learners WHERE learner_id = ?", (learner_id.strip().upper(),)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    profile = dict(row)
    profile["preferred_employers"] = [e.strip() for e in profile["preferred_employers"].split(",") if e.strip()]
    return profile


def find_matching_openings(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic placement matching - NO LLM involved, so nothing can be invented.

    A learner is matched only if (a) readiness_status is 'Ready for Placement', (b) the
    opening's employer is one of the learner's preferred employers and (c) the opening's
    track equals the learner's bootcamp track. Results are ranked: openings whose salary band
    contains the learner's EGP target first, then by the learner's own employer preference.
    """
    if profile["readiness_status"] != "Ready for Placement":
        return []
    conn = _connect_ro()
    try:
        rows = conn.execute("SELECT * FROM job_openings WHERE track = ?", (profile["bootcamp_track"],)).fetchall()
    finally:
        conn.close()
    prefs = profile["preferred_employers"]
    target = profile["target_salary_egp"]
    matches = []
    for r in rows:
        if r["employer"] not in prefs:
            continue
        in_range = r["salary_min_egp"] <= target <= r["salary_max_egp"]
        matches.append({**dict(r), "target_in_range": in_range, "_rank": (not in_range, prefs.index(r["employer"]))})
    matches.sort(key=lambda m: m["_rank"])
    for m in matches:
        m.pop("_rank")
    return matches


_FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def run_readonly_sql(sql: str, limit: int = 50) -> list[dict[str, Any]]:
    """Execute a single read-only SELECT statement (used by the ReAct agent's SQL tool)."""
    stmt = sql.strip().rstrip(";").strip()
    if not stmt.lower().startswith("select") or ";" in stmt or _FORBIDDEN_SQL.search(stmt):
        raise ValueError("Only a single SELECT statement is allowed")
    conn = _connect_ro()
    try:
        return [dict(r) for r in conn.execute(stmt).fetchmany(limit)]
    finally:
        conn.close()


def safe_get_profile(learner_id: str) -> dict[str, Any]:
    """Never-raising profile lookup. Returns {'status': 'ok', 'data': ...} or 'Unable to verify'."""
    try:
        profile = get_learner_profile(learner_id)
    except Exception as exc:  # noqa: BLE001
        return unable_to_verify("sqlite_db", f"database error: {type(exc).__name__}")
    if profile is None:
        return unable_to_verify("sqlite_db", f"no learner record found for '{learner_id}'")
    return {"status": "ok", "data": profile}


def safe_find_openings(profile: dict[str, Any]) -> dict[str, Any]:
    """Never-raising openings lookup. Returns {'status': 'ok', 'data': [...]} or 'Unable to verify'."""
    try:
        return {"status": "ok", "data": find_matching_openings(profile)}
    except Exception as exc:  # noqa: BLE001
        return unable_to_verify("sqlite_db", f"database error: {type(exc).__name__}")


def get_sql_tool():
    """Return the LangChain ``@tool`` that the ReAct agent uses to query the learner DB."""
    from langchain_core.tools import tool

    @tool
    def query_learner_db(sql: str) -> str:
        """Run ONE read-only SQL SELECT against the Sprints learner database.

        Curriculum mapping: Sprint 1 custom SQL tool for the ReAct agent.
        Use it to look up a learner's completed_modules, target_salary_egp (EGP) and
        preferred_employers. Schema:
        learners(learner_id, name, bootcamp_track, completed_modules, python_assignment_score,
        target_salary_egp, preferred_employers, readiness_status);
        job_openings(opening_id, employer, title, track, salary_min_egp, salary_max_egp).
        """
        try:
            return str(run_readonly_sql(sql))
        except Exception as exc:  # noqa: BLE001
            return str(unable_to_verify("sqlite_db", f"query failed: {type(exc).__name__}: {exc}"))

    return query_learner_db


REACT_SYSTEM_PROMPT = (
    "You are the Sprints Advisor. Answer questions about a learner ONLY from the results of the "
    "query_learner_db tool. Never invent data. If the tool returns 'Unable to verify' or no rows, "
    "reply exactly 'Unable to verify' and explain briefly. Salaries are monthly EGP.\n" + SCHEMA_DESCRIPTION
)


def build_sql_react_agent(llm):
    """Build the simple ReAct agent (reason -> SQL tool -> observe -> answer).

    Curriculum mapping: Sprint 1 'ReAct agent with a custom tool'. Requires a tool-calling
    chat model (e.g. ChatOpenAI); in offline mock mode the Advisor uses deterministic lookups.
    """
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(llm, [get_sql_tool()], prompt=REACT_SYSTEM_PROMPT)
