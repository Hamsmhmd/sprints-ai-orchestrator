"""LLM factory: real OpenAI model when a key exists, deterministic offline MOCK otherwise.

Curriculum mapping: Sprint 1 - LangChain Runnables. Both modes are plain Runnables that take a
chat prompt and return an AIMessage, so every agent uses the same
``prompt | llm | PydanticOutputParser`` chain in both modes.

MOCK mode exists so the whole system (routing, RAG loop, evaluator-optimizer, HITL, memory) can be
run, tested and demoed with no API key and no cost. It is rule-based, NOT intelligent: the
technical report states this limitation clearly. Set OPENAI_API_KEY to use a real model.
"""
from __future__ import annotations

import json
import os
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.config import PASS_SCORE

_STOP = set("a an the is are was were be to of in on for and or how what why when does do did can i me my you your it this that with as at by from about which should would could explain tell".split())
_EXPAND = {
    "memory": "checkpointer MemorySaver thread_id", "remember": "checkpointer MemorySaver thread_id",
    "loop": "conditional edge reflection", "human": "interrupt approval Command resume",
    "approval": "interrupt approval Command resume", "rag": "retrieval ChromaDB embeddings chunks",
    "chunk": "chunks 500 tokens overlap splitter", "sandbox": "execute learner code safely timeout",
    "judge": "evaluator-optimizer rubric refinement", "route": "conditional edges orchestrator intent",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_&\-]+", text.lower()) if t not in _STOP and len(t) > 2]


def _section(text: str, name: str, nxt: list[str]) -> str:
    """Extract the text after 'NAME:' up to the next known section header."""
    pattern = rf"{name}:\s*(.*?)(?=\n(?:{'|'.join(nxt)}):|\Z)" if nxt else rf"{name}:\s*(.*)"
    m = re.search(pattern, text, re.S)
    return m.group(1).strip() if m else ""


# ----------------------------------------------------------------------------- mock tasks
def _mock_route(human: str) -> str:
    msg = human.lower()
    if "```" in msg or re.search(r"\bdef \w+\(", msg) or any(k in msg for k in ("grade my", "review my code", "my submission")):
        intent = "code_grading"
    elif any(k in msg for k in ("recommend", "placement", "job", "opening", "match me", "apply", "which company", "which employer", "partner", "hire", "fit me", "fit my")):
        intent = "career_match"
    elif any(k in msg for k in ("my salary", "my target", "my modules", "my progress", "completed", "my score", "my profile", "my status", "my employers")):
        intent = "profile_question"
    else:
        intent = "curriculum_question"
    return json.dumps({"intent": intent, "reason": "keyword heuristic (mock mode)"})


def _mock_sufficiency(human: str) -> str:
    q = _section(human, "QUESTION", ["CONTEXT"])
    ctx = _section(human, "CONTEXT", []).lower()
    toks = set(_tokens(q))
    cov = sum(1 for t in toks if t in ctx) / max(len(toks), 1)
    ok = cov >= 0.5 and len(ctx) > 40
    return json.dumps({"sufficient": ok, "reason": f"keyword coverage {cov:.0%} (mock mode)"})


def _mock_rewrite(human: str) -> str:
    q = _section(human, "QUESTION", [])
    toks = _tokens(q)
    extra = list(dict.fromkeys(_EXPAND[t] for t in toks if t in _EXPAND))
    return json.dumps({"query": " ".join(toks + extra)[:200] or q})


def _mock_answer(human: str) -> str:
    q = _section(human, "QUESTION", ["CONTEXT", "HISTORY"])
    ctx = _section(human, "CONTEXT", ["HISTORY"])
    toks = set(_tokens(q))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", ctx) if len(s.strip()) > 30 and not s.startswith(("[source", "#"))]
    ranked = sorted(sents, key=lambda s: -sum(1 for t in toks if t in s.lower()))[:3]
    return "Based on the Sprints curriculum:\n" + "\n".join(f"- {s}" for s in ranked)


def _parse_exec(human: str) -> dict:
    nxt = ["STDOUT", "STDERR", "EXPECTED_OUTPUT", "CODE", "PREVIOUS_ASSESSMENT", "JUDGE_ISSUES"]
    return {
        "status": _section(human, "EXECUTION_STATUS", nxt),
        "stdout": _section(human, "STDOUT", nxt),
        "stderr": _section(human, "STDERR", nxt),
        "expected": _section(human, "EXPECTED_OUTPUT", nxt),
        "code": _section(human, "CODE", nxt),
    }


def _grade(e: dict, style_points: bool) -> dict:
    """Deterministic rubric scoring used by the mock grader / refiner."""
    status, code = e["status"], e["code"]
    has_doc = '"""' in code or "'''" in code
    has_def = "def " in code
    pts = []
    if status == "blocked":
        score = 0
        pts.append(f"Security policy violation, nothing was executed: {e['stderr'].strip()}")
    elif status == "timeout":
        score = 20
        pts.append("Execution exceeded the time limit: look for infinite loops or very slow algorithms.")
    elif status == "runtime_error":
        score = 35
        last = e["stderr"].strip().splitlines()[-1] if e["stderr"].strip() else "unknown error"
        pts.append(f"Runtime error: {last}. Fix this error first, then re-run.")
    else:
        score = 70
        if e["expected"]:
            if e["stdout"].strip() == e["expected"].strip():
                score += 15
                pts.append("The output matches the expected output.")
            else:
                score = 55
                pts.append(f"The output '{e['stdout'].strip()[:60]}' differs from the expected '{e['expected'].strip()[:60]}'.")
        else:
            score += 5
            pts.append(f"Code ran without errors; output was '{e['stdout'].strip()[:60]}'. No expected output was supplied, so correctness could not be fully verified.")
        if has_doc:
            score += 10
        if has_def:
            score += 5
        if style_points:
            pts.append("Good: functions are documented with docstrings." if has_doc else "Add docstrings to your functions to explain purpose, arguments and return values.")
    if style_points and status != "ok":
        pts.append("After fixing, add a few assert-based checks for edge cases (empty input, negative numbers).")
    elif style_points and status == "ok" and len(pts) < 2:
        pts.append("Consider adding assert-based checks for edge cases.")
    score = max(0, min(100, score))
    return {"score": score, "feedback_points": pts, "pass_status": "pass" if score >= PASS_SCORE else "fail"}


def _mock_grade(human: str) -> str:
    return json.dumps(_grade(_parse_exec(human), style_points=False))


def _mock_refine(human: str) -> str:
    return json.dumps(_grade(_parse_exec(human), style_points=True))


def _mock_judge(human: str) -> str:
    e = _parse_exec(human)
    try:
        a = json.loads(_section(human, "ASSESSMENT", ["EXECUTION_STATUS"]).splitlines()[0])
    except Exception:  # noqa: BLE001
        return json.dumps({"approved": False, "issues": ["Assessment is not valid JSON"]})
    issues = []
    if e["status"] in ("blocked", "timeout", "runtime_error") and a["score"] >= PASS_SCORE:
        issues.append("Score is inconsistent with the execution result: failed runs cannot pass.")
    if e["status"] == "ok" and a["score"] < 50 and not e["expected"]:
        issues.append("Score is too low for code that ran without errors.")
    if a["pass_status"] != ("pass" if a["score"] >= PASS_SCORE else "fail"):
        issues.append("pass_status does not match the 60-point pass mark.")
    if len(a["feedback_points"]) < 2:
        issues.append("Provide at least 2 specific feedback points (what happened + what to improve).")
    return json.dumps({"approved": not issues, "issues": issues})


_MOCK_TASKS = {
    "route": _mock_route, "sufficiency": _mock_sufficiency, "rewrite": _mock_rewrite,
    "answer": _mock_answer, "grade": _mock_grade, "refine": _mock_refine, "judge": _mock_judge,
}


def _mock_llm(prompt_value) -> AIMessage:
    msgs = prompt_value.to_messages()
    system = next((str(m.content) for m in msgs if m.type == "system"), "")
    human = str(msgs[-1].content)
    task = re.search(r"TASK=(\w+)", system)
    if not task or task.group(1) not in _MOCK_TASKS:
        raise ValueError("mock LLM: unknown task")
    return AIMessage(content=_MOCK_TASKS[task.group(1)](human))


class HashingEmbeddings:
    """Offline bag-of-words embeddings (feature hashing) used in mock mode.

    Implements the LangChain ``Embeddings`` interface (embed_documents / embed_query).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import math

        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 100) & 1 else -1.0
            vec[(h >> 20) % self.dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ----------------------------------------------------------------------------- factories
def llm_mode() -> str:
    """Return 'openai' or 'mock' based on env (LLM_MODE overrides; default by API key)."""
    forced = os.getenv("LLM_MODE", "").strip().lower()
    if forced in ("mock", "openai"):
        return forced
    return "openai" if os.getenv("OPENAI_API_KEY") else "mock"


def get_llm():
    """Return the chat model Runnable for the current mode."""
    if llm_mode() == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return RunnableLambda(_mock_llm)


def get_embeddings():
    """Return the embedding model for the current mode (OpenAI or offline hashing)."""
    if llm_mode() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HashingEmbeddings()


def stubborn_judge(base):
    """Wrap an LLM so the judge ALWAYS rejects. Used to prove the 3-iteration halt."""
    def _run(prompt_value):
        system = next((str(m.content) for m in prompt_value.to_messages() if m.type == "system"), "")
        if "TASK=judge" in system:
            return AIMessage(content=json.dumps({"approved": False, "issues": ["(stubborn judge) never satisfied"]}))
        return base.invoke(prompt_value)
    return RunnableLambda(_run)
