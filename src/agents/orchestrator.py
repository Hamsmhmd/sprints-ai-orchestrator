"""Orchestrator: unified StateGraph that routes learner queries to the Advisor or Assessor.

Curriculum mapping:
  * Sprint 2 - StateGraph + global TypedDict state, conditional routing by intent.
  * Sprint 2 - MemorySaver checkpointer: one thread_id per learner session isolates context.
  * Sprint 2 - Human-in-the-Loop: ``interrupt`` pauses the graph before a tier-1 placement
    recommendation (e.g. Fawry) is released; execution resumes with Command(resume=...).
  * Sprint 2 - Agentic protocol: all workers exchange HumanMessage / AIMessage / ToolMessage.
"""
from __future__ import annotations

import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.advisor_agent import make_advisor_nodes
from src.agents.assessor_agent import make_assessor_nodes
from src.agents.common import build_chain, safe_invoke
from src.config import TIER1_PARTNERS
from src.llm import get_llm
from src.logging_utils import agent_message, log_messages
from src.schemas import IntentDecision
from src.state import OrchestratorState, last_human_text

SYS_ROUTER = ("You are the Sprints orchestrator. Classify the learner's message into exactly one intent: "
              "curriculum_question (asks about course content), profile_question (asks about their own record: modules, "
              "salary target, employers), career_match (asks for job / placement / employer recommendations), "
              "code_grading (submits Python code to be graded).")

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
_EXPECTED = re.compile(r"expected output\s*[:=]\s*(.+)", re.I)
_LEARNER = re.compile(r"\bSPR-\d{4}\b", re.I)


def make_router_node(llm):
    """Create the intent-routing node."""
    chain = build_chain(llm, "route", SYS_ROUTER, "MESSAGE: {message}", IntentDecision)

    def router(state: OrchestratorState) -> dict:
        """Classify intent, extract code / learner id, and reset per-turn working fields.

        Per-turn fields are reset because the checkpointer keeps state between turns of a thread;
        durable facts (learner_id, messages) are intentionally NOT reset.
        """
        text = last_human_text(state["messages"])
        code = _CODE_BLOCK.search(text)
        decision = safe_invoke(chain, {"message": text})
        intent = decision.intent if decision else "curriculum_question"
        if code:  # deterministic guard: a code block always means grading
            intent = "code_grading"
        lid = _LEARNER.search(text)
        exp = _EXPECTED.search(text)
        msg = agent_message("orchestrator", f"Routing intent='{intent}'" + (f" for learner {lid.group(0).upper()}" if lid else ""))
        log_messages("orchestrator", [msg])
        update = {"intent": intent, "messages": [msg], "current_query": "", "retrieved_context": "", "retrieved_sources": [],
                  "rag_attempts": 0, "context_sufficient": False, "retrieval_failed": False,
                  "code_submission": code.group(1).strip() if code else "",
                  "expected_output": exp.group(1).strip() if exp else "",
                  "execution_result": {}, "assessment": {}, "judge_verdict": {}, "refinement_count": 0,
                  "recommendation": {}, "approval_status": "", "final_answer": ""}
        if lid:
            update["learner_id"] = lid.group(0).upper()
        return update

    return router


def hitl_gate(state: OrchestratorState) -> dict:
    """Pause for human approval when the recommended employer is a tier-1 partner.

    NOTE: LangGraph re-runs this node from the top when resumed, so nothing before ``interrupt``
    may have side effects.
    """
    rec = state["recommendation"]
    if rec["employer"] not in TIER1_PARTNERS:
        text = rec["summary"]
        msg = agent_message("orchestrator", "Human approval not required (not a tier-1 partner).")
        log_messages("orchestrator", [msg])
        return {"approval_status": "not_required", "final_answer": text, "messages": [msg]}
    decision = interrupt({"type": "placement_approval", "learner_id": rec["learner_id"], "learner": rec["name"],
                          "employer": rec["employer"], "opening_id": rec["opening_id"], "title": rec["title"],
                          "salary_band_egp": rec["salary_band_egp"],
                          "question": f"Approve recommending {rec['name']} for {rec['title']} at tier-1 partner {rec['employer']}?"})
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    note = (decision.get("note") if isinstance(decision, dict) else "") or ""
    if approved:
        text = rec["summary"] + f"\nPlacement recommendation APPROVED by a human reviewer.{' Note: ' + note if note else ''}"
        status = "approved"
    else:
        text = (f"The recommendation for {rec['employer']} was held back: a human reviewer did not approve it."
                f"{' Note: ' + note if note else ''} Your advisor will follow up with you.")
        status = "rejected"
    msg = agent_message("orchestrator", f"Human decision: {status}")
    log_messages("orchestrator", [msg])
    return {"approval_status": status, "final_answer": text, "messages": [msg, agent_message("advisor", text)]}


def build_orchestrator(llm=None, checkpointer=None, judge_llm=None):
    """Assemble and compile the unified multi-agent StateGraph.

    Args:
        llm: chat model Runnable (defaults to :func:`src.llm.get_llm`).
        checkpointer: LangGraph checkpointer (defaults to in-memory ``MemorySaver``).
        judge_llm: optional different model for the LLM-as-a-Judge node.
    """
    llm = llm or get_llm()
    adv = make_advisor_nodes(llm)
    ass = make_assessor_nodes(llm, judge_llm)

    g = StateGraph(OrchestratorState)
    g.add_node("router", make_router_node(llm))
    for name in ("rag_search", "rag_reflect", "rag_rewrite", "rag_answer", "advisor_profile", "advisor_career"):
        g.add_node(name, adv[name])
    g.add_node("hitl_gate", hitl_gate)
    for name in ("assessor_execute", "assessor_grade", "assessor_judge", "assessor_refine", "assessor_finalize"):
        g.add_node(name, ass[name])

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["intent"], {
        "curriculum_question": "rag_search", "profile_question": "advisor_profile",
        "career_match": "advisor_career", "code_grading": "assessor_execute"})
    # Advisor: agentic RAG loop
    g.add_conditional_edges("rag_search", adv["route_after_search"], {"rag_reflect": "rag_reflect", "rag_answer": "rag_answer"})
    g.add_conditional_edges("rag_reflect", adv["route_after_reflect"], {"rag_answer": "rag_answer", "rag_rewrite": "rag_rewrite"})
    g.add_edge("rag_rewrite", "rag_search")
    g.add_edge("rag_answer", END)
    g.add_edge("advisor_profile", END)
    # Advisor: career -> HITL
    g.add_conditional_edges("advisor_career", adv["route_after_career"], {"hitl_gate": "hitl_gate", "__end__": END})
    g.add_edge("hitl_gate", END)
    # Assessor: execute -> grade -> judge <-> refine (max 3)
    g.add_conditional_edges("assessor_execute", ass["route_after_execute"], {"assessor_grade": "assessor_grade", "assessor_finalize": "assessor_finalize"})
    g.add_conditional_edges("assessor_grade", ass["route_after_grade"], {"assessor_judge": "assessor_judge", "assessor_finalize": "assessor_finalize"})
    g.add_conditional_edges("assessor_judge", ass["route_after_judge"], {"assessor_finalize": "assessor_finalize", "assessor_refine": "assessor_refine"})
    g.add_edge("assessor_refine", "assessor_judge")
    g.add_edge("assessor_finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
