"""Advisor Agent: curriculum Q&A (agentic RAG), learner profile lookup and career matching.

Curriculum mapping:
  * Sprint 1 - RAG retrieval tool, SQL tool + ReAct agent.
  * Sprint 2 - Agentic RAG with a self-reflection conditional edge:
        rag_search -> rag_reflect --(insufficient)--> rag_rewrite -> rag_search ...
    The loop is capped at MAX_RAG_ATTEMPTS; if the context is still insufficient the agent
    answers 'Unable to verify' instead of guessing (anti-hallucination).
  * Career matching is DETERMINISTIC (SQL rows only) - the LLM never writes job facts.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.common import build_chain, safe_invoke
from src.config import MAX_RAG_ATTEMPTS
from src.fallback import is_unable, unable_to_verify
from src.llm import llm_mode
from src.logging_utils import agent_message, log_messages, tool_exchange
from src.schemas import QueryRewrite, SufficiencyDecision
from src.state import OrchestratorState, history_text, last_human_text
from src.tools.db_queries import build_sql_react_agent, safe_find_openings, safe_get_profile
from src.tools.rag_retriever import retrieve

SYS_REFLECT = ("You are a strict reviewer. Decide whether the CONTEXT contains enough information to fully "
               "answer the QUESTION. Answer sufficient=true only if the facts needed are explicitly present.")
SYS_REWRITE = ("The previous search did not return enough information. Write a better, more specific "
               "keyword-rich search query for the curriculum search engine (max 15 words).")
SYS_ANSWER = ("You are the Sprints Advisor. Answer the QUESTION using ONLY the CONTEXT. If the context does not "
              "contain the answer, say you are unable to verify it. Be concise and mention module names.")


def make_advisor_nodes(llm) -> dict[str, Any]:
    """Create the Advisor Agent's graph nodes bound to ``llm``."""
    reflect_chain = build_chain(llm, "sufficiency", SYS_REFLECT, "QUESTION: {question}\nCONTEXT:\n{context}", SufficiencyDecision)
    rewrite_chain = build_chain(llm, "rewrite", SYS_REWRITE, "PREVIOUS_QUERY: {previous}\nQUESTION: {question}", QueryRewrite)
    answer_chain = build_chain(llm, "answer", SYS_ANSWER, "QUESTION: {question}\nCONTEXT:\n{context}\nHISTORY:\n{history}")

    # ------------------------------------------------------------------ agentic RAG
    def rag_search(state: OrchestratorState) -> dict:
        """Retrieve curriculum chunks for the current (possibly reformulated) query."""
        question = last_human_text(state["messages"])
        query = state.get("current_query") or question
        attempt = state.get("rag_attempts", 0) + 1
        res = retrieve(query)
        summary = {k: res.get(k) for k in ("status", "sources", "latency_ms", "within_budget", "reason")}
        msgs = tool_exchange("advisor", "search_curriculum", {"query": query, "attempt": attempt}, summary)
        if is_unable(res):
            return {"messages": msgs, "rag_attempts": attempt, "retrieval_failed": True, "retrieved_context": ""}
        return {"messages": msgs, "rag_attempts": attempt, "retrieval_failed": False,
                "retrieved_context": res["context"], "retrieved_sources": res["sources"]}

    def rag_reflect(state: OrchestratorState) -> dict:
        """Self-reflection: is the retrieved context sufficient to answer?"""
        decision = safe_invoke(reflect_chain, {"question": last_human_text(state["messages"]),
                                               "context": state.get("retrieved_context", "")})
        ok = bool(decision and decision.sufficient)
        reason = decision.reason if decision else "reflection failed"
        msg = agent_message("advisor", f"Reflection (attempt {state.get('rag_attempts', 1)}): "
                                       f"{'context sufficient' if ok else 'context insufficient'} - {reason}")
        log_messages("advisor", [msg])
        return {"context_sufficient": ok, "messages": [msg]}

    def rag_rewrite(state: OrchestratorState) -> dict:
        """Reformulate the query so the next search can find better context."""
        previous = state.get("current_query") or last_human_text(state["messages"])
        res = safe_invoke(rewrite_chain, {"previous": previous, "question": last_human_text(state["messages"])})
        new_query = res.query if res else previous
        msg = agent_message("advisor", f"Reformulated query: '{new_query}'")
        log_messages("advisor", [msg])
        return {"current_query": new_query, "messages": [msg]}

    def rag_answer(state: OrchestratorState) -> dict:
        """Answer from retrieved context, or return 'Unable to verify' if it never became sufficient."""
        if state.get("retrieval_failed") or not state.get("context_sufficient"):
            text = ("Unable to verify: the Sprints curriculum documents do not contain enough information to "
                    "answer this reliably. Please rephrase, or ask a human advisor.")
        else:
            body = safe_invoke(answer_chain, {"question": last_human_text(state["messages"]),
                                              "context": state["retrieved_context"],
                                              "history": history_text(state["messages"])})
            text = (body or "Unable to verify: the answer could not be generated.")
            sources = ", ".join(state.get("retrieved_sources", []))
            text += f"\n\nSources: {sources}" if sources else ""
        msg = agent_message("advisor", text)
        log_messages("advisor", [msg])
        return {"final_answer": text, "messages": [msg]}

    def route_after_search(state: OrchestratorState) -> str:
        """Conditional edge: skip reflection if the vector store failed."""
        return "rag_answer" if state.get("retrieval_failed") else "rag_reflect"

    def route_after_reflect(state: OrchestratorState) -> str:
        """Conditional edge: answer, or reformulate and search again (max MAX_RAG_ATTEMPTS)."""
        if state.get("context_sufficient") or state.get("rag_attempts", 0) >= MAX_RAG_ATTEMPTS:
            return "rag_answer"
        return "rag_rewrite"

    # ------------------------------------------------------------------ profile / career
    def advisor_profile(state: OrchestratorState) -> dict:
        """Answer questions about the learner's own record (ReAct agent + SQL tool)."""
        lid = state.get("learner_id")
        question = last_human_text(state["messages"])
        if not lid:
            text = "Unable to verify: please tell me your learner id (e.g. SPR-1001) so I can look you up."
            return {"final_answer": text, "messages": [agent_message("advisor", text)]}
        if llm_mode() == "openai":
            try:
                agent = build_sql_react_agent(llm)
                out = agent.invoke({"messages": [HumanMessage(content=f"Learner id: {lid}. Question: {question}")]})
                new = out["messages"][1:]
                log_messages("advisor-react", new)
                text = str(out["messages"][-1].content)
                return {"final_answer": text, "messages": new}
            except Exception as exc:  # noqa: BLE001
                text = f"Unable to verify: the learner database could not be queried ({type(exc).__name__})."
                return {"final_answer": text, "messages": [agent_message("advisor", text)]}
        res = safe_get_profile(lid)
        msgs = tool_exchange("advisor", "query_learner_db", {"learner_id": lid}, res)
        if is_unable(res):
            text = f"Unable to verify: {res['reason']}."
        else:
            p = res["data"]
            text = (f"{p['name']} ({p['bootcamp_track']}): {p['completed_modules']} modules completed, Python "
                    f"assignment score {p['python_assignment_score']}, target salary {p['target_salary_egp']:,} EGP/month, "
                    f"preferred employers: {', '.join(p['preferred_employers'])}, status: {p['readiness_status']}.")
        msgs.append(agent_message("advisor", text))
        return {"final_answer": text, "messages": msgs}

    def advisor_career(state: OrchestratorState) -> dict:
        """Match a learner to a partner opening using ONLY live database records."""
        lid = state.get("learner_id")
        if not lid:
            text = "Unable to verify: please tell me your learner id (e.g. SPR-1001) so I can check your record."
            return {"final_answer": text, "recommendation": {}, "messages": [agent_message("advisor", text)]}
        res = safe_get_profile(lid)
        msgs = tool_exchange("advisor", "query_learner_db", {"learner_id": lid}, res)
        if is_unable(res):
            text = f"Unable to verify: {res['reason']}. No recommendation was made."
            msgs.append(agent_message("advisor", text))
            return {"final_answer": text, "recommendation": {}, "messages": msgs}
        p = res["data"]
        match = safe_find_openings(p)
        msgs += tool_exchange("advisor", "find_matching_openings",
                              {"learner_id": lid, "track": p["bootcamp_track"], "preferred": p["preferred_employers"]}, match)
        if is_unable(match):
            text = f"Unable to verify: {match['reason']}. No recommendation was made."
        elif p["readiness_status"] != "Ready for Placement":
            text = (f"No placement recommendation for {p['name']}: the database shows readiness status "
                    f"'{p['readiness_status']}' ({p['completed_modules']} modules completed, assignment score "
                    f"{p['python_assignment_score']}). Only learners marked 'Ready for Placement' are matched.")
        elif not match["data"]:
            text = (f"No open positions currently match {p['name']}'s track ({p['bootcamp_track']}) and preferred "
                    f"employers ({', '.join(p['preferred_employers'])}).")
        else:
            top, others = match["data"][0], match["data"][1:]
            band = f"{top['salary_min_egp']:,}-{top['salary_max_egp']:,} EGP"
            fit = "within" if top["target_in_range"] else "outside"
            text = (f"{p['name']} ({p['bootcamp_track']}, {p['completed_modules']} modules, assignment score "
                    f"{p['python_assignment_score']}, target {p['target_salary_egp']:,} EGP) best matches "
                    f"{top['title']} at {top['employer']} [{top['opening_id']}], salary band {band} "
                    f"(target {fit} band).")
            if others:
                text += " Other matches: " + "; ".join(f"{o['title']} at {o['employer']} [{o['opening_id']}]" for o in others) + "."
            rec = {"learner_id": p["learner_id"], "name": p["name"], "employer": top["employer"], "opening_id": top["opening_id"],
                   "title": top["title"], "salary_band_egp": band, "summary": text}
            msgs.append(agent_message("advisor", text))
            return {"recommendation": rec, "final_answer": "", "messages": msgs}  # final text is released by the HITL gate
        msgs.append(agent_message("advisor", text))
        return {"recommendation": {}, "final_answer": text, "messages": msgs}

    def route_after_career(state: OrchestratorState) -> str:
        """Conditional edge: only real recommendations go through the HITL gate."""
        return "hitl_gate" if state.get("recommendation") else "__end__"

    return {"rag_search": rag_search, "rag_reflect": rag_reflect, "rag_rewrite": rag_rewrite, "rag_answer": rag_answer,
            "advisor_profile": advisor_profile, "advisor_career": advisor_career,
            "route_after_search": route_after_search, "route_after_reflect": route_after_reflect,
            "route_after_career": route_after_career}
