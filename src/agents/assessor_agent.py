"""Assessor Agent: sandboxed execution + Evaluator-Optimizer grading loop.

Curriculum mapping:
  * Sprint 1 - Python REPL tool + Pydantic-parsed feedback (score, feedback_points, pass_status).
  * Sprint 2 - Evaluator-Optimizer: grade -> LLM-as-a-Judge -> (refine -> judge)* with a HARD
    limit of MAX_REFINEMENTS (3) refinement iterations.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.common import build_chain, safe_invoke
from src.config import MAX_REFINEMENTS, PASS_SCORE
from src.fallback import is_unable, unable_to_verify
from src.logging_utils import agent_message, log_messages, tool_exchange
from src.schemas import AssessmentResult, JudgeVerdict
from src.state import OrchestratorState
from src.tools.code_repl import safe_execute_python

RUBRIC = f"""Grading rubric (total 100, pass mark {PASS_SCORE}):
1. Correctness: the code runs without runtime errors and the output matches the expected output (up to 70 points).
2. Readability: meaningful names and structure (up to 5 points).
3. Documentation: functions have docstrings (up to 10 points).
4. Security: code that violates the sandbox policy scores 0. Runtime errors and timeouts must score below {PASS_SCORE}.
Feedback must be specific, actionable and consistent with the score and pass_status. Base every statement on the
execution evidence provided; never claim the code did something the evidence does not show."""

SYS_GRADE = "You are the Sprints Python Assessor. Grade the learner submission using the rubric.\n" + RUBRIC
SYS_JUDGE = ("You are an independent judge reviewing an assessment. Approve ONLY if: (a) the score is consistent with the "
             "execution evidence, (b) pass_status matches the pass mark, (c) there are at least 2 specific feedback points, "
             "(d) feedback is grounded in the evidence. Otherwise list the issues.\n" + RUBRIC)
SYS_REFINE = ("You are the Sprints Python Assessor. Improve the PREVIOUS_ASSESSMENT so that it fixes every item in "
              "JUDGE_ISSUES while following the rubric.\n" + RUBRIC)

_EVIDENCE = ("EXECUTION_STATUS: {status}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\nEXPECTED_OUTPUT: {expected}\nCODE:\n{code}")


def _evidence(state: OrchestratorState) -> dict[str, str]:
    ex = state.get("execution_result", {})
    return {"status": ex.get("status", ""), "stdout": ex.get("stdout", ""), "stderr": ex.get("stderr", ""),
            "expected": state.get("expected_output", ""), "code": state.get("code_submission", "")}


def make_assessor_nodes(llm, judge_llm=None) -> dict[str, Any]:
    """Create the Assessor Agent's graph nodes. ``judge_llm`` lets tests swap in a stricter judge."""
    grade_chain = build_chain(llm, "grade", SYS_GRADE, _EVIDENCE, AssessmentResult)
    judge_chain = build_chain(judge_llm or llm, "judge", SYS_JUDGE, "ASSESSMENT: {assessment}\n" + _EVIDENCE, JudgeVerdict)
    refine_chain = build_chain(llm, "refine", SYS_REFINE,
                               _EVIDENCE + "\nPREVIOUS_ASSESSMENT: {previous}\nJUDGE_ISSUES: {issues}", AssessmentResult)

    def assessor_execute(state: OrchestratorState) -> dict:
        """Run the learner's code in the sandbox (Python REPL tool) and capture stdout / errors."""
        code = state.get("code_submission", "").strip()
        if not code:
            return {"execution_result": {"status": "no_code"}}
        result = safe_execute_python(code)
        shown = {k: result.get(k) for k in ("status", "stdout", "stderr", "duration_ms", "reason")}
        return {"execution_result": result, "messages": tool_exchange("assessor", "python_repl_sandbox", {"code": code[:200]}, shown)}

    def assessor_grade(state: OrchestratorState) -> dict:
        """First grading pass -> Pydantic-validated AssessmentResult."""
        result = safe_invoke(grade_chain, _evidence(state))
        if result is None:
            return {"execution_result": unable_to_verify("llm_grader", "grader returned invalid output"), "refinement_count": 0}
        a = result.model_dump()
        msg = agent_message("assessor", "Initial assessment: " + json.dumps(a, ensure_ascii=False))
        log_messages("assessor", [msg])
        return {"assessment": a, "refinement_count": 0, "messages": [msg]}

    def assessor_judge(state: OrchestratorState) -> dict:
        """LLM-as-a-Judge: review the grading decision against the explicit rubric."""
        verdict = safe_invoke(judge_chain, {"assessment": json.dumps(state["assessment"], ensure_ascii=False), **_evidence(state)})
        v = verdict.model_dump() if verdict else {"approved": False, "issues": ["judge failed to respond"]}
        msg = agent_message("judge", f"Verdict after {state.get('refinement_count', 0)} refinement(s): "
                                     f"{'APPROVED' if v['approved'] else 'REJECTED - ' + '; '.join(v['issues'])}")
        log_messages("judge", [msg])
        return {"judge_verdict": v, "messages": [msg]}

    def assessor_refine(state: OrchestratorState) -> dict:
        """Optimizer step: improve the assessment using the judge's issues."""
        n = state.get("refinement_count", 0) + 1
        refined = safe_invoke(refine_chain, {**_evidence(state), "previous": json.dumps(state["assessment"], ensure_ascii=False),
                                             "issues": "; ".join(state["judge_verdict"]["issues"])})
        a = refined.model_dump() if refined else state["assessment"]
        msg = agent_message("assessor", f"Refinement {n}/{MAX_REFINEMENTS}: " + json.dumps(a, ensure_ascii=False))
        log_messages("assessor", [msg])
        return {"assessment": a, "refinement_count": n, "messages": [msg]}

    def route_after_execute(state: OrchestratorState) -> str:
        """Skip grading if the sandbox failed or no code was supplied."""
        status = state.get("execution_result", {}).get("status")
        return "assessor_finalize" if status in ("no_code", "Unable to verify") else "assessor_grade"

    def route_after_grade(state: OrchestratorState) -> str:
        """Skip the judge if grading itself failed (e.g. invalid model output)."""
        return "assessor_finalize" if state.get("execution_result", {}).get("status") == "Unable to verify" else "assessor_judge"

    def route_after_judge(state: OrchestratorState) -> str:
        """Conditional edge: halt when approved OR after MAX_REFINEMENTS refinements (hard cap)."""
        if state.get("judge_verdict", {}).get("approved") or state.get("refinement_count", 0) >= MAX_REFINEMENTS:
            return "assessor_finalize"
        return "assessor_refine"

    def assessor_finalize(state: OrchestratorState) -> dict:
        """Compose the learner-facing result (or an 'Unable to verify' message)."""
        ex = state.get("execution_result", {})
        if ex.get("status") == "no_code":
            text = "I could not find any code to grade. Please paste your solution inside a ```python code block."
        elif is_unable(ex):
            text = f"Unable to verify: the grading tools failed ({ex['source']}: {ex['reason']}). Your submission was NOT graded; please retry or ask a human grader."
        else:
            a, v, n = state["assessment"], state["judge_verdict"], state.get("refinement_count", 0)
            lines = [f"Score: {a['score']}/100 - {a['pass_status'].upper()}", "Feedback:"] + [f"- {p}" for p in a["feedback_points"]]
            if ex.get("stdout"):
                lines.append(f"Program output: {ex['stdout'].strip()[:200]}")
            lines.append(f"Quality review: approved by judge after {n} refinement(s)." if v["approved"] else
                         f"Quality review: NOT approved after the maximum of {MAX_REFINEMENTS} refinements - flagged for a human grader.")
            lines.append("Structured result: " + json.dumps(a, ensure_ascii=False))
            text = "\n".join(lines)
        msg = agent_message("assessor", text)
        log_messages("assessor", [msg])
        return {"final_answer": text, "messages": [msg]}

    return {"assessor_execute": assessor_execute, "assessor_grade": assessor_grade, "assessor_judge": assessor_judge,
            "assessor_refine": assessor_refine, "assessor_finalize": assessor_finalize,
            "route_after_execute": route_after_execute, "route_after_grade": route_after_grade, "route_after_judge": route_after_judge}
