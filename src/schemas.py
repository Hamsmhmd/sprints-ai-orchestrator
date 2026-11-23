"""Pydantic schemas for every structured LLM output in the system.

Curriculum mapping: Sprint 1 - "Pydantic prompt templates and output parsers". Each schema is
paired with a PydanticOutputParser so the LLM output is validated before any agent uses it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AssessmentResult(BaseModel):
    """Assessor Agent feedback on a learner's Python submission (client-mandated fields)."""

    score: int = Field(ge=0, le=100, description="Overall grade from 0 to 100")
    feedback_points: list[str] = Field(min_length=1, description="Short, specific, actionable feedback items")
    pass_status: Literal["pass", "fail"] = Field(description="'pass' if score >= 60 else 'fail'")


class JudgeVerdict(BaseModel):
    """LLM-as-a-Judge decision on an AssessmentResult."""

    approved: bool = Field(description="True only if the assessment satisfies every rubric item")
    issues: list[str] = Field(default_factory=list, description="Rubric violations to fix (empty if approved)")


class IntentDecision(BaseModel):
    """Orchestrator routing decision."""

    intent: Literal["curriculum_question", "profile_question", "career_match", "code_grading"]
    reason: str = Field(description="One short sentence explaining the choice")


class SufficiencyDecision(BaseModel):
    """Self-reflection of the Advisor's agentic RAG loop."""

    sufficient: bool = Field(description="True if the retrieved context can fully answer the question")
    reason: str


class QueryRewrite(BaseModel):
    """Reformulated search query for a second retrieval attempt."""

    query: str = Field(description="A more specific keyword-rich search query")
