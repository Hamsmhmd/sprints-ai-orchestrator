"""Shared helpers for agent chains.

Curriculum mapping: Sprint 1 - prompt templates + Pydantic output parsers (LCEL chains).
"""
from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_chain(llm, task: str, system: str, human: str, schema=None):
    """Build ``prompt | llm | parser``.

    Args:
        task: marker consumed by the offline mock LLM (ignored by real models).
        system: system prompt (must not contain curly braces).
        human: human prompt template with ``{variables}``.
        schema: Pydantic model -> output is validated by PydanticOutputParser; ``None`` -> plain text.
    """
    if schema is None:
        prompt = ChatPromptTemplate.from_messages([("system", f"TASK={task}\n{system}"), ("human", human)])
        return prompt | llm | StrOutputParser()
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages(
        [("system", f"TASK={task}\n{system}\n{{format_instructions}}"), ("human", human)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def safe_invoke(chain, inputs: dict[str, Any], retries: int = 1):
    """Invoke a chain, retrying once on parse/LLM errors. Returns None if every attempt fails."""
    for _ in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception:  # noqa: BLE001 - OutputParserException, API errors...
            continue
    return None
