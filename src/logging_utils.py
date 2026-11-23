"""Agentic-protocol tracing: HumanMessage / AIMessage / ToolMessage with verbose logs.

Curriculum mapping: Sprint 2 - "Standardized agent communication". Every worker writes its
steps to the shared state as typed messages (an AIMessage that carries ``tool_calls`` followed
by the ToolMessage answering that call id) and logs them, so a whole run is traceable.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.config import VERBOSE

logger = logging.getLogger("sprints.trace")


def setup_logging(verbose: bool = VERBOSE) -> None:
    """Configure console logging for agent traces."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(message)s", force=True)


def log_messages(agent: str, messages: list[BaseMessage]) -> None:
    """Log each message as '[agent] TYPE(name): content'."""
    for m in messages:
        extra = ""
        if getattr(m, "tool_calls", None):
            extra = " tool_calls=" + json.dumps([{"name": c["name"], "args": c["args"]} for c in m.tool_calls],
                                                 ensure_ascii=False, default=str)[:300]
        logger.info("  [%s] %s(%s): %s%s", agent, type(m).__name__, getattr(m, "name", None) or "-",
                    str(m.content).replace("\n", " ")[:200], extra)


def agent_message(agent: str, text: str) -> AIMessage:
    """Create a named AIMessage representing a worker's statement."""
    return AIMessage(content=text, name=agent)


def tool_exchange(agent: str, tool_name: str, args: dict[str, Any], result: Any) -> list[BaseMessage]:
    """Create the AIMessage(tool_calls) + ToolMessage pair for one tool invocation and log it."""
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    ai = AIMessage(content="", name=agent,
                   tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}])
    payload = json.dumps(result, ensure_ascii=False, default=str)
    tool = ToolMessage(content=payload[:3000], tool_call_id=call_id, name=tool_name)
    msgs = [ai, tool]
    log_messages(agent, msgs)
    return msgs
