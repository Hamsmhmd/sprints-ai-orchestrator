"""Graceful-failure helper.

Curriculum mapping: Sprint 1 tool design. The client requires a structured
'Unable to verify' answer whenever the SQLite DB or the Python REPL throws, so that the
agents never improvise (hallucinate) a result.
"""
from __future__ import annotations

from typing import Any


def unable_to_verify(source: str, reason: str) -> dict[str, Any]:
    """Build the structured 'Unable to verify' payload returned by failing tools.

    Args:
        source: Which tool failed ("sqlite_db", "python_repl", "chromadb"...).
        reason: Short human readable cause (no stack traces, no secrets).
    """
    return {"status": "Unable to verify", "source": source, "reason": reason}


def is_unable(result: Any) -> bool:
    """Return True if ``result`` is an 'Unable to verify' payload."""
    return isinstance(result, dict) and result.get("status") == "Unable to verify"
