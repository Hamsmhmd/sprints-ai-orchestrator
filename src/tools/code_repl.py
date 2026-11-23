"""Sandboxed Python execution tool for the Assessor Agent.

Curriculum mapping: Sprint 1 - "Python REPL tool". The course's stock ``PythonREPL``
utility executes code *inside the host process* with full access to the file system and
network. Sprints' VP of Engineering made strict sandboxing a non-negotiable constraint, so
this tool keeps the same contract (code in -> stdout / runtime error out) but runs the
submission in a separate, locked-down interpreter:

1. Static AST screening (import allow-list, banned calls, banned dunder attributes).
2. Fresh ``python -I`` subprocess, empty environment, temp working directory.
3. Restricted ``__builtins__`` (no open/eval/exec/getattr...) and a guarded ``__import__``.
4. Hard wall-clock timeout, CPU / memory / file-size rlimits (POSIX), truncated output.

IMPORTANT: this is defence in depth, not a perfect jail. For production, run the same runner
inside a container (no network, read-only FS). That is listed in the report as next step.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.config import SANDBOX_MAX_OUTPUT_CHARS, SANDBOX_TIMEOUT_S
from src.fallback import unable_to_verify

ALLOWED_MODULES = frozenset({
    "math", "random", "statistics", "itertools", "functools", "collections", "re", "json",
    "string", "datetime", "typing", "heapq", "bisect", "decimal", "fractions", "operator",
    "dataclasses", "enum", "textwrap", "copy", "time", "abc",
})
BLOCKED_CALLS = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit", "help",
})
BLOCKED_DUNDERS = frozenset({
    "__class__", "__subclasses__", "__bases__", "__mro__", "__globals__", "__builtins__",
    "__code__", "__dict__", "__import__", "__getattribute__", "__reduce__", "__reduce_ex__",
    "__loader__", "__spec__", "__closure__", "__func__", "__self__",
})

_RUNNER = r'''
import sys, builtins, json
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (__CPU__, __CPU__))
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
except Exception:
    pass  # Windows has no `resource`; the wall-clock timeout still applies.

ALLOWED = set(json.loads('__ALLOWED__'))
BLOCKED = set(json.loads('__BLOCKED__'))
_real_import = builtins.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in ALLOWED:
        raise ImportError("import of '%s' is not allowed in the Sprints sandbox" % name)
    return _real_import(name, globals, locals, fromlist, level)

safe = {k: v for k, v in vars(builtins).items() if k not in BLOCKED}
safe["__import__"] = _guarded_import
with open("submission.py", encoding="utf-8") as fh:
    source = fh.read()
code = compile(source, "<submission>", "exec")
exec(code, {"__builtins__": safe, "__name__": "__main__"})
'''


def static_check(code: str) -> str | None:
    """Screen a submission before running it.

    Returns:
        A human readable violation reason, or ``None`` if the code passes the screen.
        Syntax errors are *not* violations: they are reported by the interpreter at run time
        so the Assessor can quote them back to the learner.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    return f"import of module '{alias.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                return f"import from module '{node.module}' is not allowed"
        elif isinstance(node, ast.Name) and node.id in BLOCKED_CALLS:
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_DUNDERS:
            return f"access to attribute '{node.attr}' is not allowed"
    return None


def _clip(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) > SANDBOX_MAX_OUTPUT_CHARS:
        return text[:SANDBOX_MAX_OUTPUT_CHARS] + "\n...[output truncated]"
    return text


def execute_python(code: str, timeout_s: int = SANDBOX_TIMEOUT_S) -> dict[str, Any]:
    """Run learner code in the sandbox and capture stdout / runtime errors.

    Curriculum mapping: Sprint 1 Python REPL tool, consumed by the Assessor Agent.

    Returns:
        dict with ``status`` in {"ok", "runtime_error", "timeout", "blocked"}, plus
        ``stdout``, ``stderr``, ``exit_code`` and ``duration_ms``. These statuses describe the
        *learner's code*; they are not tool failures.

    Raises:
        OSError / RuntimeError: if the sandbox itself cannot start (infrastructure failure).
        Callers should use :func:`safe_execute_python` to convert that into 'Unable to verify'.
    """
    violation = static_check(code)
    if violation:
        return {"status": "blocked", "stdout": "", "stderr": f"Security policy violation: {violation}",
                "exit_code": None, "duration_ms": 0}

    runner = (_RUNNER.replace("__CPU__", str(int(timeout_s) + 1))
              .replace("__ALLOWED__", json.dumps(sorted(ALLOWED_MODULES)))
              .replace("__BLOCKED__", json.dumps(sorted(BLOCKED_CALLS))))
    env = {"PYTHONIOENCODING": "utf-8"}
    if os.environ.get("SYSTEMROOT"):  # required by Python on Windows
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    with tempfile.TemporaryDirectory(prefix="sprints_sandbox_") as tmp:
        Path(tmp, "submission.py").write_text(code, encoding="utf-8")
        Path(tmp, "runner.py").write_text(runner, encoding="utf-8")
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-I", "runner.py"], cwd=tmp, env=env,
                                  capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            return {"status": "timeout", "stdout": _clip(exc.stdout),
                    "stderr": f"Execution exceeded {timeout_s}s time limit",
                    "exit_code": None, "duration_ms": int((time.perf_counter() - start) * 1000)}
        duration = int((time.perf_counter() - start) * 1000)

    stderr = _clip(proc.stderr)
    # Hide sandbox internals from tracebacks so feedback talks about the learner's code only.
    stderr = "\n".join(l for l in stderr.splitlines() if "runner.py" not in l and "exec(code" not in l)
    return {"status": "ok" if proc.returncode == 0 else "runtime_error",
            "stdout": _clip(proc.stdout), "stderr": stderr,
            "exit_code": proc.returncode, "duration_ms": duration}


def safe_execute_python(code: str) -> dict[str, Any]:
    """Like :func:`execute_python` but never raises: tool failures become 'Unable to verify'."""
    try:
        return execute_python(code)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all fallback
        return unable_to_verify("python_repl", f"sandbox failure: {type(exc).__name__}")


def get_repl_tool():
    """Return the LangChain ``@tool`` wrapper used by the Assessor Agent (lazy import)."""
    from langchain_core.tools import tool

    @tool
    def python_repl_sandbox(code: str) -> dict:
        """Execute a learner's Python snippet in the Sprints sandbox and return stdout/errors.

        Curriculum mapping: Sprint 1 Python REPL tool for the Assessor Agent.
        """
        return safe_execute_python(code)

    return python_repl_sandbox
