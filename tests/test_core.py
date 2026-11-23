"""Tests for the stdlib-only core (sandbox + database). Run: python -m pytest -q"""
from src.tools import code_repl
from src.tools import db_queries as db


def setup_module(_):
    db.build_database()


def test_sandbox_ok_and_stdout():
    r = code_repl.execute_python("print(sum(range(5)))")
    assert r["status"] == "ok" and r["stdout"].strip() == "10"


def test_sandbox_runtime_error_is_captured():
    r = code_repl.execute_python("print(1/0)")
    assert r["status"] == "runtime_error" and "ZeroDivisionError" in r["stderr"]


def test_sandbox_blocks_os_and_open():
    assert code_repl.execute_python("import os\nprint(os.listdir('.'))")["status"] == "blocked"
    assert code_repl.execute_python("print(open('/etc/passwd').read())")["status"] == "blocked"
    assert code_repl.execute_python("print((1).__class__.__bases__)")["status"] == "blocked"


def test_sandbox_timeout():
    r = code_repl.execute_python("while True:\n    pass", timeout_s=2)
    assert r["status"] == "timeout"


def test_sandbox_runtime_import_guard():
    # dynamic import bypass attempts must not work
    r = code_repl.execute_python("m = __import__\nprint(m('os'))")
    assert r["status"] == "blocked"


def test_sandbox_infrastructure_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise OSError("no fork")
    monkeypatch.setattr(code_repl.subprocess, "run", boom)
    r = code_repl.safe_execute_python("print(1)")
    assert r["status"] == "Unable to verify"


def test_profile_and_matching():
    p = db.get_learner_profile("spr-1001")
    assert p["name"] == "Ahmed Youssef" and p["target_salary_egp"] == 30000
    m = db.find_matching_openings(p)
    assert m and m[0]["employer"] == "Fawry"


def test_not_ready_learner_gets_no_match():
    p = db.get_learner_profile("SPR-1007")
    assert db.find_matching_openings(p) == []


def test_unknown_learner_unable_to_verify():
    assert db.safe_get_profile("SPR-9999")["status"] == "Unable to verify"


def test_sql_guard():
    assert db.run_readonly_sql("SELECT COUNT(*) AS n FROM learners")[0]["n"] == 25
    for bad in ["DROP TABLE learners", "SELECT 1; DELETE FROM learners", "UPDATE learners SET name='x'"]:
        try:
            db.run_readonly_sql(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_db_failure_falls_back(monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", db.DB_PATH.parent / "missing.db")
    assert db.safe_get_profile("SPR-1001")["status"] == "Unable to verify"
