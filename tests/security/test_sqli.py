"""SQL-injection resistance: the public search box and project lookups must use
parameterized queries, and the FTS sanitizer must neutralize FTS5 operators."""
from __future__ import annotations

from kicraft.server.accounts import AccountStore


def test_fts_match_query_neutralizes_operators():
    f = AccountStore._fts_match_query
    # FTS5 operators / quotes / SQL metacharacters are stripped, leaving safe
    # double-quoted prefix phrases (or None when nothing usable remains).
    assert f('"; DROP TABLE projects; --') is not None  # tokens survive, operators gone
    out = f('foo" OR "1"="1')
    for op in ('OR', ';', '--'):
        # the only quotes in the output are the phrase-wrapping ones we added
        assert f' {op} ' not in (out or "")
    assert out is None or out.count('"') % 2 == 0  # balanced phrase quotes
    # pure-punctuation input yields no filter rather than a syntax error
    assert f("';--") is None
    assert f("") is None and f(None) is None


def test_search_runs_safely_with_injection_payloads(tmp_path):
    """End-to-end: feeding injection strings to the public search must not raise and
    must not corrupt the DB (it returns results or nothing, never executes SQL)."""
    store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    u = store.create_user("a@x.io", "pw")
    pid = store.create_project(u.id, "an esp32 sensor board", is_public=True)
    store.finish_project(pid, "ok", stem="ESP32")
    store.set_visibility(pid, True) if hasattr(store, "set_visibility") else None
    for payload in ["'; DROP TABLE projects; --", '" OR 1=1 --', "%", "esp32", "*]["]:
        results = store.list_public_projects(query=payload)
        assert isinstance(results, list)
    # the table still exists and the row survived the injection attempts
    assert store.get_project(pid) is not None


def test_queries_are_parameterized_not_fstring_built():
    """Guardrail: account queries bind via '?' placeholders, never f-string the
    user value into SQL. Spot-check the hot user-facing lookups."""
    import inspect
    src = inspect.getsource(AccountStore.authenticate)
    assert "WHERE email=?" in src  # parameterized, not f"...{email}..."
    create_src = inspect.getsource(AccountStore.create_project)
    assert "VALUES (?" in create_src
