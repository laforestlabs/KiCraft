"""Tests for kicraft/security/store.py + scans.py parsers (no external tools needed)."""
from __future__ import annotations

from kicraft.security import scans
from kicraft.security.store import SecurityResultStore, fingerprint


def test_finding_upsert_preserves_acknowledged_status(tmp_path):
    s = SecurityResultStore(tmp_path / "sec.db")
    s.record_finding(tool="bandit", severity="HIGH", rule="B602", location="a.py:10",
                     message="subprocess with shell=True")
    findings = s.list_findings(tool="bandit")
    assert len(findings) == 1 and findings[0]["severity"] == "high"  # normalized lower
    # acknowledge it, then re-run the scan (same finding) -> status preserved
    s.set_status(findings[0]["id"], "acknowledged")
    s.record_finding(tool="bandit", severity="HIGH", rule="B602", location="a.py:10",
                     message="subprocess with shell=True")
    again = s.list_findings(tool="bandit")
    assert len(again) == 1 and again[0]["status"] == "acknowledged"  # not resurfaced


def test_severity_sort_and_counts(tmp_path):
    s = SecurityResultStore(tmp_path / "sec.db")
    s.record_finding(tool="t", severity="low", rule="r1", location="x", message="m")
    s.record_finding(tool="t", severity="critical", rule="r2", location="y", message="m")
    ordered = s.list_findings()
    assert ordered[0]["severity"] == "critical"  # critical sorts first
    assert s.severity_counts() == {"critical": 1, "low": 1}


def test_scan_lifecycle(tmp_path):
    s = SecurityResultStore(tmp_path / "sec.db")
    s.start_scan("bandit-1", "bandit")
    s.finish_scan("bandit-1", "ok", {"findings": 3})
    scans_ = s.list_scans()
    assert scans_[0]["status"] == "ok" and scans_[0]["summary"]["findings"] == 3


def test_fingerprint_is_stable():
    a = fingerprint("bandit", "B602", "a.py:10", "msg")
    b = fingerprint("bandit", "B602", "a.py:10", "msg")
    c = fingerprint("bandit", "B602", "a.py:11", "msg")
    assert a == b and a != c


# --- scanner output parsers (pure, no tool needed) ---------------------------
def test_parse_bandit():
    data = {"results": [{"filename": "k/x.py", "line_number": 5,
                         "issue_severity": "MEDIUM", "test_id": "B105",
                         "issue_text": "hardcoded password"}]}
    out = scans.parse_bandit(data)
    assert out == [{"severity": "medium", "rule": "B105",
                    "location": "k/x.py:5", "message": "hardcoded password"}]


def test_parse_pip_audit():
    data = {"dependencies": [{"name": "requests", "version": "2.0.0",
                             "vulns": [{"id": "CVE-2023-1", "description": "bad",
                                        "fix_versions": ["2.31.0"]}]}]}
    out = scans.parse_pip_audit(data)
    assert out[0]["rule"] == "CVE-2023-1" and out[0]["location"] == "requests==2.0.0"
    assert "2.31.0" in out[0]["message"]


def test_parse_gitleaks():
    data = [{"RuleID": "generic-api-key", "File": ".env", "StartLine": 3,
             "Description": "API key"}]
    out = scans.parse_gitleaks(data)
    assert out[0]["severity"] == "critical" and out[0]["location"] == ".env:3"


def test_run_all_marks_missing_tools_not_installed(tmp_path, monkeypatch):
    # force every tool "absent" so run_all degrades gracefully (no hard failure)
    monkeypatch.setattr(scans, "tool_available", lambda name: False)
    s = SecurityResultStore(tmp_path / "sec.db")
    results = scans.run_all(s)
    assert {r["status"] for r in results} == {"not_installed"}
    assert all(r["findings"] == 0 for r in results)
