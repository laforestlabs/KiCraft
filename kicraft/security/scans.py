"""Wrappers around external security scanners, normalized into SecurityResultStore.

Each scanner runs as a subprocess emitting JSON, which we parse into uniform
findings ({severity, rule, location, message}). A missing scanner yields a
``not_installed`` scan status rather than a hard failure, so the suite degrades
gracefully on a box without every tool.

    python -m kicraft.security.scans            # run bandit + pip-audit + gitleaks
    python -m kicraft.security.scans --tool bandit
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .store import SecurityResultStore

_REPO = Path(__file__).resolve().parents[2]
# Vendored third-party JS we do not own; excluded from the SAST sweep.
_BANDIT_EXCLUDE = "kicraft/server/static"

# bandit/pip-audit ship as importable modules -> run via `python -m` so they work
# regardless of whether their console script is on PATH (e.g. an unactivated venv).
# gitleaks is a standalone Go binary, found on PATH.
_PY_MODULES = {"bandit": "bandit", "pip-audit": "pip_audit"}


def tool_available(name: str) -> bool:
    mod = _PY_MODULES.get(name)
    if mod is not None:
        return importlib.util.find_spec(mod) is not None
    return shutil.which(name) is not None


def _run_json(cmd: list[str], cwd: Path | None = None) -> tuple[int, dict | list | None, str]:
    """Run a scanner; return (rc, parsed_json_or_None, stderr_tail)."""
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, None, str(e)
    try:
        return proc.returncode, json.loads(proc.stdout or "null"), proc.stderr[-500:]
    except json.JSONDecodeError:
        return proc.returncode, None, (proc.stderr or proc.stdout)[-500:]


# --- parsers: each returns a list of normalized finding dicts -----------------
def parse_bandit(data: dict) -> list[dict]:
    out = []
    for r in (data or {}).get("results", []):
        out.append({
            "severity": (r.get("issue_severity") or "unknown").lower(),
            "rule": r.get("test_id") or r.get("test_name") or "bandit",
            "location": f"{r.get('filename')}:{r.get('line_number')}",
            "message": (r.get("issue_text") or "")[:300],
        })
    return out


def parse_pip_audit(data) -> list[dict]:
    deps = data.get("dependencies", data) if isinstance(data, dict) else data
    out = []
    for dep in deps or []:
        name, ver = dep.get("name"), dep.get("version")
        for v in dep.get("vulns", []) or []:
            fix = ", ".join(v.get("fix_versions", []) or []) or "no fix listed"
            out.append({
                "severity": "high",  # pip-audit does not grade; treat CVEs as high
                "rule": v.get("id") or "CVE",
                "location": f"{name}=={ver}",
                "message": f"{(v.get('description') or '')[:200]} (fix: {fix})",
            })
    return out


def parse_gitleaks(data) -> list[dict]:
    out = []
    for f in data or []:
        out.append({
            "severity": "critical",  # a committed secret is always critical
            "rule": f.get("RuleID") or "secret",
            "location": f"{f.get('File')}:{f.get('StartLine')}",
            "message": (f.get("Description") or "leaked secret")[:200],
        })
    return out


# --- scanners -----------------------------------------------------------------
def run_bandit(store: SecurityResultStore, scan_id: str) -> dict:
    if not tool_available("bandit"):
        store.finish_scan(scan_id, "not_installed")
        return {"tool": "bandit", "status": "not_installed", "findings": 0}
    rc, data, err = _run_json(
        [sys.executable, "-m", "bandit", "-r", "kicraft", "-x", _BANDIT_EXCLUDE,
         "-f", "json", "-q"], _REPO)
    findings = parse_bandit(data) if isinstance(data, dict) else []
    for f in findings:
        store.record_finding(tool="bandit", **f)
    status = "ok" if data is not None else "error"
    store.finish_scan(scan_id, status, {"findings": len(findings), "stderr": err})
    return {"tool": "bandit", "status": status, "findings": len(findings)}


def run_pip_audit(store: SecurityResultStore, scan_id: str) -> dict:
    if not tool_available("pip-audit"):
        store.finish_scan(scan_id, "not_installed")
        return {"tool": "pip-audit", "status": "not_installed", "findings": 0}
    rc, data, err = _run_json(
        [sys.executable, "-m", "pip_audit", "-f", "json", "--progress-spinner", "off"], _REPO)
    findings = parse_pip_audit(data) if data is not None else []
    for f in findings:
        store.record_finding(tool="pip-audit", **f)
    status = "ok" if data is not None else "error"
    store.finish_scan(scan_id, status, {"findings": len(findings), "stderr": err})
    return {"tool": "pip-audit", "status": status, "findings": len(findings)}


def run_gitleaks(store: SecurityResultStore, scan_id: str) -> dict:
    if not tool_available("gitleaks"):
        store.finish_scan(scan_id, "not_installed")
        return {"tool": "gitleaks", "status": "not_installed", "findings": 0}
    # gitleaks writes the report to a file; --no-git scans the working tree.
    report = _REPO / ".gitleaks-report.json"
    cmd = ["gitleaks", "detect", "--no-git", "--no-banner",
           "--report-format", "json", "--report-path", str(report)]
    try:
        subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, timeout=600)
        data = json.loads(report.read_text()) if report.exists() else []
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        store.finish_scan(scan_id, "error", {"error": str(e)})
        return {"tool": "gitleaks", "status": "error", "findings": 0}
    finally:
        report.unlink(missing_ok=True)
    findings = parse_gitleaks(data)
    for f in findings:
        store.record_finding(tool="gitleaks", **f)
    store.finish_scan(scan_id, "ok", {"findings": len(findings)})
    return {"tool": "gitleaks", "status": "ok", "findings": len(findings)}


_RUNNERS = {"bandit": run_bandit, "pip-audit": run_pip_audit, "gitleaks": run_gitleaks}


def run_all(store: SecurityResultStore, tools=None) -> list[dict]:
    tools = tools or list(_RUNNERS)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = []
    for tool in tools:
        scan_id = f"{tool}-{stamp}"
        store.start_scan(scan_id, tool)
        out.append(_RUNNERS[tool](store, scan_id))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run KiCraft security scans into the store.")
    ap.add_argument("--store", help="SecurityResultStore path")
    ap.add_argument("--tool", action="append", choices=list(_RUNNERS),
                    help="limit to specific tool(s); default all")
    args = ap.parse_args(argv)
    store = SecurityResultStore(Path(args.store) if args.store else None)
    results = run_all(store, args.tool)
    for r in results:
        print(f"  {r['tool']:<10} {r['status']:<14} findings={r['findings']}")
    # CI gating decision is left to the caller (advisory first); exit 0 here.
    return 0


if __name__ == "__main__":
    sys.exit(main())
