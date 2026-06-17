"""KiCraft security / vulnerability testing.

Two halves:
  - scans.py  : wrappers around external scanners (bandit SAST, pip-audit dep CVEs,
                gitleaks secret scan, OWASP ZAP DAST) that normalize their JSON into
                a SecurityResultStore (store.py), surfaced on /admin/security.
  - tests/security/ (pytest) : KiCraft-specific abuse tests -- capability-token
                forgery, quota/spend bypass, build-slot DoS, login rate-limit,
                prompt injection, XSS, SQLi, Stripe webhook -- run in CI.

External scanners are invoked as subprocesses and degrade with a clear "not
installed" status when absent, so the suite never hard-fails on a box missing a
tool (mirrors the FTS5-absent guard in accounts.py).
"""
