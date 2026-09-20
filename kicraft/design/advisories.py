"""RECORD-class advisories: the one string protocol the artifact and the scorecard share.

The BLOCK-vs-RECORD bar (design-yield-recovery plan §4) lets a stage record a property it
could not *prove* instead of refusing the run. Two carriers exist because two stages produce
them:

* the architecture stage's typed rows (`Architecture.advisories`), and
* the BOM commit's `advisory [<code>]: <message>` lines in ``bom.assumptions`` (§9.33/§9.34).

Both are read by the self-eval record, the board's promote provenance and the rubric's
advisory gate, so the marker, the line format and the reader live here rather than being
re-spelled in each reader.
"""
from __future__ import annotations

from typing import Any, Iterable

ADVISORY_MARKER = "advisory ["

#: Everything after this separator in an advisory line is the message, not the code.
_CODE_END = "]"


def advisory_line(code: str, message: str = "", offenders: Iterable[str] = ()) -> str:
    """One ``bom.assumptions`` advisory line naming the code, the message and the offenders."""
    line = f"{ADVISORY_MARKER}{code}{_CODE_END}: {message}".rstrip()
    extra = [str(offender) for offender in offenders if str(offender).strip()]
    return f"{line} -- {'; '.join(extra)}" if extra else line


def advisory_code(line: object) -> str | None:
    """The code inside an ``advisory [<code>]`` line, or None."""
    text = str(line or "")
    start = text.find(ADVISORY_MARKER)
    if start < 0:
        return None
    end = text.find(_CODE_END, start + len(ADVISORY_MARKER))
    if end < 0:
        return None
    code = text[start + len(ADVISORY_MARKER) : end].strip()
    return code or None


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def recorded_advisory_codes(architecture: Any = None, bom: Any = None) -> list[str]:
    """Distinct advisory codes a design shipped with, in first-seen order.

    Reads the typed architecture rows and the BOM's advisory lines. A design with neither
    carries none, and an unreadable row is skipped rather than guessed at.
    """
    codes: list[str] = []
    for row in _field(architecture, "advisories") or ():
        code = _field(row, "code")
        if code:
            codes.append(str(code))
    for line in _field(bom, "assumptions") or ():
        code = advisory_code(line)
        if code:
            codes.append(code)
    return list(dict.fromkeys(codes))
