"""Shared bounded rendering rules for LLM evidence digests.

A prompt budget is not permission to cut evidence mid-record.  Consumers pass
complete source categories as sections; this module emits a category whole or
labels it omitted.  Models may therefore use an absence claim only when the
relevant category is explicitly marked COMPLETE.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSection:
    """One independently complete source category in an evidence digest."""

    name: str
    content: str
    complete: bool = True

def _short_incomplete(budget: int) -> str:
    """Reject budgets too small to state that all evidence is omitted."""
    marker = "ALL EVIDENCE OMITTED: absence is unverified."
    if len(marker) > budget:
        raise ValueError("budget cannot hold the evidence-omission notice")
    return marker


def render_bounded_evidence(
    title: str, sections: Iterable[EvidenceSection], *, budget: int
) -> str:
    """Render whole evidence sections within ``budget`` characters.

    A section is never character-sliced.  If it cannot fit, its named omission
    marker explains that it cannot support an absence conclusion.  The compact
    fallback for an impractically small budget contains no source evidence, so
    it cannot accidentally turn a partial record into a negative fact.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")

    section_list = list(sections)
    rule = (
        f"{title}\n"
        "EVIDENCE RULE: Only a section labeled COMPLETE can establish absence. "
        "OMITTED or INCOMPLETE sections are unverified; NEVER infer absence from them."
    )
    if len(rule) > budget:
        return _short_incomplete(budget)

    # Reserve a named omission for every category before including any source.
    # Otherwise a large early section can leave later categories silently absent.
    rendered_sections = [
        f"\n\n## {section.name} — OMITTED\nAbsence is unverified."
        for section in section_list
    ]
    used = len(rule) + sum(map(len, rendered_sections))
    if used > budget:
        return _short_incomplete(budget)
    for index, section in enumerate(section_list):
        content = section.content.strip() or "No source records."
        status = "COMPLETE" if section.complete else "INCOMPLETE"
        candidate = f"\n\n## {section.name} — {status}\n{content}"
        delta = len(candidate) - len(rendered_sections[index])
        if used + delta <= budget:
            rendered_sections[index] = candidate
            used += delta
    return rule + "".join(rendered_sections)
