"""Silkscreen legibility check — does the board self-describe?

A fabricated board should carry a legend (name / rev / maker — the deterministic
KiCraft attribution line) and, ideally, at least one functional label (an IO
rating, a DIP-switch table, a usage note). This check reads the F/B.SilkS text
the build-tail placer stamped and scores presence, so a self-describing board
ranks above anonymous copper.

Silk is cosmetic: DRC silk violations are MINOR by design
(see ``drc_check.MINOR_TYPES``) and never gate a build, so this is a light
0.05-weight nudge — never a fab blocker.

Only board-level ``PCB_TEXT`` (the legend + labels, added via ``board.Add``) is
inspected; footprint reference/value silk lives in ``fp.GraphicalItems()`` and
is intentionally excluded — refdes text is not a legend.

Optional honesty input via config (best-effort; absent when scoring a bare
``.kicad_pcb``):
- ``_silk_dropped``: list of "id: reason" the placer could not fit (from
  ``state.artifacts.silk_dropped``) — docked lightly and surfaced as info.
"""
import pcbnew

from .base import LayoutCheck, CheckResult, Issue

# The deterministic attribution line always contains the maker token, so a silk
# text carrying it means the board identifies itself (name/rev/date sit on the
# same or adjacent legend line). Kept in sync with ``silk_legend.build_legend_lines``.
_MAKER_TOKEN = "kicraft"
_SILK_LAYERS = (pcbnew.F_SilkS, pcbnew.B_SilkS)


class SilkCheck(LayoutCheck):
    name = "silk"
    display_name = "Silkscreen Legend"
    weight = 0.05  # small bonus — self-describing boards rank above anonymous copper

    def run(self, board, config: dict) -> CheckResult:
        texts: list[str] = []
        for d in board.GetDrawings():
            if not isinstance(d, pcbnew.PCB_TEXT):
                continue
            if d.GetLayer() not in _SILK_LAYERS:
                continue
            s = (d.GetText() or "").strip()
            if s:
                texts.append(s)

        legend_present = any(_MAKER_TOKEN in s.lower() for s in texts)
        # Content lines = silk text that isn't the maker/attribution line. On the
        # board a title and a functional label are both just PCB_TEXT, so we can't
        # tell them apart geometrically — but the deterministic legend is always a
        # title line + the maker line, so >=2 content lines means the board carries
        # a real label beyond its title. Precise per-label classification (kind =
        # io/table/note) is the eval rubric's job; here we only need a cheap
        # board-level richness signal.
        content = [s for s in texts if _MAKER_TOKEN not in s.lower()]
        dropped = [str(x) for x in (config.get("_silk_dropped") or [])]

        issues: list[Issue] = []
        score = 0.0
        # 70 pts: the board self-identifies (name/rev/maker legend line).
        if legend_present:
            score += 70.0
        else:
            issues.append(Issue(
                "warning",
                "Board does not self-describe — no silkscreen legend (name/rev/maker)"))
        # 30 pts: a functional label beyond the title; 15 for a title only.
        if len(content) >= 2:
            score += 30.0
        elif len(content) == 1:
            score += 15.0
            issues.append(Issue(
                "info",
                "Only a title — no functional silkscreen label (IO rating / DIP table / note)"))
        else:
            issues.append(Issue(
                "info",
                "No functional silkscreen labels (IO rating / DIP table / note)"))
        if dropped:
            score = max(0.0, score - min(20.0, 5.0 * len(dropped)))
            issues.append(Issue(
                "info", f"{len(dropped)} silk label(s) dropped for lack of space"))

        summary = ("self-describes" if legend_present else "no legend")
        summary += f"; {len(content)} content line(s)"
        if dropped:
            summary += f"; {len(dropped)} dropped"

        return CheckResult(
            score=round(score, 1),
            issues=issues,
            metrics={
                "legend_present": legend_present,
                "silk_text_count": len(texts),
                "content_line_count": len(content),
                "dropped_count": len(dropped),
            },
            summary=summary,
        )
