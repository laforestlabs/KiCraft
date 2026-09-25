"""Reconcile naming differences with typed decisions, where code cannot decide alone.

The pipeline's recurring failures are naming failures, not design failures: a demanded part class
spelled with a qualifier the reviewed vocabulary does not use, a count whose subject names no
class, a claimed pin spelled as a function instead of a contact, a group that plainly implements a
demanded class but is not recognized as doing so. Each of those has a *closed* answer set somewhere
in this codebase — the reviewed class vocabulary, the classes a slot declares, the parts a unit
emitted, the pins a symbol publishes — and a closed answer is exactly what Jev answers well.

Three rules keep this honest:

- **Code shortlists, the decision model picks.** Every question offers a bounded list derived
  deterministically (token overlap against the reviewed vocabulary, the slot's own classes, the
  symbols' published pins). The model never invents a name that was not offered.
- **A wrong pick must be cheap.** Every answer carries a calibrated confidence; below the caller's
  threshold nothing is applied. Counts bind only after a confident answer, a class is rewritten
  only when confident, and the BOM/wiring reconcilers only ever *recommend* (the correction round
  still owns the change).
- **Every rewrite is visible.** An applied reconciliation records an assumption ending
  ``(defaulted)``, so the operator sees which names the machine read for them.

Decisions go through :mod:`kicraft.server.decision_layer` (Jev over OpenRouter), so budgets,
ceilings and the kill switch apply, and a batch of questions costs one call.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from kicraft.design.part_identity import (
    _DEMANDED_CLASS_ALIASES,
    _REVIEWED_FEATURE_VOCABULARY,
    class_key,
    has_reviewed_coverage,
    quantity_class_for,
    reviewed_class_variants,
)

from .decision_layer import DEFAULT_MODEL, Question, decide

#: (state, questions) -> answers keyed by question id. Injected so every caller is testable
#: without a network and so a deployment can point the reconciler at another decision model.
Decider = Callable[[Any, Sequence[Question]], Mapping[str, Any]]

#: The answer that means "none of the listed names": nothing is changed, and the stage's own
#: refusal (or its acceptance of a new class) stands.
KEEP = "none of these"

#: The answer that means "this count is a property of one part, not a count of a class".
PROPERTY = "a property of one part, not a count of a class"

DEFAULT_CONFIDENCE = 0.7


def jev_decider(
    model: str = DEFAULT_MODEL,
    *,
    recorder=None,
) -> Decider:
    """A decider backed by Jev: one call per batch of questions.

    Every failure yields no answers, which leaves the candidate exactly as the deterministic
    path left it. A stage must never depend on this decider being reachable: an unreachable
    provider must not turn a reconcilable naming difference into a crashed stage.
    """

    def decide_batch(state: Any, questions: Sequence[Question]) -> Mapping[str, Any]:
        try:
            answers = decide(state, questions, model=model, recorder=recorder)
        except Exception:
            return {}
        return {answer.key: answer for answer in answers}

    return decide_batch


# ---------------------------------------------------------------- shortlists


_STOPWORDS = frozenset({"the", "and", "with", "for", "from", "of", "on", "per", "to", "a", "an"})


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if len(token) > 2 and token not in _STOPWORDS
    )


def shortlist_reviewed_classes(demanded: str, limit: int = 12) -> tuple[str, ...]:
    """Reviewed classes that could be what ``demanded`` means, best first.

    Ranked by shared tokens (the whole name, then the head noun), with the demanded class's own
    alias targets first because the library already relates them. Candidates are reviewed classes
    only: a class the library cannot answer for is never offered as a rename.
    """
    demanded_tokens = _tokens(class_key(demanded).replace("-", " "))
    if not demanded_tokens:
        return ()
    demanded_head = class_key(demanded).split("-")[-1]
    scored: list[tuple[int, str]] = []
    for feature in sorted(_REVIEWED_FEATURE_VOCABULARY):
        if not has_reviewed_coverage(feature):
            continue
        feature_tokens = _tokens(feature.replace("-", " "))
        shared = demanded_tokens & feature_tokens
        if not shared:
            continue
        # A shared *subject* token ("relay" in "relay-driver") identifies the part; the trailing
        # head noun is usually the generic category ("driver", "module", "connector"), so it
        # ranks the class that shares a subject above the one that shares only the category.
        subject_shared = shared - {demanded_head}
        score = len(subject_shared) * 3 + (1 if demanded_head in shared else 0)
        scored.append((-score, feature))
    scored.sort()
    return tuple(feature for _score, feature in scored[:limit])


def shortlist_slot_classes(subject: str, classes: Sequence[str], limit: int = 12) -> tuple[str, ...]:
    """The classes this slot declares that ``subject`` could be counting, best first."""
    subject_tokens = _tokens(class_key(subject).replace("-", " "))
    ranked: list[tuple[int, str]] = []
    for value in classes:
        tokens = _tokens(class_key(value).replace("-", " "))
        shared = subject_tokens & tokens
        if shared:
            ranked.append((-len(shared), value))
    ranked.sort()
    return tuple(value for _score, value in ranked[:limit])


# ---------------------------------------------------------------- intent classes


def resolve_demanded_classes(
    candidate: dict,
    *,
    decider: Decider | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict, list[str]]:
    """Rewrite a demanded class the library spells differently, when Jev is sure which one it is.

    Only classes with no reviewed coverage and no deterministic spelling relation are asked
    about: those are the ones the intent detector passes silently today and the parts stage then
    cannot resolve. A confident answer rewrites the obligation (and records the reading); a
    "none of these" answer leaves the class alone, which is the honest outcome for a genuinely
    new part category.
    """
    rows = candidate.get("obligations") or []
    pending: list[tuple[int, str, tuple[str, ...]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("kind") != "physical":
            continue
        demanded = str(row.get("component_class") or "").strip()
        if not demanded or has_reviewed_coverage(demanded) or reviewed_class_variants(demanded):
            continue
        if class_key(demanded) in _DEMANDED_CLASS_ALIASES:
            continue
        options = shortlist_reviewed_classes(demanded)
        if options:
            pending.append((index, demanded, options))
    if decider is None or not pending:
        return candidate, []

    questions = [
        Question(
            key=f"class_{index}",
            prompt=(
                f"A board demands the part class {demanded!r}, which the reviewed library does "
                "not carry under that name. Which reviewed class does it denote, if any?"
            ),
            kind="choice",
            options=(*options, KEEP),
            descriptions={
                option: "a reviewed class the library can answer for" for option in options
            },
        )
        for index, demanded, options in pending
    ]
    answers = decider(candidate, questions)

    completed = None
    notes: list[str] = []
    for index, demanded, options in pending:
        answer = answers.get(f"class_{index}")
        if answer is None or not answer.is_confident(confidence):
            continue
        chosen = str(answer.value)
        if chosen == KEEP or chosen not in options:
            continue
        if completed is None:
            import copy

            completed = copy.deepcopy(candidate)
            completed["obligations"] = list(rows)
        completed["obligations"][index] = {
            **completed["obligations"][index],
            "component_class": chosen,
        }
        note = (
            f"part class {demanded!r} read as reviewed class {chosen!r} (defaulted)"
        )
        notes.append(note)
    if completed is None:
        return candidate, []
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    completed["assumptions"] = [*assumptions, *[note for note in notes if note not in assumptions]]
    return completed, notes


# ---------------------------------------------------------------- quantities


def resolve_quantity_subjects(
    candidate: dict,
    *,
    decider: Decider | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[dict, list[str]]:
    """Bind a count to the class it counts, or record that it counts a property.

    A count line whose subject names no class in the slot is refused today, which costs a
    correction round and leaves the count unenforced until the writer re-spells it. The classes
    are already listed in the slot, so which one the count belongs to is a closed question — and
    so is the honest third answer, "a property of one part" (pins on a header), which leaves the
    refusal standing rather than mis-binding eight pins to eight headers.
    """
    rows = candidate.get("obligations") or []
    classes = [
        str(row.get("component_class") or "")
        for row in rows
        if isinstance(row, dict) and row.get("kind") == "physical"
    ]
    pending: list[tuple[int, str, tuple[str, ...]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("kind") != "quantity":
            continue
        subject = str(row.get("subject") or "").strip()
        if not subject or quantity_class_for(subject, classes) is not None:
            continue
        options = shortlist_slot_classes(subject, classes)
        if options:
            pending.append((index, subject, options))
    if decider is None or not pending:
        return candidate, []

    questions = [
        Question(
            key=f"count_{index}",
            prompt=(
                f"A count row says {subject!r}. Which class in this design does the count belong "
                "to?"
            ),
            kind="choice",
            options=(*options, PROPERTY),
        )
        for index, subject, options in pending
    ]
    answers = decider(candidate, questions)

    completed = None
    notes: list[str] = []
    for index, subject, options in pending:
        answer = answers.get(f"count_{index}")
        if answer is None or not answer.is_confident(confidence):
            continue
        chosen = str(answer.value)
        if chosen not in options:
            continue
        if completed is None:
            import copy

            completed = copy.deepcopy(candidate)
            completed["obligations"] = list(rows)
        completed["obligations"][index] = {
            **completed["obligations"][index],
            "subject": chosen,
        }
        notes.append(f"count {subject!r} read as a count of {chosen!r} (defaulted)")
    if completed is None:
        return candidate, []
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    completed["assumptions"] = [*assumptions, *[note for note in notes if note not in assumptions]]
    return completed, notes


# ---------------------------------------------------------------- parts and pins


def reconcile_bom_classes(
    demanded_classes: Sequence[str],
    groups: Sequence[Any],
    *,
    decider: Decider | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """Name the emitted group that implements a class the unit left unfulfilled.

    The unit refused because no group *provably* carries the class; but the group that does is
    usually right there in the draft, written with a symbol or an MPN the class check cannot read.
    One yes/no per (class, group) pair, all in a single call, and the recommendation names the
    group — so the correction round binds it instead of re-emitting the same draft.
    """
    if decider is None or not demanded_classes or not groups:
        return []
    pairs = [
        (component_class, group)
        for component_class in demanded_classes
        for group in groups
    ]
    state = {
        "demanded_classes": list(demanded_classes),
        "groups": [
            {
                "id": str(getattr(group, "id", "")),
                "value": str(getattr(group, "value", "")),
                "symbol": str(getattr(group, "symbol", "")),
                "footprint": str(getattr(group, "footprint", "")),
                "mpn": str(getattr(group, "mpn", "") or ""),
                "quantity": int(getattr(group, "quantity", 1) or 1),
            }
            for group in groups
        ],
    }
    questions = [
        Question(
            key=f"implements_{index}",
            prompt=(
                f"Does group {str(getattr(group, 'id', ''))!r} "
                f"(value={str(getattr(group, 'value', ''))!r}, "
                f"symbol={str(getattr(group, 'symbol', ''))!r}, "
                f"mpn={str(getattr(group, 'mpn', '') or '')!r}) implement the demanded part "
                f"class {component_class!r}?"
            ),
            kind="noul",
        )
        for index, (component_class, group) in enumerate(pairs)
    ]
    answers = decider(state, questions)

    recommendations: list[str] = []
    for component_class in demanded_classes:
        confirming: list[tuple[Any, float]] = []
        for index, (pair_class, group) in enumerate(pairs):
            if pair_class != component_class:
                continue
            answer = answers.get(f"implements_{index}")
            if answer is not None and answer.value and answer.is_confident(confidence):
                confirming.append((group, answer.confidence))
        if len(confirming) == 1:
            group, sure = confirming[0]
            recommendations.append(
                f"{component_class}: the emitted group {str(getattr(group, 'id', ''))!r} "
                f"({str(getattr(group, 'symbol', ''))!r}"
                + (f", mpn={str(getattr(group, 'mpn', '') or '')!r}" if getattr(group, "mpn", None) else "")
                + f") implements this class (confidence {sure:.2f}); bind the group to the "
                "obligation instead of re-emitting it"
            )
    return recommendations


def reconcile_declared_port_pins(
    claims: Sequence[Mapping[str, Any]],
    *,
    decider: Decider | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """Name the contact a declared port's free-text pin actually means.

    A claim states a function ("18 V DC positive input") and a contact; when the contact string is
    neither a pin number nor a unique pin name, the symbol's own inventory is the closed answer
    set. One choice per claim, one call for the batch, and the recommendation names the contact so
    the correction is mechanical.
    """
    pending = [
        claim
        for claim in claims
        if claim.get("pins") and str(claim.get("function") or "").strip()
    ]
    if decider is None or not pending:
        return []
    questions = [
        Question(
            key=f"pin_{index}",
            prompt=(
                f"Claim {claim.get('key')!r} of {claim.get('symbol')!r} states the function "
                f"{claim.get('function')!r} on contact {claim.get('claimed')!r}, which is not one "
                "of that symbol's published contacts."
                + (
                    " Published contacts: "
                    + ", ".join(
                        f"{number}={name}" if name else number
                        for number, name in claim.get("contacts") or ()
                    )
                    + "."
                    if claim.get("contacts")
                    else ""
                )
                + " Which published contact (answer with its number) does it mean?"
            ),
            kind="choice",
            options=tuple(str(pin) for pin in claim.get("pins") or ()) + (KEEP,),
        )
        for index, claim in enumerate(pending)
    ]
    state = {"claims": [dict(claim) for claim in pending]}
    answers = decider(state, questions)

    recommendations: list[str] = []
    for index, claim in enumerate(pending):
        answer = answers.get(f"pin_{index}")
        if answer is None or not answer.is_confident(confidence):
            continue
        chosen = str(answer.value)
        if chosen == KEEP or chosen not in {str(pin) for pin in claim.get("pins") or ()}:
            continue
        recommendations.append(
            f"{claim.get('requirement_id')}:{claim.get('key')}: claimed contact "
            f"{claim.get('claimed')!r} reads as contact {chosen!r} of {claim.get('symbol')!r} "
            f"(confidence {answer.confidence:.2f}); state that contact"
        )
    return recommendations
