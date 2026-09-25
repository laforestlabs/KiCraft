"""Audit a design draft before the compiler's contracts see it.

The deterministic contracts are the last line: they refuse a draft outright, and a refusal costs
the stage a correction round (or, before the design-contract rounds, the whole stage). An audit
asks a model the same questions the contracts answer — but as typed decisions over the draft, so
the answer arrives as findings a correction round can act on *together*, instead of one refusal
at a time.

Three kinds of question, one call, and only confident answers are reported:

- board-level yes/no facts the contracts enforce (a USB socket's rail, a rail that names no
  declared requirement, contacts left unnumbered);
- per-requirement identity: does this requirement's family/part name resolve to a real family or
  reviewed part, and if not, which one was meant (a closed choice over the catalogue);
- per-requirement part choice against the rail it runs on: which of the reviewed candidates can
  take that voltage and current.

Every answer is advisory in the sense that the contracts still decide; the audit only moves the
conversation one round earlier. It is fail-soft by construction: no client, no payload or a low
confidence yields no findings, never a blocked stage.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from kicraft.design import models
from kicraft.design.part_identity import (
    canonical_physical_features,
    reviewed_part,
    reviewed_parts_for_feature,
    reviewed_supply_voltage_limits,
)
from kicraft.design.recipes import get_recipe, recipe_summaries

from .decision_layer import DEFAULT_MODEL, DecisionUnavailable, Question, decide

#: How sure the auditor must be before its finding is reported. A finding costs a correction
#: round, so a guess is worse than silence.
DEFAULT_CONFIDENCE = 0.7

_NOT_LISTED = "none of these"


def _curated_families() -> frozenset[str]:
    """Every family the compiler can build with: the recipes plus the generic lowerers.

    Asking `get_recipe` is not the test — an unknown name raises, and a lowerer family (a screw
    terminal, a pin header) is not a recipe at all while being perfectly buildable. The audit
    needs one set of "families the pipeline knows", so it composes it from both registries.
    """
    from kicraft.design.lowering import lowerer_summaries

    families = {str(row["family"]) for row in recipe_summaries() if row.get("family")}
    for row in lowerer_summaries():
        families.update(str(name) for name in row.get("families") or [] if name)
    return frozenset(families)


def _catalogue() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The closed lists the audit may choose from: curated families, then reviewed identities."""
    families = tuple(sorted(_curated_families()))
    identities = tuple(
        sorted(
            {
                str(part.identity)
                for feature in (
                    "microcontroller",
                    "motor-driver",
                    "voltage-regulator",
                    "status-led",
                    "led",
                    "screw-terminal",
                    "jst-ph",
                    "pin-header",
                    "usb-c-receptacle",
                    "wire-to-board-connector",
                )
                for part in reviewed_parts_for_feature(feature)
            }
        )
    )
    return families, identities


def _rail_voltage(candidate: Mapping[str, Any], rail_name: Any) -> float | None:
    rail = ((candidate.get("power") or {}).get("rails") or {}).get(rail_name)
    if not isinstance(rail, Mapping):
        return None
    voltage = rail.get("voltage")
    try:
        return float(voltage)
    except (TypeError, ValueError):
        return None


def architecture_questions(candidate: Mapping[str, Any]) -> list[Question]:
    """The questions this draft is audited with. Empty when the draft offers nothing to ask."""
    questions: list[Question] = []
    requirements = [
        row
        for row in candidate.get("requirements") or []
        if isinstance(row, Mapping) and row.get("id")
    ]
    rails = (candidate.get("power") or {}).get("rails") or {}
    declared = {str(row.get("id")) for row in requirements}

    signals = [row for row in candidate.get("signals") or [] if isinstance(row, Mapping)]
    carries_usb_data = any(
        isinstance(reference, str) and reference.rsplit(".", 1)[-1].casefold() in {"usb_dm", "usb_dp"}
        for signal in signals
        for reference in (
            [signal.get("from"), *(signal.get("to") or [])]
            if isinstance(signal.get("to"), list)
            else [signal.get("from"), signal.get("to")]
        )
    )
    if carries_usb_data:
        questions.append(
            Question(
                key="usb_socket_rail",
                prompt=(
                    "This draft wires a chip's USB data pair to an off-board connector. Does the "
                    "draft also declare the ~5 V rail that connector's VBUS pin carries?"
                ),
                kind="noul",
            )
        )

    dangling = sorted(
        str(rail.get("from"))
        for rail in rails.values()
        if isinstance(rail, Mapping)
        and isinstance(rail.get("from"), str)
        and str(rail["from"]).partition(".")[0] not in declared
    )
    if dangling:
        questions.append(
            Question(
                key="dangling_rail_sources",
                prompt=(
                    "These rails name a part that the draft never declares: "
                    + ", ".join(dangling)
                    + ". Is that a mistake in the draft (rather than a part declared elsewhere)?"
                ),
                kind="noul",
            )
        )

    families, identities = _catalogue()
    curated = _curated_families()
    for requirement in requirements:
        rid = str(requirement["id"])
        family = str(requirement.get("family") or "")
        exact = str(requirement.get("exact_part") or "")
        if family and family not in curated:
            questions.append(
                Question(
                    key=f"identity_{rid}",
                    prompt=(
                        f"Requirement {rid!r} uses family {family!r}, which is not a curated "
                        "family or lowerer, so the compiler cannot build it. Which curated "
                        "family or reviewed part should implement this requirement instead? "
                        f"Answer {_NOT_LISTED!r} only when no listed entry implements it, and "
                        "then also answer the companion yes/no question."
                    ),
                    kind="choice",
                    options=(*identities, *families, _NOT_LISTED),
                )
            )
            questions.append(
                Question(
                    key=f"carrier_exists_{rid}",
                    prompt=(
                        f"Does any curated family or reviewed part implement requirement {rid!r} "
                        f"(demanded class {family!r}), so the compiler can build it without new "
                        "part research?"
                    ),
                    kind="noul",
                    descriptions={
                        "no": "the catalogue has no carrier: a real part must be researched and added",
                    },
                )
            )
        voltage = _rail_voltage(candidate, requirement.get("supply"))
        if exact and voltage is not None:
            record = reviewed_part(exact)
            limits = [
                row for row in reviewed_supply_voltage_limits(record) if row[2] is not None
            ]
            if limits and voltage > max(row[2] for row in limits):
                questions.append(
                    Question(
                        key=f"rated_{rid}",
                        prompt=(
                            f"Requirement {rid!r} runs on {voltage:g} V and its part {exact!r} is "
                            f"rated at most {max(row[2] for row in limits):g} V. Is that part "
                            "within its own rating on this rail?"
                        ),
                        kind="noul",
                    )
                )
                questions.append(
                    Question(
                        key=f"part_choice_{rid}",
                        prompt=(
                            f"Requirement {rid!r} runs on {voltage:g} V but its part {exact!r} is "
                            f"rated at most {max(row[2] for row in limits):g} V. Which listed part "
                            "should implement it instead, or is it none of these?"
                        ),
                        kind="choice",
                        options=(
                            *sorted(
                                {
                                    str(part.identity)
                                    for feature in canonical_physical_features(
                                        str(requirement.get("family") or "")
                                    )
                                    for part in reviewed_parts_for_feature(feature)
                                }
                            ),
                            exact,
                            _NOT_LISTED,
                        ),
                    )
                )
    return questions


def _finding(code: str, message: str, evidence: Sequence[str]) -> models.StageDiagnostic:
    return models.StageDiagnostic(
        code=code,
        severity="repair_required",
        message=message,
        evidence=list(evidence),
    )


def audit_architecture(
    candidate: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    confidence: float = DEFAULT_CONFIDENCE,
    recorder=None,
) -> list[models.StageDiagnostic]:
    """Typed findings about a draft, or nothing when the audit cannot be taken.

    The draft is the STATE of one Jev decision call: every question is answered in the same pass,
    so the audit costs one call per draft (input-only pricing, a fraction of a cent for a draft
    this size). A finding is reported only above the confidence threshold, and any transport or
    protocol failure yields no findings — a broken auditor must never block a stage.
    """
    questions = architecture_questions(candidate)
    if not questions:
        return []
    try:
        answers = decide(candidate, questions, model=model, recorder=recorder)
    except DecisionUnavailable:
        return []

    findings: list[models.StageDiagnostic] = []
    for answer in answers:
        if not answer.is_confident(confidence):
            continue
        if answer.key == "usb_socket_rail" and answer.value is False:
            findings.append(
                _finding(
                    "audit_usb_socket_rail_missing",
                    "A USB data connector has no declared 5 V rail, so the compiler will drop it "
                    "and report the chip's USB pins as unwired.",
                    ["declare the rail the socket exposes: VBUS or +5V at ~5 V"],
                )
            )
        elif answer.key == "dangling_rail_sources" and answer.value is True:
            findings.append(
                _finding(
                    "audit_rail_source_undeclared",
                    "A declared rail is sourced from a part this draft never declares.",
                    [f"confidence={answer.confidence:.2f}"],
                )
            )
        elif answer.key.startswith("carrier_exists_") and answer.value is False:
            requirement_id = answer.key[len("carrier_exists_") :]
            findings.append(
                _finding(
                    "audit_no_carrier_in_catalogue",
                    f"No curated family or reviewed part implements requirement {requirement_id!r}, "
                    "so the parts stage cannot resolve it without new part research.",
                    [f"confidence={answer.confidence:.2f}"],
                )
            )
        elif answer.key.startswith("rated_") and answer.value is False:
            requirement_id = answer.key[len("rated_") :]
            findings.append(
                _finding(
                    "audit_part_over_rating",
                    f"Requirement {requirement_id!r} runs above the rating of the part chosen for "
                    "it.",
                    [f"confidence={answer.confidence:.2f}"],
                )
            )
        elif answer.key.startswith("identity_") and answer.value not in ("", _NOT_LISTED):
            requirement_id = answer.key[len("identity_") :]
            findings.append(
                _finding(
                    "audit_identity_unresolved",
                    f"Requirement {requirement_id!r} does not name a curated family or reviewed "
                    "part; the audit reads it as this one.",
                    [f"audit choice: {answer.value}", f"confidence={answer.confidence:.2f}"],
                )
            )
        elif answer.key.startswith("part_choice_") and answer.value not in ("", _NOT_LISTED):
            requirement_id = answer.key[len("part_choice_") :]
            findings.append(
                _finding(
                    "audit_part_over_rating",
                    f"Requirement {requirement_id!r} runs above its part's rating; the audit "
                    "proposes a different reviewed part.",
                    [f"audit choice: {answer.value}", f"confidence={answer.confidence:.2f}"],
                )
            )
    return findings


def by_code(findings: Iterable[models.StageDiagnostic]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.code] = counts.get(finding.code, 0) + 1
    return counts
