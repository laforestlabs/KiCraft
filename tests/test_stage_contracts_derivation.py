"""The intent-shaped architecture slot keeps the committed block that asked for the pad field.

A prototyping area owns no part, no net and no pin: the only thing that can own the functional
block naming it is the requirement the derivation writes. That block's name lives in the
committed functional-spec slot and nowhere else, so the architecture normalization has to hand
the slot through -- without it the derived requirement owns no block,
`check_every_block_has_sheet` reports the block unowned, and the stage refuses a board the
brief asked for.
"""
from __future__ import annotations

import pytest

from kicraft.design.models import Architecture, FunctionalSpec
from kicraft.design.synthesis.validation import check_every_block_has_sheet
from kicraft.server.stage_contracts import (
    StageSchemaError,
    _derive_intent_payload,
    _normalize_stage_response,
)

_PROTOTYPING_AREA_OBLIGATION = {
    "kind": "fabrication",
    "original_obligation_id": "prototyping_area",
    "feature": "prototyping-area",
}


def _intent_shaped_slot() -> dict:
    return {
        "mcu_present": False,
        "sheets": [],
        "requirements": [],
        "signals": [],
        "obligations": [dict(_PROTOTYPING_AREA_OBLIGATION)],
    }


def _spec(name: str) -> dict:
    return {
        "blocks": [
            {
                "name": name,
                "category": "interface",
                "purpose": "Bare pad field the user solders through-hole parts into.",
                "count": 1,
            }
        ],
        "connections": [],
    }


def _prototyping_requirement(payload: dict):
    architecture = Architecture.model_validate(payload)
    return next(row for row in architecture.requirements if row.id == "prototyping_area")


def test_the_derived_requirement_claims_the_committed_block_that_names_it():
    payload = _derive_intent_payload(_intent_shaped_slot(), _spec("PAD FIELD"))
    assert _prototyping_requirement(payload).functional_blocks == ["PAD FIELD"]


def test_no_block_name_is_invented_when_none_matches():
    for spec in (None, _spec("POWER INPUT"), {"blocks": []}):
        payload = _derive_intent_payload(_intent_shaped_slot(), spec)
        assert _prototyping_requirement(payload).functional_blocks == []


def test_architecture_normalization_hands_the_committed_spec_to_the_derivation():
    spec = _spec("PAD FIELD")
    payload, _expanded = _normalize_stage_response(
        "architecture",
        _intent_shaped_slot(),
        {"intent": _intent_shaped_slot(), "functional_spec": spec},
    )
    architecture = _prototyping_requirement(payload)
    assert architecture.functional_blocks == ["PAD FIELD"]
    # The block the spec declared is owned, which is the gate the stage commits on.
    committed = Architecture.model_validate(payload)
    assert check_every_block_has_sheet(FunctionalSpec.model_validate(spec), committed).ok

    # The same design with no committed spec still derives the board feature; the
    # requirement simply owns no block.
    without_spec, _expanded = _normalize_stage_response(
        "architecture", _intent_shaped_slot(), {"intent": _intent_shaped_slot()}
    )
    assert _prototyping_requirement(without_spec).functional_blocks == []


def test_ordinary_question_options_are_normalized_and_require_two_choices():
    payload, _expanded = _normalize_stage_response(
        "intent",
        {
            "questions": [
                {
                    "text": "Which enclosure style should the board use?",
                    "stage": "intent",
                    "options": [
                        "  Use the compact enclosure  ",
                        "Use the larger enclosure",
                        "Use the compact enclosure",
                        "Use a panel-mount enclosure",
                        "Use a custom enclosure",
                        "Ignored after the fourth choice",
                    ],
                }
            ]
        },
        {},
    )

    assert payload["questions"][0]["options"] == [
        "Use the compact enclosure",
        "Use the larger enclosure",
        "Use a panel-mount enclosure",
        "Use a custom enclosure",
    ]

    with pytest.raises(
        StageSchemaError,
        match="Clarification questions require at least two distinct options.",
    ):
        _normalize_stage_response(
            "intent",
            {
                "questions": [
                    {
                        "text": "Which enclosure style should the board use?",
                        "stage": "intent",
                        "options": ["Only one choice"],
                    }
                ]
            },
            {},
        )


def test_bom_reconciliation_question_does_not_require_user_options():
    payload, _expanded = _normalize_stage_response(
        "wiring",
        {
            "questions": [
                {
                    "text": "Add the required decoupling capacitor.",
                    "stage": "wiring",
                    "reconcile_target": "bom",
                    "options": [],
                }
            ]
        },
        {},
    )

    assert payload["questions"][0]["options"] == []


def test_a_multi_finding_refusal_stays_a_loadable_stage_diagnostic():
    """An aggregate refusal must not make the run's own state unreadable.

    Live runs KC-HPD3YF (seed 25) and KC-P4E2PH (seed 27) each died on a draft with several
    contract defects at once. The wrapper row that carried them named no `severity` and left the
    findings in `evidence` as bare dicts, so `ConversationState.model_validate` refused the state
    those runs had just written -- and `stage_driver replay`, the tool that exists to iterate on
    exactly those runs, could not open them.
    """
    from kicraft.design.architecture_intent import ArchitectureIntentError, IntentDiagnostic
    from kicraft.design.models import ConversationState, StageDiagnostic
    from kicraft.server.stage_contracts import _aggregate_intent_diagnostic

    error = ArchitectureIntentError(
        [
            IntentDiagnostic(
                code="unknown_part_refused",
                message="requirement 'jack' (dc005) has no curated recipe and declares no interface",
                requirement_id="jack",
                sheet="POWER INPUT",
                evidence=["dc005"],
            ),
            IntentDiagnostic(
                code="unsupported_lowerer_contract",
                message="known lowerer led-current-resistor@1 does not implement the exact part",
                requirement_id="led",
                sheet="OUTPUT INTERFACE",
            ),
        ]
    )
    row = _aggregate_intent_diagnostic(error)
    diag = StageDiagnostic.model_validate(row)
    assert diag.code == "multiple_intent_contracts"
    assert diag.severity == "repair_required"
    assert [finding.code for finding in diag.findings] == [
        "unknown_part_refused",
        "unsupported_lowerer_contract",
    ]
    assert diag.findings[0].requirement_id == "jack"
    assert "unknown_part_refused" in " ".join(diag.evidence)

    # ...and the state it is written into round-trips, which is what replay depends on.
    state = ConversationState.model_validate(
        {"stage_status": {"architecture": {"diagnostics": [row], "ok": False}}}
    )
    reloaded = ConversationState.model_validate(state.model_dump())
    assert reloaded.stage_status["architecture"].diagnostics[0].findings[1].code == (
        "unsupported_lowerer_contract"
    )


def test_a_single_finding_refusal_is_its_own_typed_row():
    """One defect is stored as itself, not wrapped: the model sees the finding, not a summary."""
    from kicraft.design.architecture_intent import ArchitectureIntentError, IntentDiagnostic
    from kicraft.design.models import StageDiagnostic
    from kicraft.server.stage_contracts import _aggregate_intent_diagnostic

    row = _aggregate_intent_diagnostic(
        ArchitectureIntentError(
            [IntentDiagnostic(code="unknown_supply_rail", message="rail '+9V' is undeclared")]
        )
    )
    assert row["code"] == "unknown_supply_rail"
    assert StageDiagnostic.model_validate(row).severity == "repair_required"


def test_rows_written_before_the_fix_still_load():
    """A state written by the old writer stays readable: severity defaulted, evidence coerced.

    119 saved states carry an architecture row with no severity, a dict in `evidence`, and the
    refusal's own scope (`requirement_id`/`sheet`/`recipe`). Losing the structure of a nested row
    is a report detail; refusing to load the state loses the run.
    """
    from kicraft.design.models import ConversationState, StageDiagnostic

    legacy = {
        "code": "multiple_intent_contracts",
        "message": "two defects",
        "evidence": [
            {"code": "unknown_part_refused", "message": "jack", "evidence": ["dc005"]},
            {"code": "declared_signal_port_tied", "message": "pin1", "evidence": []},
        ],
    }
    diag = StageDiagnostic.model_validate(legacy)
    assert diag.severity is None
    assert diag.evidence == ["unknown_part_refused: jack", "declared_signal_port_tied: pin1"]

    scoped = StageDiagnostic.model_validate(
        {
            "code": "unavailable_recipe_gpio",
            "message": "mcu has no allocatable pin",
            "requirement_id": "mcu",
            "sheet": "ESP32 C3",
            "recipe": "esp32-c3-mini-1-minimal@1",
        }
    )
    assert (scoped.requirement_id, scoped.sheet, scoped.recipe) == (
        "mcu",
        "ESP32 C3",
        "esp32-c3-mini-1-minimal@1",
    )
    ConversationState.model_validate(
        {"stage_status": {"architecture": {"diagnostics": [legacy, scoped], "ok": False}}}
    )
