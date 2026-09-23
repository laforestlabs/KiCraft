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
