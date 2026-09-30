"""A provider draft cannot invent obligations, and an ownership refusal stays repairable.

Two measured live failures drove this: `encoder-oled-panel` and `stepper-a4988` ended their
architecture stage as `invalid_schema` with "Architecture obligations must be owned by a
requirement: listed_at_top_level_only=[('physical','mounting-holes')]" / `[('physical',
'motor-connector')]` -- a design-contract refusal delivered as unusable provider output, so the
driver answered with a generic re-emit instead of the targeted attach-this-obligation repair.
"""

from __future__ import annotations

from pydantic import ValidationError

from kicraft.server import stage_contracts as sc

COMMITTED = [
    {"kind": "physical", "original_obligation_id": "jack", "component_class": "audio-jack-3-5mm"}
]
INVENTED = {
    "kind": "physical",
    "original_obligation_id": "mounting-holes",
    "component_class": "mounting-holes",
}


def _payload() -> dict:
    return {
        "requirements": [{"id": "jack1", "obligation_ids": ["jack"]}],
        "obligations": [*COMMITTED, INVENTED],
    }


def _ownership_error() -> ValidationError:
    return ValidationError.from_exception_data(
        "Architecture",
        [
            {
                "type": "value_error",
                "loc": (),
                "input": {},
                "ctx": {
                    "error": ValueError(
                        "Architecture obligations must be owned by a requirement: "
                        "listed_at_top_level_only=[('physical', 'jack')] — attach each "
                        "obligation to the requirement that implements it"
                    )
                },
            }
        ],
    )


def test_a_provider_draft_cannot_add_an_obligation_row_of_its_own():
    """`replace` makes the committed set the list, so an invented row never reaches the validator."""
    state = {"intent": {"obligations": COMMITTED}}
    kept = sc.restore_source_obligations(_payload(), state, stage="architecture")
    assert [row["original_obligation_id"] for row in kept["obligations"]] == ["jack"]
    replaced = sc.restore_source_obligations(
        _payload(), state, stage="architecture", replace=True
    )
    assert [row["original_obligation_id"] for row in replaced["obligations"]] == ["jack"]


def test_an_empty_committed_set_clears_the_drafts_invented_rows():
    """The committed set *is* the list, including when it is empty."""
    cleared = sc.restore_source_obligations(
        _payload(), {"intent": {"obligations": []}}, stage="architecture", replace=True
    )
    assert cleared["obligations"] == []
    # Without `replace` (the canonical replay path) the payload is left alone.
    untouched = sc.restore_source_obligations(
        _payload(), {"intent": {"obligations": []}}, stage="architecture"
    )
    assert [row["original_obligation_id"] for row in untouched["obligations"]] == [
        "jack",
        "mounting-holes",
    ]


def test_an_ownership_refusal_is_a_repairable_contract_defect_not_a_schema_error():
    state = {"intent": {"obligations": COMMITTED}}
    refusal = sc._ownership_refusal(_ownership_error(), {"requirements": []}, state)
    assert refusal is not None
    assert refusal.diagnostic["code"] == "source_obligation_not_retained"
    assert refusal.diagnostic["severity"] == "repair_required"
    assert [row["original_obligation_id"] for row in refusal.diagnostic["evidence"]] == ["jack"]
    assert "jack" in refusal.diagnostic["candidate_requirement_ids"]


def test_a_refusal_that_is_not_about_ownership_is_left_alone():
    assert sc._ownership_refusal(ValueError("something else"), _payload(), {}) is None
    assert sc._ownership_refusal(_ownership_error(), _payload(), {"intent": {"obligations": []}}) is None
