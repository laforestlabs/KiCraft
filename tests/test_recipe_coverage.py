import pytest

from kicraft.eval.recipe_coverage import analyze_recipe_coverage


def test_coverage_attributes_parts_pins_calls_and_cost_exactly_once():
    state = {
        "bom": {
            "parts": [
                {"ref": "U1", "recipe_id": "esp32@1"},
                {"ref": "R1", "resolution_source": "lowerer"},
            ],
            "connections": [
                {
                    "net_name": "SIG",
                    "endpoints": [
                        {"ref": "U1", "pin": "1"},
                        {"ref": "R1", "pin": "1"},
                    ],
                }
            ],
            "no_connect_pins": [{"ref": "U1", "pin": "2"}],
            "recipe_ownership": [
                {
                    "recipe": "esp32@1",
                    "instance": "main",
                    "refs": ["U1"],
                    "pins": [
                        {"ref": "U1", "pin": "1", "owner": "recipe"},
                        {"ref": "U1", "pin": "2", "owner": "allocator"},
                    ],
                    "internal_nets": [],
                }
            ],
        }
    }
    attempt = {
        "kind": "work_unit_attempt",
        "stage": "wiring",
        "unit_id": "wiring:novel",
        "provider_attempt": 1,
        "cost_usd": 0.125,
    }
    report = analyze_recipe_coverage(
        state,
        [
            {
                "kind": "work_unit_plan",
                "stage": "wiring",
                "unit_id": "wiring:r1",
                "source": "deterministic_architecture_lowering",
                "refs": ["R1"],
            },
            {
                "kind": "work_unit_plan",
                "stage": "wiring",
                "unit_id": "wiring:novel",
                "source": "llm",
                "refs": [],
            },
            attempt,
            dict(attempt),
        ],
    )
    assert report["parts"] == {
        "total": 2,
        "recipe": 1,
        "allocator": 0,
        "lowerer": 1,
        "reuse": 0,
        "llm": 0,
    }
    assert report["pins"] == {
        "total": 3,
        "recipe": 1,
        "allocator": 1,
        "lowerer": 1,
        "reuse": 0,
        "llm": 0,
    }
    assert report["calls"]["total"] == report["calls"]["llm"] == 1
    assert report["cost_usd"]["total"] == pytest.approx(0.125)
    assert report["work_units"] == [
        {
            "stage": "wiring",
            "unit_id": "wiring:novel",
            "source": "llm",
            "calls": 1,
            "cost_usd": 0.125,
        },
        {
            "stage": "wiring",
            "unit_id": "wiring:r1",
            "source": "lowerer",
            "calls": 0,
            "cost_usd": 0.0,
        },
    ]


def test_coverage_rejects_duplicate_pin_ownership():
    state = {
        "bom": {
            "parts": [],
            "connections": [],
            "no_connect_pins": [],
            "recipe_ownership": [
                {
                    "pins": [
                        {"ref": "U1", "pin": "1", "owner": "recipe"},
                        {"ref": "U1", "pin": "1", "owner": "allocator"},
                    ]
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="coverage_duplicate_pin_owner: U1.1"):
        analyze_recipe_coverage(state)
