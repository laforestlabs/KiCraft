from pathlib import Path
import json

from kicraft.eval.stage_reliability import (
    exact_zero_failure_lower_bound,
    load_corpus,
    statistical_release_gate,
)


def test_committed_corpus_has_full_stage_denominators():
    corpus = load_corpus(Path("kicraft/eval/stage_reliability_corpus.json"))
    counts = {stage: 0 for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")}
    for case in corpus["cases"]:
        counts[case["stage"]] += 1
    assert set(counts.values()) == {306}


def test_exact_zero_failure_bound_and_release_gate():
    assert exact_zero_failure_lower_bound(299) > 0.99
    rows = [
        {
            "commit_ok": True,
            "semantic_clean": True,
            "cost_usd": 1.0,
            "wall_s": 1.0,
            "valid": True,
        }
        for _ in range(299)
    ]
    result = statistical_release_gate(rows, baseline_p95_cost=1.0, baseline_p95_wall_s=1.0)
    assert result["passed"]
    rows[0]["semantic_clean"] = False
    assert not statistical_release_gate(rows)["gates"]["semantic_reliability"]


def test_recovery_baseline_fixture_covers_each_failure_mechanism_without_reasoning():
    fixture = json.loads(
        Path("tests/fixtures/stage_reliability/recovery_baseline_20260910.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["privacy"] == {
        "contains_reasoning": False,
        "contains_credentials": False,
        "contains_binary_artifacts": False,
    }
    assert {case["mechanism"] for case in fixture["cases"]} == {
        "functional_spec_collection_overflow",
        "architecture_invalid_schema",
        "architecture_collection_overflow",
        "unresolved_footprint",
        "unresolved_symbol",
        "symbol_pad_mismatch",
        "absent_mpn_sentinel",
        "named_part_omission",
        "recipe_duplicate",
        "protected_identity_collision",
        "bom_partial_wiring_coverage",
        "wiring_pin_inventory",
        "aggregate_dangling_signal",
    }
    serialized = json.dumps(fixture).lower()
    assert "openrouter_api_key" not in serialized
    assert "reasoning_delta" not in serialized
    assert "answer_delta" not in serialized
