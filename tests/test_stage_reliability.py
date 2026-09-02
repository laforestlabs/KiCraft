from pathlib import Path

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
