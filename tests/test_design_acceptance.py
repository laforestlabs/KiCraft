import json

import pytest

from kicraft.eval.design_acceptance import verify_campaign
from kicraft.eval.self_eval import _stable_hash
from kicraft.server.stage_contracts import DESIGN_STAGES
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS


def _campaign(tmp_path):
    entry = BENCHMARK_PROMPTS[0]
    slug = entry["slug"]
    rundir = tmp_path / "run"
    (rundir / ".kicraft").mkdir(parents=True)
    corpus = [{"index": 1, "slug": slug, "brief_hash": _stable_hash(entry["brief"])}]
    manifest = {
        "immutable": {
            "corpus": corpus,
            "corpus_hash": _stable_hash(corpus),
            "repeats": 1,
            "source_fingerprint": "sha256:fixture",
            "caps": {"project_usd": 0.15},
            "llm_mode": "live",
        }
    }
    summary = {
        "finished_at": "2026-09-11T12:00:00Z",
        "wall_s": 1,
        "design_only": True,
        "judge": False,
        "resumed": False,
        "source_unchanged": True,
        "n": 1,
        "design_committed": 1,
        "runs": [
            {
                "slug": slug,
                "index": 1,
                "prompt": entry["brief"],
                "rundir": str(rundir),
                "design_committed": True,
                "design_cost_usd": 0.05,
            }
        ],
        "fresh_output_directory": True,
    }
    (tmp_path / "campaign_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (rundir / ".kicraft/state.json").write_text(
        json.dumps(
            {
                "stage_status": {stage: {"ok": True} for stage in DESIGN_STAGES},
            }
        )
    )
    (rundir / "events.jsonl").write_text(
        "\n".join(
            json.dumps({"kind": "stage_done", "stage": stage, "ok": True})
            for stage in DESIGN_STAGES
        )
    )
    return slug, summary, rundir


def test_acceptance_rejects_missing_stage_evidence_despite_green_summary(tmp_path):
    slug, _, rundir = _campaign(tmp_path)
    assert verify_campaign(tmp_path, [slug]) == []
    events = rundir / "events.jsonl"
    events.write_text("\n".join(events.read_text().splitlines()[:-1]))
    assert any("events do not prove" in error for error in verify_campaign(tmp_path, [slug]))


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "substituted_brief", "resumed", "source_changed", "over_budget", "unfinished"],
)
def test_acceptance_rejects_false_green_campaign(tmp_path, defect):
    slug, summary, _ = _campaign(tmp_path)
    if defect == "duplicate":
        summary["runs"].append(summary["runs"][0].copy())
    elif defect == "substituted_brief":
        summary["runs"][0]["prompt"] = "A different design"
    elif defect == "resumed":
        summary["resumed"] = True
    elif defect == "source_changed":
        summary["source_unchanged"] = False
    elif defect == "over_budget":
        summary["runs"][0]["design_cost_usd"] = 0.151
    else:
        del summary["finished_at"]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert verify_campaign(tmp_path, [slug])


def test_acceptance_allows_recovered_stage_but_rejects_later_terminal_failure(tmp_path):
    slug, _, rundir = _campaign(tmp_path)
    events = rundir / "events.jsonl"
    rows = events.read_text().splitlines()
    failed = json.dumps({"kind": "stage_done", "stage": "functional_spec", "ok": False})
    rows.insert(1, failed)
    events.write_text("\n".join(rows))
    assert verify_campaign(tmp_path, [slug]) == []

    events.write_text("\n".join([*rows, failed]))
    assert any("events do not prove" in error for error in verify_campaign(tmp_path, [slug]))
