"""Pin kicraft.cli.triage against artifact-schema drift.

The investigate skill used to carry this logic as untested inline heredocs and
it rotted silently (the worst: testing ``routed_validation is not None`` on a
field that is always a dict, which mis-tiered every never-routed run). These
tests pin the reading logic AND the producer contracts it depends on, so the
next compactor/serializer change breaks a test here instead of the skill.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kicraft.cli import triage


# ---------------------------------------------------------------------------
# fixture builders — a minimal on-disk run shaped like the real artifacts
# ---------------------------------------------------------------------------

def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def make_run(tmp_path: Path, *, stem: str = "BOARD") -> tuple[Path, Path, Path]:
    """(run_dir, stem_dir, experiments_dir) skeleton."""
    run = tmp_path / "1" / "999"
    sd = run / "generated" / stem
    exp = sd / ".experiments"
    exp.mkdir(parents=True)
    return run, sd, exp


def make_parent_round(exp: Path, n: int, state: dict) -> Path:
    return _write(
        exp / "hierarchical_autoexperiment" / f"round_{n:04d}" / "parent_pipeline.json",
        {"state": state},
    )


# ---------------------------------------------------------------------------
# the routed_validation predicate (the bug that motivated this module)
# ---------------------------------------------------------------------------

def test_pick_parent_round_skips_empty_routed_validation(tmp_path):
    _run, _sd, exp = make_run(tmp_path)
    make_parent_round(exp, 1, {"routed_validation": {"accepted": False,
                                                     "rejection_reasons": ["x"]}})
    # a later round that never routed: routed_validation is {} — NEVER None
    make_parent_round(exp, 2, {"routed_validation": {}})
    pp, st, how = triage.pick_parent_round(triage.parent_rounds(exp))
    assert how == "routed"
    assert pp.parent.name == "round_0001"
    assert st["routed_validation"]["accepted"] is False


def test_pick_parent_round_all_empty_falls_back_to_last(tmp_path):
    _run, _sd, exp = make_run(tmp_path)
    make_parent_round(exp, 1, {"routed_validation": {}})
    make_parent_round(exp, 2, {"routed_validation": {}})
    pp, _st, how = triage.pick_parent_round(triage.parent_rounds(exp))
    assert how == "last_attempted"
    assert pp.parent.name == "round_0002"


def test_scan_tiers_never_routed_as_route_fail(tmp_path):
    """A run whose rounds all have routed_validation == {} (and no routed
    board) must tier route_fail — the old `is not None` test filed it under
    'unknown'."""
    run, _sd, exp = make_run(tmp_path)
    make_parent_round(exp, 1, {"routed_validation": {}})
    data = triage.collect_scan([tmp_path])
    assert data["run_count"] == 1
    assert data["tiers"] == {"route_fail (no routed parent, rc6 family)": 1}
    assert run  # silence unused

def test_scan_home_fetched_buckets(tmp_path, monkeypatch):
    """Provenance flags aggregate into the cross-run library-coverage
    buckets, keyed by slug with the run tags that emitted them."""
    run, _sd, exp = make_run(tmp_path)
    make_parent_round(exp, 1, {"routed_validation": {}})
    monkeypatch.setattr(triage, "collect_library_provenance", lambda _r: {
        "rows": [], "tiers": {}, "flagged": [
            ("J1", "home-fetched", "pj-320d"),
            ("U7", "missing-lib", "tl074x"),
        ]})
    data = triage.collect_scan([tmp_path])
    assert data["home_fetched"] == {"pj-320d": ["1/999"]}
    assert data["missing_libs"] == {"tl074x": ["1/999"]}
    assert run  # silence unused


def test_scan_home_fetched_ranked_by_designs_then_latest(tmp_path, monkeypatch):
    """Bucket order: most designs first, ties broken by most-recent run."""
    for pid, mtime in ((998, 0), (999, 86400)):
        run = tmp_path / "1" / str(pid)
        sd = run / "generated" / "BOARD"
        exp = sd / ".experiments"
        exp.mkdir(parents=True)
        _write(exp / "hierarchical_autoexperiment" / "round_0001"
               / "parent_pipeline.json",
               {"state": {"routed_validation": {}}})
        os.utime(run, (mtime, mtime))

    def fake_prov(run: Path):
        if run.name == "998":
            return {"rows": [], "tiers": {}, "flagged": [
                ("J1", "home-fetched", "older-slug"),
                ("J2", "home-fetched", "shared-slug")]}
        return {"rows": [], "tiers": {}, "flagged": [
            ("J1", "home-fetched", "shared-slug"),
            ("J2", "home-fetched", "newer-slug")]}

    monkeypatch.setattr(triage, "collect_library_provenance", fake_prov)
    data = triage.collect_scan([tmp_path])
    assert list(data["home_fetched"]) == ["shared-slug", "newer-slug", "older-slug"]


# ---------------------------------------------------------------------------
# unconnected-net classification
# ---------------------------------------------------------------------------

def test_classify_unconnected_with_names():
    cls = triage.classify_unconnected(["GND", "GPIO2"], ["GND", "VBUS"])
    assert cls["cross_leaf"] == ["GND"]
    assert cls["leaf_internal"] == ["GPIO2"]
    assert cls["unclassified"] == []


def test_classify_unconnected_predates_key():
    cls = triage.classify_unconnected(["GND"], None)
    assert cls["cross_leaf"] is None and cls["leaf_internal"] is None
    assert cls["unclassified"] == ["GND"]
    assert "predates" in cls["note"]


# ---------------------------------------------------------------------------
# reason normalization — new failure families must collapse to one row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("connector_stranded:J1@-4.41mm(left)", "connector_stranded:<ref>"),
    ("connector_misoriented:J2(mouth 90deg vs left outward 180deg)",
     "connector_misoriented:<ref>"),
    ("form-factor non-conformant (4/32 standard header pins present)",
     "form-factor non-conformant"),
    ("outline-shape non-conformant (requested circle, delivered rect 61x44mm)",
     "outline-shape non-conformant"),
    ("unconnected_nets", "unconnected_nets"),
])
def test_norm_reason(raw, expected):
    assert triage.norm_reason(raw) == expected


# ---------------------------------------------------------------------------
# promote provenance — an rc6 partial promote must never read as routed
# ---------------------------------------------------------------------------

def test_partial_promote_is_not_reported_routed(tmp_path):
    run, sd, exp = make_run(tmp_path)
    (sd / "BOARD.kicad_pcb").write_text("(kicad_pcb)")
    _write(sd / "BOARD.provenance.json",
           {"source_kind": "partial", "fresh": True, "run_id": "abc"})
    make_parent_round(exp, 1, {"routed_validation": {}})
    data = triage.collect_run(run)
    prov = data["promotion"]["provenance"]
    assert prov["source_kind"] == "partial"
    assert "rc6 family" in data["verdict"]
    assert "partial" in data["verdict"]


def test_build_done_ok_wins_over_dirty_round(tmp_path):
    """The promote-time verify is authoritative: a dirty last round on a run
    whose build_done says ok must not be reported rc7."""
    run, _sd, exp = make_run(tmp_path)
    make_parent_round(exp, 1, {"routed_validation": {
        "accepted": False, "rejection_reasons": ["unconnected_nets"],
        "drc": {"unconnected": 1, "unconnected_nets": ["CTRL_4"]}}})
    (run / "events.jsonl").write_text(json.dumps({"kind": "build_done", "ok": True}) + "\n")
    data = triage.collect_run(run)
    assert "fab-ready" in data["verdict"]


# ---------------------------------------------------------------------------
# the autorouter fingerprints
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# leaves: replica stubs and the parent artifact must not be counted as leaves
# ---------------------------------------------------------------------------

def test_leaves_classifies_replica_and_parent(tmp_path):
    _run, _sd, exp = make_run(tmp_path)
    _write(exp / "subcircuits" / "leaf_a" / "debug.json", {
        "metadata": {"sheet_name": "POWER"},
        "extra": {"leaf_acceptance_structured": {
            "accepted": False, "rejection_reasons": ["no_unconnected"],
            "gate_results": {"no_unconnected": {
                "passed": False, "unconnected_total": 2,
                "unconnected_nets": ["A", "B"],
                "signal_unconnected_nets": ["A"],
                "ignored_interface_nets": ["B"],
                "failure_class": "router_fail"}}}}
    })
    _write(exp / "subcircuits" / "leaf_b" / "debug.json",
           {"replicated_from": "/leaf_a", "sheet_name": "POWER 2"})
    _write(exp / "subcircuits" / "subcircuit__p" / "debug.json",
           {"routing_result": {}, "composition_state": {}})
    leaves = triage.collect_leaves(exp)
    kinds = sorted(lf["kind"] for lf in leaves)
    assert kinds == ["leaf", "replica"]  # the parent artifact is excluded
    leaf = next(lf for lf in leaves if lf["kind"] == "leaf")
    assert leaf["unconnected"]["signal"] == ["A"]


# ---------------------------------------------------------------------------
# intent adherence — the no-template path must not raise (old skill NameError)
# ---------------------------------------------------------------------------

def test_intent_adherence_mech_signal_without_template(tmp_path):
    run, sd, _exp = make_run(tmp_path)
    _write(run / ".kicraft" / "state.json", {
        "intent": {"brief": "A sensor board that fits enclosure X, 50 x 40 mm"},
        "bom": {"parts": []},
    })
    (sd / "BOARD.kicad_pcb").write_text("(kicad_pcb)")
    ia = triage.collect_intent_adherence(run)
    assert ia["explicit_dims"]
    assert "GAP" in ia["verdict"] or "detection gap" in ia["verdict"]


# ---------------------------------------------------------------------------
# MPN relatedness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,related", [
    ("WJ126V-5.0-2P", "WJ126V-5.0-02P-14-00A", True),
    ("PC817C", "PC817C", True),
    ("SS14", "SS14", True),
    ("TPS5430", "LM2596S-5.0", False),
])
def test_mpn_related(a, b, related):
    assert triage._mpn_related(a, b) is related


# ---------------------------------------------------------------------------
# drift guards — the producer contracts triage depends on
# ---------------------------------------------------------------------------

def test_parent_state_to_dict_carries_every_key_triage_reads():
    from kicraft.cli._compose_state import ParentCompositionState
    d = ParentCompositionState(project_dir=".", spacing_mm=1.0).to_dict()
    missing = [k for k in triage.PARENT_STATE_KEYS if k not in d]
    assert not missing, (
        f"ParentCompositionState.to_dict() no longer emits {missing} — "
        "update triage.py AND the investigate skill together")


def test_compact_routed_validation_keeps_repair_evidence():
    from kicraft.cli.compose_subcircuits import _compact_routed_validation
    full = {
        "accepted": False,
        "rejection_reasons": ["unconnected_nets"],
        "drc": {"unconnected": 1, "report_text": "x" * 5000},
        "post_route_repairs": {"gnd_islands": {}},
        "signal_unconnected_repair": {"ran": True},
        "illegal_geometry_repair": {"ran": True},
    }
    out = _compact_routed_validation(full)
    for k in ("post_route_repairs", "signal_unconnected_repair",
              "illegal_geometry_repair"):
        assert k in out, f"compactor dropped repair-evidence key {k}"
    assert "report_text" not in out["drc"]


def test_leaf_gate_detail_contract():
    """The no_unconnected gate detail keys triage reads, produced by the real
    gate code — not a hand-rolled fixture."""
    from kicraft.autoplacer.brain.leaf_acceptance import (
        LeafAcceptanceConfig, _gate_no_unconnected)
    validation = {
        "drc": {"unconnected": 2, "unconnected_nets": ["A", "IFACE"]},
        "interface_port_names": ["IFACE"],
    }
    passed, detail = _gate_no_unconnected(
        validation, {}, LeafAcceptanceConfig(max_unconnected=0))
    assert passed is False
    for k in ("signal_unconnected_nets", "ignored_interface_nets",
              "failure_class"):
        assert k in detail, f"gate detail lost key {k} triage reads"
    assert detail["failure_class"] == "router_fail"
