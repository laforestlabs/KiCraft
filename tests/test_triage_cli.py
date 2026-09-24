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


# ---------------------------------------------------------------------------
# LLM design stages — the failure class that dies before any board exists
# ---------------------------------------------------------------------------

def make_stage_run(tmp_path, *, status: dict, events: list[dict],
                   stem: str = "BOARD") -> Path:
    """A run whose LLM stages are its only artifacts (no .experiments, no ERC
    report) — exactly the shape the layout scan cannot see."""
    run = tmp_path / "1" / "999"
    run.mkdir(parents=True, exist_ok=True)
    (run / ".kicraft").mkdir(exist_ok=True)
    _write(run / ".kicraft" / "state.json", {"stage_status": status})
    (run / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))
    (run / "generated" / stem).mkdir(parents=True, exist_ok=True)
    return run


def _stage_events(stage: str, retries: list[dict], *, done: dict | None = None):
    evs = [{"kind": "stage_start", "stage": stage, "model": "m"}]
    evs += retries
    evs.append(done or {"kind": "stage_done", "stage": stage, "ok": False})
    return evs


def test_stage_failure_kind_falls_back_to_state_status(tmp_path):
    """Older runs' stage_done carries only {stage, ok, cost, attempts}. The
    classification lives in state.json — reading the event alone loses it and
    the failure reads as unclassified."""
    run = make_stage_run(
        tmp_path,
        status={"wiring": {"ok": False, "attempts": 6, "failure_kind": "commit_rejected"}},
        events=_stage_events("wiring", [
            {"kind": "retry", "stage": "wiring",
             "errors": ["9.15 no dangling signal nets: 39 signal net(s) wire a single pin"]},
        ], done={"kind": "stage_done", "stage": "wiring", "ok": False,
                 "cost": 0.0055, "attempts": 6}),
    )
    d = triage.collect_stages(run)
    assert d["terminal_stage"] == "wiring"
    assert d["terminal_failure_kind"] == "commit_rejected"
    assert d["terminal_family"] == "gate-rejection"


def test_contract_rejection_is_not_reported_as_malformed_json(tmp_path):
    """A pre-`contract_rejected` run reports `invalid_schema` for BOTH malformed
    provider output and a semantic/recipe contract rejection. With a diagnostic
    code present the failure is the contract's, and chasing JSON formatting is
    the wrong investigation."""
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": False, "attempts": 4,
                                 "failure_kind": "invalid_schema",
                                 "provider_ok": True, "schema_ok": True}},
        events=_stage_events("architecture", [{
            "kind": "retry", "stage": "architecture",
            "errors": ["provider response did not satisfy the required JSON schema"],
            "failure_kind": "invalid_schema",
            "diagnostic": {
                "code": "multiple_recipe_contracts",
                "evidence": [{"code": "unsupported_protected_variant"},
                             {"code": "unsupported_recipe_endpoint"}]},
        }]),
    )
    d = triage.collect_stages(run)
    term = d["stages"][0]
    assert d["terminal_family"] == "contract/recipe"
    assert term["contract_rejection"] is True
    assert term["legacy_schema_label"] is True
    assert term["diagnostics"][0]["code"] == "multiple_recipe_contracts"
    assert term["diagnostics"][0]["sub_codes"] == [
        "unsupported_protected_variant", "unsupported_recipe_endpoint"]
    assert term["attempt_budget_floor"] == 4   # max(2, architecture 3) + 1


def test_invalid_schema_without_diagnostic_stays_schema_output(tmp_path):
    """No diagnostic code => a genuine provider/schema failure, family
    schema-output. The distinction is the whole point of the label."""
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": False, "failure_kind": "invalid_schema"}},
        events=_stage_events("architecture", [{
            "kind": "retry", "stage": "architecture",
            "errors": ["provider response did not satisfy the required JSON schema"],
            "failure_kind": "invalid_schema"}]),
    )
    term = triage.collect_stages(run)["stages"][0]
    assert term["family"] == "schema-output"
    assert term["contract_rejection"] is False
    assert term["legacy_schema_label"] is False


def test_contract_rejected_kind_classifies_when_the_label_is_honest(tmp_path):
    """The stage driver labels a contract rejection `contract_rejected`, so the
    reader must classify it without needing the diagnostic-code backstop."""
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": False, "failure_kind": "contract_rejected",
                                 "attempts": 4}},
        events=_stage_events("architecture", [{
            "kind": "retry", "stage": "architecture",
            "errors": ["the reply was schema-valid but a deterministic design contract refused it"],
            "failure_kind": "contract_rejected",
            "diagnostic": {"code": "unsupported_recipe_endpoint"}}]),
    )
    d = triage.collect_stages(run)
    term = d["stages"][0]
    assert d["terminal_failure_kind"] == "contract_rejected"
    assert d["terminal_family"] == "contract/recipe"
    assert term["contract_rejection"] is True
    assert term["legacy_schema_label"] is False


def test_rejection_signature_carries_the_detector_codes(tmp_path):
    """Runs 904/905/907 printed "5 DISTINCT diagnostics under ONE signature".

    A contract rejection repeats the same schema-clean wrapper text on every rung while
    the diagnostic moves, so grouping on the wrapper alone could not tell convergence
    from ping-pong. The codes are part of the signature: one row per attempt-level
    situation, and its count is how often that situation repeated.
    """
    wrapper = ("the reply was schema-valid but a deterministic design contract refused it "
               "(see the diagnostic)")
    retry = lambda mode, codes: {
        "kind": "retry",
        "stage": "architecture",
        "errors": [wrapper],
        "failure_kind": "contract_rejected",
        "call_mode": mode,
        "diagnostic": {
            "code": codes[0],
            "evidence": [{"code": code} for code in codes[1:]],
        },
    }
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": False, "attempts": 3,
                                 "failure_kind": "contract_rejected"}},
        events=_stage_events("architecture", [
            retry("normal", ["multiple_intent_contracts", "declared_signal_port_tied"]),
            retry("clean_slate", ["multiple_intent_contracts", "unsupported_lowerer_contract"]),
            # The same situation twice: one row, count 2.
            retry("normal", ["multiple_intent_contracts", "declared_signal_port_tied"]),
        ]),
    )

    term = triage.collect_stages(run)["stages"][0]
    assert [(r["codes"], r["count"]) for r in term["rejections"]] == [
        (["multiple_intent_contracts", "declared_signal_port_tied"], 2),
        (["multiple_intent_contracts", "unsupported_lowerer_contract"], 1),
    ]
    # The label names the situation, not just the generic wrapper text.
    assert term["rejections"][0]["label"] == (
        "(no gate id: a semantic contract refused a schema-clean candidate)"
        " + multiple_intent_contracts + declared_signal_port_tied"
    )


def test_resolved_stage_failure_is_not_a_terminal_failure(tmp_path):
    """A stage that failed and was later re-run reads ok in stage_status (last
    attempt wins). The stale ok=false stage_done must not resurrect it."""
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": True, "attempts": 2}},
        events=_stage_events("architecture", [], done={
            "kind": "stage_done", "stage": "architecture", "ok": False}) + [
            {"kind": "stage_start", "stage": "architecture"},
            {"kind": "stage_done", "stage": "architecture", "ok": True}],
    )
    d = triage.collect_stages(run)
    assert d["terminal_stage"] is None
    assert d["all_committed"] is False  # the other four still uncommitted


def test_interrupted_stage_is_reported(tmp_path):
    """An exception can escape between stage_start and stage_done; the
    unmatched start identifies the interrupted stage."""
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": True}},
        events=[{"kind": "stage_start", "stage": "architecture"},
                {"kind": "stage_done", "stage": "architecture", "ok": True},
                {"kind": "stage_start", "stage": "bom"},
                {"kind": "retry", "stage": "bom", "errors": ["x"]}],
    )
    d = triage.collect_stages(run)
    assert d["interrupted"] == ["bom"]
    assert d["terminal_stage"] == "bom"


def test_run_verdict_names_the_stage_not_the_schematic(tmp_path):
    """The mis-route this reader exists to fix: an uncommitted stage means the
    build never ran, so 'investigate the schematic' is a dead end."""
    run = make_stage_run(
        tmp_path,
        status={"functional_spec": {"ok": True},
                "architecture": {"ok": False, "failure_kind": "invalid_schema"}},
        events=_stage_events("architecture", [{
            "kind": "retry", "stage": "architecture", "errors": ["x"],
            "diagnostic": {"code": "missing_recipe_port"}}]),
    )
    data = triage.collect_run(run)
    assert "LLM STAGE FAILURE at architecture" in data["verdict"]
    assert "schematic" not in data["verdict"]


def test_scan_sees_stage_only_failures_the_layout_scan_cannot(tmp_path):
    """A run with an event stream and no layout artifact must rank in the
    stage buckets — and must NOT inflate the layout tier count."""
    make_stage_run(
        tmp_path,
        status={"bom": {"ok": False, "failure_kind": "unit_repair_exhausted"}},
        events=_stage_events("bom", [
            {"kind": "retry", "stage": "bom", "errors": ["9.33 spec-named part accountability"]}]),
    )
    data = triage.collect_scan([tmp_path])
    assert data["run_count"] == 0            # no layout artifact anywhere
    assert data["pipeline_run_count"] == 1
    assert data["stage_fail_run_count"] == 1
    assert data["stage_kinds"] == {"bom: unit_repair_exhausted": ["1/999"]}
    assert data["stage_sigs"] == {"bom: 9.33": ["1/999"]}


@pytest.mark.parametrize("raw,expected", [
    ("9.15 no dangling signal nets: 39 signal net(s) wire a single pin",
     "9.15 no dangling signal nets: <n> signal net(s) wire a single pin"),
    ("9.26 BOM part(s) not orderable: U1 needs stock",
     "9.26 BOM part(s) not orderable: <ref> needs stock"),
])
def test_norm_stage_error_keeps_the_gate_code(raw, expected):
    """The gate code names the contract and must survive; per-instance counts
    and refdes must not, or one family splits into a row per design."""
    assert triage.norm_stage_error(raw) == expected


def test_unit_repair_failure_reports_the_failing_unit(tmp_path):
    """A unit stage's stage-level error is a summary string. The actionable
    evidence is the failing work unit's own validation error, which only the
    work_unit_attempt events carry."""
    bad = "work unit bom-s001 invalid: missing-requirement-implementation=['db9_can']"
    run = make_stage_run(
        tmp_path,
        status={"bom": {"ok": False, "attempts": 5, "rounds": 8,
                        "failure_kind": "unit_repair_exhausted"}},
        events=_stage_events("bom", [], done={
            "kind": "stage_done", "stage": "bom", "ok": False}) + [
            {"kind": "work_unit_attempt", "stage": "bom", "unit_id": "bom-s000",
             "unit_sheet": "STM32 MCU", "outcome": "candidate"},
            {"kind": "work_unit_attempt", "stage": "bom", "unit_id": "bom-s001",
             "unit_sheet": "DB9 CAN INTERFACE", "outcome": "invalid_work_unit",
             "schema_error": bad},
            {"kind": "work_unit_attempt", "stage": "bom", "unit_id": "bom-s001",
             "unit_sheet": "DB9 CAN INTERFACE", "outcome": "invalid_work_unit",
             "schema_error": bad},
        ],
    )
    term = triage.collect_stages(run)["stages"][0]
    assert term["family"] == "unit-repair"
    assert [u["unit_id"] for u in term["units"]] == ["bom-s000", "bom-s001"]
    bad_unit = [u for u in term["units"] if not u["ok"]]
    assert len(bad_unit) == 1
    assert bad_unit[0]["outcomes"] == {"invalid_work_unit": 2}
    assert bad_unit[0]["errors"] == [bad]


def test_ladder_reports_the_clean_slate_escape_as_terminal(tmp_path):
    """A rejected clean-slate escape ends the stage BY POLICY with budget unspent.

    Reading `attempts < budget` as "it had attempts left, so it was breadth"
    would send a reader to raise a budget that never governs this path.
    """
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": False, "attempts": 3,
                                 "failure_kind": "contract_rejected"}},
        events=_stage_events("architecture", [
            {"kind": "retry", "stage": "architecture", "errors": ["a"],
             "failure_kind": "contract_rejected", "call_mode": "normal",
             "diagnostic": {"code": "native_usb_connector_required"}},
            {"kind": "serialization_recovery", "stage": "architecture",
             "failure_kind": "contract_rejected"},
            {"kind": "retry", "stage": "architecture", "errors": ["b"],
             "failure_kind": "contract_rejected", "call_mode": "clean_slate",
             "diagnostic": {"code": "unknown_recipe_port_net"}},
        ]),
    )
    term = triage.collect_stages(run)["stages"][0]

    assert [r["mode"] for r in term["ladder"]["rungs"]] == [
        "normal", "serialization", "clean_slate"]
    assert term["ladder"]["clean_slate_rejected"] is True
    assert term["ladder"]["inferred"] is False
    assert term["terminal_by_policy"] is True
    assert term["budget_spent"] is False          # a slot was never spendable
    note = triage._budget_note(term)
    assert "BY POLICY" in note and "BREADTH" not in note


def test_ladder_infers_rungs_on_artifacts_without_call_mode(tmp_path):
    """`call_mode` on retry events is new; an older run must still yield the
    ladder, and must say that it was inferred."""
    run = make_stage_run(
        tmp_path,
        status={"architecture": {"ok": False, "attempts": 3,
                                 "failure_kind": "invalid_schema"}},
        events=_stage_events("architecture", [
            {"kind": "retry", "stage": "architecture", "errors": ["a"],
             "failure_kind": "invalid_schema"},
        ]) + [{"kind": "serialization_recovery", "stage": "architecture",
               "failure_kind": "invalid_schema"}],
    )
    term = triage.collect_stages(run)["stages"][0]

    assert [r["mode"] for r in term["ladder"]["rungs"]] == [
        "normal", "serialization", "clean_slate"]
    assert term["ladder"]["inferred"] is True
    assert term["terminal_by_policy"] is True


def test_budget_note_does_not_misread_a_unit_repair_stop(tmp_path):
    """attempts == budget is a coincidence for a unit stage: the stop rule is
    the per-unit repair loop, not the provider-call budget. Calling that
    'breadth' would send the reader to raise a budget that is not the limit."""
    unit_row = {"budget_spent": True, "attempts": 5, "attempt_budget_floor": 5,
                "failure_kind": "unit_repair_exhausted", "rounds": 8,
                "units": [{"unit_id": "bom-s001"}]}
    note = triage._budget_note(unit_row)
    assert "per-unit repair loop" in note and "BREADTH" not in note
    # ... while the schema path, which has no stall rule, really is breadth.
    for kind in ("invalid_schema", "contract_rejected", "truncated_json"):
        assert "BREADTH" in triage._budget_note(
            {"budget_spent": True, "attempts": 4, "attempt_budget_floor": 4,
             "failure_kind": kind, "units": []})
    # A commit rejection has a stall rule, so "breadth" would mislead there even
    # when the diagnosis bucket is a contract family.
    commit = triage._budget_note(
        {"budget_spent": True, "attempts": 8, "attempt_budget_floor": 8,
         "failure_kind": "commit_rejected", "family": "contract/recipe", "units": []})
    assert "DIFFERENT rejection" in commit and "BREADTH" not in commit
    assert triage._budget_note({"budget_spent": False}) is None


@pytest.mark.parametrize("raw,expected", [
    ("work unit bom-s001 invalid: missing-requirement-implementation=['db9_can']",
     "work unit <id> invalid: missing-requirement-implementation=…"),
    ("work unit b-1 invalid: no-parts=['J1', 'U2']",
     "work unit <id> invalid: no-parts=…"),
])
def test_norm_unit_error_is_a_stable_cross_run_key(raw, expected):
    """Unit id and argued values are per-instance; the failing check is not, or
    the same contract gap ranks as a separate row per design."""
    assert triage.norm_unit_error(raw) == expected

