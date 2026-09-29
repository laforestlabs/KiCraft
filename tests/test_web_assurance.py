"""Phase E: the honest assurance reading of a project (web._project_assurance).

The UI must distinguish a software-verified complete export, generated-but-
review-required output, a failed/partial or stale preview, awaiting
clarification, and a recorded capability limitation -- from persisted evidence
only, so a reopen derives the same answer. Two rules are load-bearing and are
pinned here:

* a downloadable package, a completed review stage or parts coverage never
  implies verified fulfilment (only a CURRENT, artifact-hash-bound verification
  does); and
* a substitution ledger entry (or an auto-defaulted answer) is NOT the user's
  consent to change a part the USER required.

Pure-data assertions cover the derivation; the last block drives the real page to
show the honest export label and the withheld download on the actual surface.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server import web
from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION

STATE_FIXTURE = Path(__file__).parent / "fixtures" / "bmp280_reader_state.json"

_FULL_STAGE_STATUS = {stage: {"ok": True} for stage in web.DESIGN_STAGES}
_CLEAN_GATE = {"fab_acceptable": True, "shorts": 0, "unconnected": 0,
               "courtyard": 0, "keepout": 0}


def _signals(state: dict, *, gate: dict | None = None, synth: dict | None = None,
             flow: dict | None = None, artifacts_present: bool = True) -> dict:
    """The durable-signal reading a presentation passes to the derivation."""
    signals = dict(web._EMPTY_SIGNALS)
    artifacts = state.get("artifacts") or {}
    signals.update(
        state=state,
        stage_status=state.get("stage_status") or {},
        review_findings=list(state.get("review_findings")
                             or artifacts.get("review_findings") or []),
        build_warnings=list(artifacts.get("build_warnings") or []),
        build_gate=dict(gate or {}),
        synth_check=dict(synth or {}),
        fulfillment=dict(flow or {}),
        sheets=artifacts_present,
        pcb=artifacts_present,
        zip_ok=artifacts_present,
        generated="/tmp/generated" if artifacts_present else None,
    )
    return signals


def _derived(**over) -> dict:
    derived = {stage: "done" for stage in web.DESIGN_STAGES}
    derived.update({"synthesize": "done", "place_route": "done",
                    "electrical_review": "done", "fab": "done"})
    derived.update(over)
    return derived


def _clean_state(**over) -> dict:
    state = {
        "project_stem": "X",
        "stage_status": {k: v for k, v in _FULL_STAGE_STATUS.items()},
        "intent": {"named_parts": ["TP4056"], "assumptions": []},
        "architecture": {"requirements": [{"id": "charger", "exact_part": "TP4056"}],
                         "assumptions": [], "advisories": []},
        "bom": {"parts": [{"ref": "U1"}], "substitutions": [], "assumptions": []},
        "artifacts": {"status": "ok"},
    }
    state.update(over)
    return state


def _assurance(state, *, status="complete", derived=None, gate=None, synth=None,
               flow=None, artifacts=True) -> dict:
    return web._project_assurance(
        status=status,
        derived=derived if derived is not None else _derived(),
        signals=_signals(state, gate=gate if gate is not None else _CLEAN_GATE,
                         synth=synth if synth is not None else {"status": "ok"},
                         flow=flow, artifacts_present=artifacts))


# ---- 1. straightforward delivery --------------------------------------------

def test_clean_delivery_is_fabrication_clean_but_not_verified():
    """A clean build is fabrication-gate clean; it is NOT software-verified, and
    the label says exactly that."""
    a = _assurance(_clean_state())
    assert a["level"] == "generated"
    assert a["verified_export"] is False
    assert a["stale_preview"] is False
    assert "No independent" in a["summary"]
    assert any("fabrication gate passed" in line for line in a["evidence"])
    assert any("Synthesis checks passed" in line for line in a["evidence"])
    assert a["limitations"] == []


def test_only_a_current_verification_promotes_to_verified():
    a = _assurance(_clean_state(),
                   flow={"verified": True, "carrier": "an independent audit", "stale": None})
    assert a["level"] == "verified"
    assert a["verified_export"] is True
    assert "Independently verified" in a["summary"]


def test_recorded_assumptions_alone_do_not_downgrade_the_label():
    """Assumptions/defaults are decisions KiCraft records and shows -- they are not
    review gaps, and they are never presented as the user's approval."""
    state = _clean_state(
        intent={"named_parts": ["TP4056"], "assumptions": ["USB-C 5 V input (defaulted)"]})
    state["open_questions"] = [{"text": "Charger current?", "stage": "bom",
                                "blocking": False, "material": False,
                                "default_applied": "1 A"}]
    a = _assurance(state)
    assert a["level"] == "generated"
    assert any("USB-C 5 V" in line for line in a["assumptions"])
    assert any("Defaulted a cosmetic choice" in line for line in a["auto_actions"])


# ---- 2. recovered delivery --------------------------------------------------

def test_recovered_delivery_reports_the_action_it_took():
    state = _clean_state()
    state["stage_status"] = {
        **_FULL_STAGE_STATUS,
        "build_recovery": {
            "ok": True, "recovery_run_id": "r1", "recovery_max_attempts": 3,
            "recovery_events": [{
                "action": "backtrack_bom",
                "reason": "U1 unorderable; chose a reviewed alternative",
                "outcome": "applied", "failure_fingerprint": "a" * 64,
            }],
        },
    }
    a = _assurance(state)
    assert a["recovered"] is True
    assert a["level"] == "generated"
    assert any("Recovered automatically" in line and "component choice" in line
               for line in a["auto_actions"])
    assert "recovered automatically" in a["summary"]


# ---- 3. exact-part limitation ----------------------------------------------

def test_recorded_unrealizable_requirement_is_a_capability_limitation():
    state = _clean_state()
    state["stage_status"] = {
        "intent": {"ok": True},
        "architecture": {
            "ok": False, "failure_kind": "contract_rejected", "repair_required": True,
            "diagnostics": [{
                "code": "unavailable_recipe_gpio",
                "severity": "repair_required",
                "message": "no reviewed recipe covers the named CP2102N",
                "requirement_id": "cp2102n_bridge",
            }],
        },
    }
    a = _assurance(state, status="failed", derived=_derived(architecture="failed"),
                   gate={}, synth={}, artifacts=False)
    assert a["level"] == "limited"
    assert a["capability"] and "cp2102n_bridge" in a["capability"][0]
    # The wording must not claim physical impossibility.
    assert "could not realize" in a["summary"]
    assert "may resolve it" in a["summary"]


# ---- 4. clarification -------------------------------------------------------

def test_unanswered_question_is_clarification_not_a_delivery():
    state = _clean_state(open_questions=[{
        "text": "Which connector?", "stage": "architecture", "blocking": True}])
    a = _assurance(state, status="awaiting_input",
                   derived=_derived(architecture="parked"), gate={}, synth={},
                   artifacts=False)
    assert a["level"] == "clarification"


# ---- 5. exhausted build -----------------------------------------------------

def test_exhausted_build_is_partial_and_names_why():
    state = _clean_state()
    state["stage_status"] = {
        **_FULL_STAGE_STATUS,
        "build_recovery": {
            "ok": False, "recovery_run_id": "r2", "recovery_max_attempts": 2,
            "failure_kind": "wall_stall",
            "recovery_events": [{
                "action": "repair_wiring",
                "reason": "unconnected net after rebuild",
                "outcome": "exhausted", "failure_fingerprint": "b" * 64,
            }],
        },
    }
    a = _assurance(state, status="failed",
                   derived=_derived(place_route="failed", fab="failed"),
                   gate={"fab_acceptable": False, "shorts": 1, "reasons": ["shorts"]})
    assert a["level"] == "partial"
    assert a["exhausted_reason"] == "unconnected net after rebuild"
    assert "exhausted" in a["summary"]


# ---- 6. review-required -----------------------------------------------------

def test_recorded_advisory_requires_review_without_blocking_the_export():
    state = _clean_state()
    state["architecture"] = {
        "requirements": [],
        "advisories": [{"code": "unverified_stock",
                        "message": "retail stock unverified for U1"}],
    }
    a = _assurance(state)
    assert a["level"] == "review_required"
    assert any("unverified_stock" in line for line in a["limitations"])
    assert a["verified_export"] is False


def test_review_blocker_is_a_review_gap_and_blocks_verification():
    state = _clean_state()
    state["review_findings"] = [{"severity": "blocker", "area": "power",
                                 "issue": "no bulk capacitance on VIN"}]
    a = _assurance(state, flow={"verified": True, "carrier": "audit", "stale": None})
    assert a["level"] == "review_required"
    assert a["verified_export"] is False
    assert any("blocker" in line for line in a["limitations"])


def test_missing_synthesis_check_record_is_a_gap_not_a_pass():
    """A legacy export with no synthesis-check summary cannot claim the checks
    passed -- the absence is recorded."""
    a = _assurance(_clean_state(), synth={})
    assert a["level"] == "review_required"
    assert any("No synthesis-check summary" in line for line in a["limitations"])
    assert not any("Synthesis checks passed" in line for line in a["evidence"])


# ---- consent ----------------------------------------------------------------

def test_substitution_ledger_alone_is_not_consent_for_a_required_part():
    state = _clean_state(
        intent={"named_parts": ["CP2102N"], "assumptions": []},
        bom={"parts": [{"ref": "U1"}],
             "substitutions": [{"wanted": "CP2102N", "got": "CH340C",
                                "reason": "out of stock"}],
             "assumptions": []})
    a = _assurance(state)
    assert a["level"] == "review_required"
    assert a["verified_export"] is False
    (deviation,) = a["deviations"]
    assert deviation["required"] is True and deviation["consent"] is None
    assert any("no recorded user approval" in line for line in a["limitations"])


def test_recorded_user_answer_is_consent_for_a_required_part():
    state = _clean_state(
        intent={"named_parts": ["CP2102N"], "assumptions": []},
        bom={"parts": [{"ref": "U1"}],
             "substitutions": [{"wanted": "CP2102N", "got": "CH340C",
                                "reason": "out of stock"}],
             "assumptions": []},
        open_questions=[{"text": "CP2102N is out of stock; use CH340C?",
                         "stage": "bom", "blocking": True,
                         "answer": "yes, use CH340C"}])
    a = _assurance(state)
    (deviation,) = a["deviations"]
    assert deviation["consent"] == "yes, use CH340C"
    assert a["level"] == "generated"  # approved: no longer a review gap
    assert any("User-approved change" in line for line in a["evidence"])


def test_model_selected_substitution_is_an_automatic_action_not_a_limitation():
    state = _clean_state(
        bom={"parts": [{"ref": "U1"}],
             "substitutions": [{"wanted": "generic 10k", "got": "RC0603FR-0710KL",
                                "reason": "cheapest in stock"}],
             "assumptions": []})
    a = _assurance(state)
    assert a["level"] == "generated"
    assert a["limitations"] == []
    (deviation,) = a["deviations"]
    assert deviation["required"] is False
    assert any("Chose an available alternative" in line for line in a["auto_actions"])


# ---- stale preview ----------------------------------------------------------

def test_artifacts_from_an_earlier_design_are_a_stale_preview():
    a = _assurance(_clean_state(), status="stale",
                   derived=_derived(intent="pending", architecture="pending",
                                    bom="pending", wiring="pending"))
    assert a["stale_preview"] is True
    assert a["level"] == "partial"
    assert "does not match" in a["summary"]


def test_reopen_derives_the_same_reading_from_persistence():
    """The derivation is a pure function of the persisted evidence: reading the
    same files twice -- as a reopen does -- yields the same status and the same
    accepted deviations."""
    state = _clean_state(
        intent={"named_parts": ["CP2102N"], "assumptions": ["USB-C 5 V (defaulted)"]},
        bom={"parts": [{"ref": "U1"}],
             "substitutions": [{"wanted": "CP2102N", "got": "CH340C",
                                "reason": "out of stock"}],
             "assumptions": []})
    first = _assurance(state)
    second = _assurance(json.loads(json.dumps(state)))
    assert first == second
    assert [(d["wanted"], d["consent"]) for d in first["deviations"]] == [("CP2102N", None)]


# ---- integration: real audit file, hash-bound --------------------------------

def _write_durable_project(store, user_id: int, brief: str) -> tuple[int, Path]:
    """A finished project laid down as build-in-place does: .kicraft/state.json, a
    generated tree, and the fab package."""
    pid = store.create_project(user_id, brief)
    root = store.projects_dir / str(user_id) / str(pid)
    (root / ".kicraft").mkdir(parents=True)
    (root / "brief.txt").write_text(brief, encoding="utf-8")
    state = json.loads(STATE_FIXTURE.read_text(encoding="utf-8"))
    state["project_stem"] = "board"
    state["obligations"] = {}
    state["stage_status"] = {stage: {"ok": True} for stage in web.DESIGN_STAGES}
    (root / ".kicraft" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    # A genuinely clean build's own verdicts: these are the durable evidence the
    # assurance reading derives from (never a second store).
    (root / ".kicraft" / "build_gate.json").write_text(
        json.dumps({"fab_acceptable": True, "shorts": 0, "unconnected": 0,
                    "courtyard": 0, "keepout": 0}), encoding="utf-8")
    (root / ".kicraft" / "synthesis_check.json").write_text(
        json.dumps({"status": "ok", "failed_checks": []}), encoding="utf-8")
    gen = root / "generated" / "board"
    gen.mkdir(parents=True)
    (gen / "board.kicad_sch").write_text("(kicad_sch)", encoding="utf-8")
    (gen / "board.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")
    (root / "kicraft_project.zip").write_bytes(b"package")
    store.finish_project(pid, "ok", stem="board", dir_path=str(root),
                         zip_path=str(root / "kicraft_project.zip"))
    return pid, root


_ARCHIVE_REL = "generated/board/board_fab_20260922.zip"


def _write_product_audit(root: Path, brief: str) -> None:
    """Write the evaluator's OWN acceptance audit for this run (the real writer).

    Mirrors tests/test_product_acceptance.py's artifact harness so the audit that
    lands is a genuine `evaluate_product` success, hash-bound to these files."""
    import hashlib
    import zipfile

    from kicraft.eval import product_acceptance as product

    for relative in (".kicraft/build_gate.json", ".kicraft/synthesis_check.json",
                     "generated/board/board.kicad_pcb"):
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
    archive_path = root / _ARCHIVE_REL
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "board-job.gbrjob",
            json.dumps({
                "GeneralSpecs": {"ProjectId": {"Name": "board"}, "LayerNumber": 2},
                "FilesAttributes": [
                    {"Path": "board-F_Cu.gtl", "FileFunction": "Copper,L1,Top"},
                    {"Path": "board-B_Cu.gbl", "FileFunction": "Copper,L2,Bot"},
                    {"Path": "board-Edge_Cuts.gm1", "FileFunction": "Profile"},
                ],
            }),
        )
        archive.writestr("board-PTH.drl", "M48\nM30")
        for filename in ("board-F_Cu.gtl", "board-B_Cu.gbl", "board-Edge_Cuts.gm1"):
            archive.writestr(filename, "%FSLAX46Y46*%\nM02*")
    with zipfile.ZipFile(archive_path) as archive:
        receipt = {
            "schema_version": 1,
            "board": "board.kicad_pcb",
            "board_sha256": hashlib.sha256(
                (archive_path.parent / "board.kicad_pcb").read_bytes()).hexdigest(),
            "archive": archive_path.name,
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "files": {name: hashlib.sha256(archive.read(name)).hexdigest()
                      for name in archive.namelist()},
        }
    archive_path.with_suffix(".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    facts = {
        "board_loaded": True,
        "bom_present": True,
        "delivered_board_path": "generated/board/board.kicad_pcb",
        "part_classes": {"bnc_connector": 2, "microcontroller": 0},
        "part_inventory": [
            {
                "mpn": "RC0603FR-0710KL",
                "manufacturer": "Yageo",
                "source_url": "https://example.test/part",
                "package": "0603",
                "rated_limits": {"voltage_v": 50},
            }
        ],
        "pin_mapping": [{"symbol_pin": "R1.1", "footprint_pad": "R1.1"}],
        "verified_export_paths": [_ARCHIVE_REL],
        "artifacts": [_ARCHIVE_REL],
        "build_gate": {"fab_acceptable": True},
        "gates": {
            "erc": "pass",
            "complete_required_connections": "pass",
            "geometry": "pass",
        },
    }
    record = {
        "slug": "local", "design_committed": True, "design_status": "ok",
        "original_brief_hash": web._stable_hash(brief),
        "execution_brief_hash": web._stable_hash(brief),
        "manual_intervention": False, "clarification_assisted": False,
        "ledger_cost_usd": 0.10, "duration_s": 12.0, "park_rounds": 0, "build_rc": 0,
    }
    policy = {"max_cost_usd": 1.0, "max_duration_s": 60.0,
              "max_park_rounds": 2, "build_timeout_s": 30.0}
    obligation = {"id": "local.bnc-count",
                  "check": {"kind": "part_class_count", "part_class": "bnc_connector",
                            "minimum": 2}}

    with _patched(product, "extract_artifact_facts",
                  lambda rundir, contract: (dict(facts), [
                      {"kind": "artifact", "path": ".kicraft/state.json"},
                      {"kind": "artifact", "path": ".kicraft/build_gate.json"},
                      {"kind": "artifact", "path": "generated/board/board.kicad_pcb"},
                      {"kind": "artifact", "path": _ARCHIVE_REL},
                  ])):

        def certify_drc(rundir, extracted, evidence, timeout_s):
            report = rundir / "eval/product_drc_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"violations": [], "unconnected_items": []}),
                              encoding="utf-8")
            evidence.append({"kind": "tool-report", "path": "eval/product_drc_report.json"})
            extracted["drc_certification"] = {"report_path": "eval/product_drc_report.json"}
            extracted.setdefault("gates", {})["drc"] = "pass"
            return []

        with _patched(product, "_certify_drc", certify_drc):
            result = product.evaluate_product(root, record, [obligation], policy)
    assert result["product_success"] is True, result["product_errors"]


class _patched:
    """Temporary attribute swap (no pytest monkeypatch: the audit writer runs
    inside a helper this module calls directly)."""

    def __init__(self, target, name, value):
        self.target, self.name, self.value = target, name, value

    def __enter__(self):
        self.previous = getattr(self.target, self.name)
        setattr(self.target, self.name, self.value)

    def __exit__(self, *exc):
        setattr(self.target, self.name, self.previous)
        return False


def _presentation(store, pid: int) -> dict:
    web._DISK_SIGNALS.clear()  # a test mutates files faster than mtime resolution
    return web._project_presentation(store.get_project(pid))


def test_current_hash_bound_audit_promotes_the_project_to_verified(tmp_path):
    store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    acct = store.create_user("a@example.com", "hunter2hunter2")
    pid, root = _write_durable_project(store, acct.id, "bmp280 reader")
    previous = web._STORE
    web._STORE = store
    try:
        # A clean, fabrication-gate-clean export with no independent verification:
        # downloadable, honestly labelled as unverified.
        before = _presentation(store, pid)
        assert before["assurance"]["level"] == "generated"
        assert "No independent" in before["assurance"]["summary"]
        assert before["download_ready"] is True

        _write_product_audit(root, "bmp280 reader")
        pres = _presentation(store, pid)
        assert pres["assurance"]["level"] == "verified"
        assert pres["assurance"]["verified_export"] is True
        assert pres["download_ready"] is True

        # An artifact that changes after verification makes the recorded success
        # stale: the export may still be downloaded, but it is no longer verified.
        (root / "generated" / "board" / "board.kicad_pcb").write_text("(changed)",
                                                                     encoding="utf-8")
        pres = _presentation(store, pid)
        assert pres["assurance"]["level"] == "review_required"
        assert pres["assurance"]["verified_export"] is False
        assert any("changed since it was verified" in line
                   for line in pres["assurance"]["limitations"])
    finally:
        web._STORE = previous


# ---- integration: the real page ----------------------------------------------

EMAIL, PASSWORD = "phase-e@example.com", "hunter2hunter2"
WEB = "kicraft.server.web"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def page_harness(tmp_path):
    prev = {key: os.environ.get(key)
            for key in ("KICRAFT_WORK_DIR", "OPENROUTER_API_KEY")}
    work = tmp_path / "work"
    work.mkdir()
    os.environ["KICRAFT_WORK_DIR"] = str(work)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-not-used")
    async with user_simulation() as u:
        mod = sys.modules.get(WEB)
        web_mod = importlib.reload(mod) if mod else importlib.import_module(WEB)
        store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        web_mod._STORE = store
        real_fetch = web_mod._safe_fetch
        web_mod._safe_fetch = lambda key: web_mod._FETCH_ERROR
        acct = store.create_user(EMAIL, PASSWORD)
        store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, web_mod, store, acct
        finally:
            web_mod._safe_fetch = real_fetch
            web_mod._STORE = None
            web_mod._LIVE_RUNS.clear()
            web_mod._DISK_SIGNALS.clear()
            for key, value in prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


async def _login(u):
    await u.open("/login")
    u.find("Email").type(EMAIL)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")


@pytest.mark.anyio
async def test_completed_project_labels_its_export_honestly(page_harness):
    """A finished project offers its package AND says what that package is: a
    downloadable board is not a verified delivery."""
    u, web_mod, store, acct = page_harness
    pid, _root = _write_durable_project(store, acct.id, "bmp280 reader")
    await _login(u)
    await u.open(f"/?project={pid}")
    await u.should_see("Fabrication-ready package")
    await u.should_see("Download KiCad project (.zip)")
    await u.should_see("No independent fulfilment verification is recorded")


@pytest.mark.anyio
async def test_projects_list_labels_each_row_with_what_the_evidence_supports(page_harness):
    """The list must not imply a fabrication-ready board from a status badge or a
    Download button alone."""
    u, web_mod, store, acct = page_harness
    _write_durable_project(store, acct.id, "bmp280 reader")
    await _login(u)
    await u.open("/projects")
    await u.should_see("Fabrication-ready package")
    await u.should_see("No independent fulfilment verification is recorded")
    await u.should_see("Download")


@pytest.mark.anyio
async def test_verified_export_is_labelled_verified_and_not_more(page_harness):
    """With a CURRENT artifact-hash-bound verification the export says so -- and
    still does not claim hardware qualification."""
    u, web_mod, store, acct = page_harness
    brief = "bmp280 reader"
    pid, root = _write_durable_project(store, acct.id, brief)
    _write_product_audit(root, brief)
    await _login(u)
    await u.open(f"/?project={pid}")
    await u.should_see("Software-verified complete export")
    await u.should_see("Hardware qualification")
    await u.should_see("Download KiCad project (.zip)")


@pytest.mark.anyio
async def test_stale_preview_is_labelled_and_withholds_the_download(page_harness):
    """A board left over from an earlier accepted design is a preview: the page
    must not offer it as this design's export.

    The invalidation is the REAL one (`invalidate_downstream`, what the web's edit
    flow calls): an upstream change clears the dependent design slots, so the
    artifacts on disk no longer match the accepted design."""
    from kicraft.design.stage_state import invalidate_downstream

    u, web_mod, store, acct = page_harness
    pid, root = _write_durable_project(store, acct.id, "bmp280 reader")
    state_path = root / ".kicraft" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    invalidate_downstream(state, "intent")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    await _login(u)
    await u.open(f"/?project={pid}")
    await u.should_see("Stale preview")
    await u.should_not_see("Download KiCad project (.zip)")
    await u.should_see("Continue design")
