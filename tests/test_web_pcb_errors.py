from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from kicraft.autoplacer import freerouting_runner
from kicraft.cli import render_drc_overlay
from kicraft.design.cli_app import build_pcb_errors
from kicraft.design.models import ArtifactPaths, PcbError, PcbViolation
from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION
from kicraft.server.session import derive_stage_statuses
from kicraft.server.web import _pcb_error_section


WEB = "kicraft.server.web"
EMAIL, PASSWORD = "pcb-errors@example.com", "hunter2hunter2"
STATE_FIXTURE = Path(__file__).parent / "fixtures" / "bmp280_reader_state.json"
pytestmark = pytest.mark.anyio


def test_pcb_error_model_round_trips_through_json(tmp_path):
    artifacts = ArtifactPaths(
        project_dir=tmp_path,
        project_stem="BOARD",
        root_sch=tmp_path / "BOARD.kicad_sch",
        leaf_schs=[],
        kicad_pro=tmp_path / "BOARD.kicad_pro",
        autoplacer_json=tmp_path / "BOARD_autoplacer.json",
        pcb_errors=[PcbError(
            stage="verify",
            code="unconnected",
            title="Open connections remain",
            explanation="Routing left one open connection.",
            details=["Unconnected item(s): 1"],
            counts={"unconnected": 1},
            nets=["SDA"],
            footprint_refs=["J1"],
            violations=[PcbViolation(
                type="unconnected_items", x_mm=12.3, y_mm=8.4,
                net1="SDA", footprint_refs=["J1"], description="open",
            )],
            next_action="Route SDA again.",
            overlay_path=tmp_path / "overlay.png",
        )],
    )
    restored = ArtifactPaths.model_validate_json(artifacts.model_dump_json())
    error = restored.pcb_errors[0]
    assert error.violations[0].x_mm == 12.3
    assert error.violations[0].footprint_refs == ["J1"]
    assert restored.pcb_errors[0].overlay_path == tmp_path / "overlay.png"


def test_verify_mapping_is_actionable_and_bounded():
    errors = build_pcb_errors({
        "unconnected": 1,
        "unconnected_nets": ["SDA"],
        "violations": [{
            "type": "unconnected_items", "x_mm": 12.3, "y_mm": 8.4,
            "net1": "SDA", "footprint_refs": ["J1"],
            "description": "Missing connection",
        }],
    }, stage="verify")
    assert len(errors) == 1
    assert errors[0].code == "unconnected"
    assert "Routing left 1 open connection" in errors[0].explanation
    assert errors[0].nets == ["SDA"]
    assert errors[0].footprint_refs == ["J1"]
    assert errors[0].violations[0].x_mm == 12.3


def test_empty_layout_evidence_gets_explicit_explanation():
    errors = build_pcb_errors({}, stage="place_route")
    assert errors and errors[0].code == "layout_failure"
    assert "No precise PCB location" in errors[0].explanation
    assert errors[0].next_action


def test_fr_hang_evidence_names_net_and_timeout():
    """The rc6 failure card must name the REAL cause -- FreeRouting froze on
    net VOUT_2 and was killed at the timeout -- instead of raw log fragments,
    and must carry no foreign violations or placement blame."""
    errors = build_pcb_errors({
        "evidence": {
            "parent_route_stderr_tail": (
                "  FreeRouting crash (rc=-1), retrying with 10 passes...\n"
                "  [dsn-sanitize] opened 7 locked-wire cycle(s)\n"
                "error: parent routing failed: FreeRouting hung and was killed "
                "at the 120 s timeout; its last output was: "
                "\"The normalization of net 'VOUT_2' failed.\"\n"
            ),
        },
    }, stage="place_route")
    assert len(errors) == 1
    err = errors[0]
    assert err.code == "layout_failure"
    assert "VOUT_2" in err.explanation
    assert "timeout" in err.explanation
    assert err.violations == []
    assert err.footprint_refs == []
    assert err.overlay_path is None
    assert "spread" not in err.next_action and "placement" not in err.next_action


def test_warn_only_silk_violations_never_attached():
    """A summary whose only violations are warn-only silkscreen clips (the
    KC-Z879KB case: two clips on LED D1 at (145, 118) mm, physically off the
    24x59 mm board) must produce a card with zero violations attached."""
    errors = build_pcb_errors({
        "violations": [
            {"type": "silk_clip", "x_mm": 145.0, "y_mm": 118.0,
             "footprint_refs": ["D1"],
             "description": "Silkscreen clipped by pad"},
            {"type": "silk_overlap", "x_mm": 10.0, "y_mm": 10.0,
             "footprint_refs": [], "description": "Silkscreen overlap"},
        ],
    }, stage="place_route")
    assert len(errors) == 1
    err = errors[0]
    assert err.violations == []
    assert err.footprint_refs == []
    assert "No board location exists for this failure." in err.details


def test_drc_parser_keeps_both_net_forms_and_footprints(monkeypatch, tmp_path):
    report = (
        "[shorting_items]: Items shorting two nets\n"
        "    @(12.3000 mm, 8.4000 mm): Track [Net 1](SDA) of J1\n"
        "    @(13.3000 mm, 9.4000 mm): Pad 1 [SCL] of J2\n"
        "[unconnected_items]: Missing connection\n"
        "    @(22.1000 mm, 4.2000 mm): Pad 1 [SDA] of J1\n"
    )

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        report_path = cmd[cmd.index("-o") + 1]
        Path(report_path).write_text(report, encoding="utf-8")
        return Result()

    monkeypatch.setattr(freerouting_runner.subprocess, "run", fake_run)
    parsed = freerouting_runner._run_kicad_cli_drc(str(tmp_path / "BOARD.kicad_pcb"))
    short, unconnected = parsed["violations"]
    assert (short["x_mm"], short["y_mm"]) == (12.3, 8.4)
    assert (short["net1"], short["net2"]) == ("SDA", "SCL")
    assert short["footprint_refs"] == ["J1", "J2"]
    assert unconnected["net1"] == "SDA"
    assert unconnected["footprint_refs"] == ["J1"]
    assert parsed["unconnected_nets"] == ["SDA"]


def test_overlay_arrows_are_actionable_and_capped():
    violations = [
        {"type": "unconnected_items", "x_mm": 12.3, "y_mm": 8.4,
         "net1": "SDA", "footprint_refs": []},
        {"type": "courtyards_overlap", "x_mm": 20.0, "y_mm": 10.0,
         "net1": None, "net2": None, "footprint_refs": ["J1"]},
        {"type": "drc_generic", "x_mm": 30.0, "y_mm": 10.0,
         "net1": None, "net2": None, "footprint_refs": []},
        {"type": "unconnected_items", "x_mm": None, "y_mm": None,
         "net1": "SCL", "footprint_refs": []},
    ]
    arrows = render_drc_overlay.actionable_arrow_geometry(
        violations, board_x0=0.0, board_y0=0.0, scale=10.0, ox=5.0, oy=7.0,
    )
    assert len(arrows) == 2
    assert (arrows[0]["x2"], arrows[0]["y2"]) == (128.0, 91.0)
    assert arrows[1]["head"][0] == (205.0, 107.0)


def test_overlay_command_emits_arrows_and_legend(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(render_drc_overlay, "parse_edge_cuts_aabb",
                        lambda _path: (0.0, 0.0, 40.0, 30.0))
    monkeypatch.setattr(render_drc_overlay, "render_pcb",
                        lambda *_args, **_kwargs: (0.0, 0.0, 40.0, 30.0))

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"png")

    monkeypatch.setattr(render_drc_overlay.subprocess, "run", fake_run)
    output = tmp_path / "overlay.png"
    ok = render_drc_overlay.render_overlay(
        str(tmp_path / "BOARD.kicad_pcb"),
        [
            {"type": "unconnected_items", "x_mm": 12.3, "y_mm": 8.4,
             "net1": "SDA", "footprint_refs": []},
            {"type": "courtyards_overlap", "x_mm": 20.0, "y_mm": 10.0,
             "net1": None, "net2": None, "footprint_refs": ["J1"]},
            {"type": "drc_generic", "x_mm": 25.0, "y_mm": 10.0,
             "net1": None, "net2": None, "footprint_refs": []},
        ],
        str(output),
        canvas_px=400,
    )
    assert ok and output.is_file()
    command = commands[0]
    assert sum(1 for item in command if str(item).startswith("polygon ")) == 2
    assert any("FAILURE LOCATIONS (red arrows): 2" in item for item in command)

def test_overlay_raises_when_external_tool_missing(monkeypatch, tmp_path):
    """A missing kicad-cli / ImageMagick is a loud error, never a silent no-overlay."""
    violations = [{"type": "unconnected_items", "x_mm": 1.0, "y_mm": 2.0,
                   "net1": "SDA", "footprint_refs": []}]

    # kicad-cli present, but neither ImageMagick binary available.
    monkeypatch.setattr(
        render_drc_overlay.shutil, "which",
        lambda n: "/usr/bin/kicad-cli" if n == "kicad-cli" else None)
    with pytest.raises(RuntimeError, match="ImageMagick"):
        render_drc_overlay.render_overlay(
            str(tmp_path / "B.kicad_pcb"), violations, str(tmp_path / "o.png"))

    # ImageMagick present, kicad-cli missing.
    monkeypatch.setattr(
        render_drc_overlay.shutil, "which",
        lambda n: "/usr/bin/convert" if n == "convert" else None)
    with pytest.raises(RuntimeError, match="kicad-cli"):
        render_drc_overlay.render_overlay(
            str(tmp_path / "B.kicad_pcb"), violations, str(tmp_path / "o.png"))


def test_imagemagick_bin_prefers_magick_then_convert(monkeypatch):
    from kicraft.cli.render_drc_overlay import _imagemagick_bin

    monkeypatch.setattr(
        render_drc_overlay.shutil, "which",
        lambda n: "/x/magick" if n == "magick" else None)
    assert _imagemagick_bin() == ["magick"]

    monkeypatch.setattr(
        render_drc_overlay.shutil, "which",
        lambda n: "/x/convert" if n == "convert" else None)
    assert _imagemagick_bin() == ["convert"]

    monkeypatch.setattr(render_drc_overlay.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="ImageMagick"):
        _imagemagick_bin()


def test_pcb_renderer_magick_bin_recognizes_im6(monkeypatch):
    from kicraft.render import pcb_renderer

    monkeypatch.setattr(
        pcb_renderer.shutil, "which",
        lambda n: "/usr/bin/convert" if n == "convert" else None)
    assert pcb_renderer.magick_bin() == ["convert"]
    assert pcb_renderer._magick_available() is True



def test_inspector_rejects_outside_or_non_png_overlay(tmp_path):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"png")
    section = _pcb_error_section({
        "artifacts": {
            "status": "failed",
            "pcb_errors": [{
                "stage": "verify", "code": "unconnected",
                "explanation": "Routing left one open connection.",
                "next_action": "Route again.", "overlay_path": str(outside),
            }],
        },
    }, tmp_path, [], "TOK")
    assert section is not None and section["overlay_url"] is None


def test_stage_statuses_keep_verify_board_inspectable():
    state = {"stage_status": {
        name: {"ok": True}
        for name in ("intent", "functional_spec", "architecture", "bom", "wiring")
    }, "artifacts": {"pcb_errors": [{"stage": "verify"}]}}
    statuses = derive_stage_statuses(
        state, project_status="failed", sheets_exist=True,
        pcb_ready=True, zip_ok=True,
    )
    assert statuses["place_route"] == "done"
    assert statuses["fab"] == "failed"



def test_legacy_failed_project_gets_location_unavailable_card():
    section = _pcb_error_section(
        {"artifacts": {"status": "failed"}},
        None,
        ["error: layout/route engine exited 6"],
    )
    assert section is not None
    assert section["title"] == "PCB place/route failed"
    assert "No precise PCB location" in section["explanation"]

# The UI simulation is intentionally small: it verifies the persisted state is
# enough to repaint the reopened card, without starting KiCad or a worker.
user_simulation = pytest.importorskip("nicegui.testing.user_simulation").user_simulation


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def web_harness(tmp_path):
    async with user_simulation() as user:
        mod = sys.modules.get(WEB)
        web = importlib.reload(mod) if mod else importlib.import_module(WEB)
        store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        web._STORE = store
        web._safe_fetch = lambda _key: web._FETCH_ERROR
        account = store.create_user(EMAIL, PASSWORD)
        store.record_consent(account.id, LEGAL_VERSION)
        try:
            yield user, web, store, account
        finally:
            web._STORE = None
            web._LIVE_RUNS.clear()


async def test_reopened_persisted_pcb_error_is_prominent(web_harness):
    user, _web, store, account = web_harness
    store.set_tier(EMAIL, "pro")
    pid = store.create_project(account.id, "PCB diagnostics")
    base = store.projects_dir / str(account.id) / str(pid)
    (base / ".kicraft").mkdir(parents=True)
    state = json.loads(STATE_FIXTURE.read_text(encoding="utf-8"))
    generated = base / "generated" / "USB_BMP280_READER"
    generated.mkdir(parents=True)
    (generated / "USB_BMP280_READER.kicad_sch").write_text("(sch)", encoding="utf-8")
    (generated / "USB_BMP280_READER.kicad_pcb").write_text("(pcb)", encoding="utf-8")
    overlay = generated / "USB_BMP280_READER_pcb_error_overlay.png"
    overlay.write_bytes(b"png")
    state["artifacts"] = {
        "project_dir": str(generated),
        "project_stem": "USB_BMP280_READER",
        "root_sch": str(generated / "USB_BMP280_READER.kicad_sch"),
        "leaf_schs": [],
        "kicad_pro": str(generated / "USB_BMP280_READER.kicad_pro"),
        "autoplacer_json": str(generated / "USB_BMP280_READER_autoplacer.json"),
        "status": "failed",
        "pcb_errors": [{
            "stage": "verify", "code": "unconnected",
            "title": "Open connections remain",
            "explanation": "Routing left one open connection.",
            "details": ["Unconnected item(s): 1"],
            "counts": {"unconnected": 1}, "nets": ["SDA"],
            "footprint_refs": ["J1"],
            "violations": [{"type": "unconnected_items", "x_mm": 12.3,
                            "y_mm": 8.4, "net1": "SDA", "net2": None,
                            "footprint_refs": ["J1"], "description": "open"}],
            "next_action": "Route SDA again.", "overlay_path": str(overlay),
        }],
    }
    (base / ".kicraft" / "state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    store.finish_project(pid, "failed", stem="USB_BMP280_READER", dir_path=str(base))

    await user.open("/login")
    user.find("Email").type(EMAIL)
    user.find("Password").type(PASSWORD).trigger("keydown.enter")
    await user.should_see("PCB verification failed", retries=30)
    await user.should_see("SDA", retries=10)
    await user.should_see("J1", retries=10)
    await user.should_see("Failure locations (red arrows)", retries=30)


async def test_live_terminal_run_repaints_same_pcb_error(web_harness):
    user, web, store, account = web_harness
    store.set_tier(EMAIL, "pro")
    pid = store.create_project(account.id, "Live PCB diagnostics")
    base = store.projects_dir / str(account.id) / str(pid)
    generated = base / "generated" / "LIVE"
    generated.mkdir(parents=True)
    (generated / "LIVE.kicad_sch").write_text("(sch)", encoding="utf-8")
    (generated / "LIVE.kicad_pcb").write_text("(pcb)", encoding="utf-8")
    overlay = generated / "LIVE_pcb_error_overlay.png"
    overlay.write_bytes(b"png")
    (base / ".kicraft").mkdir(parents=True)
    (base / ".kicraft" / "state.json").write_text(json.dumps({
        "project_stem": "LIVE",
        "artifacts": {
            "status": "failed", "pcb_errors": [{
                "stage": "verify", "code": "unconnected",
                "explanation": "Routing left one open connection.",
                "details": ["Unconnected item(s): 1"],
                "counts": {"unconnected": 1}, "nets": ["SDA"],
                "footprint_refs": ["J1"], "violations": [],
                "next_action": "Route SDA again.",
                "overlay_path": str(overlay),
            }],
        },
    }), encoding="utf-8")
    live = web._fresh_run_state()
    live.update(
        running=False, done=True, ok=False, user_id=account.id,
        project_id=pid, ws=str(base), project_dir=str(generated),
        stem="LIVE", pcb_ready=True, token=web._register_project_dir(generated),
        brief="Live PCB diagnostics", status="failed",
    )
    web._LIVE_RUNS[pid] = live
    await user.open("/login")
    user.find("Email").type(EMAIL)
    user.find("Password").type(PASSWORD).trigger("keydown.enter")
    await user.should_see("PCB verification failed", retries=30)
    await user.should_see("SDA", retries=10)
    await user.should_see("Failure locations (red arrows)", retries=30)
