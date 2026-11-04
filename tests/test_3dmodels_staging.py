"""Tests for synthesis 3D-model staging (kicraft.design.synthesis.models3d).

Bundle footprints reference models as ``${KIPRJMOD}/3dmodels/<lib>/<file>``;
``stage_3d_models`` copies each used bundle's ``3d/`` files into the
generated project so that reference resolves. The pcbnew test proves the
full chain: a board built by ``write_empty_pcb`` embeds the bundle's stanza
verbatim AND the staged file sits exactly where ``${KIPRJMOD}`` points.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.design.synthesis.models3d import stage_3d_models

WIDGET_FOOTPRINT = (
    '(footprint "WidgetFP"\n'
    "\t(version 20231120)\n"
    '\t(generator "test")\n'
    '\t(layer "F.Cu")\n'
    '\t(model "${KIPRJMOD}/3dmodels/widget/WidgetBody.wrl"\n'
    "\t\t(offset (xyz 0 0 0))\n"
    "\t\t(scale (xyz 1 1 1))\n"
    "\t\t(rotate (xyz 0 0 0))\n"
    "\t)\n"
    ")\n"
)


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch) -> Path:
    """A project root whose project tier ships a 'widget' bundle with models.

    The resolver only needs ``<root>/.kicraft/parts/widget/widget.pretty/``
    to exist; no manifest is required for path resolution.
    """
    monkeypatch.delenv("KICRAFT_EXTRA_PARTS_DIRS", raising=False)
    root = tmp_path / "proj"
    bundle = root / ".kicraft" / "parts" / "widget"
    pretty = bundle / "widget.pretty"
    pretty.mkdir(parents=True)
    (pretty / "WidgetFP.kicad_mod").write_text(WIDGET_FOOTPRINT)
    model_dir = bundle / "3d"
    model_dir.mkdir()
    (model_dir / "WidgetBody.wrl").write_text("fake wrl")
    (model_dir / "WidgetBody.step").write_text("fake step")
    (model_dir / "notes.txt").write_text("not a model")
    return root


def _bom(*footprints: str):
    return SimpleNamespace(
        parts=[SimpleNamespace(footprint=f) for f in footprints]
    )


def test_stages_bundle_models_into_project(project_root: Path, tmp_path: Path):
    out = tmp_path / "generated"
    out.mkdir()
    staged = stage_3d_models(
        out, _bom("widget:WidgetFP"), project_root=project_root
    )
    assert sorted(p.name for p in staged) == [
        "WidgetBody.step", "WidgetBody.wrl",
    ]
    # Exactly where the stanza's ${KIPRJMOD} reference points, and nothing
    # besides model files came along.
    assert (out / "3dmodels" / "widget" / "WidgetBody.wrl").is_file()
    assert not (out / "3dmodels" / "widget" / "notes.txt").exists()


def test_unknown_and_stock_libraries_stage_nothing(
    project_root: Path, tmp_path: Path
):
    out = tmp_path / "generated"
    out.mkdir()
    # 'no-such-lib' resolves nowhere; stock libs (if present on this host)
    # have no sibling 3d/ dir. Neither creates a 3dmodels/ dir.
    staged = stage_3d_models(
        out,
        _bom("no-such-lib:Foo", "Resistor_SMD:R_0603_1608Metric"),
        project_root=project_root,
    )
    assert staged == []
    assert not (out / "3dmodels").exists()


def test_none_bom_is_noop(tmp_path: Path):
    assert stage_3d_models(tmp_path, None, project_root=tmp_path) == []


def test_untracked_3d_model_is_not_overwritten(project_root: Path, tmp_path: Path):
    out = tmp_path / "generated"
    target = out / "3dmodels" / "widget" / "WidgetBody.step"
    target.parent.mkdir(parents=True)
    target.write_text("my model")
    with pytest.raises(ValueError, match="untracked or modified"):
        stage_3d_models(out, _bom("widget:WidgetFP"), project_root=project_root)
    assert target.read_text() == "my model"


def test_tracked_model_can_update_but_user_edits_are_preserved(project_root: Path, tmp_path: Path):
    out = tmp_path / "generated"
    out.mkdir()
    bom = _bom("widget:WidgetFP")
    paths = stage_3d_models(out, bom, project_root=project_root)
    owned = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    source = project_root / ".kicraft" / "parts" / "widget" / "3d" / "WidgetBody.step"
    source.write_text("new bundle model")
    stage_3d_models(out, bom, project_root=project_root, owned_files=owned)
    target = out / "3dmodels" / "widget" / "WidgetBody.step"
    assert target.read_text() == "new bundle model"
    target.write_text("user edited model")
    with pytest.raises(ValueError, match="untracked or modified"):
        stage_3d_models(out, bom, project_root=project_root, owned_files=owned)
    assert target.read_text() == "user edited model"

def test_attribution_stages_with_model_and_is_hash_protected(project_root: Path, tmp_path: Path):
    from zipfile import ZipFile
    from kicraft.server.web import _zip_generated

    source = project_root / ".kicraft" / "parts" / "widget" / "3d"
    (source / "LICENSE.txt").write_text("model attribution and license")
    (source / "WidgetExtra.STEP").write_text("uppercase model")
    out = tmp_path / "generated"
    out.mkdir()
    written = []
    paths = stage_3d_models(out, _bom("widget:WidgetFP"), project_root=project_root,
                            on_written=written.append)
    license_path = out / "3dmodels" / "widget" / "LICENSE.txt"
    assert license_path.read_text() == "model attribution and license"
    assert (license_path.parent / "WidgetExtra.STEP").read_text() == "uppercase model"
    assert not (license_path.parent / "notes.txt").exists()
    owned = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with ZipFile(_zip_generated(tmp_path, out)) as archive:
        assert archive.read("generated/3dmodels/widget/LICENSE.txt") == b"model attribution and license"
        assert archive.read("generated/3dmodels/widget/WidgetExtra.STEP") == b"uppercase model"
    license_path.write_text("user edited attribution")
    with pytest.raises(ValueError, match="untracked or modified"):
        stage_3d_models(out, _bom("widget:WidgetFP"), project_root=project_root,
                        owned_files=owned)
    assert license_path.read_text() == "user edited attribution"


def test_untracked_attribution_is_not_overwritten(project_root: Path, tmp_path: Path):
    source = project_root / ".kicraft" / "parts" / "widget" / "3d"
    (source / "LICENSE.txt").write_text("bundle license")
    out = tmp_path / "generated"
    target = out / "3dmodels" / "widget" / "LICENSE.txt"
    target.parent.mkdir(parents=True)
    target.write_text("user attribution")
    with pytest.raises(ValueError, match="untracked or modified"):
        stage_3d_models(out, _bom("widget:WidgetFP"), project_root=project_root)
    assert target.read_text() == "user attribution"





def test_partial_staging_reports_completed_models_only(project_root: Path, tmp_path: Path):
    out = tmp_path / "generated"
    target = out / "3dmodels" / "widget" / "WidgetBody.wrl"
    target.parent.mkdir(parents=True)
    target.write_text("user model")
    completed = []
    with pytest.raises(ValueError, match="untracked or modified"):
        stage_3d_models(out, _bom("widget:WidgetFP"), project_root=project_root,
                        on_written=completed.append)
    assert [path.name for path in completed] == ["WidgetBody.step"]
    assert target.read_text() == "user model"


def test_board_embeds_stanza_and_staged_file_matches(
    project_root: Path, tmp_path: Path
):
    """Full chain: write_empty_pcb embeds the ${KIPRJMOD} stanza into the
    saved board, and staging puts the file where that stanza resolves."""
    pcbnew = pytest.importorskip("pcbnew")

    from kicraft.design.models import BOM, BomPart, NetConnection, PinEndpoint
    from kicraft.design.synthesis.kicad_pcb_stub import write_empty_pcb

    bom = BOM(
        parts=[
            BomPart(ref="U1", value="widget", symbol="widget:Widget",
                    footprint="widget:WidgetFP", sheet="MAIN"),
            BomPart(ref="R1", value="10k", symbol="Device:R",
                    footprint="Resistor_SMD:R_0805_2012Metric", sheet="MAIN"),
            BomPart(ref="R2", value="10k", symbol="Device:R",
                    footprint="Resistor_SMD:R_0805_2012Metric", sheet="MAIN"),
        ],
        connections=[
            NetConnection(net_name="N1", sheet="MAIN",
                          endpoints=[PinEndpoint(ref="R1", pin="1"),
                                     PinEndpoint(ref="R2", pin="1")]),
        ],
    )
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    from kicraft.design.synthesis.footprint_library import (
        FootprintNotFoundError, lookup_footprint,
    )
    try:
        lookup_footprint("Resistor_SMD:R_0805_2012Metric", project_root=project_root)
    except FootprintNotFoundError as exc:
        pytest.skip(f"stock resistor prerequisite unavailable: {exc}")
    out = write_empty_pcb(out_dir, "TEST", bom, project_root=project_root)
    stage_3d_models(out_dir, bom, project_root=project_root)

    board = pcbnew.LoadBoard(str(out))
    u1 = board.FindFootprintByReference("U1")
    assert u1 is not None
    models = list(u1.Models())
    assert len(models) == 1
    model_path = models[0].m_Filename
    assert model_path == "${KIPRJMOD}/3dmodels/widget/WidgetBody.wrl"
    # ${KIPRJMOD} resolves to the board's directory.
    assert (out.parent / "3dmodels" / "widget" / "WidgetBody.wrl").is_file()


def test_step_coverage_distinguishes_nested_names_and_missing_bodies(tmp_path):
    from kicraft.design.synthesis.models3d import collect_step_coverage
    step = tmp_path / "board.step"
    step.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#10 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','U1','',#1,#2,$);\n"
        "#11 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('2','U2','',#2,#3,$);\n"
        "#12 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('3','PCB','',#1,#4,$);\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )
    assert collect_step_coverage(step, ["U1", "U2"]) == {
        "status": "observed", "observed_refs": ["U1"], "missing_refs": ["U2"],
    }


def test_step_coverage_unknown_for_truncated_or_ambiguous_assembly(tmp_path):
    from kicraft.design.synthesis.models3d import collect_step_coverage
    step = tmp_path / "board.step"
    step.write_text("ISO-10303-21;\nDATA;\n")
    assert collect_step_coverage(step, ["U1"])["missing_refs"] is None
    step.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#10 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('1','U1','',#1,#2,$);\n"
        "#11 = NEXT_ASSEMBLY_USAGE_OCCURRENCE('2','U2','',#3,#4,$);\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )
    assert collect_step_coverage(step, ["U1"])["status"] == "unknown"
