"""Tests for fetch-3d, the model-stanza helpers, and validate-part --check-3d.

All hermetic: the EasyEDA API / importer / exporter classes are monkeypatched
where the network path would be exercised, and the CLI is invoked in-process
(``cli_app.main``) so the patches apply.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.design.cli_app import (
    _check_3d_model_paths,
    _model_stanza_paths,
    _rewrite_model_stanza,
    main,
)
from kicraft.parts_library import load_manifest, verify_content_hash

from .conftest import write_valid_part

# A footprint with the broken bare path easyeda2kicad leaves when exported
# without a model path, plus a non-trivial transform that must survive.
FOOTPRINT_BROKEN_MODEL = (
    '(footprint "WidgetFP"\n'
    "\t(version 20231120)\n"
    '\t(generator "kicraft_parts_library")\n'
    '\t(layer "F.Cu")\n'
    '\t(model "/OLD-NAME.wrl"\n'
    "\t\t(offset (xyz 0.000 0.000 0.000))\n"
    "\t\t(scale (xyz 1 1 1))\n"
    "\t\t(rotate (xyz 0 0 180))\n"
    "\t)\n"
    ")\n"
)

FOOTPRINT_NO_MODEL = (
    '(footprint "WidgetFP"\n'
    "\t(version 20231120)\n"
    '\t(generator "kicraft_parts_library")\n'
    '\t(layer "F.Cu")\n'
    ")\n"
)

FOOTPRINT_STOCK_MODEL = (
    '(footprint "WidgetFP"\n'
    "\t(version 20231120)\n"
    '\t(layer "F.Cu")\n'
    '\t(model "${KICAD9_3DMODEL_DIR}/Package_TO_SOT_SMD.3dshapes/SOT-23.step"\n'
    "\t\t(offset (xyz 0 0 0))\n"
    "\t\t(scale (xyz 1 1 1))\n"
    "\t\t(rotate (xyz 0 0 0))\n"
    "\t)\n"
    ")\n"
)


def _kiprjmod_footprint(name: str, model_file: str) -> str:
    return (
        '(footprint "WidgetFP"\n'
        "\t(version 20231120)\n"
        '\t(layer "F.Cu")\n'
        f'\t(model "${{KIPRJMOD}}/3dmodels/{name}/{model_file}"\n'
        "\t\t(offset (xyz 0 0 0))\n"
        "\t\t(scale (xyz 1 1 1))\n"
        "\t\t(rotate (xyz 0 0 0))\n"
        "\t)\n"
        ")\n"
    )


# ---------- stanza helpers ----------


def test_model_stanza_paths_finds_all():
    text = FOOTPRINT_BROKEN_MODEL + FOOTPRINT_STOCK_MODEL
    paths = _model_stanza_paths(text)
    assert paths == [
        "/OLD-NAME.wrl",
        "${KICAD9_3DMODEL_DIR}/Package_TO_SOT_SMD.3dshapes/SOT-23.step",
    ]


def test_rewrite_replaces_path_and_keeps_transform():
    new_text, n = _rewrite_model_stanza(
        FOOTPRINT_BROKEN_MODEL, "${KIPRJMOD}/3dmodels/widget/NEW.wrl"
    )
    assert n == 1
    assert '(model "${KIPRJMOD}/3dmodels/widget/NEW.wrl"' in new_text
    assert "/OLD-NAME.wrl" not in new_text
    # The original transform is untouched.
    assert "(rotate (xyz 0 0 180))" in new_text
    assert "(offset (xyz 0.000 0.000 0.000))" in new_text


def test_rewrite_appends_stanza_when_missing():
    new_text, n = _rewrite_model_stanza(
        FOOTPRINT_NO_MODEL, "${KIPRJMOD}/3dmodels/widget/NEW.wrl"
    )
    assert n == 1
    assert '(model "${KIPRJMOD}/3dmodels/widget/NEW.wrl"' in new_text
    assert "(rotate (xyz 0 0 0))" in new_text
    # Still a balanced footprint expression.
    assert new_text.count("(") == new_text.count(")")
    assert new_text.rstrip().endswith(")")


# ---------- _check_3d_model_paths ----------


def test_check_3d_accepts_stock_and_backed_kiprjmod(tmp_path: Path):
    (tmp_path / "3d").mkdir()
    (tmp_path / "3d" / "NEW.wrl").write_text("wrl")
    text = FOOTPRINT_STOCK_MODEL + _kiprjmod_footprint("widget", "NEW.wrl")
    assert _check_3d_model_paths(tmp_path, "widget", text) == []


def test_check_3d_rejects_bare_and_unbacked_paths(tmp_path: Path):
    problems = _check_3d_model_paths(tmp_path, "widget", FOOTPRINT_BROKEN_MODEL)
    assert len(problems) == 1 and "/OLD-NAME.wrl" in problems[0]
    # KIPRJMOD path with no file behind it in 3d/.
    problems = _check_3d_model_paths(
        tmp_path, "widget", _kiprjmod_footprint("widget", "GHOST.wrl")
    )
    assert len(problems) == 1 and "no backing file" in problems[0]
    # KIPRJMOD path namespaced under a different part name.
    problems = _check_3d_model_paths(
        tmp_path, "widget", _kiprjmod_footprint("other-part", "NEW.wrl")
    )
    assert len(problems) == 1 and "resolves nowhere" in problems[0]


# ---------- validate-part 3D check ----------


def test_validate_part_3d_pass(tmp_path: Path, capsys):
    part_dir = write_valid_part(
        tmp_path,
        footprint=_kiprjmod_footprint("widget", "NEW.wrl"),
        models={"NEW.wrl": "wrl", "NEW.step": "step"},
    )
    assert main(["validate-part", str(part_dir)]) == 0


def test_validate_part_3d_fails_on_broken_path(tmp_path: Path, capsys):
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_BROKEN_MODEL)
    assert main(["validate-part", str(part_dir)]) == 2
    assert "/OLD-NAME.wrl" in capsys.readouterr().err


def test_validate_part_3d_passes_without_any_stanza(tmp_path: Path):
    """No (model ...) stanza means nothing to resolve: still valid (the
    bundle just renders without a body until a model is fetched)."""
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_NO_MODEL)
    assert main(["validate-part", str(part_dir)]) == 0


# ---------- fetch-3d (hermetic fakes) ----------


class _FakeApi:
    """Records LCSC lookups; returns a sentinel CAD blob."""

    calls: list[str] = []

    def __init__(self, use_cache: bool = False):
        pass

    def get_cad_data_of_component(self, lcsc_id: str):
        _FakeApi.calls.append(lcsc_id)
        return {"fake": lcsc_id}


class _FakeEeModel:
    # Name carries characters _sanitize_kicad_name must strip.
    name = "Widget 3D#Model"


class _FakeImporter:
    output: object | None = _FakeEeModel()

    def __init__(self, easyeda_cp_cad_data, download_raw_3d_model):
        assert download_raw_3d_model is True
        self.output = _FakeImporter.output


class _FakeKiModel:
    def __init__(self, name: str):
        self.name = name


class _FakeExporter:
    def __init__(self, model_3d):
        self.output = _FakeKiModel(model_3d.name) if model_3d else None

    def export(self, output_dir: str) -> bool:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.output.name}.wrl").write_text("fake wrl")
        (out / f"{self.output.name}.step").write_text("fake step")
        return True


@pytest.fixture
def fake_easyeda(monkeypatch):
    _FakeApi.calls = []
    _FakeImporter.output = _FakeEeModel()
    monkeypatch.setattr(
        "easyeda2kicad.easyeda.easyeda_api.EasyedaApi", _FakeApi
    )
    monkeypatch.setattr(
        "easyeda2kicad.easyeda.easyeda_importer.Easyeda3dModelImporter",
        _FakeImporter,
    )
    monkeypatch.setattr(
        "easyeda2kicad.kicad.export_kicad_3d_model.Exporter3dModelKicad",
        _FakeExporter,
    )
    return _FakeApi


def test_fetch_3d_downloads_and_rewires_bundle(
    tmp_path: Path, fake_easyeda, capsys
):
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_BROKEN_MODEL)
    assert main(["fetch-3d", str(part_dir)]) == 0
    assert fake_easyeda.calls == ["C999999"]  # from make_valid_manifest

    # Files written with the sanitized name (space and '#' stripped).
    assert (part_dir / "3d" / "Widget3DModel.wrl").is_file()
    assert (part_dir / "3d" / "Widget3DModel.step").is_file()
    assert not (part_dir / "3d" / "Widget 3D#Model.wrl").exists()

    fp_text = (
        part_dir / "widget.pretty" / "WidgetFP.kicad_mod"
    ).read_text()
    assert (
        '(model "${KIPRJMOD}/3dmodels/widget/Widget3DModel.wrl"' in fp_text
    )
    assert "(rotate (xyz 0 0 180))" in fp_text  # transform preserved

    # Hash was re-blessed in the same run, and a renamed model warns.
    manifest = load_manifest(part_dir)
    assert verify_content_hash(part_dir, manifest)
    assert "model name changed" in capsys.readouterr().err
    # The result now passes the strict 3D validation.
    assert main(["validate-part", str(part_dir)]) == 0


def test_fetch_3d_skips_already_fetched(tmp_path: Path, fake_easyeda, capsys):
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_BROKEN_MODEL)
    assert main(["fetch-3d", str(part_dir)]) == 0
    assert main(["fetch-3d", str(part_dir)]) == 0
    # The second run classified it as already-fetched: one network call total.
    assert fake_easyeda.calls == ["C999999"]
    assert "already has 3D" in capsys.readouterr().out


def test_fetch_3d_skips_stock_reference(tmp_path: Path, fake_easyeda, capsys):
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_STOCK_MODEL)
    assert main(["fetch-3d", str(part_dir)]) == 0
    assert fake_easyeda.calls == []
    assert "stock KiCad model" in capsys.readouterr().out
    assert not (part_dir / "3d").exists()


def test_fetch_3d_no_model_available(tmp_path: Path, fake_easyeda, capsys):
    _FakeImporter.output = None
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_BROKEN_MODEL)
    before = (part_dir / "widget.pretty" / "WidgetFP.kicad_mod").read_text()
    assert main(["fetch-3d", str(part_dir)]) == 0
    assert "no 3D model on EasyEDA" in capsys.readouterr().out
    # Nothing was changed: footprint intact, hash still valid.
    after = (part_dir / "widget.pretty" / "WidgetFP.kicad_mod").read_text()
    assert after == before
    assert verify_content_hash(part_dir, load_manifest(part_dir))


def test_fetch_3d_report_is_offline_and_readonly(tmp_path: Path, capsys):
    # No fakes installed: --report must not touch the network at all.
    part_dir = write_valid_part(tmp_path, footprint=FOOTPRINT_BROKEN_MODEL)
    manifest_before = (part_dir / "manifest.json").read_bytes()
    assert main(["fetch-3d", "--report", str(part_dir)]) == 0
    assert "needs fetch" in capsys.readouterr().out
    assert (part_dir / "manifest.json").read_bytes() == manifest_before


def test_fetch_3d_requires_input(capsys):
    assert main(["fetch-3d"]) == 2
    assert "no part directories" in capsys.readouterr().err
