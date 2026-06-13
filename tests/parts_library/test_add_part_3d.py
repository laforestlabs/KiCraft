"""Tests for the add-part --from-lcsc 3D-model flow.

Hermetic: every easyeda2kicad class the LCSC path touches is monkeypatched
and the CLI runs in-process (``cli_app.main``). The fakes mirror the real
library's contract: the footprint exporter emits a ``(model ...)`` stanza
iff ``output.model_3d`` is not None, with the path joined as
``{model_3d_path}/{name}.wrl``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.design.cli_app import main
from kicraft.parts_library import (
    load_manifest,
    project_parts_dir,
    verify_content_hash,
)

from .conftest import VALID_SYMBOL

MODEL_RAW_NAME = "Widget 3D#Model"
MODEL_SAFE_NAME = "Widget3DModel"


class _FakeApi:
    def __init__(self, use_cache: bool = False):
        pass

    def get_cad_data_of_component(self, lcsc_id: str):
        return {"lcsc": lcsc_id}


class _FakeSymbolImporter:
    def __init__(self, easyeda_cp_cad_data):
        pass

    def get_symbol(self):
        return SimpleNamespace(
            info=SimpleNamespace(
                name="Widget",
                mpn="WIDGET-3D",
                manufacturer="ACME",
                datasheet=None,
                description="A widget.",
            )
        )


class _FakeFootprintImporter:
    def __init__(self, easyeda_cp_cad_data):
        pass

    def get_footprint(self):
        return SimpleNamespace(info=SimpleNamespace(name="WidgetFP"))


class _FakeSymbolExporter:
    def __init__(self, symbol, lib_path, custom_fields):
        pass

    def save_to_lib(self, lib_path, footprint_lib_name, overwrite):
        Path(lib_path).write_text(VALID_SYMBOL)
        return True


class _FakeFootprintExporter:
    """Mirrors ExporterFootprintKicad's stanza contract."""

    def __init__(self, footprint):
        self.output = SimpleNamespace(
            model_3d=SimpleNamespace(
                name=MODEL_RAW_NAME,
                translation=SimpleNamespace(x=0, y=0, z=0),
                rotation=SimpleNamespace(x=0, y=0, z=180),
            )
        )

    def export(self, footprint_full_path, model_3d_path):
        body = (
            '(footprint "WidgetFP"\n'
            "\t(version 20231120)\n"
            '\t(layer "F.Cu")\n'
        )
        m = self.output.model_3d
        if m is not None:
            body += (
                f'\t(model "{model_3d_path}/{m.name}.wrl"\n'
                f"\t\t(offset (xyz {m.translation.x} {m.translation.y} "
                f"{m.translation.z}))\n"
                "\t\t(scale (xyz 1 1 1))\n"
                f"\t\t(rotate (xyz {m.rotation.x} {m.rotation.y} "
                f"{m.rotation.z}))\n"
                "\t)\n"
            )
        Path(footprint_full_path).write_text(body + ")\n")


class _Fake3dImporter:
    output: object | None = SimpleNamespace(name=MODEL_RAW_NAME)
    raise_exc: Exception | None = None

    def __init__(self, easyeda_cp_cad_data, download_raw_3d_model):
        if _Fake3dImporter.raise_exc is not None:
            raise _Fake3dImporter.raise_exc
        self.output = _Fake3dImporter.output


class _Fake3dExporter:
    def __init__(self, model_3d):
        self.output = (
            SimpleNamespace(name=model_3d.name) if model_3d else None
        )

    def export(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.output.name}.wrl").write_text("fake wrl")
        (out / f"{self.output.name}.step").write_text("fake step")
        return True


@pytest.fixture
def fake_easyeda(monkeypatch):
    _Fake3dImporter.output = SimpleNamespace(name=MODEL_RAW_NAME)
    _Fake3dImporter.raise_exc = None
    for target, fake in [
        ("easyeda2kicad.easyeda.easyeda_api.EasyedaApi", _FakeApi),
        (
            "easyeda2kicad.easyeda.easyeda_importer.EasyedaSymbolImporter",
            _FakeSymbolImporter,
        ),
        (
            "easyeda2kicad.easyeda.easyeda_importer.EasyedaFootprintImporter",
            _FakeFootprintImporter,
        ),
        (
            "easyeda2kicad.easyeda.easyeda_importer.Easyeda3dModelImporter",
            _Fake3dImporter,
        ),
        (
            "easyeda2kicad.kicad.export_kicad_symbol.ExporterSymbolKicad",
            _FakeSymbolExporter,
        ),
        (
            "easyeda2kicad.kicad.export_kicad_footprint.ExporterFootprintKicad",
            _FakeFootprintExporter,
        ),
        (
            "easyeda2kicad.kicad.export_kicad_3d_model.Exporter3dModelKicad",
            _Fake3dExporter,
        ),
    ]:
        monkeypatch.setattr(target, fake)


@pytest.fixture
def project(monkeypatch, tmp_path: Path) -> Path:
    proj = tmp_path / "project"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def _add(*extra: str) -> int:
    return main(
        ["add-part", "--from-lcsc", "C42", "--name", "widget-three", *extra]
    )


def _bundle(project: Path) -> Path:
    return project_parts_dir(project) / "widget-three"


def test_add_part_fetches_3d_by_default(
    isolated_home, clean_extras_env, fake_easyeda, project
):
    assert _add() == 0
    part_dir = _bundle(project)

    # Model files on disk under the sanitized name.
    assert (part_dir / "3d" / f"{MODEL_SAFE_NAME}.wrl").is_file()
    assert (part_dir / "3d" / f"{MODEL_SAFE_NAME}.step").is_file()
    assert not (part_dir / "3d" / f"{MODEL_RAW_NAME}.wrl").exists()

    # Stanza points at the sanitized file, transform preserved.
    fp_text = (
        part_dir / "widget-three.pretty" / "WidgetFP.kicad_mod"
    ).read_text()
    assert (
        '(model "${KIPRJMOD}/3dmodels/widget-three/'
        f'{MODEL_SAFE_NAME}.wrl"'
    ) in fp_text
    assert "(rotate (xyz 0 0 180))" in fp_text

    # The bundle passes strict validation (3D paths + hash).
    assert verify_content_hash(part_dir, load_manifest(part_dir))
    assert main(["validate-part", str(part_dir)]) == 0


def test_add_part_no_3d_flag_skips_fetch_and_stanza(
    isolated_home, clean_extras_env, fake_easyeda, project
):
    assert _add("--no-3d") == 0
    part_dir = _bundle(project)
    assert not (part_dir / "3d").exists()
    fp_text = (
        part_dir / "widget-three.pretty" / "WidgetFP.kicad_mod"
    ).read_text()
    assert "(model " not in fp_text
    assert main(["validate-part", str(part_dir)]) == 0


def test_add_part_no_model_available(
    isolated_home, clean_extras_env, fake_easyeda, project, capsys
):
    _Fake3dImporter.output = None
    assert _add() == 0
    part_dir = _bundle(project)
    assert not (part_dir / "3d").exists()
    fp_text = (
        part_dir / "widget-three.pretty" / "WidgetFP.kicad_mod"
    ).read_text()
    assert "(model " not in fp_text
    assert "no 3D model for C42" in capsys.readouterr().err
    assert main(["validate-part", str(part_dir)]) == 0


def test_add_part_3d_fetch_error_does_not_fail_fetch(
    isolated_home, clean_extras_env, fake_easyeda, project, capsys
):
    _Fake3dImporter.raise_exc = RuntimeError("boom")
    assert _add() == 0
    part_dir = _bundle(project)
    fp_text = (
        part_dir / "widget-three.pretty" / "WidgetFP.kicad_mod"
    ).read_text()
    assert "(model " not in fp_text
    assert "3D model fetch failed for C42" in capsys.readouterr().err
    assert main(["validate-part", str(part_dir)]) == 0
