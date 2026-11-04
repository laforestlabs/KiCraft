from types import SimpleNamespace

from scripts import replay_corpus
from scripts import promote_to_sample


def _workspace(root):
    ws = root / "board"
    ws.mkdir()
    for suffix in (".kicad_pro", ".kicad_pcb", ".kicad_sch"):
        (ws / f"valid{suffix}").touch()
    return ws


def test_update_reports_replay_failure(tmp_path, monkeypatch):
    _workspace(tmp_path)

    def fail(*args):
        raise RuntimeError("replay failed")

    monkeypatch.setattr(replay_corpus, "_run_leaf", fail)
    assert replay_corpus.main(["--root", str(tmp_path), "--mode", "leaf", "--update"]) == 1
    assert not (tmp_path / "board.leaf.golden.json").exists()


def test_replay_selects_complete_project_stem(tmp_path, monkeypatch):
    ws = tmp_path / "board"
    ws.mkdir()
    (ws / "orphan.kicad_pro").touch()
    for suffix in (".kicad_pro", ".kicad_pcb", ".kicad_sch"):
        (ws / f"valid{suffix}").touch()
    seen = []
    monkeypatch.setattr(replay_corpus, "_check_mode",
                        lambda mode, ws, stem, scratch, root, update: seen.append(stem) or False)
    assert replay_corpus.main(["--root", str(tmp_path), "--mode", "parent"]) == 0
    assert seen == ["valid"]


def test_leaf_replay_pins_subprocess_environment(tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    calls = []
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    monkeypatch.setattr(replay_corpus.subprocess, "run",
                        lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(replay_corpus, "_leaf_geometry", lambda dest: {})
    replay_corpus._run_leaf(ws, scratch)
    assert all(calls[0]["env"][k] == v for k, v in replay_corpus.PINNED_ENV.items())


def test_model_resolver_reports_missing_project_model(tmp_path, monkeypatch):
    path = "${KIPRJMOD}/3dmodels/missing/foo.wrl"
    (tmp_path / "board.kicad_pcb").write_text(f'(model "{path}")')
    monkeypatch.setattr(promote_to_sample, "_MODEL_ROOTS", ())
    assert promote_to_sample._resolve_model_refs(tmp_path, "board") == (0, [path])


def test_model_resolver_preserves_existing_colliding_basename(tmp_path, monkeypatch):
    for lib in ("a", "b"):
        directory = tmp_path / "3dmodels" / lib
        directory.mkdir(parents=True)
        (directory / "foo.wrl").write_text(lib)
    path = "${KIPRJMOD}/3dmodels/a/foo.wrl"
    pcb = tmp_path / "board.kicad_pcb"
    original = f'(model "{path}")'
    pcb.write_text(original)
    monkeypatch.setattr(promote_to_sample, "_MODEL_ROOTS", ())
    assert promote_to_sample._resolve_model_refs(tmp_path, "board") == (0, [])
    assert pcb.read_text() == original
