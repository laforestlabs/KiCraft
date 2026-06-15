"""Discover the frozen corpus of synthesized workspaces + brief-level split.

A workspace is any directory holding exactly one ``<stem>.kicad_{pro,pcb,sch}``
triple (same shape ``ab_compose``/``replay_corpus`` detect). The tuner re-runs
place+route on these at $0 LLM cost.

Overfitting guard: the train/holdout split is **by brief**, not by workspace, so
re-syntheses of the same brief never straddle the split. A ``manifest.json`` in
the corpus root records each workspace's brief + split; absent that, the brief
is recovered from ``.kicraft/state.json`` / ``brief.txt`` and the split is hashed
deterministically.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kicraft.tuning.workspace import PATH_TOKEN, discover_stem

MANIFEST_NAME = "manifest.json"


@dataclass
class Workspace:
    path: Path
    name: str          # directory name; used as the EvalResult `board` key
    stem: str          # the .kicad_pro/.pcb/.sch stem
    brief: str = ""    # natural-language brief (for brief-level splitting)
    split: str = ""    # "train" | "holdout" | ""


def _is_workspace(d: Path) -> str | None:
    try:
        return discover_stem(d)
    except Exception:  # noqa: BLE001
        return None


def _recover_brief(d: Path) -> str:
    """Best-effort brief text from a workspace, for splitting/grouping."""
    state = d / ".kicraft" / "state.json"
    if state.is_file():
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
            for key in ("brief", "prompt", "intent"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except Exception:  # noqa: BLE001
            pass
    for fname in ("brief.txt", "prompt.txt"):
        f = d / fname
        if f.is_file():
            try:
                t = f.read_text(encoding="utf-8").strip()
                if t:
                    return t
            except OSError:
                pass
    return ""


def discover_corpus(roots: Sequence[str | Path]) -> list[Workspace]:
    """All synthesized workspaces under ``roots`` (recurses one level via
    ``generated/`` like self-eval runs, and also accepts direct workspace dirs).
    Honors a ``manifest.json`` in each root for brief/split when present."""
    out: list[Workspace] = []
    seen: set[str] = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        manifest = _load_manifest(root)
        for d in _candidate_dirs(root):
            stem = _is_workspace(d)
            if stem is None or d.name in seen:
                continue
            seen.add(d.name)
            entry = manifest.get(d.name, {}) if manifest else {}
            brief = entry.get("brief") or _recover_brief(d)
            split = entry.get("split", "")
            out.append(Workspace(path=d, name=d.name, stem=stem,
                                  brief=brief, split=split))
    return sorted(out, key=lambda w: w.name)


def _candidate_dirs(root: Path):
    """Yield dirs that might be workspaces: the root's children, plus one hop
    through ``generated/`` (self-eval layout: run_NN/generated/<STEM>/)."""
    yield root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        yield child
        gen = child / "generated"
        if gen.is_dir():
            for g in sorted(p for p in gen.iterdir() if p.is_dir()):
                yield g


def split_by_brief(
    workspaces: Sequence[Workspace], *, holdout_frac: float = 0.3, seed: int = 0
) -> None:
    """Assign ``.split`` in place, holding out whole briefs (not workspaces).

    Deterministic: a brief goes to holdout iff its salted hash falls in the
    bottom ``holdout_frac`` of the unit interval. Workspaces with no recovered
    brief are grouped individually (by name) so each is split on its own."""
    for w in workspaces:
        key = w.brief.strip() or f"__noname__:{w.name}"
        h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        w.split = "holdout" if frac < holdout_frac else "train"


def train(workspaces: Sequence[Workspace]) -> list[Workspace]:
    return [w for w in workspaces if w.split == "train"]


def holdout(workspaces: Sequence[Workspace]) -> list[Workspace]:
    return [w for w in workspaces if w.split == "holdout"]


# --- manifest -------------------------------------------------------------

def _load_manifest(root: Path) -> dict[str, dict]:
    f = root / MANIFEST_NAME
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    entries = data.get("workspaces", data) if isinstance(data, dict) else {}
    return {k: v for k, v in entries.items() if isinstance(v, dict)}


def write_manifest(root: str | Path, workspaces: Sequence[Workspace]) -> Path:
    root = Path(root)
    payload = {
        "workspaces": {
            w.name: {"brief": w.brief, "split": w.split, "stem": w.stem}
            for w in workspaces
        }
    }
    f = root / MANIFEST_NAME
    f.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return f


# --- freeze (Phase 0): snapshot synthesized runs into a relocatable corpus ---

def freeze_workspace(
    src: str | Path, dest: str | Path, *, brief: str = "", split: str = "",
    lean: bool = False,
) -> Workspace:
    """Copy a synthesized workspace to ``dest`` and tokenize its frozen-leaf
    artifacts so the corpus is relocatable.

    The synthesized ``.experiments`` JSONs bake in the source project's absolute
    path; we replace it with ``PATH_TOKEN`` (the same convention the committed
    fixtures use), which ``workspace.prepare_scratch`` rewrites to the scratch
    dir at eval time. The result is a board the tuner can re-place/re-route from
    any location at $0 LLM.

    ``lean=True`` drops the bulky ``.experiments`` tree (routed leaves, DSN
    files): in replay mode the tuner regenerates leaves from the schematic, so a
    lean freeze (schematics + seed PCB + project) is all it needs and is small
    enough to commit. Lean freezes need no tokenization (nothing references the
    source path)."""
    src, dest = Path(src), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    if lean:
        shutil.copytree(src, dest,
                        ignore=shutil.ignore_patterns(".experiments"))
    else:
        shutil.copytree(src, dest)
        src_abs = str(src.resolve())
        exp = dest / ".experiments"
        if exp.is_dir():
            for jf in exp.rglob("*.json"):
                try:
                    text = jf.read_text(encoding="utf-8")
                except OSError:
                    continue
                if src_abs in text:
                    jf.write_text(text.replace(src_abs, PATH_TOKEN),
                                  encoding="utf-8")
    return Workspace(path=dest, name=dest.name, stem=discover_stem(dest),
                     brief=brief, split=split)


def freeze_corpus(
    run_roots: Sequence[str | Path], dest_root: str | Path, *,
    holdout_frac: float = 0.3, split_seed: int = 0,
) -> list[Workspace]:
    """Discover synthesized workspaces under ``run_roots`` (e.g. self-eval batch
    dirs), assign a brief-level train/holdout split, freeze each into
    ``dest_root/<name>/``, and write the corpus ``manifest.json``."""
    src_ws = discover_corpus(run_roots)
    if not src_ws:
        raise RuntimeError(f"no synthesized workspaces under {list(run_roots)}")
    split_by_brief(src_ws, holdout_frac=holdout_frac, seed=split_seed)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    frozen = [
        freeze_workspace(w.path, dest_root / w.name, brief=w.brief, split=w.split)
        for w in src_ws
    ]
    write_manifest(dest_root, frozen)
    return frozen


def corpus_stats(workspaces: Sequence[Workspace]) -> dict:
    briefs = {w.brief.strip() or w.name for w in workspaces}
    return {
        "n_workspaces": len(workspaces),
        "n_distinct_briefs": len(briefs),
        "n_train": len(train(workspaces)),
        "n_holdout": len(holdout(workspaces)),
    }
