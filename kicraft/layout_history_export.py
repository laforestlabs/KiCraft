"""Recover complete gzip members and validate version-one layout histories."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator
import zlib
import argparse
import base64
import fcntl
import hashlib
import os
import re
import time
import uuid
import warnings

from kicraft.layout_history import SCHEMA_VERSION, atomic_json, json_bytes


@dataclass
class RecoveredScope:
    header: dict = field(default_factory=dict)
    objects: dict[str, dict] = field(default_factory=dict)
    templates: dict[str, dict] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    last_seq: int = 0
    ended: bool = False
    members: list[dict] = field(default_factory=list)
    phases: dict[str, int] = field(default_factory=dict)
    baseline_seq: int | None = None


def iter_members(path: Path) -> Iterator[tuple[int, bytes, bytes]]:
    """Yield (file offset, compressed member, decoded records). Reject bad tails."""
    with path.open("rb") as stream:
        pending = b""
        offset = 0
        while True:
            if not pending:
                pending = stream.read(65536)
                if not pending:
                    return
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            compressed = bytearray()
            decoded = bytearray()
            while not decoder.eof:
                if not pending:
                    pending = stream.read(65536)
                    if not pending:
                        raise ValueError("truncated gzip member; buffered tail unavailable")
                try:
                    decoded.extend(decoder.decompress(pending))
                except zlib.error as error:
                    raise ValueError("corrupt gzip member") from error
                consumed = len(pending) - len(decoder.unused_data)
                compressed.extend(pending[:consumed])
                pending = decoder.unused_data
            yield offset, bytes(compressed), bytes(decoded)
            offset += len(compressed)


def apply_record(scope: RecoveredScope, record: dict, *, capture_id: str | None = None) -> None:
    kind = record.get("kind")
    if not scope.header:
        if kind != "header" or record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported or missing header/schema")
        if capture_id and record.get("capture_id") != capture_id:
            raise ValueError("capture identity mismatch")
        scope.header = record
        return
    if scope.ended:
        raise ValueError("record after scope end")
    if record.get("seq") != scope.last_seq + 1:
        raise ValueError("out-of-order sequence")
    if record.get("gap"):
        scope.gaps.append(str(record["gap"]))
    scope.gaps.extend(str(gap) for gap in record.get("gaps", []))
    if kind == "define":
        key, geometry = record["template_id"], record["geometry"]
        if key in scope.templates and scope.templates[key] != geometry:
            raise ValueError("immutable template redefined")
        scope.templates[key] = geometry
    elif kind in ("scene", "delta"):
        objects = record.get("objects", []) if kind == "scene" else record.get("put", [])
        remove = record.get("remove", [])
        existing = {} if kind == "scene" else scope.objects
        if len(set(remove)) != len(remove) or any(key not in existing for key in remove):
            raise ValueError("unknown or duplicate object removal")
        for obj in objects:
            if obj.get("template_id") and obj["template_id"] not in scope.templates:
                raise ValueError("missing template: " + obj["template_id"])
            pose = obj.get("pose")
            if pose is not None and not all(isinstance(pose.get(key), (int, float)) for key in ("x", "y", "rotation")):
                raise ValueError("invalid transform")
        if kind == "scene":
            scope.objects.clear()
            if scope.baseline_seq is None:
                scope.baseline_seq = record["seq"]
        for key in remove:
            del scope.objects[key]
        for obj in objects:
            scope.objects[obj["id"]] = obj
    elif kind == "end":
        scope.ended = True
    elif kind != "decision":
        raise ValueError("unknown record kind: " + str(kind))
    scope.last_seq = record["seq"]
    if record.get("phase"):
        scope.phases.setdefault(record["phase"], scope.last_seq)


def recover_scope(path: Path, *, capture_id: str | None = None) -> RecoveredScope:
    scope = RecoveredScope()
    try:
        for offset, compressed, data in iter_members(path):
            first = scope.last_seq + 1
            corrupt = False
            for line in data.splitlines():
                try:
                    apply_record(scope, json.loads(line), capture_id=capture_id)
                except (ValueError, KeyError, TypeError) as error:
                    scope.gaps.append(str(error))
                    corrupt = True
                    break
            scope.members.append({"offset": offset, "bytes": len(compressed), "first_seq": first,
                                  "last_seq": scope.last_seq})
            if corrupt:
                break
    except (ValueError, KeyError, TypeError, OSError) as error:
        scope.gaps.append(str(error))
    if not scope.ended:
        scope.gaps.append("scope incomplete: missing end; last usable step " + str(scope.last_seq))
    return scope


def _capture_dir(project_dir: Path, capture_id: str) -> Path:
    if not isinstance(capture_id, str) or not re.fullmatch(r"[0-9a-f]{32}", capture_id):
        raise ValueError("invalid capture identity")
    root = project_dir.resolve()
    directory = root / "layout-history" / capture_id
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError("capture outside project or missing")
    return directory


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("invalid manifest")
    return value


def published_capture(project_dir: Path, capture_id: str | None = None) -> tuple[dict, Path] | None:
    """Resolve fixed files with ownership, containment and digest checks."""
    try:
        root = project_dir.resolve()
        pointer = _load_json(root / "layout-history.json") if capture_id is None else None
        capture_id = pointer["capture_id"] if pointer else capture_id
        directory = _capture_dir(root, capture_id)
        manifest_path = directory / "manifest.json"
        if manifest_path.resolve() != manifest_path:
            return None
        manifest = _load_json(manifest_path)
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("capture_id") != capture_id:
            return None
        expected = f"layout-history/{capture_id}/timelapse.html"
        if manifest.get("html_path") != expected:
            return None
        target = directory / "timelapse.html"
        if target.resolve() != target or not target.is_file():
            return None
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest.get("html_sha256"):
            return None
        if pointer and any(pointer.get(key) != manifest.get(key) for key in ("html_sha256", "generation", "run_id")):
            return None
        return manifest, target
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _references(value):
    if isinstance(value, dict):
        source = value.get("source_scene")
        if isinstance(source, dict) and source.get("scope_id"):
            yield source["scope_id"], source.get("seq")
        board_scene = value.get("board_scene")
        if isinstance(board_scene, dict) and board_scene.get("scope_id"):
            yield board_scene["scope_id"], board_scene.get("seq")
        if value.get("source_scope_id"):
            yield value["source_scope_id"], None
        for nested in value.values():
            yield from _references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _references(nested)


def _html_write(stream, value):
    stream.write(value.encode("utf-8") if isinstance(value, str) else value)


def finalize_capture(project_dir: Path, capture_id: str, *, build_rc: int | None,
                     interrupted: bool = False) -> Path | None:
    """Publish once under an advisory lock. Failed publication keeps all spools."""
    started = time.monotonic()
    project_dir = project_dir.resolve()
    try:
        directory = _capture_dir(project_dir, capture_id)
        with (directory / ".publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = published_capture(project_dir, capture_id)
            if existing:
                manifest = existing[0]
                pointer_path = project_dir / "layout-history.json"
                pointer = _load_json(pointer_path) if pointer_path.is_file() else {}
                current = {key: value for key, value in manifest.items() if key != "scope_ids"}
                if pointer.get("capture_id") == capture_id and pointer != current:
                    atomic_json(pointer_path, current)
                scope_ids = manifest.get("scope_ids", [scope["header"]["scope_id"] for scope in manifest.get("scopes", [])])
                for scope_id in scope_ids:
                    if re.fullmatch(r"[0-9a-f]{32}", scope_id):
                        (directory / f"scope-{scope_id}.jsonl.gz").unlink(missing_ok=True)
                return directory / "manifest.json"
            identity = _load_json(directory / "capture.json")
            if identity.get("capture_id") != capture_id or identity.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("capture ownership mismatch")
            scopes, references = [], []
            gaps = ["capture interrupted after process termination"] if interrupted else []
            for path in sorted(directory.glob("scope-*.jsonl.gz")):
                scope = recover_scope(path, capture_id=capture_id)
                if not scope.header:
                    gaps.extend(f"{path.name}: {gap}" for gap in scope.gaps)
                    continue
                if path.name != f"scope-{scope.header.get('scope_id')}.jsonl.gz":
                    scope.gaps.append("scope file identity mismatch")
                if scope.members:
                    for offset, compressed, data in iter_members(path):
                        for line in data.splitlines():
                            try:
                                record = json.loads(line)
                            except ValueError:
                                break
                            if record.get("kind") != "header" and record.get("seq", 0) > scope.last_seq:
                                break
                            references.extend(_references(record))
                        if offset >= scope.members[-1]["offset"]:
                            break
                scopes.append({"header": scope.header, "gaps": scope.gaps,
                               "last_seq": scope.last_seq, "members": scope.members, "path": path,
                               "phases": scope.phases, "baseline_seq": scope.baseline_seq})
            scopes.sort(key=lambda scope: scope["header"].get("started_ns", 0))
            by_id = {scope["header"]["scope_id"]: scope for scope in scopes}
            for scope in scopes:
                parent = scope["header"].get("parent_scope_id")
                if parent and parent not in by_id:
                    scope["gaps"].append("missing parent scope " + parent)
                gaps.extend(f"{scope['header']['scope_id']}: {gap}" for gap in scope["gaps"])
            for source, sequence in references:
                if source not in by_id or (sequence is not None and sequence > by_id[source]["last_seq"]):
                    gaps.append("missing source scene " + str(source))
            if not scopes:
                gaps.append("no recoverable scope")
            final_board = _load_json(directory / "final-board.json") if (directory / "final-board.json").is_file() else {}
            selection = final_board.get("source_scene")
            if selection and selection.get("scope_id") not in by_id:
                gaps.append("delivered board scene unavailable")
            status = "partial" if gaps else "complete"
            assets = Path(__file__).parent / "server" / "static"
            css = (assets / "layout-history.css").read_text()
            javascript = (assets / "layout-history.js").read_text()
            temporary = directory / ("timelapse." + uuid.uuid4().hex + ".tmp")
            try:
                with temporary.open("wb") as stream:
                    _html_write(stream, '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KiCraft layout history</title><style>')
                    _html_write(stream, css)
                    _html_write(stream, '</style></head><body><main id="layout-history-player"></main><script type="application/json" id="layout-history-data">')
                    prefix = {"schema_version": 1, "capture_id": capture_id, "status": status,
                              "gaps": gaps, "selection": selection, "build_rc": build_rc,
                              "selection_role": final_board.get("final_board_role")}
                    _html_write(stream, json_bytes(prefix)[:-1].replace(b"<", b"\\u003c"))
                    _html_write(stream, b',"scopes":[')
                    total_bytes = 0
                    for index, scope in enumerate(scopes):
                        if index:
                            _html_write(stream, b",")
                        metadata = {key: scope[key] for key in ("header", "gaps", "last_seq", "phases", "baseline_seq")}
                        _html_write(stream, json_bytes(metadata)[:-1].replace(b"<", b"\\u003c"))
                        _html_write(stream, b',"members":[')
                        with scope["path"].open("rb") as spool:
                            for member_index, member in enumerate(scope["members"]):
                                if member_index:
                                    _html_write(stream, b",")
                                _html_write(stream, json_bytes(member)[:-1])
                                _html_write(stream, b',"data":"')
                                spool.seek(member["offset"])
                                remaining = member["bytes"]
                                while remaining:
                                    chunk = spool.read(min(49152, remaining))
                                    if not chunk:
                                        raise ValueError("spool changed during export")
                                    remaining -= len(chunk)
                                    total_bytes += len(chunk)
                                    _html_write(stream, base64.b64encode(chunk))
                                _html_write(stream, b'"}')
                        _html_write(stream, b"]}")
                    _html_write(stream, b"]}")
                    _html_write(stream, '</script><script>')
                    _html_write(stream, javascript)
                    _html_write(stream, '</script></body></html>')
                temporary.replace(directory / "timelapse.html")
            finally:
                temporary.unlink(missing_ok=True)
            target = directory / "timelapse.html"
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = {**identity, "status": status, "build_rc": build_rc,
                        "html_path": str(target.relative_to(project_dir)),
                        "html_sha256": digest, "html_bytes": target.stat().st_size,
                        "compressed_bytes": total_bytes, "event_count": sum(s["last_seq"] for s in scopes),
                        "scope_count": len(scopes), "gaps": gaps,
                        "scope_ids": [scope["header"]["scope_id"] for scope in scopes],
                        **final_board, "publication_s": round(time.monotonic() - started, 4)}
            atomic_json(directory / "manifest.json", manifest)
            pointer_path = project_dir / "layout-history.json"
            pointer = _load_json(pointer_path) if pointer_path.is_file() else {}
            if pointer.get("capture_id") == capture_id:
                atomic_json(pointer_path, {key: value for key, value in manifest.items() if key != "scope_ids"})
            for scope in scopes:
                scope["path"].unlink()
            return directory / "manifest.json"
    except Exception as error:
        warnings.warn(f"[layout-history] publication failed: {error}", RuntimeWarning)
        try:
            pointer_path = project_dir / "layout-history.json"
            pointer = _load_json(pointer_path)
            if pointer.get("capture_id") == capture_id and pointer.get("status") == "recording":
                atomic_json(pointer_path, {**pointer, "status": "unavailable",
                                          "build_rc": build_rc,
                                          "gaps": ["publication_failed:" + type(error).__name__]})
        except (OSError, ValueError):
            pass
        return None


def recover_terminated_capture(project_dir: Path | None, *, generation=None,
                               build_rc=None, interrupted=True) -> Path | None:
    """Call only AFTER confirming the owning job/process has terminated."""
    if project_dir is None:
        return None
    try:
        pointer = _load_json(project_dir / "layout-history.json")
        if pointer.get("status") != "recording" or pointer.get("generation") != generation:
            return None
        return finalize_capture(project_dir, pointer["capture_id"], build_rc=build_rc,
                                interrupted=interrupted)
    except (OSError, ValueError, KeyError):
        return None


def _embedded_data(path: Path) -> dict:
    # Check CLI only: UI polling never opens or decompresses the HTML.
    html = path.read_text()
    marker = '<script type="application/json" id="layout-history-data">'
    return json.loads(html.split(marker, 1)[1].split("</script>", 1)[0])


def check_publication(project_dir: Path) -> list[str]:
    publication = published_capture(project_dir)
    if not publication:
        return ["publication missing or hash/identity mismatch"]
    manifest, path = publication
    data = _embedded_data(path)
    gaps = list(manifest.get("gaps", []))
    selected = None
    for item in data["scopes"]:
        scope = RecoveredScope()
        try:
            for member in item["members"]:
                decoded = zlib.decompress(base64.b64decode(member["data"]), 16 + zlib.MAX_WBITS)
                for line in decoded.splitlines():
                    record = json.loads(line)
                    if record.get("seq", 0) > item["last_seq"]:
                        break
                    apply_record(scope, record, capture_id=manifest["capture_id"])
            if (manifest.get("source_scene") or {}).get("scope_id") == scope.header.get("scope_id"):
                selected = scope
        except (ValueError, KeyError, zlib.error) as error:
            gaps.append(str(error))
        gaps.extend(scope.gaps)
    board_name = manifest.get("final_board_path")
    if board_name:
        board = (project_dir / board_name).resolve()
        if not board.is_relative_to(project_dir.resolve()) or not board.is_file():
            gaps.append("bound board missing")
        elif hashlib.sha256(board.read_bytes()).hexdigest() != manifest.get("final_board_sha256"):
            gaps.append("bound board hash mismatch")
        elif selected is None:
            gaps.append("bound board scene missing")
        else:
            gaps.extend(_check_native_geometry(board, selected))
    return gaps


def _check_native_geometry(board: Path, selected: RecoveredScope) -> list[str]:
    import subprocess
    from kicraft.autoplacer.hardware.adapter import _pcbnew_subprocess_env
    script = '''import json, pcbnew, sys
from kicraft.autoplacer.hardware.history_geometry import extract_native_scene
class Definitions:
 def __init__(self): self.templates={}
 def define(self,g,**kw):
  from kicraft.layout_history import compact,json_bytes
  import hashlib
  g=compact(g); key=hashlib.sha256(json_bytes(g)).hexdigest(); self.templates[key]=g; return key
d=Definitions()
objects=extract_native_scene(pcbnew.LoadBoard(sys.argv[1]),d,phase="check")
print(json.dumps({"objects":objects,"templates":d.templates}))
'''
    python = os.environ.get("KICRAFT_PCBNEW_PYTHON", "/usr/bin/python3")
    result = subprocess.run([python, "-c", script, str(board)], env=_pcbnew_subprocess_env(),
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        return ["native geometry check failed: " + result.stderr[-1000:]]
    from kicraft.layout_history import compact
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    actual = {obj["id"]: compact(obj) for obj in measured["objects"]}
    referenced = {obj.get("template_id") for obj in selected.objects.values()}
    templates = {key: value for key, value in selected.templates.items() if key in referenced}
    if actual != selected.objects or compact(measured["templates"]) != templates:
        return ["selected final native geometry mismatch at display precision"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    gaps = check_publication(args.project.resolve())
    print(json.dumps({"ok": not gaps, "gaps": gaps}))
    return int(bool(gaps))


if __name__ == "__main__":
    raise SystemExit(main())
