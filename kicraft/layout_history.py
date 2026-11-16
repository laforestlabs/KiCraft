"""Best-effort, single-writer layout history. No pcbnew dependency.

Each flush is a complete gzip member. A killed writer may lose its buffered tail;
only complete members are recoverable. Recording never changes build verdicts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Mapping

SCHEMA_VERSION = 1
BUFFER_BYTES = 256 * 1024
ENV_DIR = "KICRAFT_LAYOUT_HISTORY_DIR"
ENV_ID = "KICRAFT_LAYOUT_HISTORY_ID"
ENV_PARENT = "KICRAFT_LAYOUT_HISTORY_PARENT"
_context: ContextVar[HistoryDescriptor | None] = ContextVar("layout_history", default=None)


@contextmanager
def history_context(descriptor):
    """In-process ownership only; children still receive an explicit environment."""
    token = _context.set(descriptor)
    try:
        yield
    finally:
        _context.reset(token)


def compact(value: Any) -> Any:
    """Display precision only; never modifies the supplied geometry."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 4)
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def public_value(value: Any, project_dir: str, key: str = "") -> Any:
    """Keep instance identities, but never export filesystem roots or credentials."""
    if isinstance(value, dict):
        return {str(k): public_value(v, project_dir, str(k)) for k, v in value.items()
                if not any(secret in str(k).lower() for secret in ("token", "password", "secret", "credential"))}
    if isinstance(value, (list, tuple)):
        return [public_value(v, project_dir, key) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        value = value.replace(project_dir, ".")
        if value.startswith(("/home/", "/tmp/", "/opt/", "/var/")):
            return Path(value).name
    return value


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(json_bytes(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class HistoryDescriptor:
    directory: str
    capture_id: str
    parent_scope_id: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HistoryDescriptor | None:
        if (os.environ if env is None else env).get("KICRAFT_LAYOUT_HISTORY") == "0":
            return None
        if env is None and _context.get() is not None:
            return _context.get()
        env = os.environ if env is None else env
        if env.get("KICRAFT_LAYOUT_HISTORY") == "0":
            return None
        directory, capture_id = env.get(ENV_DIR), env.get(ENV_ID)
        if not directory or not capture_id:
            return None
        return cls(directory, capture_id, env.get(ENV_PARENT))

    def child_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        result = dict(os.environ if env is None else env)
        result[ENV_DIR] = self.directory
        result[ENV_ID] = self.capture_id
        if self.parent_scope_id:
            result[ENV_PARENT] = self.parent_scope_id
        else:
            result.pop(ENV_PARENT, None)
        return result

    def writer(self, scope_kind: str, label: str, identity: dict | None = None) -> HistoryWriter:
        return HistoryWriter(self, scope_kind=scope_kind, label=label, identity=identity)


class HistoryWriter:
    def __init__(self, descriptor: HistoryDescriptor, *, scope_kind: str, label: str,
                 identity: dict | None = None):
        self.descriptor = descriptor
        self.scope_id = uuid.uuid4().hex
        self.path = Path(descriptor.directory) / f"scope-{self.scope_id}.jsonl.gz"
        self._project_dir = str(self.path.parent.parent.parent)
        self.seq = 0
        self.disabled = False
        self.closed = False
        self.cpu_ns = 0
        self.bytes_written = 0
        self._buffer = bytearray()
        self._last_flush = time.monotonic()
        self._objects: dict[str, dict] = {}
        self._templates: set[str] = set()
        self._placement_templates: dict[str, str] = {}
        self._groups: dict[str, dict] = {}
        self._has_scene = False
        self._record({"kind": "header", "schema_version": SCHEMA_VERSION,
                      "capture_id": descriptor.capture_id, "scope_id": self.scope_id,
                      "parent_scope_id": descriptor.parent_scope_id,
                      "scope_kind": scope_kind, "label": label,
                      "identity": identity or {}, "started_ns": time.monotonic_ns()})
        self.flush()

    def _fail(self, error: Exception) -> None:
        if not self.disabled:
            self.disabled = True
            warnings.warn(f"[layout-history] disabled writer {self.scope_id}: {error}", RuntimeWarning)
        self._buffer.clear()

    def _record(self, record: dict) -> None:
        if self.disabled or self.closed:
            return
        started = time.process_time_ns()
        try:
            self._buffer.extend(json_bytes(compact(public_value(record, self._project_dir))))
            self._buffer.extend(b"\n")
            if len(self._buffer) >= BUFFER_BYTES or time.monotonic() - self._last_flush >= 1:
                self.flush()
        except Exception as error:
            self._fail(error)
        finally:
            self.cpu_ns += time.process_time_ns() - started

    def emit(self, kind: str, *, phase: str, **fields: Any) -> None:
        if self.closed or self.disabled:
            return
        self.seq += 1
        self._record({"seq": self.seq, "t_ns": time.monotonic_ns(), "kind": kind,
                      "phase": phase, **fields})

    def flush(self) -> None:
        if self.disabled or not self._buffer:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            member = gzip.compress(self._buffer, compresslevel=1, mtime=0)
            with self.path.open("ab") as stream:
                stream.write(member)
            self.bytes_written += len(member)
            self._buffer.clear()
            self._last_flush = time.monotonic()
        except Exception as error:
            self._fail(error)

    def child_descriptor(self) -> HistoryDescriptor:
        return HistoryDescriptor(self.descriptor.directory, self.descriptor.capture_id, self.scope_id)

    def child_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        return self.child_descriptor().child_env(env)

    def child(self, scope_kind: str, label: str, identity: dict | None = None) -> HistoryWriter:
        return self.child_descriptor().writer(scope_kind, label, identity)

    def define(self, geometry: dict, *, phase: str = "geometry") -> str:
        geometry = compact(geometry)
        template_id = hashlib.sha256(json_bytes(geometry)).hexdigest()
        if template_id not in self._templates:
            self._templates.add(template_id)
            self.emit("define", phase=phase, template_id=template_id, geometry=geometry)
        return template_id

    def scene(self, objects: list[dict], *, phase: str, geometry_source: str,
              **metadata: Any) -> None:
        objects = compact(objects)
        self._objects = {obj["id"]: obj for obj in objects}
        self._has_scene = True
        self.emit("scene", phase=phase, objects=objects, geometry_source=geometry_source, **metadata)
        self.flush()

    def delta(self, put: list[dict] | None = None, remove: list[str] | None = None,
              *, phase: str, **metadata: Any) -> None:
        put, remove = compact(put or []), remove or []
        for object_id in remove:
            self._objects.pop(object_id, None)
        for obj in put:
            self._objects[obj["id"]] = obj
        if put or remove:
            self.emit("delta", phase=phase, put=put, remove=remove, **metadata)

    def update_scene(self, objects: list[dict], *, phase: str, geometry_source: str,
                     **metadata: Any) -> None:
        if not self._has_scene:
            self.scene(objects, phase=phase, geometry_source=geometry_source, **metadata)
            return
        objects_by_id = {obj["id"]: compact(obj) for obj in objects}
        self.delta([obj for key, obj in objects_by_id.items() if self._objects.get(key) != obj],
                   [key for key in self._objects if key not in objects_by_id],
                   phase=phase, geometry_source=geometry_source, **metadata)

    def decision(self, action: str, *, phase: str, reason: str | None = None, **fields: Any) -> None:
        self.emit("decision", phase=phase, action=action, reason=reason, **fields)
        self.flush()

    def register_group(self, ref: str, source_scene: dict, origin=None, **metadata) -> None:
        self._groups[ref] = {"source_scene": source_scene, "origin": origin or [0, 0], **metadata}

    def placement(self, components: dict, *, phase: str, iteration: int | None = None,
                  refs=None, metrics=None) -> None:
        if self.disabled or self.closed:
            return
        started = time.process_time_ns()
        try:
            from kicraft.autoplacer.brain.subcircuit_artifacts import serialize_component
            objects = []
            for ref in components if refs is None else refs:
                component = components.get(ref)
                if component is None:
                    continue
                if ref in self._groups:
                    from kicraft.autoplacer.brain.placement_utils import _world_artifact_origin
                    origin = _world_artifact_origin(component)
                    objects.append({"id": "footprint:" + ref, "type": "group", "ref": ref,
                                    **self._groups[ref],
                                    "pose": {"x": origin.x, "y": origin.y, "rotation": component.rotation,
                                             "side": "back" if int(component.layer) == 1 else "front"}})
                    continue
                if ref not in self._placement_templates:
                    model = serialize_component(component)
                    rotation = math.radians(float(component.rotation))
                    co, si = math.cos(rotation), math.sin(rotation)
                    pads = []
                    for index, pad in enumerate(model["pads"]):
                        px, py = pad["pos"]["x"] - component.pos.x, pad["pos"]["y"] - component.pos.y
                        pads.append({**pad, "id": str(index), "pos": [co * px + si * py, -si * px + co * py]})
                    self._placement_templates[ref] = self.define({
                        "geometry_source": "placement_model", "proxy": "world-AABB pad/body proxy",
                        "width_mm": model["width_mm"], "height_mm": model["height_mm"], "pads": pads,
                    }, phase=phase)
                side = "back" if "back" in str(component.layer).lower() or str(getattr(component.layer, "value", "")) in ("B.Cu", "1") else "front"
                objects.append({"id": "footprint:" + ref, "type": "footprint", "ref": ref,
                                "value": component.value, "template_id": self._placement_templates[ref],
                                "pose": {"x": component.pos.x, "y": component.pos.y,
                                         "rotation": component.rotation, "side": side}})
            metadata = {"iteration": iteration, "metrics": metrics}
            if not self._has_scene:
                self.scene(objects, phase=phase, geometry_source="placement_model", **metadata)
            else:
                changed = [obj for obj in compact(objects) if self._objects.get(obj["id"]) != obj]
                remove = [] if refs is not None else [key for key in self._objects if key.startswith("footprint:") and key[10:] not in components]
                self.delta(changed, remove, phase=phase, **metadata)
                if not changed and not remove and iteration is not None:
                    self.emit("delta", phase=phase, put=[], remove=[], **metadata)
        except Exception as error:
            self._fail(error)
        finally:
            self.cpu_ns += time.process_time_ns() - started

    def end(self, outcome: str, *, phase: str = "end", **fields: Any) -> None:
        if self.closed:
            return
        self.emit("end", phase=phase, outcome=outcome, recorder_cpu_ns=self.cpu_ns, **fields)
        self.flush()
        self.closed = True
        self._objects.clear()
        self._templates.clear()
        self._placement_templates.clear()
        self._groups.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end("error" if exc_type else "complete", error=exc_type.__name__ if exc_type else None)
        return False


def history_from_env(scope_kind: str, label: str, identity: dict | None = None) -> HistoryWriter | None:
    descriptor = HistoryDescriptor.from_env()
    return descriptor.writer(scope_kind, label, identity) if descriptor else None


def start_capture(project_dir: Path, *, run_id: str, generation: int | str | None = None) -> HistoryWriter | None:
    if os.environ.get("KICRAFT_LAYOUT_HISTORY") == "0":
        try:
            atomic_json(project_dir / "layout-history.json", {
                "schema_version": SCHEMA_VERSION, "capture_id": uuid.uuid4().hex,
                "run_id": run_id, "generation": generation, "status": "disabled",
            })
        except OSError as error:
            warnings.warn(f"[layout-history] cannot publish disabled status: {error}", RuntimeWarning)
        return None
    capture_id = uuid.uuid4().hex
    directory = project_dir / "layout-history" / capture_id
    try:
        directory.mkdir(parents=True)
        identity = {"schema_version": SCHEMA_VERSION, "capture_id": capture_id,
                    "run_id": run_id, "generation": generation, "status": "recording"}
        atomic_json(directory / "capture.json", identity)
        atomic_json(project_dir / "layout-history.json", identity)
        return HistoryDescriptor(str(directory.resolve()), capture_id).writer("run", "Layout and routing", identity)
    except Exception as error:
        warnings.warn(f"[layout-history] cannot start capture: {error}", RuntimeWarning)
        return None


def run_capture(project_dir: Path, board: Path, callback):
    """Own an invocation, including failure exits; never change its return code."""
    import subprocess
    from kicraft.cli.artifact_paths import ensure_run_context, read_provenance
    from kicraft.layout_history_export import finalize_capture
    run_id, _ = ensure_run_context()
    generation = os.environ.get("KICRAFT_ATTEMPT_GENERATION") or next(
        (part[8:] for part in project_dir.parts if part.startswith("attempt-")), None,
    )
    if generation is not None and generation.isdigit():
        generation = int(generation)
    root = start_capture(project_dir, run_id=run_id, generation=generation)
    if root is None:
        return callback(), None
    rc = None
    manifest = None
    invocation_started_at = time.time()
    try:
        with history_context(root.child_descriptor()):
            rc = callback()
        return_code = rc
    finally:
        try:
            provenance = read_provenance(board) or {}
            selected_board = board if provenance.get("run_id") == run_id and provenance.get("fresh") else None
            role = "delivered"
            if selected_board is None:
                from kicraft.cli import artifact_paths
                preview = (artifact_paths.resolve_parent_board(project_dir, kind="placed")
                           or artifact_paths.resolve_best_leaf_board(project_dir))
                if preview is not None and preview.stat().st_mtime >= invocation_started_at:
                    selected_board = preview
                    role = "partial_preview"
            if selected_board is not None:
                from kicraft.autoplacer.hardware.adapter import _pcbnew_subprocess_env
                env = root.child_env(_pcbnew_subprocess_env())
                result = subprocess.run(
                    [os.environ.get("KICRAFT_PCBNEW_PYTHON", "/usr/bin/python3"), "-m",
                     "kicraft.autoplacer.hardware.history_geometry", "--board", str(selected_board),
                     "--phase", "delivered" if role == "delivered" else "selected-partial-preview"],
                    capture_output=True, text=True, env=env, timeout=120,
                )
                if result.returncode:
                    root.decision("skip", phase="delivered", gap="native_delivery_extraction_failed",
                                  reason=result.stderr[-1000:])
                else:
                    extraction = json.loads(result.stdout.strip().splitlines()[-1])
                    source = {"scope_id": extraction["scope_id"]}
                    digest = hashlib.sha256(selected_board.read_bytes()).hexdigest()
                    binding = {"source_scene": source,
                               "final_board_path": str(selected_board.relative_to(project_dir)),
                               "final_board_sha256": digest, "final_board_role": role}
                    atomic_json(Path(root.descriptor.directory) / "final-board.json", binding)
                    root.decision("select", phase="promotion" if role == "delivered" else "partial_preview",
                                  reason=provenance.get("source_kind") if role == "delivered" else "artifact resolver partial preview; not promoted",
                                  **binding)
            else:
                root.decision("skip", phase="promotion", reason="No fresh board was delivered by this invocation")
            from kicraft.autoplacer.hardware.history_geometry import close_capture_history
            close_capture_history(root.descriptor.capture_id)
            root.end("interrupted" if rc is None else "complete", build_rc=rc)
            manifest = finalize_capture(project_dir, root.descriptor.capture_id, build_rc=rc,
                                        interrupted=rc is None)
        except Exception as error:
            warnings.warn(f"[layout-history] finalization failed: {error}", RuntimeWarning)
    return return_code, manifest
