#!/usr/bin/env python3
"""Replay a frozen self-eval corpus under FreeRouting and KiCadRoutingTools.

The synthesized projects, placement seeds, prompts, and non-router configuration
are shared.  This driver runs only the real ``replay --quality good`` build tail;
it never calls synthesis, an LLM, a judge, or tuning code.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "bin" / "python"
DRIVER = Path(__file__).resolve()
EXPECTED_BRIEFS_N = 34
EXPECTED_BRIEFS_SHA256 = "fe68f47e23a9cbd4958d439f7414fd2a843397ed2dd06159346e456adb67d258"
WATCHDOG_S = 2400
BUILD_MAX_WALL_S = 2160
FREEROUTING_JAR = Path("/home/kicraft/.local/lib/freerouting-1.9.0.jar")
JAVA_BIN = Path("/home/kicraft/.local/lib/jre/bin/java")
KRT_ROOT = Path("/tmp/KiCadRoutingTools")
KRT_PYTHON = Path("/tmp/krt-venv/bin/python")
KRT_COMMIT = "3ceb773722bea67aa3685e7ee430c0c0d17ef38d"
KRT_SOURCE_VERSION = "0.20.2"
KRT_NATIVE_VERSION = "0.20.1"
BACKENDS = ("freerouting", "krt")
BACKEND_DISPLAY = {"freerouting": "FreeRouting", "krt": "KiCadRoutingTools"}
PINNED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "KICRAFT_BUILD_MAX_WALL_S": str(BUILD_MAX_WALL_S),
}
FR_OVERLAY = {
    "routing_backend": "freerouting",
    "freerouting_jar": str(FREEROUTING_JAR),
    "java_bin": str(JAVA_BIN),
}
KRT_OVERLAY = {
    "routing_backend": "kicad-routing-tools",
    "kicad_routing_tools_path": str(KRT_ROOT),
    "kicad_routing_tools_python": str(KRT_PYTHON),
}
BACKEND_CONFIG_KEYS = frozenset({
    "routing_backend",
    "freerouting_jar",
    "java_bin",
    "freerouting_timeout_s",
    "freerouting_max_passes",
    "leaf_freerouting_max_passes",
    "parent_dense_max_passes",
    "parent_dense_timeout_s",
    "parent_s_per_interconnect",
    "kicad_routing_tools_path",
    "kicad_routing_tools_python",
    "kicad_routing_tools_timeout_s",
    "kicad_routing_tools_max_iterations",
    "kicad_routing_tools_max_ripup",
    "kicad_routing_tools_ordering",
    "kicad_routing_tools_clearance_mm",
    "kicad_routing_tools_layers",
})
MARKERS = {
    "layout": "[build] 2/5 place + route",
    "leaf": "[build]   leaf phase:",
    "parent": "[build]   parent phase:",
    "promote": "[build] 3/5 promoted",
    "verify": "[build] 4/5 verify:",
    "done": "REPLAY COMPLETE",
}
RE_MASTER_SEED = re.compile(r"Master seed:\s+(\d+)")
RE_ROUND_TIMING = re.compile(r"\[timing\] round (\d+) ([A-Za-z0-9_]+)=([\d.]+)s")


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (set, frozenset)):
        return [jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - artifact identity, not cryptographic trust
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path, *, explicit: list[Path] | None = None) -> tuple[str, int]:
    paths = explicit if explicit is not None else [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
    rel_paths = sorted(paths, key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    count = 0
    for path in rel_paths:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            data = b"SYMLINK\0" + os.readlink(path).encode("utf-8", "surrogateescape")
        else:
            data = path.read_bytes()
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
        count += 1
    return h.hexdigest(), count


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def relative_evidence(path: Path, batch_dir: Path) -> str:
    if not is_within(path, batch_dir):
        raise ValueError(f"evidence path escapes batch: {path}")
    return path.resolve().relative_to(batch_dir.resolve()).as_posix()


def command_output(cmd: list[str], *, cwd: Path = REPO, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": cmd, "error": repr(exc)}
    return {
        "command": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout.decode("utf-8", "replace").strip(),
        "stderr": proc.stderr.decode("utf-8", "replace").strip(),
    }


def runtime_tree_identity() -> dict[str, Any]:
    paths = sorted((REPO / "kicraft").rglob("*.py"))
    paths.extend([REPO / "pyproject.toml", DRIVER])
    digest, count = tree_sha256(REPO, explicit=paths)
    return {"sha256": digest, "file_count": count, "driver_sha256": sha256_file(DRIVER)}


def git_identity() -> dict[str, Any]:
    head = command_output(["git", "rev-parse", "HEAD"])
    branch = command_output(["git", "branch", "--show-current"])
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO, capture_output=True, check=True
    ).stdout
    status_proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    parts = status_proc.stdout.split(b"\0")
    status_records: list[str] = []
    status_paths: set[str] = set()
    i = 0
    while i < len(parts):
        raw = parts[i]
        i += 1
        if not raw:
            continue
        text = raw.decode("utf-8", "surrogateescape")
        status_records.append(text)
        if len(text) >= 4:
            status_paths.add(text[3:])
        xy = text[:2]
        if ("R" in xy or "C" in xy) and i < len(parts) and parts[i]:
            other = parts[i].decode("utf-8", "surrogateescape")
            i += 1
            status_records.append(other)
            status_paths.add(other)
    path_hashes: dict[str, Any] = {}
    for rel in sorted(status_paths):
        path = REPO / rel
        if path.is_symlink():
            path_hashes[rel] = {"symlink": os.readlink(path)}
        elif path.is_file():
            path_hashes[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        else:
            path_hashes[rel] = {"exists": path.exists()}
    return {
        "head": head.get("stdout"),
        "branch": branch.get("stdout"),
        "status_sha256": sha256_bytes(status_proc.stdout),
        "status_records": status_records,
        "diff_binary_head_sha256": sha256_bytes(diff),
        "dirty_path_hashes": path_hashes,
    }


def host_identity() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
    }


def backend_overlay(backend: str) -> dict[str, Any]:
    return dict(FR_OVERLAY if backend == "freerouting" else KRT_OVERLAY)


def effective_config(base: dict[str, Any], backend: str) -> dict[str, Any]:
    from kicraft.autoplacer.config import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(copy.deepcopy(base))
    cfg.update(backend_overlay(backend))
    return cfg


def preflight_identity() -> dict[str, Any]:
    from kicraft.autoplacer.config import DEFAULT_CONFIG
    from kicraft.autoplacer.routing_backends import preflight_routing_backend

    fr_cfg = copy.deepcopy(DEFAULT_CONFIG)
    fr_cfg.update(FR_OVERLAY)
    krt_cfg = copy.deepcopy(DEFAULT_CONFIG)
    krt_cfg.update(KRT_OVERLAY)
    fr = preflight_routing_backend(fr_cfg)
    krt = preflight_routing_backend(krt_cfg)
    if Path(fr["jar"]).resolve() != FREEROUTING_JAR.resolve():
        raise RuntimeError(f"FreeRouting jar drift: {fr['jar']} != {FREEROUTING_JAR}")
    if Path(fr["java"]).resolve() != JAVA_BIN.resolve():
        raise RuntimeError(f"Java drift: {fr['java']} != {JAVA_BIN}")
    expected_krt = {
        "root": str(KRT_ROOT.resolve()),
        "python": str(KRT_PYTHON),
        "version": KRT_SOURCE_VERSION,
        "commit": KRT_COMMIT,
        "native_version": KRT_NATIVE_VERSION,
    }
    for key, expected in expected_krt.items():
        if str(krt.get(key)) != expected:
            raise RuntimeError(f"KRT {key} drift: {krt.get(key)!r} != {expected!r}")
    return {
        "freerouting": {
            **fr,
            "jar_sha256": sha256_file(FREEROUTING_JAR),
            "version": "1.9.0",
        },
        "kicad_routing_tools": krt,
        "kicad_cli": command_output(["kicad-cli", "--version"]),
        "pcbnew": command_output(
            [str(PYTHON), "-c", "import pcbnew; print(pcbnew.GetBuildVersion())"]
        ),
        "java": command_output([str(JAVA_BIN), "-version"]),
    }


def frozen_briefs() -> tuple[list[dict[str, Any]], str]:
    from kicraft.eval.self_eval import BRIEFS

    briefs = copy.deepcopy(list(BRIEFS))
    digest = sha256_bytes(canonical_bytes(briefs))
    if len(briefs) != EXPECTED_BRIEFS_N or digest != EXPECTED_BRIEFS_SHA256:
        raise RuntimeError(
            f"self-eval corpus drift: {len(briefs)} {digest}; expected "
            f"{EXPECTED_BRIEFS_N} {EXPECTED_BRIEFS_SHA256}"
        )
    return briefs, digest


def classify_source(source_batch: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = source_batch / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"completed source summary missing: {summary_path}")
    if not (source_batch / "summary.md").is_file():
        raise FileNotFoundError(f"completed source markdown missing: {source_batch / 'summary.md'}")
    summary = load_json(summary_path)
    runs = summary.get("runs")
    if (
        summary.get("n") != EXPECTED_BRIEFS_N
        or summary.get("n_briefs") != EXPECTED_BRIEFS_N
        or not isinstance(runs, list)
        or len(runs) != EXPECTED_BRIEFS_N
        or int(summary.get("n_errored", 0) or 0) != 0
        or not summary.get("finished_at")
    ):
        raise RuntimeError(
            "source batch is incomplete: require finished n=34, n_briefs=34, "
            "34 run records, and n_errored=0"
        )
    briefs, briefs_sha = frozen_briefs()
    by_index = {int(r.get("index", -1)): r for r in runs if isinstance(r, dict)}
    if sorted(by_index) != list(range(1, EXPECTED_BRIEFS_N + 1)):
        raise RuntimeError("source batch does not contain exactly one record for each index 1..34")

    rows: list[dict[str, Any]] = []
    for index in range(1, EXPECTED_BRIEFS_N + 1):
        rec = by_index[index]
        entry = briefs[index - 1]
        reasons: list[str] = []
        if rec.get("slug") != entry.get("slug"):
            reasons.append("slug_mismatch")
        if rec.get("prompt") != entry.get("brief"):
            reasons.append("prompt_mismatch")
        if rec.get("design_status") != "ok":
            reasons.append(f"design_status={rec.get('design_status') or 'missing'}")
        rundir = Path(str(rec.get("rundir") or "")).resolve()
        if not is_within(rundir, source_batch):
            reasons.append("rundir_outside_source_batch")
        generated = rundir / "generated"
        projects = sorted(p for p in generated.iterdir() if p.is_dir()) if generated.is_dir() else []
        if len(projects) != 1:
            reasons.append(f"generated_projects={len(projects)}")
        configs = sorted(projects[0].glob("*_autoplacer.json")) if len(projects) == 1 else []
        if len(configs) != 1:
            reasons.append(f"autoplacer_configs={len(configs)}")
        seed = projects[0] / ".experiments" / "pre_promote_seed.kicad_pcb" if len(projects) == 1 else None
        if seed is None or not seed.is_file():
            reasons.append("missing_pre_promote_seed")

        row: dict[str, Any] = {
            "index": index,
            "slug": entry["slug"],
            "archetype": entry["archetype"],
            "brief": entry["brief"],
            "outline_shape": entry.get("outline_shape"),
            "route_eligible": not reasons,
            "eligibility_reasons": reasons,
            "source": {
                "design_status": rec.get("design_status"),
                "design_error": rec.get("design_error"),
                "build_rc": rec.get("build_rc"),
                "build_label": rec.get("build_label"),
                "grade": rec.get("grade"),
                "final": rec.get("final"),
                "gates": rec.get("gates") or [],
                "design_cost_usd": rec.get("design_cost_usd"),
                "judge_cost_usd": rec.get("judge_cost_usd"),
                "outline_check": rec.get("outline_check"),
                "rundir": str(rundir),
            },
        }
        if not reasons:
            assert len(projects) == 1 and len(configs) == 1 and seed is not None
            project = projects[0].resolve()
            config_path = configs[0].resolve()
            base = load_json(config_path)
            pcb_name = base.get("pcb_file")
            if not isinstance(pcb_name, str) or not pcb_name or Path(pcb_name).name != pcb_name:
                raise RuntimeError(f"invalid canonical pcb_file in {config_path}: {pcb_name!r}")
            canonical = project / pcb_name
            if not canonical.is_file():
                raise RuntimeError(f"canonical PCB missing from eligible project: {canonical}")
            project_hash, project_files = tree_sha256(project)
            base_non_backend = {k: v for k, v in base.items() if k not in BACKEND_CONFIG_KEYS}
            row.update({
                "project": str(project),
                "project_tree_sha256": project_hash,
                "project_file_count": project_files,
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "config_non_backend_sha256": sha256_bytes(canonical_bytes(base_non_backend)),
                "base_config": jsonable(base),
                "canonical_pcb": pcb_name,
                "pre_promote_seed": str(seed.resolve()),
                "pre_promote_seed_sha256": sha256_file(seed),
                "effective_configs": {
                    backend: jsonable(effective_config(base, backend)) for backend in BACKENDS
                },
            })
        rows.append(row)

    metadata = {
        "path": str(source_batch.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "briefs_sha256": briefs_sha,
        "n": summary["n"],
        "n_briefs": summary["n_briefs"],
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "wall_s": summary.get("wall_s"),
        "design_model": summary.get("design_model"),
        "judge": summary.get("judge"),
        "judge_model": summary.get("judge_model"),
        "rubric_version": summary.get("rubric_version"),
        "total_cost_usd": summary.get("total_cost_usd"),
        "parallel": summary.get("parallel"),
        "build_slots": summary.get("build_slots"),
        "full_events": summary.get("full_events"),
    }
    return metadata, rows


def build_matrix(prompt_rows: list[dict[str, Any]], seeds: list[int], selected_slugs: list[str]) -> list[dict[str, Any]]:
    selected = set(selected_slugs)
    eligible = [r for r in prompt_rows if r["route_eligible"] and (not selected or r["slug"] in selected)]
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        for row in eligible:
            order = ("freerouting", "krt") if (seed + row["index"]) % 2 else ("krt", "freerouting")
            for backend in order:
                token = "freerouting" if backend == "freerouting" else "krt"
                cells.append({
                    "cell": f"s{seed}_{row['index']:02d}_{row['slug']}_{token}",
                    "seed": seed,
                    "index": row["index"],
                    "slug": row["slug"],
                    "backend": backend,
                })
    return cells


def build_identity(source_batch: Path, seeds: list[int], selected_slugs: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source, prompt_rows = classify_source(source_batch)
    known_slugs = {r["slug"] for r in prompt_rows}
    unknown = sorted(set(selected_slugs) - known_slugs)
    if unknown:
        raise ValueError(f"unknown --slugs: {', '.join(unknown)}")
    matrix = build_matrix(prompt_rows, seeds, selected_slugs)
    if not matrix:
        raise RuntimeError("no route-eligible prompts selected")
    toolchains = preflight_identity()
    identity = {
        "schema_version": "krt-self-eval-router-ab-v1",
        "source": source,
        "prompts": prompt_rows,
        "seeds": seeds,
        "selected_slugs": selected_slugs,
        "matrix": matrix,
        "arm_order_rule": (
            "FreeRouting first when (seed + self_eval_index) is odd; "
            "KiCadRoutingTools first when even"
        ),
        "execution": {
            "cwd": str(REPO),
            "command_template": [
                str(PYTHON), "-m", "kicraft.design.cli_app", "replay",
                "--project", "<cell>/project", "--quality", "good",
                "--seed", "<seed>", "--no-fab",
            ],
            "environment": PINNED_ENV,
            "watchdog_s": WATCHDOG_S,
            "watchdog_reset_marker": "[build] build slot acquired",
            "quality": "good",
            "fab": False,
            "llm": False,
            "synthesis": False,
            "judge": False,
        },
        "arm_overlays": {
            "freerouting": FR_OVERLAY,
            "krt": KRT_OVERLAY,
        },
        "toolchains": toolchains,
        "host": host_identity(),
        "repository": git_identity(),
        "runtime_tree": runtime_tree_identity(),
    }
    return jsonable(identity), prompt_rows, matrix


def ensure_manifest(batch_dir: Path, identity: dict[str, Any]) -> dict[str, Any]:
    path = batch_dir / "manifest.json"
    if path.is_file():
        manifest = load_json(path)
        if manifest.get("identity") != identity:
            old = sha256_bytes(canonical_bytes(manifest.get("identity")))
            new = sha256_bytes(canonical_bytes(identity))
            raise RuntimeError(
                f"manifest identity drift for {batch_dir}: recorded={old} current={new}; "
                "start a new batch instead of mixing evidence"
            )
        return manifest
    manifest = {
        "schema_version": "krt-self-eval-router-ab-manifest-v1",
        "created_at": utc_iso(),
        "identity": identity,
    }
    atomic_write_json(path, manifest)
    return manifest


def read_result_rows(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_path.is_file():
        return rows
    for line_no, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row.setdefault("_line", line_no)
            rows.append(row)
    return rows


def validate_result_evidence(row: dict[str, Any], batch_dir: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    checksums = row.get("evidence_checksums")
    if not isinstance(checksums, dict) or not checksums:
        return False, ["missing_evidence_checksums"]
    for rel, expected in sorted(checksums.items()):
        path = batch_dir / rel
        if not is_within(path, batch_dir):
            reasons.append(f"outside_batch:{rel}")
        elif not path.is_file():
            reasons.append(f"missing:{rel}")
        elif sha256_file(path) != expected:
            reasons.append(f"checksum:{rel}")
    return not reasons, reasons


def reusable_cells(results_path: Path, batch_dir: Path) -> set[str]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_result_rows(results_path):
        if row.get("cell"):
            latest[str(row["cell"])] = row
    done: set[str] = set()
    for cell, row in latest.items():
        if row.get("status") == "harness_error":
            continue
        valid, _ = validate_result_evidence(row, batch_dir)
        if valid:
            done.add(cell)
    return done


def prepare_cell(cell: dict[str, Any], prompt: dict[str, Any], batch_dir: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    cell_dir = batch_dir / "cells" / cell["cell"]
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    project = cell_dir / "project"
    project.parent.mkdir(parents=True, exist_ok=True)
    source_project = Path(prompt["project"])
    if tree_sha256(source_project)[0] != prompt["project_tree_sha256"]:
        raise RuntimeError(f"source project drift before {cell['cell']}")
    shutil.copytree(source_project, project, symlinks=True)

    config_source = Path(prompt["config"])
    config_rel = config_source.relative_to(source_project)
    config_path = project / config_rel
    base = load_json(config_path)
    if sha256_bytes(canonical_bytes({k: v for k, v in base.items() if k not in BACKEND_CONFIG_KEYS})) != prompt["config_non_backend_sha256"]:
        raise RuntimeError(f"non-backend config drift in copied {cell['cell']}")
    canonical = project / str(base["pcb_file"])
    copied_seed = project / ".experiments" / "pre_promote_seed.kicad_pcb"
    if not copied_seed.is_file():
        raise FileNotFoundError(f"copied frozen seed missing: {copied_seed}")
    if sha256_file(copied_seed) != prompt["pre_promote_seed_sha256"]:
        raise RuntimeError(f"copied seed checksum drift in {cell['cell']}")
    shutil.copyfile(copied_seed, canonical)

    experiments = project / ".experiments"
    shutil.rmtree(experiments)
    for path in project.glob("*.provenance.json"):
        path.unlink()
    fab = project / "fab"
    if fab.exists():
        shutil.rmtree(fab)
    for path in project.glob("*_fab_*.zip"):
        path.unlink()
    for board in project.glob("*.kicad_pcb"):
        if board.resolve() != canonical.resolve():
            board.unlink()

    cfg = {k: v for k, v in base.items() if k not in BACKEND_CONFIG_KEYS}
    cfg.update(backend_overlay(cell["backend"]))
    atomic_write_json(config_path, cfg)
    effective = effective_config(base, cell["backend"])
    input_evidence = {
        "source_project_tree_sha256": prompt["project_tree_sha256"],
        "source_config_sha256": prompt["config_sha256"],
        "source_seed_sha256": prompt["pre_promote_seed_sha256"],
        "canonical_seed_sha256": sha256_file(canonical),
        "non_backend_config_sha256": sha256_bytes(
            canonical_bytes({k: v for k, v in cfg.items() if k not in BACKEND_CONFIG_KEYS})
        ),
        "cell_config_sha256": sha256_file(config_path),
        "canonical_pcb": canonical.name,
        "config": config_path.name,
    }
    if input_evidence["canonical_seed_sha256"] != prompt["pre_promote_seed_sha256"]:
        raise RuntimeError(f"canonical seed mismatch in {cell['cell']}")
    if input_evidence["non_backend_config_sha256"] != prompt["config_non_backend_sha256"]:
        raise RuntimeError(f"paired config mismatch in {cell['cell']}")
    atomic_write_json(cell_dir / "input.json", input_evidence)
    return cell_dir, canonical, effective, input_evidence


def loadavg() -> list[float]:
    try:
        return [round(v, 3) for v in os.getloadavg()]
    except OSError:
        return []


def stream_replay(cmd: list[str], env: dict[str, str], log_path: Path) -> dict[str, Any]:
    from kicraft.build_slots import ACQUIRED_MARKER, WAITING_MARKER
    from kicraft.proc_tree import kill_tree

    started = time.monotonic()
    deadline = [started + WATCHDOG_S]
    acquired_elapsed: list[float | None] = [None]
    deadline_lock = threading.Lock()
    timed_out = threading.Event()
    marks: dict[str, float] = {}
    lines: list[str] = []
    master_seeds: list[int] = []
    round_timings: list[dict[str, Any]] = []
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
        bufsize=1,
    )

    def watchdog() -> None:
        while proc.poll() is None:
            with deadline_lock:
                expired = time.monotonic() >= deadline[0]
            if expired:
                timed_out.set()
                kill_tree(proc.pid)
                return
            time.sleep(0.25)

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        assert proc.stdout is not None
        for raw in proc.stdout:
            elapsed = time.monotonic() - started
            log.write(f"{elapsed:9.1f}  {raw}")
            line = raw.rstrip("\n")
            lines.append(line)
            if WAITING_MARKER in line and "slot_wait_start" not in marks:
                marks["slot_wait_start"] = round(elapsed, 3)
            if ACQUIRED_MARKER in line and "slot_acquired" not in marks:
                marks["slot_acquired"] = round(elapsed, 3)
                acquired_elapsed[0] = elapsed
                with deadline_lock:
                    deadline[0] = time.monotonic() + WATCHDOG_S
            for name, marker in MARKERS.items():
                if marker in line and name not in marks:
                    marks[name] = round(elapsed, 3)
            seed_match = RE_MASTER_SEED.search(line)
            if seed_match:
                master_seeds.append(int(seed_match.group(1)))
            timing_match = RE_ROUND_TIMING.search(line)
            if timing_match:
                round_timings.append({
                    "round": int(timing_match.group(1)),
                    "stage": timing_match.group(2),
                    "seconds": float(timing_match.group(3)),
                })
    rc = proc.wait()
    wall_s = round(time.monotonic() - started, 3)
    thread.join(timeout=1)
    slot_wait_s = None
    if "slot_wait_start" in marks and "slot_acquired" in marks:
        slot_wait_s = round(marks["slot_acquired"] - marks["slot_wait_start"], 3)
    elif "slot_acquired" in marks:
        slot_wait_s = 0.0
    phases = {
        "leaf_s": (
            round(marks["parent"] - marks["leaf"], 3)
            if "leaf" in marks and "parent" in marks else None
        ),
        "parent_s": (
            round(marks["promote"] - marks["parent"], 3)
            if "parent" in marks and "promote" in marks else None
        ),
        "promote_s": (
            round(marks["verify"] - marks["promote"], 3)
            if "promote" in marks and "verify" in marks else None
        ),
        "verify_s": (
            round(wall_s - marks["verify"], 3) if "verify" in marks else None
        ),
    }
    return {
        "rc": rc,
        "timed_out": timed_out.is_set(),
        "signal": signal.Signals(-rc).name if rc < 0 and -rc in signal.Signals._value2member_map_ else None,
        "wall_s": wall_s,
        "slot_wait_s": slot_wait_s,
        "marks": marks,
        "phases": phases,
        "master_seeds": master_seeds,
        "round_timings": round_timings,
        "watchdog_reset_elapsed_s": acquired_elapsed[0],
        "lines": lines,
    }


def status_for(rc: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if rc < 0:
        return "crash"
    return {0: "fab_ready", 3: "input_error", 5: "synth_fail", 6: "route_fail", 7: "gate_fail", 8: "determinism_fail"}.get(rc, f"error_rc{rc}")


def strip_raw_streams(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: strip_raw_streams(v)
            for k, v in value.items()
            if k not in {"_raw_stdout", "_raw_stderr"}
        }
    if isinstance(value, list):
        return [strip_raw_streams(v) for v in value]
    return value


def strip_transient_validation_paths(validation: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(validation)
    drc = out.get("drc")
    if isinstance(drc, dict):
        drc.pop("report_path", None)
    return strip_raw_streams(out)


def collect_path_strings(artifacts: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    if isinstance(artifacts.get("project"), str):
        paths.append(artifacts["project"])
    if isinstance(artifacts.get("promoted"), str):
        paths.append(artifacts["promoted"])
    prov = artifacts.get("promoted_provenance")
    if isinstance(prov, dict):
        for key in ("promoted_pcb", "source_board"):
            if isinstance(prov.get(key), str):
                paths.append(prov[key])
    entries = artifacts.get("artifacts")
    if isinstance(entries, dict):
        for entry in entries.values():
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
    return paths


def run_artifact_resolver(project: Path, cell_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        str(PYTHON), "-m", "kicraft.design.cli_app", "artifacts",
        "--project", str(project), "--kind", "all", "--json",
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"artifacts command failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-1000:]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"artifacts command returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("artifacts command returned a non-object")
    atomic_write_json(cell_dir / "artifacts.json", data)
    atomic_write_json(cell_dir / "artifacts-command.json", {
        "command": cmd, "rc": proc.returncode, "stderr": proc.stderr,
    })
    return data


def artifact_consistency(artifacts: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    reasons: list[str] = []
    for raw in collect_path_strings(artifacts):
        if not is_within(Path(raw), cell_dir):
            reasons.append(f"path_outside_cell:{raw}")
    promoted_raw = artifacts.get("promoted")
    promoted = Path(promoted_raw) if isinstance(promoted_raw, str) else None
    prov = artifacts.get("promoted_provenance") if isinstance(artifacts.get("promoted_provenance"), dict) else None
    routed = (artifacts.get("artifacts") or {}).get("routed") if isinstance(artifacts.get("artifacts"), dict) else None
    promoted_md5 = md5_file(promoted) if promoted and promoted.is_file() and is_within(promoted, cell_dir) else None
    promotion_present = bool(prov)
    if promoted is None or not promoted.is_file():
        reasons.append("promoted_board_missing")
    if not prov:
        reasons.append("promoted_provenance_missing")
    else:
        if prov.get("fresh") is not True:
            reasons.append("promotion_not_fresh")
        if prov.get("source_kind") != "routed":
            reasons.append(f"promotion_source_kind={prov.get('source_kind')}")
        if promoted_md5 != prov.get("md5"):
            reasons.append("promoted_md5_mismatch")
        source_raw = prov.get("source_board")
        source = Path(source_raw) if isinstance(source_raw, str) else None
        if source is None or not source.is_file() or not is_within(source, cell_dir):
            reasons.append("provenance_source_missing_or_outside")
        elif promoted_md5 != md5_file(source):
            reasons.append("provenance_source_md5_mismatch")
        if not isinstance(routed, dict):
            reasons.append("resolved_routed_missing")
        else:
            if routed.get("run_id") != prov.get("run_id"):
                reasons.append("resolved_routed_run_id_mismatch")
            if routed.get("md5") != promoted_md5:
                reasons.append("resolved_routed_md5_mismatch")
            if routed.get("matches_promoted") is not True:
                reasons.append("resolved_routed_not_promoted_run")
    return {
        "ok": not reasons,
        "promotion_present": promotion_present,
        "promoted_md5": promoted_md5,
        "reasons": reasons,
    }


def run_drc(promoted: Path, cell_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    output = cell_dir / "drc.json"
    cmd = [
        "kicad-cli", "pcb", "drc", "--format", "json", "--severity-error",
        "--output", str(output), str(promoted),
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    if not output.is_file():
        raise RuntimeError(
            f"kicad-cli DRC produced no JSON rc={proc.returncode}: {(proc.stderr or proc.stdout)[-1000:]}"
        )
    data = load_json(output)
    atomic_write_json(cell_dir / "drc-command.json", {
        "command": cmd, "rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
    })
    violations = data.get("violations") if isinstance(data.get("violations"), list) else []
    unconnected = data.get("unconnected_items") if isinstance(data.get("unconnected_items"), list) else []
    by_type = Counter(str(v.get("type") or v.get("rule") or "unknown") for v in violations if isinstance(v, dict))
    return {
        "rc": proc.returncode,
        "violation_count": len(violations),
        "unconnected_count": len(unconnected),
        "violation_types": dict(sorted(by_type.items())),
        "kicad_version": data.get("kicad_version"),
    }


def safe_json_path(raw: Any, cell_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_file() and is_within(path, cell_dir) else None


def collect_hierarchy(project: Path, cell_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    from kicraft.cli.artifact_paths import resolve_parent_board

    named_paths: list[Path] = []
    summary_path = project / ".experiments" / "hierarchical_summary.json"
    if not summary_path.is_file():
        evidence = {"hierarchical_summary": None, "parent": None, "selected_leaves": []}
        atomic_write_json(cell_dir / "hierarchy_evidence.json", evidence)
        return evidence, named_paths
    named_paths.append(summary_path)
    hierarchy = load_json(summary_path)
    best = hierarchy.get("best_round") if isinstance(hierarchy.get("best_round"), dict) else {}
    compact_summary = {
        "status": hierarchy.get("status"),
        "run_id": hierarchy.get("run_id"),
        "master_seed": hierarchy.get("master_seed"),
        "best_score": best.get("score", hierarchy.get("best_score")),
        "best_round": best.get("round_num"),
        "leaf_total": best.get("leaf_total"),
        "leaf_accepted": best.get("leaf_accepted"),
        "parent_composed": best.get("parent_composed"),
        "parent_routed": best.get("parent_routed"),
        "latest_stage": best.get("latest_stage"),
        "parent_pipeline_path": best.get("parent_output_json"),
    }
    parent_board = resolve_parent_board(project, kind="routed")
    parent_debug: dict[str, Any] | None = None
    parent_debug_path: Path | None = None
    if parent_board is not None and parent_board.is_file() and is_within(parent_board, cell_dir):
        named_paths.append(parent_board)
        candidate = parent_board.parent / "debug.json"
        if candidate.is_file():
            parent_debug_path = candidate
            named_paths.append(candidate)
            parent_debug = load_json(candidate)
    parent = None
    if parent_debug is not None:
        routing_result = parent_debug.get("routing_result") if isinstance(parent_debug.get("routing_result"), dict) else {}
        parent = {
            "board": str(parent_board),
            "debug_json": str(parent_debug_path),
            "validation": strip_raw_streams(parent_debug.get("validation")),
            "routing_result": strip_raw_streams(routing_result),
            "copper_verification": strip_raw_streams(routing_result.get("copper_verification")),
            "copper_accounting": strip_raw_streams(routing_result.get("copper_accounting")),
            "routing_stats": strip_raw_streams(
                routing_result.get("routing_stats") or routing_result.get("freerouting_stats") or {}
            ),
        }

    pipeline_path = safe_json_path(best.get("parent_output_json"), cell_dir)
    selected_leaves: list[dict[str, Any]] = []
    if pipeline_path is not None:
        named_paths.append(pipeline_path)
        pipeline = load_json(pipeline_path)
        artifacts = pipeline.get("artifacts") if isinstance(pipeline.get("artifacts"), list) else []
        for item in artifacts:
            artifact = item.get("artifact") if isinstance(item, dict) and isinstance(item.get("artifact"), dict) else {}
            source_files = artifact.get("source_files") if isinstance(artifact.get("source_files"), dict) else {}
            debug_path = safe_json_path(source_files.get("debug_json"), cell_dir)
            sub_id = artifact.get("subcircuit_id") if isinstance(artifact.get("subcircuit_id"), dict) else {}
            leaf: dict[str, Any] = {
                "sheet_name": sub_id.get("sheet_name"),
                "instance_path": sub_id.get("instance_path"),
                "debug_json": str(debug_path) if debug_path else None,
                "validation": None,
                "routing": None,
                "routing_stats": {},
            }
            if debug_path is not None:
                named_paths.append(debug_path)
                debug = load_json(debug_path)
                extra = debug.get("extra") if isinstance(debug.get("extra"), dict) else {}
                routing = extra.get("best_round_routing") if isinstance(extra.get("best_round_routing"), dict) else {}
                leaf["validation"] = strip_raw_streams(
                    extra.get("leaf_acceptance_structured") or extra.get("leaf_acceptance")
                )
                leaf["routing"] = strip_raw_streams(routing)
                leaf["routing_stats"] = strip_raw_streams(
                    routing.get("routing_stats") or routing.get("freerouting_stats") or {}
                )
            selected_leaves.append(leaf)
    evidence = {
        "hierarchical_summary": compact_summary,
        "parent": parent,
        "selected_leaves": selected_leaves,
    }
    atomic_write_json(cell_dir / "hierarchy_evidence.json", evidence)
    return evidence, named_paths


def stats_router_seconds(stats: dict[str, Any]) -> float | None:
    for key in ("elapsed_s", "router_time_s"):
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    vals = [stats.get("routing_seconds"), stats.get("optimization_seconds")]
    numeric = [float(v) for v in vals if isinstance(v, (int, float))]
    return sum(numeric) if numeric else None


def preservation_check(stats: dict[str, Any], label: str) -> dict[str, Any]:
    if not stats:
        return {"label": label, "applicable": True, "ok": False, "reason": "routing_stats_missing"}
    preservation = stats.get("input_copper_preservation")
    traces = preservation.get("traces") if isinstance(preservation, dict) and isinstance(preservation.get("traces"), dict) else {}
    vias = preservation.get("vias") if isinstance(preservation, dict) and isinstance(preservation.get("vias"), dict) else {}
    trace_missing = traces.get("missing_count")
    via_missing = vias.get("missing_count")
    backend_ok = stats.get("backend") == "kicad-routing-tools"
    ok = backend_ok and trace_missing == 0 and via_missing == 0
    return {
        "label": label,
        "applicable": True,
        "ok": ok,
        "backend": stats.get("backend"),
        "missing_traces": trace_missing,
        "missing_vias": via_missing,
        "reason": None if ok else "backend_or_input_copper_preservation_failed",
    }


def safety_evidence(backend: str, hierarchy: dict[str, Any], artifact: dict[str, Any], outline: dict[str, Any] | None, status: str, lines: list[str]) -> dict[str, Any]:
    adapter_checks: list[dict[str, Any]] = []
    parent = hierarchy.get("parent") if isinstance(hierarchy.get("parent"), dict) else None
    if backend == "krt":
        if parent is not None:
            adapter_checks.append(preservation_check(parent.get("routing_stats") or {}, "parent"))
        for leaf in hierarchy.get("selected_leaves") or []:
            routing = leaf.get("routing") if isinstance(leaf, dict) and isinstance(leaf.get("routing"), dict) else {}
            if routing.get("skipped") and routing.get("reason") == "no_internal_nets":
                adapter_checks.append({
                    "label": f"leaf:{leaf.get('sheet_name')}",
                    "applicable": False,
                    "ok": True,
                    "reason": "no_internal_nets_router_not_invoked",
                })
            else:
                adapter_checks.append(
                    preservation_check(leaf.get("routing_stats") or {}, f"leaf:{leaf.get('sheet_name')}")
                )
    copper = parent.get("copper_verification") if parent else None
    child_check: dict[str, Any]
    if isinstance(copper, dict) and copper:
        missing_traces = int(copper.get("expected_child_traces", 0) or 0) - int(copper.get("matched_child_traces", 0) or 0)
        missing_vias = int(copper.get("expected_child_vias", 0) or 0) - int(copper.get("matched_child_vias", 0) or 0)
        child_check = {
            "available": True,
            "ok": missing_traces == 0 and missing_vias == 0 and not (copper.get("issues") or []),
            "missing_traces": missing_traces,
            "missing_vias": missing_vias,
            "status": copper.get("status"),
            "issues": copper.get("issues") or [],
        }
    else:
        child_check = {"available": False, "ok": False, "missing_traces": None, "missing_vias": None}
    completed_route = status in {"fab_ready", "gate_fail"}
    adapter_error_in_log = any(
        "failed to preserve input copper" in line.lower() or "RoutingCopperPreservationError" in line
        for line in lines
    )
    return {
        "adapter_checks": adapter_checks,
        "adapter_ok": (
            all(c.get("ok") for c in adapter_checks if c.get("applicable"))
            and not adapter_error_in_log
            and (bool([c for c in adapter_checks if c.get("applicable")]) if completed_route and backend == "krt" else True)
        ),
        "adapter_error_in_log": adapter_error_in_log,
        "child_copper": child_check,
        "artifact": artifact,
        "outline": outline,
        "completed_route": completed_route,
    }


def collect_cell_evidence(
    cell: dict[str, Any], prompt: dict[str, Any], cell_dir: Path, canonical: Path,
    effective: dict[str, Any], env: dict[str, str], replay: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    from kicraft.autoplacer.freerouting_runner import validate_routed_board
    from kicraft.eval.outline_check import evaluate_outline_shape

    project = cell_dir / "project"
    artifacts = run_artifact_resolver(project, cell_dir, env)
    artifact = artifact_consistency(artifacts, cell_dir)
    named_paths = [
        cell_dir / "artifacts.json",
        cell_dir / "artifacts-command.json",
    ]
    promoted_raw = artifacts.get("promoted")
    promoted = Path(promoted_raw) if isinstance(promoted_raw, str) else canonical
    drc_metrics: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    outline: dict[str, Any] | None = None
    if promoted.is_file() and is_within(promoted, cell_dir):
        drc_metrics = run_drc(promoted, cell_dir, env)
        named_paths.extend([cell_dir / "drc.json", cell_dir / "drc-command.json", promoted])
        validation = strip_transient_validation_paths(
            validate_routed_board(str(promoted), cfg=effective)
        )
        atomic_write_json(cell_dir / "validation.json", validation)
        named_paths.append(cell_dir / "validation.json")
        if prompt.get("outline_shape"):
            outline = evaluate_outline_shape(promoted, prompt["outline_shape"])
            atomic_write_json(cell_dir / "outline.json", outline)
            named_paths.append(cell_dir / "outline.json")
    hierarchy, hierarchy_paths = collect_hierarchy(project, cell_dir)
    named_paths.extend([cell_dir / "hierarchy_evidence.json", *hierarchy_paths])
    for raw in collect_path_strings(artifacts):
        path = Path(raw)
        if path.is_file() and is_within(path, cell_dir):
            named_paths.append(path)

    drc = validation.get("drc") if isinstance(validation, dict) and isinstance(validation.get("drc"), dict) else {}
    tracks = validation.get("track_summary") if isinstance(validation, dict) and isinstance(validation.get("track_summary"), dict) else {}
    raw_clearance = int(drc.get("clearance", 0) or 0) if drc else None
    waived_clearance = int(validation.get("footprint_internal_clearance_count", 0) or 0) if validation else None
    genuine_clearance = max(0, raw_clearance - waived_clearance) if raw_clearance is not None and waived_clearance is not None else None
    router_times: list[float] = []
    parent = hierarchy.get("parent") if isinstance(hierarchy.get("parent"), dict) else None
    if parent:
        value = stats_router_seconds(parent.get("routing_stats") or {})
        if value is not None:
            router_times.append(value)
    for leaf in hierarchy.get("selected_leaves") or []:
        value = stats_router_seconds(leaf.get("routing_stats") or {})
        if value is not None:
            router_times.append(value)
    status = status_for(int(replay["rc"]), bool(replay["timed_out"]))
    safety = safety_evidence(cell["backend"], hierarchy, artifact, outline, status, replay["lines"])
    evidence = {
        "artifacts": artifacts,
        "artifact_consistency": artifact,
        "drc": drc_metrics,
        "validation": validation,
        "hierarchy": hierarchy,
        "outline": outline,
        "metrics": {
            "accepted": validation.get("accepted") if validation else None,
            "rejection_reasons": validation.get("rejection_reasons") if validation else [],
            "shorts": int(drc.get("shorts", 0) or 0) if drc else None,
            "unconnected": int(drc.get("unconnected", 0) or 0) if drc else None,
            "unconnected_nets": drc.get("unconnected_nets") or [] if drc else [],
            "clearance_raw": raw_clearance,
            "clearance_waived": waived_clearance,
            "clearance_genuine": genuine_clearance,
            "drc_total": int(drc.get("total", 0) or 0) if drc else None,
            "severity_error_drc_total": drc_metrics.get("violation_count") if drc_metrics else None,
            "malformed_board_geometry": validation.get("malformed_board_geometry") if validation else None,
            "illegal_routed_geometry": validation.get("obviously_illegal_routed_geometry") if validation else None,
            "footprints": tracks.get("footprints"),
            "traces": tracks.get("traces"),
            "vias": tracks.get("vias"),
            "routed_length_mm": tracks.get("total_length_mm"),
            "router_time_s": round(sum(router_times), 3) if router_times else None,
            "outline_level": outline.get("level") if outline else None,
        },
        "safety": safety,
    }
    atomic_write_json(cell_dir / "evidence.json", evidence)
    named_paths.append(cell_dir / "evidence.json")
    return evidence, named_paths


def run_cell(cell: dict[str, Any], prompt: dict[str, Any], batch_dir: Path, say) -> dict[str, Any]:
    cell_dir, canonical, effective, input_evidence = prepare_cell(cell, prompt, batch_dir)
    env = dict(os.environ)
    env.update(PINNED_ENV)
    env.pop("KICRAFT_QUALITY_PRESETS", None)
    cmd = [
        str(PYTHON), "-m", "kicraft.design.cli_app", "replay",
        "--project", str(cell_dir / "project"), "--quality", "good",
        "--seed", str(cell["seed"]), "--no-fab",
    ]
    command_record: dict[str, Any] = {
        "command": cmd,
        "cwd": str(REPO),
        "environment": PINNED_ENV,
        "effective_config": jsonable(effective),
        "input": input_evidence,
        "started_at": utc_iso(),
        "loadavg_start": loadavg(),
    }
    atomic_write_json(cell_dir / "command.json", command_record)
    say(f"[{cell['cell']}] start backend={BACKEND_DISPLAY[cell['backend']]} seed={cell['seed']}")
    replay = stream_replay(cmd, env, cell_dir / "replay.log")
    status = status_for(replay["rc"], replay["timed_out"])
    command_record.update({
        "finished_at": utc_iso(),
        "loadavg_end": loadavg(),
        "rc": replay["rc"],
        "timed_out": replay["timed_out"],
        "signal": replay["signal"],
        "wall_s": replay["wall_s"],
    })
    atomic_write_json(cell_dir / "command.json", command_record)
    evidence, named_paths = collect_cell_evidence(
        cell, prompt, cell_dir, canonical, effective, env, replay
    )
    named_paths.extend([
        cell_dir / "input.json", cell_dir / "command.json", cell_dir / "replay.log",
    ])
    checksums: dict[str, str] = {}
    for path in sorted(set(p.resolve() for p in named_paths if p.is_file()), key=str):
        rel = relative_evidence(path, batch_dir)
        checksums[rel] = sha256_file(path)
    row = {
        **cell,
        "backend_display": BACKEND_DISPLAY[cell["backend"]],
        "started_at": command_record["started_at"],
        "finished_at": command_record["finished_at"],
        "rc": replay["rc"],
        "status": status,
        "timed_out": replay["timed_out"],
        "signal": replay["signal"],
        "wall_s": replay["wall_s"],
        "slot_wait_s": replay["slot_wait_s"],
        "phase_wall_s": replay["phases"],
        "marks": replay["marks"],
        "master_seeds": replay["master_seeds"],
        "round_timings": replay["round_timings"],
        "watchdog_reset_elapsed_s": replay["watchdog_reset_elapsed_s"],
        "loadavg_start": command_record["loadavg_start"],
        "loadavg_end": command_record["loadavg_end"],
        "input": input_evidence,
        "metrics": evidence["metrics"],
        "safety": evidence["safety"],
        "evidence_paths": {
            "cell": relative_evidence(cell_dir, batch_dir),
            "replay_log": relative_evidence(cell_dir / "replay.log", batch_dir),
            "artifacts": relative_evidence(cell_dir / "artifacts.json", batch_dir),
            "drc": relative_evidence(cell_dir / "drc.json", batch_dir) if (cell_dir / "drc.json").is_file() else None,
            "validation": relative_evidence(cell_dir / "validation.json", batch_dir) if (cell_dir / "validation.json").is_file() else None,
            "hierarchy": relative_evidence(cell_dir / "hierarchy_evidence.json", batch_dir),
            "evidence": relative_evidence(cell_dir / "evidence.json", batch_dir),
        },
        "evidence_checksums": checksums,
    }
    say(
        f"[{cell['cell']}] {status} rc={replay['rc']} wall={replay['wall_s']}s "
        f"shorts={row['metrics']['shorts']} unconnected={row['metrics']['unconnected']} "
        f"genuine_clearance={row['metrics']['clearance_genuine']}"
    )
    return row


def median(values: list[float | int | None], digits: int = 3) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return round(statistics.median(nums), digits) if nums else None


def percentile(values: list[float | int | None], p: float, digits: int = 3) -> float | None:
    nums = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not nums:
        return None
    if len(nums) == 1:
        return round(nums[0], digits)
    pos = (len(nums) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    value = nums[lo] + (nums[hi] - nums[lo]) * (pos - lo)
    return round(value, digits)


def backend_rollup(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    threshold = len(seeds) // 2 + 1
    statuses = [r.get("status") for r in rows]
    metrics = [r.get("metrics") or {} for r in rows]
    fab_n = sum(s == "fab_ready" for s in statuses)
    completed_n = sum(s not in {"timeout", "crash", "harness_error"} for s in statuses)
    crash_timeout_n = sum(s in {"timeout", "crash"} for s in statuses)
    short_free_n = sum(
        s not in {"timeout", "crash", "harness_error"} and m.get("shorts") == 0
        for s, m in zip(statuses, metrics)
    )
    shorts_n = sum(isinstance(m.get("shorts"), (int, float)) and m.get("shorts") > 0 for m in metrics)
    return {
        "n": len(rows),
        "required_n": len(seeds),
        "majority_threshold": threshold,
        "statuses": statuses,
        "status_counts": dict(sorted(Counter(statuses).items(), key=lambda item: str(item[0]))),
        "majority_fab_ready": fab_n >= threshold,
        "majority_completed": completed_n >= threshold,
        "majority_crash_timeout": crash_timeout_n >= threshold,
        "majority_short_free": short_free_n >= threshold,
        "majority_has_shorts": shorts_n >= threshold,
        "fab_ready_n": fab_n,
        "median": {
            "wall_s": median([r.get("wall_s") for r in rows]),
            "router_time_s": median([m.get("router_time_s") for m in metrics]),
            "shorts": median([m.get("shorts") for m in metrics]),
            "unconnected": median([m.get("unconnected") for m in metrics]),
            "clearance_genuine": median([m.get("clearance_genuine") for m in metrics]),
            "drc_total": median([m.get("drc_total") for m in metrics]),
            "severity_error_drc_total": median([m.get("severity_error_drc_total") for m in metrics]),
            "outline_level": median([m.get("outline_level") for m in metrics]),
        },
        "instability": len(set(statuses)) > 1 or (0 < fab_n < len(rows)),
    }


def quality_tuple(rollup: dict[str, Any]) -> tuple[float, float, float]:
    med = rollup.get("median") or {}
    return tuple(float(med.get(key)) if isinstance(med.get(key), (int, float)) else math.inf for key in ("shorts", "unconnected", "clearance_genuine"))


def latest_result_map(batch_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_result_rows(batch_dir / "results.jsonl"):
        cell = row.get("cell")
        if cell:
            latest[str(cell)] = row
    return latest


def classify_experiment(prompt_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]], prompt_summary: list[dict[str, Any]], missing_cells: list[str]) -> dict[str, Any]:
    eligible_n = sum(r["route_eligible"] for r in prompt_rows)
    qualifier = f"{eligible_n}/34 fresh self-eval prompts route-eligible"
    if missing_cells:
        return {
            "label": "incomplete",
            "display": "Incomplete",
            "qualifier": qualifier,
            "reasons": [f"{len(missing_cells)} required router cells missing or invalid"],
        }
    hard_reasons: list[str] = []
    for row in result_rows:
        if row.get("backend") != "krt":
            continue
        safety = row.get("safety") or {}
        if safety.get("adapter_error_in_log"):
            hard_reasons.append(f"{row['cell']}: adapter reported input-copper loss")
        if safety.get("completed_route") and not safety.get("adapter_ok"):
            hard_reasons.append(f"{row['cell']}: KRT adapter preservation evidence failed or missing")
        child = safety.get("child_copper") or {}
        if safety.get("completed_route") and (not child.get("available") or not child.get("ok")):
            hard_reasons.append(f"{row['cell']}: composed child-copper preservation failed or missing")
        artifact = safety.get("artifact") or {}
        if artifact.get("promotion_present") and not artifact.get("ok"):
            hard_reasons.append(f"{row['cell']}: stale or byte-inconsistent routed promotion")
        outline = safety.get("outline")
        if isinstance(outline, dict) and outline.get("level") != 4:
            hard_reasons.append(f"{row['cell']}: shaped outline degraded to level {outline.get('level')}")
    for prompt in prompt_summary:
        if not prompt.get("route_eligible"):
            continue
        fr = prompt["backends"]["freerouting"]
        krt = prompt["backends"]["krt"]
        if fr["majority_completed"] and fr["majority_short_free"] and (
            krt["majority_crash_timeout"] or krt["majority_has_shorts"]
        ):
            hard_reasons.append(
                f"{prompt['slug']}: KRT majority crashed/timed out or had shorts while FreeRouting completed short-free"
            )
    if hard_reasons:
        return {
            "label": "not_viable",
            "display": "Not viable",
            "qualifier": qualifier,
            "reasons": sorted(set(hard_reasons)),
        }

    keep_reasons: list[str] = []
    eligible_prompts = [p for p in prompt_summary if p.get("route_eligible")]
    fr_fab = sum(p["backends"]["freerouting"]["majority_fab_ready"] for p in eligible_prompts)
    krt_fab = sum(p["backends"]["krt"]["majority_fab_ready"] for p in eligible_prompts)
    losses = [
        p["slug"] for p in eligible_prompts
        if p["backends"]["freerouting"]["majority_fab_ready"]
        and not p["backends"]["krt"]["majority_fab_ready"]
    ]
    gains = [
        p["slug"] for p in eligible_prompts
        if p["backends"]["krt"]["majority_fab_ready"]
        and not p["backends"]["freerouting"]["majority_fab_ready"]
    ]
    if losses:
        keep_reasons.append(f"FreeRouting-only majority fab-ready prompts: {', '.join(losses)}")
    if krt_fab < fr_fab:
        keep_reasons.append(f"KRT majority fab-ready count {krt_fab} is below FreeRouting {fr_fab}")
    regressions: list[str] = []
    for p in eligible_prompts:
        fr_med = p["backends"]["freerouting"]["median"]
        krt_med = p["backends"]["krt"]["median"]
        for metric in ("shorts", "unconnected", "clearance_genuine"):
            a, b = fr_med.get(metric), krt_med.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a:
                regressions.append(f"{p['slug']}:{metric} {b}>{a}")
    if regressions:
        keep_reasons.append("KRT per-prompt routing-quality regressions: " + ", ".join(regressions))
    if keep_reasons:
        return {
            "label": "keep_freerouting_default",
            "display": "Keep FreeRouting default",
            "qualifier": qualifier,
            "reasons": keep_reasons,
        }

    paired = [
        p for p in eligible_prompts
        if p["backends"]["freerouting"]["majority_completed"]
        and p["backends"]["krt"]["majority_completed"]
        and isinstance(p["backends"]["freerouting"]["median"]["wall_s"], (int, float))
        and isinstance(p["backends"]["krt"]["median"]["wall_s"], (int, float))
    ]
    fr_corpus_wall = median([p["backends"]["freerouting"]["median"]["wall_s"] for p in paired])
    krt_corpus_wall = median([p["backends"]["krt"]["median"]["wall_s"] for p in paired])
    krt_faster = sum(
        p["backends"]["krt"]["median"]["wall_s"] < p["backends"]["freerouting"]["median"]["wall_s"]
        for p in paired
    )
    speed_condition = (
        isinstance(fr_corpus_wall, (int, float))
        and isinstance(krt_corpus_wall, (int, float))
        and krt_corpus_wall <= 0.8 * fr_corpus_wall
        and paired
        and krt_faster / len(paired) >= 2 / 3
    )
    gain_condition = bool(gains) and not losses
    if krt_fab >= fr_fab and (gain_condition or speed_condition):
        reason = (
            f"KRT gains majority fab-ready prompts without losses: {', '.join(gains)}"
            if gain_condition
            else f"KRT corpus median wall {krt_corpus_wall}s is at least 20% below FreeRouting {fr_corpus_wall}s and is faster on {krt_faster}/{len(paired)} paired prompts"
        )
        return {
            "label": "migration_candidate",
            "display": "Migration candidate",
            "qualifier": qualifier,
            "reasons": [reason],
        }
    return {
        "label": "keep_freerouting_default",
        "display": "Keep FreeRouting default",
        "qualifier": qualifier,
        "reasons": ["KRT passed safety and quality gates but did not meet the pre-registered gain or speed threshold"],
    }


def summarize(batch_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("manifest identity missing")
    source = identity.get("source") or {}
    source_summary = Path(str(source.get("summary_path") or ""))
    if not source_summary.is_file() or sha256_file(source_summary) != source.get("summary_sha256"):
        raise RuntimeError("frozen source summary is missing or changed")
    prompt_rows = identity.get("prompts") if isinstance(identity.get("prompts"), list) else []
    if len(prompt_rows) != EXPECTED_BRIEFS_N:
        raise RuntimeError("manifest must contain exactly 34 prompt rows")
    matrix = identity.get("matrix") if isinstance(identity.get("matrix"), list) else []
    seeds = identity.get("seeds") if isinstance(identity.get("seeds"), list) else []
    latest = latest_result_map(batch_dir)
    valid_rows: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    invalid_evidence: dict[str, list[str]] = {}
    for cell in matrix:
        name = cell["cell"]
        row = latest.get(name)
        if not row or row.get("status") == "harness_error":
            missing_cells.append(name)
            continue
        valid, reasons = validate_result_evidence(row, batch_dir)
        if not valid:
            missing_cells.append(name)
            invalid_evidence[name] = reasons
            continue
        valid_rows.append(row)
    by_prompt_backend: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in valid_rows:
        by_prompt_backend.setdefault((int(row["index"]), str(row["backend"])), []).append(row)
    for rows in by_prompt_backend.values():
        rows.sort(key=lambda r: int(r["seed"]))

    prompt_summary: list[dict[str, Any]] = []
    for prompt in sorted(prompt_rows, key=lambda r: int(r["index"])):
        out = {
            "index": prompt["index"],
            "slug": prompt["slug"],
            "archetype": prompt["archetype"],
            "brief": prompt["brief"],
            "outline_shape": prompt.get("outline_shape"),
            "route_eligible": prompt["route_eligible"],
            "eligibility_reasons": prompt.get("eligibility_reasons") or [],
            "source": prompt.get("source") or {},
        }
        if not prompt["route_eligible"]:
            out["router_comparison"] = "not_applicable"
            out["backends"] = None
        else:
            backends = {
                backend: backend_rollup(by_prompt_backend.get((prompt["index"], backend), []), seeds)
                for backend in BACKENDS
            }
            fr, krt = backends["freerouting"], backends["krt"]
            if fr["majority_fab_ready"] and not krt["majority_fab_ready"]:
                comparison = "freerouting_only"
            elif krt["majority_fab_ready"] and not fr["majority_fab_ready"]:
                comparison = "krt_only"
            elif quality_tuple(krt) > quality_tuple(fr):
                comparison = "krt_quality_regression"
            elif quality_tuple(krt) < quality_tuple(fr):
                comparison = "krt_quality_improvement"
            else:
                comparison = "tie"
            out["router_comparison"] = comparison
            out["backends"] = backends
            out["instability"] = {
                backend: backends[backend]["instability"] for backend in BACKENDS
            }
        prompt_summary.append(out)

    eligible = [p for p in prompt_summary if p["route_eligible"]]
    exclusions = Counter(reason for p in prompt_summary if not p["route_eligible"] for reason in p["eligibility_reasons"])
    fr_only = [p["slug"] for p in eligible if p["backends"]["freerouting"]["majority_fab_ready"] and not p["backends"]["krt"]["majority_fab_ready"]]
    krt_only = [p["slug"] for p in eligible if p["backends"]["krt"]["majority_fab_ready"] and not p["backends"]["freerouting"]["majority_fab_ready"]]
    paired_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    row_key = {(r["seed"], r["index"], r["backend"]): r for r in valid_rows}
    for prompt in eligible:
        for seed in seeds:
            fr = row_key.get((seed, prompt["index"], "freerouting"))
            krt = row_key.get((seed, prompt["index"], "krt"))
            if fr and krt:
                paired_rows.append((fr, krt))
    fr_walls = [a["wall_s"] for a, _ in paired_rows]
    krt_walls = [b["wall_s"] for _, b in paired_rows]
    fr_router = [(a.get("metrics") or {}).get("router_time_s") for a, _ in paired_rows]
    krt_router = [(b.get("metrics") or {}).get("router_time_s") for _, b in paired_rows]
    aggregate = {
        "eligible_n": len(eligible),
        "excluded_n": EXPECTED_BRIEFS_N - len(eligible),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "expected_cells": len(matrix),
        "complete_cells": len(valid_rows),
        "missing_cells": missing_cells,
        "invalid_evidence": invalid_evidence,
        "majority_fab_ready": {
            backend: sum(p["backends"][backend]["majority_fab_ready"] for p in eligible)
            for backend in BACKENDS
        },
        "freerouting_only_prompts": fr_only,
        "krt_only_prompts": krt_only,
        "status_counts": {
            backend: dict(sorted(Counter(r["status"] for r in valid_rows if r["backend"] == backend).items()))
            for backend in BACKENDS
        },
        "paired_timing": {
            "n": len(paired_rows),
            "wall_median_s": {"freerouting": median(fr_walls), "krt": median(krt_walls)},
            "wall_p90_s": {"freerouting": percentile(fr_walls, 0.9), "krt": percentile(krt_walls, 0.9)},
            "router_median_s": {"freerouting": median(fr_router), "krt": median(krt_router)},
            "router_p90_s": {"freerouting": percentile(fr_router, 0.9), "krt": percentile(krt_router, 0.9)},
            "faster_wall_cells": {
                "freerouting": sum(a["wall_s"] < b["wall_s"] for a, b in paired_rows),
                "krt": sum(b["wall_s"] < a["wall_s"] for a, b in paired_rows),
                "ties": sum(a["wall_s"] == b["wall_s"] for a, b in paired_rows),
            },
        },
        "safety_regressions": {
            backend: {
                "artifact": sum(not (r.get("safety") or {}).get("artifact", {}).get("ok", False) for r in valid_rows if r["backend"] == backend and (r.get("safety") or {}).get("artifact", {}).get("promotion_present")),
                "child_copper": sum((r.get("safety") or {}).get("completed_route") and not (r.get("safety") or {}).get("child_copper", {}).get("ok", False) for r in valid_rows if r["backend"] == backend),
                "adapter": sum((r.get("safety") or {}).get("completed_route") and not (r.get("safety") or {}).get("adapter_ok", True) for r in valid_rows if r["backend"] == backend),
                "outline": sum(isinstance((r.get("safety") or {}).get("outline"), dict) and (r.get("safety") or {}).get("outline", {}).get("level") != 4 for r in valid_rows if r["backend"] == backend),
            }
            for backend in BACKENDS
        },
    }

    archetypes: list[dict[str, Any]] = []
    for archetype in sorted({p["archetype"] for p in prompt_summary}):
        prompts = [p for p in eligible if p["archetype"] == archetype]
        archetypes.append({
            "archetype": archetype,
            "eligible_n": len(prompts),
            "excluded_n": sum(p["archetype"] == archetype and not p["route_eligible"] for p in prompt_summary),
            "majority_fab_ready": {
                backend: sum(p["backends"][backend]["majority_fab_ready"] for p in prompts)
                for backend in BACKENDS
            },
            "median_wall_s": {
                backend: median([p["backends"][backend]["median"]["wall_s"] for p in prompts])
                for backend in BACKENDS
            },
            "median_shorts": {
                backend: median([p["backends"][backend]["median"]["shorts"] for p in prompts])
                for backend in BACKENDS
            },
            "median_unconnected": {
                backend: median([p["backends"][backend]["median"]["unconnected"] for p in prompts])
                for backend in BACKENDS
            },
            "median_genuine_clearance": {
                backend: median([p["backends"][backend]["median"]["clearance_genuine"] for p in prompts])
                for backend in BACKENDS
            },
        })

    classification = classify_experiment(prompt_rows, valid_rows, prompt_summary, missing_cells)
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "schema_version": "krt-self-eval-router-ab-summary-v1",
        "manifest_sha256": manifest_sha,
        "source": source,
        "seeds": seeds,
        "coverage": {
            "eligible_n": len(eligible),
            "excluded_n": EXPECTED_BRIEFS_N - len(eligible),
            "qualifier": f"{len(eligible)}/34 fresh self-eval prompts route-eligible",
        },
        "aggregate": aggregate,
        "archetypes": archetypes,
        "prompts": prompt_summary,
        "seed_rows": sorted(valid_rows, key=lambda r: (int(r["seed"]), int(r["index"]), BACKENDS.index(r["backend"]))),
        "classification": classification,
    }
    summary_path = batch_dir / "summary.json"
    report_path = batch_dir / "report.md"
    atomic_write_json(summary_path, summary)
    atomic_write_text(report_path, render_report(summary, batch_dir))
    return summary_path, report_path, summary


def fmt(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def render_report(summary: dict[str, Any], batch_dir: Path) -> str:
    source = summary["source"]
    aggregate = summary["aggregate"]
    classification = summary["classification"]
    lines = [
        f"# Self-eval router A/B — {batch_dir.name}",
        "",
        f"**Classification: {classification['display']} — {classification['qualifier']}.**",
        "",
        *[f"- {reason}" for reason in classification.get("reasons") or []],
        "",
        "## Frozen source",
        "",
        f"- Batch: `{source['path']}`",
        f"- Summary SHA-256: `{source['summary_sha256']}`",
        f"- Design model: `{source.get('design_model')}`; judge: `{source.get('judge_model')}`; rubric: `{source.get('rubric_version')}`",
        f"- Cost: `${source.get('total_cost_usd')}`; source wall: `{source.get('wall_s')}s`",
        f"- Coverage: **{summary['coverage']['qualifier']}**; {aggregate['excluded_n']} shared upstream failures.",
        "",
        "## Aggregate",
        "",
        "| Metric | FreeRouting | KiCadRoutingTools |",
        "|---|---:|---:|",
        f"| Majority fab-ready prompts | {aggregate['majority_fab_ready']['freerouting']}/{aggregate['eligible_n']} | {aggregate['majority_fab_ready']['krt']}/{aggregate['eligible_n']} |",
        f"| Paired wall median | {fmt(aggregate['paired_timing']['wall_median_s']['freerouting'], 's')} | {fmt(aggregate['paired_timing']['wall_median_s']['krt'], 's')} |",
        f"| Paired wall p90 | {fmt(aggregate['paired_timing']['wall_p90_s']['freerouting'], 's')} | {fmt(aggregate['paired_timing']['wall_p90_s']['krt'], 's')} |",
        f"| Paired router median | {fmt(aggregate['paired_timing']['router_median_s']['freerouting'], 's')} | {fmt(aggregate['paired_timing']['router_median_s']['krt'], 's')} |",
        f"| Faster paired cells | {aggregate['paired_timing']['faster_wall_cells']['freerouting']} | {aggregate['paired_timing']['faster_wall_cells']['krt']} |",
        f"| Artifact regressions | {aggregate['safety_regressions']['freerouting']['artifact']} | {aggregate['safety_regressions']['krt']['artifact']} |",
        f"| Child-copper regressions | {aggregate['safety_regressions']['freerouting']['child_copper']} | {aggregate['safety_regressions']['krt']['child_copper']} |",
        "",
        f"FreeRouting-only prompts: `{', '.join(aggregate['freerouting_only_prompts']) or 'none'}`.  ",
        f"KRT-only prompts: `{', '.join(aggregate['krt_only_prompts']) or 'none'}`.",
        "",
        "## Prompt outcomes",
        "",
        "| # | Prompt | Archetype | Upstream | Eligible | FreeRouting majority / med S-U-C / wall | KRT majority / med S-U-C / wall | Comparison |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for prompt in summary["prompts"]:
        upstream = prompt["source"]
        if not prompt["route_eligible"]:
            eligible = "no: " + ", ".join(prompt["eligibility_reasons"])
            fr_text = krt_text = "N/A"
        else:
            eligible = "yes"
            texts = {}
            for backend in BACKENDS:
                roll = prompt["backends"][backend]
                med = roll["median"]
                texts[backend] = (
                    f"{'fab' if roll['majority_fab_ready'] else 'not-fab'} / "
                    f"{fmt(med['shorts'])}-{fmt(med['unconnected'])}-{fmt(med['clearance_genuine'])} / "
                    f"{fmt(med['wall_s'], 's')}"
                )
            fr_text, krt_text = texts["freerouting"], texts["krt"]
        lines.append(
            f"| {prompt['index']} | `{prompt['slug']}` | {prompt['archetype']} | "
            f"{upstream.get('grade')}/{upstream.get('build_label')} | {eligible} | "
            f"{fr_text} | {krt_text} | {prompt['router_comparison']} |"
        )
    lines += [
        "",
        "S-U-C = median shorts, unconnected items, genuine clearances. Source grade appears once as shared upstream context; it does not choose the router winner.",
        "",
        "## Per archetype",
        "",
        "| Archetype | eligible/excluded | FreeRouting fab | KRT fab | FreeRouting wall | KRT wall | FreeRouting S/U/C | KRT S/U/C |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["archetypes"]:
        fr_suc = f"{fmt(row['median_shorts']['freerouting'])}/{fmt(row['median_unconnected']['freerouting'])}/{fmt(row['median_genuine_clearance']['freerouting'])}"
        krt_suc = f"{fmt(row['median_shorts']['krt'])}/{fmt(row['median_unconnected']['krt'])}/{fmt(row['median_genuine_clearance']['krt'])}"
        lines.append(
            f"| {row['archetype']} | {row['eligible_n']}/{row['excluded_n']} | "
            f"{row['majority_fab_ready']['freerouting']} | {row['majority_fab_ready']['krt']} | "
            f"{fmt(row['median_wall_s']['freerouting'], 's')} | {fmt(row['median_wall_s']['krt'], 's')} | {fr_suc} | {krt_suc} |"
        )
    lines += [
        "",
        "## Seed-level evidence",
        "",
        "| Seed | Prompt | Backend | Status | rc | wall | router | shorts | unconnected | genuine clearance | DRC total | artifact | child copper | Evidence |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in summary["seed_rows"]:
        metrics = row.get("metrics") or {}
        safety = row.get("safety") or {}
        lines.append(
            f"| {row['seed']} | `{row['slug']}` | {BACKEND_DISPLAY[row['backend']]} | {row['status']} | {row['rc']} | "
            f"{fmt(row.get('wall_s'), 's')} | {fmt(metrics.get('router_time_s'), 's')} | {fmt(metrics.get('shorts'))} | "
            f"{fmt(metrics.get('unconnected'))} | {fmt(metrics.get('clearance_genuine'))} | {fmt(metrics.get('drc_total'))} | "
            f"{(safety.get('artifact') or {}).get('ok')} | {(safety.get('child_copper') or {}).get('ok')} | "
            f"`{row['evidence_paths']['cell']}` |"
        )
    lines += [
        "",
        "## Exclusions",
        "",
    ]
    excluded = [p for p in summary["prompts"] if not p["route_eligible"]]
    if excluded:
        lines += [f"- `{p['slug']}`: {', '.join(p['eligibility_reasons'])}; upstream `{p['source'].get('build_label')}`." for p in excluded]
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Method boundary",
        "",
        "This is an end-to-end production-pipeline comparison over one frozen synthesized batch. FreeRouting retains KiCraft's DSN/SES conversion and dedicated workarounds; KiCadRoutingTools routes `.kicad_pcb` directly. It is not an isolated router-algorithm benchmark. Every cell used `quality=good`, the same seed and non-backend project configuration, no synthesis, no LLM, no judge, no fab export, and a 2400-second build-slot-aware watchdog.",
        "",
        f"Machine verdict: `{(batch_dir / 'summary.json').relative_to(batch_dir)}`. Manifest: `{(batch_dir / 'manifest.json').relative_to(batch_dir)}`.",
        "",
    ]
    return "\n".join(lines)


def parse_seeds(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("seeds must be non-negative integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", help="completed 34-prompt self-eval batch")
    parser.add_argument("--batch-dir", help="router batch directory to create or resume")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated placement seeds")
    parser.add_argument("--slugs", default="", help="comma-separated prompt slug restriction")
    parser.add_argument("--limit", type=int, default=0, help="cap pending cells (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="preflight and print pending matrix only")
    parser.add_argument("--summarize", metavar="BATCH_DIR", help="regenerate reports without routing")
    args = parser.parse_args(argv)

    if args.summarize:
        if args.source_batch or args.batch_dir:
            parser.error("--summarize cannot be combined with --source-batch or --batch-dir")
        summary_path, report_path, summary = summarize(Path(args.summarize).resolve())
        print(f"wrote {summary_path}")
        print(f"wrote {report_path}")
        print(f"classification: {summary['classification']['display']} — {summary['classification']['qualifier']}")
        return 0
    if not args.source_batch or not args.batch_dir:
        parser.error("normal and dry runs require --source-batch and --batch-dir")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    selected_slugs = [token.strip() for token in args.slugs.split(",") if token.strip()]
    source_batch = Path(args.source_batch).resolve()
    batch_dir = Path(args.batch_dir).resolve()

    identity, prompt_rows, matrix = build_identity(source_batch, seeds, selected_slugs)
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest(batch_dir, identity)
    results_path = batch_dir / "results.jsonl"
    done = reusable_cells(results_path, batch_dir)
    pending = [cell for cell in matrix if cell["cell"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(
        f"source={source_batch} eligible={sum(r['route_eligible'] for r in prompt_rows)}/34 "
        f"matrix={len(matrix)} done={len(done)} pending_selected={len(pending)}"
    )
    for cell in pending:
        print(
            f"{cell['cell']} seed={cell['seed']} index={cell['index']} "
            f"backend={BACKEND_DISPLAY[cell['backend']]} timeout={WATCHDOG_S}s"
        )
    if args.dry_run:
        print(f"manifest={batch_dir / 'manifest.json'} sha256={sha256_file(batch_dir / 'manifest.json')}")
        return 0

    batch_log_path = batch_dir / "batch.log"
    with batch_log_path.open("a", encoding="utf-8", buffering=1) as batch_log:
        def say(message: str) -> None:
            line = f"{utc_iso()} {message}"
            print(line, flush=True)
            batch_log.write(line + "\n")

        say(
            f"batch start manifest_sha256={sha256_file(batch_dir / 'manifest.json')} "
            f"matrix={len(matrix)} done={len(done)} selected={len(pending)}"
        )
        by_index = {int(row["index"]): row for row in prompt_rows}
        for ordinal, cell in enumerate(pending, 1):
            say(f"cell {ordinal}/{len(pending)} {cell['cell']}")
            try:
                row = run_cell(cell, by_index[int(cell["index"])], batch_dir, say)
            except Exception as exc:  # collection/setup failure, not a router verdict
                row = {
                    **cell,
                    "status": "harness_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_iso(),
                }
                say(f"[{cell['cell']}] harness_error: {row['error']}")
            append_jsonl(results_path, row)
            try:
                summary_path, report_path, summary = summarize(batch_dir)
                say(
                    f"reports {summary_path.name}/{report_path.name} "
                    f"classification={summary['classification']['label']}"
                )
            except Exception as exc:
                say(f"summary regeneration failed: {type(exc).__name__}: {exc}")
        summary_path, report_path, summary = summarize(batch_dir)
        say(
            f"batch complete complete_cells={summary['aggregate']['complete_cells']}/"
            f"{summary['aggregate']['expected_cells']} classification={summary['classification']['label']}"
        )
        say(f"summary={summary_path} report={report_path}")
    _ = manifest
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
