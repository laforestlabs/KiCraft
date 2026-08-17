"""Configurable PCB routing backends.

FreeRouting remains the default.  KiCadRoutingTools is an experimental direct
``.kicad_pcb`` backend, deliberately isolated from the DSN/SES transformations
in :mod:`freerouting_runner`.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


KICAD_ROUTING_TOOLS_VERSION = "0.20.2"
KICAD_ROUTING_TOOLS_COMMIT = "3ceb773722bea67aa3685e7ee430c0c0d17ef38d"
SUPPORTED_BACKENDS = ("freerouting", "kicad-routing-tools")
_KRT_NATIVE_VERSION = "0.20.1"
_KRT_PREFLIGHT_CACHE: dict[tuple[str, str], dict[str, str]] = {}



class RoutingBackendUnavailableError(RuntimeError):
    """The selected routing backend is not installed or usable."""


class RoutingCopperPreservationError(RuntimeError):
    """KiCadRoutingTools returned a board missing authoritative input copper."""

    def __init__(self, message: str, stats: dict[str, Any]) -> None:
        super().__init__(message)
        self.stats = stats


def routing_backend(config: dict[str, Any] | None = None) -> str:
    name = str((config or {}).get("routing_backend", "freerouting")).strip().lower()
    aliases = {"kicadroutingtools": "kicad-routing-tools", "krt": "kicad-routing-tools"}
    name = aliases.get(name, name)
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown routing_backend {name!r}; choose one of {', '.join(SUPPORTED_BACKENDS)}"
        )
    return name


def _krt_root(config: dict[str, Any]) -> Path:
    raw = config.get("kicad_routing_tools_path", "")
    if not raw and "kicad_routing_tools_path" in config:
        raw = os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PATH", "")
    raw = str(raw or "").strip()
    if not raw:
        raise RoutingBackendUnavailableError(
            "KiCadRoutingTools is selected but kicad_routing_tools_path is unset. "
            f"Clone commit {KICAD_ROUTING_TOOLS_COMMIT} and configure its repository path."
        )
    return Path(os.path.expanduser(raw)).resolve()


def preflight_routing_backend(config: dict[str, Any] | None = None) -> dict[str, str]:
    if config is None:
        from kicraft.autoplacer.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    backend = routing_backend(config)
    if backend == "freerouting":
        from kicraft.autoplacer.freerouting_runner import preflight_routing_toolchain
        java, jar = preflight_routing_toolchain(config)
        return {"backend": backend, "java": java, "jar": jar}
    root = _krt_root(config)
    raw_python = config.get("kicad_routing_tools_python", sys.executable)
    if (not raw_python and "kicad_routing_tools_python" in config):
        raw_python = os.environ.get(
            "KICRAFT_KICAD_ROUTING_TOOLS_PYTHON", sys.executable
        )
    raw_python = str(raw_python or sys.executable)
    python = shutil.which(os.path.expanduser(raw_python))
    if python is None:
        candidate = Path(os.path.expanduser(raw_python))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            python = str(candidate.absolute())
    if python is None:
        raise RoutingBackendUnavailableError(
            f"KiCadRoutingTools Python interpreter not found: {raw_python}"
        )
    # Keep a virtualenv's executable symlink intact: resolving it to the base
    # interpreter discards the virtualenv dependency context at process launch.
    python = os.path.abspath(python)
    cache_key = (str(root), python)
    cached = _KRT_PREFLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    route_script = root / "py_router" / "route.py"
    version_file = root / "VERSION"
    problems: list[str] = []
    if not route_script.is_file():
        problems.append(f"route CLI not found: {route_script}")

    source_version = ""
    if not version_file.is_file():
        problems.append(f"VERSION file not found: {version_file}")
    else:
        try:
            source_version = version_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            problems.append(f"could not read VERSION file {version_file}: {exc}")
        else:
            if source_version != KICAD_ROUTING_TOOLS_VERSION:
                problems.append(
                    f"VERSION is {source_version!r}, "
                    f"expected pinned {KICAD_ROUTING_TOOLS_VERSION!r}"
                )

    revision = ""
    if not (root / ".git").exists():
        problems.append(f"Git checkout metadata not found: {root / '.git'}")
    else:
        try:
            revision_check = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"could not verify Git checkout revision: {exc}")
        else:
            revision = revision_check.stdout.strip()
            if revision_check.returncode or revision != KICAD_ROUTING_TOOLS_COMMIT:
                detail = revision or revision_check.stderr.strip() or "unknown"
                problems.append(
                    f"checkout is {detail}, "
                    f"expected pinned commit {KICAD_ROUTING_TOOLS_COMMIT}"
                )

    native_version = ""
    try:
        startup_check = subprocess.run(
            [
                python,
                "-c",
                "from py_router.startup_checks import run_all_checks; "
                "print(run_all_checks())",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        problems.append(
            "upstream startup checks failed"
            + (f": {detail}" if detail else f": {exc}")
        )
    else:
        output_lines = [line.strip() for line in startup_check.stdout.splitlines() if line.strip()]
        native_version = output_lines[-1] if output_lines else ""
        if native_version != _KRT_NATIVE_VERSION:
            problems.append(
                f"native router is {native_version or 'unknown'!r}, "
                f"expected pinned {_KRT_NATIVE_VERSION!r}"
            )

    if problems:
        raise RoutingBackendUnavailableError(
            "KiCadRoutingTools backend unavailable:\n  - " + "\n  - ".join(problems)
        )
    result = {
        "backend": backend,
        "root": str(root),
        "python": python,
        "version": source_version,
        "commit": revision,
        "native_version": native_version,
    }
    _KRT_PREFLIGHT_CACHE[cache_key] = result
    return dict(result)


def _krt_command(
    input_path: str, output_path: str, config: dict[str, Any]
) -> list[str]:
    root = _krt_root(config)
    python = (
        config.get("kicad_routing_tools_python")
        or (
            os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PYTHON", sys.executable)
            if "kicad_routing_tools_python" in config else sys.executable
        )
    )
    cmd = [
        str(python),
        str(root / "py_router" / "route.py"),
        str(Path(input_path).resolve()),
        str(Path(output_path).resolve()),
        "--nets", "*",
        "--no-fix-drc-settings",
        "--keep-input-copper",
        "--max-iterations", str(config.get("kicad_routing_tools_max_iterations", 200000)),
        "--max-ripup", str(config.get("kicad_routing_tools_max_ripup", 3)),
        "--ordering", str(config.get("kicad_routing_tools_ordering", "mps")),
    ]
    clearance = config.get("kicad_routing_tools_clearance_mm")
    if clearance is not None:
        cmd.extend(["--clearance", str(clearance)])
    layers = config.get("kicad_routing_tools_layers")
    if layers:
        cmd.extend(["--layers", *map(str, layers)])
    return cmd


def _fingerprint_multiset(items: list[Any], fingerprint: Any) -> Counter:
    return Counter(fingerprint(item) for item in items)


def _fingerprint_rows(counter: Counter) -> list[dict[str, Any]]:
    return [
        {"fingerprint": list(value), "count": count}
        for value, count in sorted(counter.items(), key=lambda item: repr(item[0]))
    ]


def _preservation_group(
    expected_items: list[Any], actual_items: list[Any], fingerprint: Any
) -> dict[str, Any]:
    expected = _fingerprint_multiset(expected_items, fingerprint)
    actual = _fingerprint_multiset(actual_items, fingerprint)
    matched = expected & actual
    missing = expected - actual
    return {
        "expected_count": sum(expected.values()),
        "matched_count": sum(matched.values()),
        "missing_count": sum(missing.values()),
        "expected": _fingerprint_rows(expected),
        "matched": _fingerprint_rows(matched),
        "missing": _fingerprint_rows(missing),
    }


def _krt_json_summaries(stdout: str, stderr: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            if "JSON_SUMMARY:" not in line:
                continue
            try:
                summary = json.loads(line.split("JSON_SUMMARY:", 1)[1].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(summary, dict):
                summaries.append(summary)
    return summaries


def route_with_kicad_routing_tools(
    kicad_pcb_path: str,
    output_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Route directly while preserving input copper and project design rules."""
    from kicraft.autoplacer.brain.copper_accounting import (
        fingerprint_trace,
        fingerprint_via,
    )
    from kicraft.autoplacer.freerouting_runner import (
        import_routed_copper,
        propagate_sibling_project_rules,
    )

    input_board = Path(kicad_pcb_path).resolve()
    output_board = Path(output_path).resolve()
    if input_board == output_board:
        raise ValueError(
            "KiCadRoutingTools input and output board paths must be distinct"
        )

    runtime = preflight_routing_backend(config)
    root = Path(runtime["root"])
    output_board.unlink(missing_ok=True)
    input_copper = import_routed_copper(str(input_board))

    expected_project = input_board.with_suffix(".kicad_pro")
    candidates: list[Path] = []
    configured_board = str(config.get("pcb_path", "") or "").strip()
    if configured_board:
        candidates.append(
            Path(os.path.expanduser(configured_board)).absolute().with_suffix(
                ".kicad_pro"
            )
        )
    candidates.extend(sorted(input_board.parent.glob("*.kicad_pro")))
    source_project = (
        expected_project
        if expected_project.is_file()
        else next((path for path in candidates if path.is_file()), None)
    )
    if source_project is None:
        raise RoutingBackendUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro before routing "
            f"{input_board}"
        )

    temporary_sidecars: list[Path] = []
    for suffix in (".kicad_pro", ".kicad_dru"):
        destination = input_board.with_suffix(suffix)
        if not destination.exists():
            temporary_sidecars.append(destination)
    propagate_sibling_project_rules(
        str(source_project.with_suffix(".kicad_pcb")), str(input_board)
    )
    if not expected_project.is_file():
        raise RoutingBackendUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro; "
            f"could not stage {source_project} beside {input_board}"
        )

    command = _krt_command(str(input_board), str(output_board), config)
    timeout_s = int(
        config.get(
            "kicad_routing_tools_timeout_s",
            config.get("freerouting_timeout_s", 120),
        )
    )
    environment = os.environ.copy()
    environment["KICAD_RIP_PREEXISTING"] = "0"
    environment["KICAD_PLANE_FINALIZE"] = "0"
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
        if output_board.is_file():
            propagate_sibling_project_rules(str(input_board), str(output_board))
    finally:
        for sidecar in temporary_sidecars:
            sidecar.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    if timed_out:
        raise RuntimeError(f"KiCadRoutingTools timed out after {timeout_s}s")
    if proc.returncode != 0 or not output_board.is_file():
        detail = (stderr or stdout or "no output").strip()[-4000:]
        raise RuntimeError(f"KiCadRoutingTools failed (rc={proc.returncode}): {detail}")
    if not output_board.with_suffix(".kicad_pro").is_file():
        raise RoutingBackendUnavailableError(
            "KiCadRoutingTools requires a sibling .kicad_pro on its routed output; "
            f"could not propagate project rules to {output_board}"
        )

    output_copper = import_routed_copper(str(output_board))
    preservation = {
        "traces": _preservation_group(
            input_copper.get("traces", []),
            output_copper.get("traces", []),
            fingerprint_trace,
        ),
        "vias": _preservation_group(
            input_copper.get("vias", []),
            output_copper.get("vias", []),
            fingerprint_via,
        ),
    }
    preserved = (
        preservation["traces"]["missing_count"] == 0
        and preservation["vias"]["missing_count"] == 0
    )
    json_summaries = _krt_json_summaries(stdout, stderr)
    summary = json_summaries[0] if json_summaries else {}
    stats = {
        "backend": "kicad-routing-tools",
        "version": runtime["version"],
        "commit": runtime["commit"],
        "source_version": runtime["version"],
        "source_commit": runtime["commit"],
        "native_version": runtime["native_version"],
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 3),
        "successful_nets": summary.get("successful"),
        "failed_nets": summary.get("failed"),
        "total_vias": summary.get("total_vias"),
        "router_time_s": summary.get("total_time"),
        "json_summaries": json_summaries,
        "input_copper_preservation": preservation,
        "preserved_existing_copper": preserved,
        "command": command,
        "_raw_stdout": stdout,
        "_raw_stderr": stderr,
    }
    if not preserved:
        raise RoutingCopperPreservationError(
            "KiCadRoutingTools failed to preserve input copper "
            f"(missing traces={preservation['traces']['missing_count']}, "
            f"vias={preservation['vias']['missing_count']}); "
            f"routed output retained at {output_board}",
            stats,
        )
    return stats


def route_board(
    kicad_pcb_path: str,
    output_path: str,
    config: dict[str, Any],
    *,
    jar_path: str | None = None,
) -> dict[str, Any]:
    """Route using the configured backend and return normalized backend stats."""
    backend = routing_backend(config)
    if backend == "kicad-routing-tools":
        return route_with_kicad_routing_tools(kicad_pcb_path, output_path, config)
    from kicraft.autoplacer.freerouting_runner import route_with_freerouting
    resolved_jar = jar_path or str(config.get("freerouting_jar", ""))
    if not resolved_jar:
        raise RoutingBackendUnavailableError("FreeRouting requires freerouting_jar")
    stats = route_with_freerouting(kicad_pcb_path, output_path, resolved_jar, config)
    stats = dict(stats or {})
    stats.setdefault("backend", "freerouting")
    return stats
