"""Mechanical self-validation checks (§9 of the contract doc).

Each check is a pure function over the project directory contents. The
top-level `run_validations` aggregates §9.1-§9.6 and raises
`SynthesisValidationError` on any failure. §9.7 (solve-subcircuits smoke)
lives in its own function so callers can opt in or out.

The contract doc specifies these as shell one-liners — this is the Python
equivalent that the synthesis stage runs unconditionally before returning.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_SCHEMATIC_VERSION = 20250114
VALID_PIN_DIRECTIONS = frozenset({"input", "output", "bidirectional", "passive"})

_VERSION_RE = re.compile(r"\(version\s+(\d+)\)")
_PIN_RE = re.compile(r'^\s*\(pin\s+"[^"]+"\s+(\w+)', re.MULTILINE)
_SHEETFILE_RE = re.compile(r'\(property\s+"Sheetfile"\s+"([^"]+)"')
_REFERENCE_RE = re.compile(r'\(property\s+"Reference"\s+"([A-Z]+[0-9]+[A-Z0-9_-]*)"')
_INSTANCE_REF_RE = re.compile(r'\(property\s+"Reference"\s+"([^"]+)"')
_INSTANCE_FOOTPRINT_RE = re.compile(r'\(property\s+"Footprint"\s+"([^"]*)"')
_REAL_REF_RE = re.compile(r'^[A-Z]+[0-9]+[A-Z0-9_-]*$')


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""
    offenders: list[str] = field(default_factory=list)


class SynthesisValidationError(RuntimeError):
    """Aggregates one or more failed validation checks."""

    def __init__(self, failures: list[CheckResult]):
        self.failures = failures
        lines = [f"synthesis validation failed ({len(failures)} check(s)):"]
        for f in failures:
            lines.append(f"  - {f.name}: {f.message}")
            for off in f.offenders[:10]:
                lines.append(f"      * {off}")
            if len(f.offenders) > 10:
                lines.append(f"      ... and {len(f.offenders) - 10} more")
        super().__init__("\n".join(lines))


# ---------- individual checks ----------


def check_schematic_version(project_dir: Path) -> CheckResult:
    """§9.1 — every .kicad_sch has version >= 20250114 (KiCad 9)."""
    bad: list[str] = []
    for sch in sorted(project_dir.glob("*.kicad_sch")):
        m = _VERSION_RE.search(sch.read_text())
        if not m:
            bad.append(f"{sch.name}: no (version ...) line")
            continue
        version = int(m.group(1))
        if version < REQUIRED_SCHEMATIC_VERSION:
            bad.append(f"{sch.name}: version {version} < {REQUIRED_SCHEMATIC_VERSION}")
    return CheckResult(
        name="9.1 schematic version",
        ok=not bad,
        message=("all schematics are KiCad 9" if not bad else f"{len(bad)} file(s) below KiCad 9"),
        offenders=bad,
    )


def _strip_lib_symbols_block(text: str) -> str:
    """Return `text` with any (lib_symbols ...) block(s) replaced by whitespace.

    Symbol library definitions inside `(lib_symbols ...)` often carry an empty
    `Footprint` property as the template default — KiCad expects the placed
    instance to override it. Only instance-level empties matter for §9.2.
    """
    out = text
    while True:
        idx = out.find("(lib_symbols")
        if idx == -1:
            return out
        # Walk parens to find the matching close.
        depth = 0
        end = -1
        for i in range(idx, len(out)):
            c = out[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return out  # malformed; let other checks catch it
        out = out[:idx] + (" " * (end - idx)) + out[end:]


def _iter_symbol_instance_blocks(text: str):
    """Yield each `(symbol ...)` block at the file's top level (outside lib_symbols)."""
    stripped = _strip_lib_symbols_block(text)
    needle = "(symbol"
    pos = 0
    while True:
        idx = stripped.find(needle, pos)
        if idx == -1:
            return
        # Reject (symbol_instances ...), (symbol_lib_table ...), etc.
        next_ch = stripped[idx + len(needle)] if idx + len(needle) < len(stripped) else ""
        if next_ch not in " \t\n\r(":
            pos = idx + 1
            continue
        # Find matching close.
        depth = 0
        end = -1
        for i in range(idx, len(stripped)):
            c = stripped[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return
        yield idx, stripped[idx:end]
        pos = end


def check_footprints_nonempty(project_dir: Path) -> CheckResult:
    """§9.2 — every placed component instance has a non-empty Footprint.

    Skips:
    - `(lib_symbols ...)` template definitions (their empty Footprint is the default
      that the placed instance overrides).
    - Power-flag pseudo-symbols (refs like `#PWR0042`, `#FLG...`) — they are net
      markers, not real components.
    """
    bad: list[str] = []
    for sch in sorted(project_dir.glob("*.kicad_sch")):
        text = sch.read_text()
        for offset, block in _iter_symbol_instance_blocks(text):
            ref_m = _INSTANCE_REF_RE.search(block)
            if not ref_m or not _REAL_REF_RE.match(ref_m.group(1)):
                continue  # power flag / pseudo-symbol
            fp_m = _INSTANCE_FOOTPRINT_RE.search(block)
            if fp_m is None or fp_m.group(1) == "":
                bad.append(f"{sch.name}:{ref_m.group(1)} empty/missing Footprint")
    return CheckResult(
        name="9.2 footprints non-empty",
        ok=not bad,
        message=("all Footprint properties populated" if not bad else "empty Footprint(s) found"),
        offenders=bad,
    )


def check_pin_directions(project_dir: Path) -> CheckResult:
    """§9.3 — every sheet pin has direction in {input,output,bidirectional,passive}."""
    bad: list[str] = []
    for sch in sorted(project_dir.glob("*.kicad_sch")):
        for m in _PIN_RE.finditer(sch.read_text()):
            direction = m.group(1)
            if direction not in VALID_PIN_DIRECTIONS:
                bad.append(f"{sch.name}: pin direction {direction!r}")
    return CheckResult(
        name="9.3 pin directions",
        ok=not bad,
        message=("all pin directions valid" if not bad else "invalid pin direction(s) found"),
        offenders=bad,
    )


def check_sheetfile_refs_resolve(project_dir: Path) -> CheckResult:
    """§9.4 — every Sheetfile property names a file that exists in the same dir."""
    bad: list[str] = []
    total = 0
    for sch in sorted(project_dir.glob("*.kicad_sch")):
        text = sch.read_text()
        for m in _SHEETFILE_RE.finditer(text):
            total += 1
            target = m.group(1)
            if not (project_dir / target).is_file():
                bad.append(f"{sch.name} references missing {target}")
    return CheckResult(
        name="9.4 Sheetfile refs resolve",
        ok=not bad,
        message=(
            f"all {total} Sheetfile ref(s) resolve"
            if not bad
            else f"{len(bad)} unresolved Sheetfile ref(s)"
        ),
        offenders=bad,
    )


def check_autoplacer_is_valid_json(project_dir: Path, project_stem: str) -> CheckResult:
    """§9.5 — `<PROJECT>_autoplacer.json` is parseable JSON."""
    path = project_dir / f"{project_stem}_autoplacer.json"
    if not path.is_file():
        return CheckResult(
            name="9.5 autoplacer.json is JSON",
            ok=False,
            message=f"{path.name} missing",
        )
    try:
        json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return CheckResult(
            name="9.5 autoplacer.json is JSON",
            ok=False,
            message=f"JSON parse error: {e}",
            offenders=[str(path)],
        )
    return CheckResult(
        name="9.5 autoplacer.json is JSON", ok=True, message=f"{path.name} parses"
    )


def _collect_refs_from_schematics(project_dir: Path) -> set[str]:
    refs: set[str] = set()
    for sch in project_dir.glob("*.kicad_sch"):
        for m in _REFERENCE_RE.finditer(sch.read_text()):
            refs.add(m.group(1))
    return refs


def check_named_refs_exist(project_dir: Path, project_stem: str) -> CheckResult:
    """§9.6 — every ref named in autoplacer.json appears in some .kicad_sch."""
    cfg_path = project_dir / f"{project_stem}_autoplacer.json"
    if not cfg_path.is_file():
        return CheckResult(
            name="9.6 autoplacer refs in schematic",
            ok=False,
            message=f"{cfg_path.name} missing",
        )
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as e:
        return CheckResult(
            name="9.6 autoplacer refs in schematic",
            ok=False,
            message=f"autoplacer.json not parseable (see §9.5): {e}",
        )
    named: set[str] = set()
    for ic, members in cfg.get("ic_groups", {}).items():
        named.add(ic)
        named.update(members)
    named.update(cfg.get("thermal_refs", []))
    named.update(cfg.get("signal_flow_order", []))
    named.update(cfg.get("component_zones", {}).keys())

    refs_in_sch = _collect_refs_from_schematics(project_dir)
    missing = sorted(named - refs_in_sch)
    return CheckResult(
        name="9.6 autoplacer refs in schematic",
        ok=not missing,
        message=(
            f"all {len(named)} named ref(s) found in schematic"
            if not missing
            else f"{len(missing)} ref(s) not in any schematic"
        ),
        offenders=missing,
    )


# ---------- aggregator ----------


def run_validations(project_dir: Path, project_stem: str) -> list[CheckResult]:
    """Run §9.1-§9.6 and raise SynthesisValidationError if any failed."""
    results = [
        check_schematic_version(project_dir),
        check_footprints_nonempty(project_dir),
        check_pin_directions(project_dir),
        check_sheetfile_refs_resolve(project_dir),
        check_autoplacer_is_valid_json(project_dir, project_stem),
        check_named_refs_exist(project_dir, project_stem),
    ]
    failures = [r for r in results if not r.ok]
    if failures:
        raise SynthesisValidationError(failures)
    return results


# ---------- §9.7 separate smoke (opt-in by caller) ----------


def run_solve_subcircuits_smoke(
    project_dir: Path, project_stem: str, timeout_s: float = 60.0
) -> CheckResult:
    """§9.7 — `solve-subcircuits <PROJECT>.kicad_sch` exits 0.

    Runs in a subprocess with a timeout. Returns a CheckResult; does NOT
    raise so the caller can decide to surface or skip.
    """
    root_sch = project_dir / f"{project_stem}.kicad_sch"
    if not root_sch.is_file():
        return CheckResult(
            name="9.7 solve-subcircuits smoke",
            ok=False,
            message=f"{root_sch.name} missing",
        )
    try:
        proc = subprocess.run(
            ["python", "-m", "kicraft.cli.solve_subcircuits", str(root_sch)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(project_dir),
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="9.7 solve-subcircuits smoke",
            ok=False,
            message=f"timed out after {timeout_s}s",
        )
    except FileNotFoundError as e:
        return CheckResult(
            name="9.7 solve-subcircuits smoke",
            ok=False,
            message=f"could not invoke python: {e}",
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).splitlines()[-5:]
        return CheckResult(
            name="9.7 solve-subcircuits smoke",
            ok=False,
            message=f"exit {proc.returncode}",
            offenders=tail,
        )
    return CheckResult(
        name="9.7 solve-subcircuits smoke", ok=True, message="exit 0"
    )
