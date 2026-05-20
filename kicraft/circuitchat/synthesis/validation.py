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

# Electrical types that are intentionally floating in the symbol — these
# don't have to appear in a NetConnection or no_connect_pins for §9.11 to
# pass. Net coverage is a real concern for signal/power pins, not for
# pins the symbol itself declares disconnected.
_COVERAGE_EXEMPT_ELECTRICAL_TYPES = frozenset({"no_connect", "free"})

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


# ---------- §9.7 ref uniqueness (leaf-library reuse + general hygiene) ----------


def check_refdes_uniqueness(project_dir: Path, project_stem: str) -> CheckResult:
    """§9.7 — every refdes is globally unique across schematic + autoplacer.

    Catches renumber-map bugs that would otherwise surface as silent
    refdes collisions between library-imported and from-scratch sheets.
    """
    import json as _json
    import re

    ref_re = re.compile(r'\(property\s+"Reference"\s+"([A-Z]+[0-9]+)"')
    ap_path = project_dir / f"{project_stem}_autoplacer.json"
    refs_by_origin: dict[str, list[str]] = {}

    for sch in sorted(project_dir.glob("*.kicad_sch")):
        if sch.name == f"{project_stem}.kicad_sch":
            continue  # root has no symbol refs in our emitter
        text = sch.read_text(encoding="utf-8")
        for ref in ref_re.findall(text):
            refs_by_origin.setdefault(ref, []).append(f"sch:{sch.name}")

    # Collect refs from autoplacer.json (ic_groups keys + members,
    # thermal_refs, signal_flow_order, component_zones keys).
    if ap_path.exists():
        try:
            ap = _json.loads(ap_path.read_text(encoding="utf-8"))
        except Exception:
            ap = {}
        for key in ("ic_groups", "group_labels", "component_zones"):
            d = ap.get(key, {})
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(k, str) and re.match(r"^[A-Z]+[0-9]+$", k):
                        refs_by_origin.setdefault(k, []).append(f"ap:{key}")
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and re.match(r"^[A-Z]+[0-9]+$", item):
                                refs_by_origin.setdefault(item, []).append(f"ap:{key}:member")
        for key in ("thermal_refs", "signal_flow_order"):
            lst = ap.get(key, [])
            if isinstance(lst, list):
                for item in lst:
                    if isinstance(item, str) and re.match(r"^[A-Z]+[0-9]+$", item):
                        refs_by_origin.setdefault(item, []).append(f"ap:{key}")

    # Origins are recorded as a list of where the ref was *seen*; for
    # uniqueness, what matters is whether the SAME ref appears in two
    # different .kicad_sch files (the schematic side is the source of
    # truth for ref ownership). Autoplacer refs reference what should
    # be a unique sch ref, so they may legitimately appear N times for
    # one schematic ref.
    sch_origins_by_ref: dict[str, set[str]] = {}
    for ref, origins in refs_by_origin.items():
        for o in origins:
            if o.startswith("sch:"):
                sch_origins_by_ref.setdefault(ref, set()).add(o)
    collisions = [
        (ref, sorted(origins))
        for ref, origins in sch_origins_by_ref.items()
        if len(origins) > 1
    ]
    if collisions:
        return CheckResult(
            name="9.7 refdes uniqueness",
            ok=False,
            message=f"{len(collisions)} ref(s) appear in multiple schematics",
            offenders=[f"{r}: {', '.join(o)}" for r, o in collisions],
        )
    return CheckResult(
        name="9.7 refdes uniqueness",
        ok=True,
        message=f"{len(sch_origins_by_ref)} unique refs across schematics",
    )


# ---------- §9.8 library interface match ----------


def check_library_interface_match(
    project_dir: Path, project_stem: str
) -> CheckResult:
    """§9.8 — every library-backed sheet's hierarchical labels match
    the manifest's declared interface exactly.

    Failure mode: the leaf on disk was edited between architecture and
    synthesis, or the renumber/copy step corrupted the labels.
    """
    import json as _json
    import re

    ap_path = project_dir / f"{project_stem}_autoplacer.json"
    if not ap_path.exists():
        return CheckResult(
            name="9.8 library interface match",
            ok=True,
            message="no library_leaves (autoplacer.json absent)",
        )
    try:
        ap = _json.loads(ap_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult(
            name="9.8 library interface match",
            ok=False,
            message=f"could not parse autoplacer.json: {exc}",
        )
    library_leaves = ap.get("library_leaves") or {}
    if not isinstance(library_leaves, dict) or not library_leaves:
        return CheckResult(
            name="9.8 library interface match",
            ok=True,
            message="no library-backed sheets",
        )

    try:
        from kicraft.leaf_library import LeafLibrary
    except ImportError:
        return CheckResult(
            name="9.8 library interface match",
            ok=False,
            message="kicraft.leaf_library not importable",
        )

    lib = LeafLibrary.from_env()
    label_re = re.compile(
        r'\(hierarchical_label\s+"([A-Z][A-Z0-9_]*)"\s+\(shape\s+(\w+)\)'
    )
    shape_to_direction = {
        "input": "input",
        "output": "output",
        "bidirectional": "bidirectional",
        "passive": "passive",
        "tri_state": "bidirectional",
    }

    mismatches: list[str] = []
    for sheet_name, entry in library_leaves.items():
        if not isinstance(entry, dict):
            continue
        slug = entry.get("source")
        if not isinstance(slug, str):
            continue
        leaf = lib.find(slug)
        if leaf is None:
            mismatches.append(f"{sheet_name}: leaf {slug} not loadable")
            continue
        # Find this sheet's stem from autoplacer.json or fall back to
        # scanning for a matching label set. The synthesis stage writes
        # the sheet's stem to <stem>.kicad_sch, but we don't have the
        # stem in library_leaves. Scan every leaf .kicad_sch and match.
        leaf_iface = {
            (lbl.name, lbl.direction)
            for lbl in leaf.manifest.interface.hierarchical_labels
        }
        # Find a sheet whose labels match -- since stems and sheet_uuid
        # are not in library_leaves we accept any leaf .kicad_sch whose
        # labels are a superset of the expected set.
        found = False
        for sch in sorted(project_dir.glob("*.kicad_sch")):
            if sch.name == f"{project_stem}.kicad_sch":
                continue
            text = sch.read_text(encoding="utf-8")
            labels = {
                (m.group(1), shape_to_direction.get(m.group(2), "passive"))
                for m in label_re.finditer(text)
            }
            if labels == leaf_iface:
                found = True
                break
        if not found:
            mismatches.append(
                f"{sheet_name}: no leaf .kicad_sch has interface "
                f"{sorted(leaf_iface)}"
            )

    if mismatches:
        return CheckResult(
            name="9.8 library interface match",
            ok=False,
            message=f"{len(mismatches)} mismatch(es)",
            offenders=mismatches,
        )
    return CheckResult(
        name="9.8 library interface match",
        ok=True,
        message=f"{len(library_leaves)} library-backed sheet(s) match manifest",
    )


# ---------- §9.10 pin existence ----------


def check_pin_existence(bom) -> CheckResult:
    """§9.10 — every (ref, pin) in BOM.connections + no_connect_pins
    references a pin that actually exists on the part's KiCad symbol.

    Catches LLM pin-number hallucination at the wiring stage.
    """
    from .symbol_pinout import SymbolNotFoundError, lookup_pins

    bad: list[str] = []
    symbol_by_ref = {p.ref: p.symbol for p in bom.parts}
    pins_by_ref_cache: dict[str, set[str]] = {}

    def _pins_for_ref(ref: str) -> set[str] | None:
        if ref in pins_by_ref_cache:
            return pins_by_ref_cache[ref]
        sym = symbol_by_ref.get(ref)
        if sym is None:
            return None
        try:
            info = lookup_pins(sym)
        except (SymbolNotFoundError, ValueError) as exc:
            bad.append(f"{ref} ({sym}): {exc}")
            pins_by_ref_cache[ref] = set()
            return pins_by_ref_cache[ref]
        nums = {p["number"] for p in info["pins"]}
        pins_by_ref_cache[ref] = nums
        return nums

    def _check(ep, ctx: str) -> None:
        nums = _pins_for_ref(ep.ref)
        if nums is None:
            bad.append(f"{ctx}: ref {ep.ref!r} not in BOM.parts")
            return
        if ep.pin not in nums:
            bad.append(
                f"{ctx}: pin {ep.pin!r} not in symbol {symbol_by_ref[ep.ref]!r} "
                f"for {ep.ref} (known: {sorted(nums)[:8]}…)"
            )

    for c in bom.connections:
        for ep in c.endpoints:
            _check(ep, f"connection {c.net_name!r}")
    for ep in bom.no_connect_pins:
        _check(ep, "no_connect_pins")

    return CheckResult(
        name="9.10 pin existence",
        ok=not bad,
        message=("every endpoint pin exists in its symbol" if not bad else "missing pin(s)"),
        offenders=bad,
    )


# ---------- §9.11 net coverage ----------


def check_net_coverage(bom) -> CheckResult:
    """§9.11 — every part pin defined by the symbol must appear in either a
    NetConnection.endpoints entry or in no_connect_pins. No silent drops.

    Pins whose electrical type is ``no_connect`` or ``free`` in the
    symbol are exempt — the symbol itself declares them disconnected.
    """
    from .symbol_pinout import SymbolNotFoundError, lookup_pins

    bad: list[str] = []
    connected: dict[str, set[str]] = {}
    for c in bom.connections:
        for ep in c.endpoints:
            connected.setdefault(ep.ref, set()).add(ep.pin)
    for ep in bom.no_connect_pins:
        connected.setdefault(ep.ref, set()).add(ep.pin)

    for part in bom.parts:
        try:
            info = lookup_pins(part.symbol)
        except (SymbolNotFoundError, ValueError) as exc:
            bad.append(f"{part.ref} ({part.symbol}): {exc}")
            continue
        accounted = connected.get(part.ref, set())
        for pin in info["pins"]:
            if pin["electrical_type"] in _COVERAGE_EXEMPT_ELECTRICAL_TYPES:
                continue
            if pin["number"] not in accounted:
                bad.append(
                    f"{part.ref}.{pin['number']} ({pin['name']!r}, "
                    f"{pin['electrical_type']}) not in connections or no_connect_pins"
                )

    return CheckResult(
        name="9.11 net coverage",
        ok=not bad,
        message=(
            "every part pin accounted for" if not bad else "uncovered pin(s)"
        ),
        offenders=bad,
    )


# ---------- §9.9 connectivity (Stage B) ----------


_LIB_ID_RE = re.compile(r'\(lib_id\s+"([^"]+)"')


def check_connectivity(project_dir: Path, project_stem: str) -> CheckResult:
    """§9.9 — every leaf sheet with ≥2 component symbols must contain at
    least one ``(wire …)`` or at least one ``(symbol (lib_id "power:…")…)``
    instance. A leaf with components but zero electrical artifacts is a
    Stage-B regression (or a Stage-A pre-wiring snapshot, which is gated
    out by the caller).
    """
    bad: list[str] = []
    root_name = f"{project_stem}.kicad_sch"
    for sch in sorted(project_dir.glob("*.kicad_sch")):
        if sch.name == root_name:
            continue  # root has no components
        text = sch.read_text()
        non_power_components = 0
        power_symbols = 0
        for _offset, block in _iter_symbol_instance_blocks(text):
            lib_id_m = _LIB_ID_RE.search(block)
            if lib_id_m and lib_id_m.group(1).startswith("power:"):
                power_symbols += 1
            else:
                non_power_components += 1
        if non_power_components < 2:
            continue
        # Wire count (top-level only is fine — wires never appear inside
        # lib_symbols).
        wire_count = text.count("(wire")
        if wire_count == 0 and power_symbols == 0:
            bad.append(
                f"{sch.name}: {non_power_components} components, "
                f"0 wires, 0 power symbols"
            )
    return CheckResult(
        name="9.9 connectivity",
        ok=not bad,
        message=(
            "every leaf has wires or power symbols"
            if not bad
            else "leaf(s) without electrical connectivity"
        ),
        offenders=bad,
    )


# ---------- §9.12 ERC (Stage B) ----------


def check_erc(project_dir: Path, project_stem: str) -> CheckResult:
    """§9.12 — ``kicad-cli sch erc`` reports 0 errors.

    Skips gracefully when ``kicad-cli`` is not installed. Treats only
    severity=error as failing; warnings are tolerated per the spec's
    v1 non-goal of ERC zero-warnings.
    """
    root_sch = project_dir / f"{project_stem}.kicad_sch"
    if not root_sch.is_file():
        return CheckResult(
            name="9.12 ERC", ok=False, message=f"{root_sch.name} missing"
        )
    out_path = project_dir / f"{project_stem}_erc.rpt"
    try:
        proc = subprocess.run(
            ["kicad-cli", "sch", "erc",
             "--output", str(out_path), str(root_sch)],
            capture_output=True, text=True, timeout=60.0,
        )
    except FileNotFoundError:
        return CheckResult(
            name="9.12 ERC", ok=True,
            message="kicad-cli not available; ERC skipped",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="9.12 ERC", ok=False, message="kicad-cli timed out after 60s",
        )

    if not out_path.exists():
        return CheckResult(
            name="9.12 ERC", ok=False,
            message=(
                f"kicad-cli sch erc exit {proc.returncode}; no report at "
                f"{out_path.name}"
            ),
        )
    report_text = out_path.read_text()
    error_lines: list[str] = []
    try:
        report = json.loads(report_text)
        for sheet in report.get("sheets", []) or []:
            for v in sheet.get("violations", []) or []:
                if str(v.get("severity", "")).lower() == "error":
                    desc = v.get("description", "")
                    error_lines.append(f"{sheet.get('path', '?')}: {desc}")
    except (json.JSONDecodeError, AttributeError):
        # Text report fallback: count lines that look like errors.
        for line in report_text.splitlines():
            if re.search(r"\b(severity\s*[:=]\s*)?error\b", line, re.IGNORECASE):
                error_lines.append(line.strip())

    if error_lines:
        return CheckResult(
            name="9.12 ERC", ok=False,
            message=f"{len(error_lines)} ERC error(s)",
            offenders=error_lines[:20],
        )
    return CheckResult(name="9.12 ERC", ok=True, message="ERC clean (0 errors)")


# ---------- aggregator ----------


def run_validations(
    project_dir: Path,
    project_stem: str,
    bom=None,
) -> list[CheckResult]:
    """Run §9.1-§9.8 and raise SynthesisValidationError if any failed.

    When ``bom`` is provided AND has a non-empty ``connections`` list,
    §9.10 (pin existence), §9.11 (net coverage), §9.9 (connectivity),
    and §9.12 (ERC) also run. The latter two are Stage-B checks that
    only make sense once schematic wires + power symbols are being
    emitted.
    """
    results = [
        check_schematic_version(project_dir),
        check_footprints_nonempty(project_dir),
        check_pin_directions(project_dir),
        check_sheetfile_refs_resolve(project_dir),
        check_autoplacer_is_valid_json(project_dir, project_stem),
        check_named_refs_exist(project_dir, project_stem),
        check_refdes_uniqueness(project_dir, project_stem),
        check_library_interface_match(project_dir, project_stem),
    ]
    if bom is not None and bom.connections:
        results.append(check_pin_existence(bom))
        results.append(check_net_coverage(bom))
        results.append(check_connectivity(project_dir, project_stem))
        results.append(check_erc(project_dir, project_stem))
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
