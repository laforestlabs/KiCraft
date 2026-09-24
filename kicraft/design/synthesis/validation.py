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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable, Iterable

from kicraft.design.models import (
    GND_NET_PATTERNS,
    POWER_NET_PATTERNS,
    FunctionalSpec,
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    SheetPin,
    is_power_or_ground_name,
    obligation_requires_requirement_owner,
)
from kicraft.design.part_identity import canonical_physical_features


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
_REAL_REF_RE = re.compile(r"^[A-Z]+[0-9]+[A-Z0-9_-]*$")


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""
    offenders: list[str] = field(default_factory=list)


class SynthesisValidationError(RuntimeError):
    """Aggregates one or more failed validation checks."""

    def __init__(self, failures: list[CheckResult], *, artifacts=None, results=None):
        self.failures = failures
        # artifacts: the ArtifactPaths for the files that WERE written (with
        # status="failed"); results: every check that ran (not only failures).
        # Both let the caller persist a useful record despite the failure.
        self.artifacts = artifacts
        self.results = results if results is not None else list(failures)
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
    return CheckResult(name="9.5 autoplacer.json is JSON", ok=True, message=f"{path.name} parses")


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

    # Full REF_RE grammar (models.py): suffixed refs like D1A / J1-PWR are
    # exactly what leaf-library renumbering emits, so the duplicate scan must
    # see them too -- requiring the closing quote right after the digits left
    # the gate blind to collisions among suffixed refs.
    ref_re = re.compile(r'\(property\s+"Reference"\s+"([A-Z]+[0-9]+[A-Z0-9_-]*)"')
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
                    if isinstance(k, str) and re.match(r"^[A-Z]+[0-9]+[A-Z0-9_-]*$", k):
                        refs_by_origin.setdefault(k, []).append(f"ap:{key}")
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and re.match(
                                r"^[A-Z]+[0-9]+[A-Z0-9_-]*$", item
                            ):
                                refs_by_origin.setdefault(item, []).append(f"ap:{key}:member")
        for key in ("thermal_refs", "signal_flow_order"):
            lst = ap.get(key, [])
            if isinstance(lst, list):
                for item in lst:
                    if isinstance(item, str) and re.match(r"^[A-Z]+[0-9]+[A-Z0-9_-]*$", item):
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
        (ref, sorted(origins)) for ref, origins in sch_origins_by_ref.items() if len(origins) > 1
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


def check_library_interface_match(project_dir: Path, project_stem: str) -> CheckResult:
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
    label_re = re.compile(r'\(hierarchical_label\s+"([A-Z][A-Z0-9_]*)"\s+\(shape\s+(\w+)\)')
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
            (lbl.name, lbl.direction) for lbl in leaf.manifest.interface.hierarchical_labels
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
                f"{sheet_name}: no leaf .kicad_sch has interface {sorted(leaf_iface)}"
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
            info = lookup_pins(sym, all_units=True)
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


# ---------- duplicate-pad (N') auto-bridge ----------


def _pin_base(number: str) -> str:
    """The terminal a (possibly primed) pin number belongs to.

    easyeda2kicad represents a terminal landed on several pads with the KiCad
    convention ``N``, ``N'``, ``N''`` — each prime is a *duplicate pad of the
    same internally-shorted terminal* (a 4-pad tactile switch: 1/1' are one
    leaf-frame contact, 2/2' the other). Stripping the trailing apostrophes
    yields the shared terminal key.
    """
    return number.rstrip("'")


def bridge_duplicate_pins(bom) -> list[str]:
    """Put every duplicate pad of an internally-shorted terminal on its net.

    A part whose symbol exposes ``N`` and ``N'`` has two pads the package shorts
    together; the net must reach both, but a wiring stage routinely wires only
    ``N`` and forgets ``N'`` — which §9.11 then (correctly) flags as an uncovered
    pin, sending the model whack-a-moling. Instead, copy the wired sibling's net
    onto every un-wired pad of the same terminal. This is always electrically
    correct (the pads are one node) and a no-op once the netlist covers them.

    A terminal whose pads are wired to *different* nets is left alone — that is a
    real short for the gates to surface, not a coverage gap to paper over; a pad
    the model explicitly marked ``no_connect`` is respected; and a terminal with
    no pad wired is left for §9.11 (we never invent a net).

    Mutates ``bom.connections`` in place; returns the ``ref.pin -> net`` bridges
    made, for logging.
    """
    from .symbol_pinout import SymbolNotFoundError, lookup_pins

    # ref -> {pin -> the connection that wires it}
    wired: dict[str, dict[str, NetConnection]] = defaultdict(dict)
    for c in bom.connections:
        for ep in c.endpoints:
            wired[ep.ref].setdefault(ep.pin, c)
    nc: dict[str, set[str]] = defaultdict(set)
    for ep in bom.no_connect_pins:
        nc[ep.ref].add(ep.pin)

    bridged: list[str] = []
    for part in bom.parts:
        try:
            info = lookup_pins(part.symbol, all_units=True)
        except (SymbolNotFoundError, ValueError):
            continue  # §9.11 reports unresolvable symbols
        groups: dict[str, list[str]] = defaultdict(list)
        for pin in info["pins"]:
            groups[_pin_base(pin["number"])].append(pin["number"])
        part_wired = wired[part.ref]
        part_nc = nc[part.ref]
        for nums in groups.values():
            if len(nums) < 2:
                continue  # single pad — nothing to bridge
            on_net = {n: part_wired[n] for n in nums if n in part_wired}
            if not on_net:
                continue  # terminal entirely unwired — not ours to invent a net
            if len({c.net_name for c in on_net.values()}) > 1:
                continue  # pads on different nets: a real short, leave it for the gates
            target = next(iter(on_net.values()))
            for n in nums:
                if n not in part_wired and n not in part_nc:
                    target.endpoints.append(PinEndpoint(ref=part.ref, pin=n))
                    part_wired[n] = target
                    bridged.append(f"{part.ref}.{n} -> {target.net_name}")
    return bridged


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
            info = lookup_pins(part.symbol, all_units=True)
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
        message=("every part pin accounted for" if not bad else "uncovered pin(s)"),
        offenders=bad,
    )


# ---------- §9.13 sheet population + §9.14 inter-sheet net coverage ----------
#
# Cross-stage model-data checks (architecture x bom) run at the BOM and wiring
# stage commits, so a weak model gets a precise retry signal BEFORE the
# schematic is emitted. Without them, an architecture inter-sheet net that the
# wiring stage never realizes is caught only by §9.12 ERC at synthesis time --
# "sheet pin <NET> has no matching hierarchical label inside the sheet" -- which
# aborts the build with no actionable per-stage feedback.


def check_collection_bounds(
    field: str,
    items: Iterable,
    *,
    total: int,
    per_group: int | None = None,
    group_key: Callable | None = None,
) -> CheckResult:
    """Check total and optional grouped cardinality for a response collection."""
    materialized = list(items)
    offenders: list[str] = []
    if len(materialized) > total:
        offenders.append(f"{field} total ({len(materialized)} items, > {total})")

    if per_group is not None:
        if group_key is None:
            raise ValueError("group_key is required when per_group is set")
        group_counts = Counter(str(group_key(item)) for item in materialized)
        over_groups = [(group, count) for group, count in group_counts.items() if count > per_group]
        for group, count in sorted(over_groups, key=lambda pair: (-pair[1], pair[0])):
            offenders.append(f"{group} ({count} items, > {per_group})")

    return CheckResult(
        name=f"collection bounds: {field}",
        ok=not offenders,
        message=(
            f"{field} cardinality is within configured bounds"
            if not offenders
            else f"{field} exceeds configured cardinality bounds"
        ),
        offenders=offenders,
    )


def check_bom_size(bom) -> CheckResult:
    """§9.35 — reject canonical BOM cardinality beyond commit limits."""
    from kicraft.server.config import BOM_SHEET_PART_LIMIT, BOM_TOTAL_PART_LIMIT

    result = check_collection_bounds(
        "parts",
        bom.parts,
        total=BOM_TOTAL_PART_LIMIT,
        per_group=BOM_SHEET_PART_LIMIT,
        group_key=lambda part: part.sheet,
    )
    return CheckResult(
        name="9.35 BOM emission bounds",
        ok=result.ok,
        message=result.message,
        offenders=result.offenders,
    )


def check_sheets_have_parts(architecture, bom) -> CheckResult:
    """§9.13 -- every from-scratch sheet has at least one BOM part.

    An architecture sheet with no parts emits a blank leaf; if any
    inter-sheet net routes through it, its sheet pins have no pin to land a
    hierarchical label on (the empty COIL DRIVER sheet on the wireless
    charger). Library-backed sheets (``from_library`` set) are exempt:
    their parts come from the leaf installer, and §9.8 checks their
    interface separately.
    """
    parts_per_sheet = Counter(p.sheet for p in bom.parts)
    bad = [
        f"{s.name!r} (stem {s.stem}) has no parts"
        for s in architecture.sheets
        if s.from_library is None and parts_per_sheet.get(s.name, 0) == 0
    ]
    return CheckResult(
        name="9.13 sheet population",
        ok=not bad,
        message=(
            "every from-scratch sheet has parts"
            if not bad
            else f"{len(bad)} sheet(s) declared in architecture but left empty by the BOM"
        ),
        offenders=bad,
    )


def bom_parts_on_unknown_sheets(architecture, bom) -> list[tuple[str, str]]:
    """Return BOM parts whose sheet is absent from the architecture."""
    sheet_names = {sheet.name for sheet in architecture.sheets}
    return [(part.ref, part.sheet) for part in bom.parts if part.sheet not in sheet_names]


def check_bom_parts_reference_architecture_sheets(architecture, bom) -> CheckResult:
    """§9.13 -- every BOM part belongs to a declared architecture sheet."""
    bad = bom_parts_on_unknown_sheets(architecture, bom)
    return CheckResult(
        name="9.13 BOM sheet references",
        ok=not bad,
        message=(
            "every BOM part references a declared architecture sheet"
            if not bad
            else "BOM part(s) reference undeclared architecture sheet(s)"
        ),
        offenders=[f"{ref} -> {sheet!r}" for ref, sheet in bad],
    )


def _reconciled_endpoints(sheets: set[str], declared) -> list[SheetPin]:
    """Endpoints for a reconciled inter-sheet net: declared sheets first (so
    their direction hints survive), then any newly-realized sheet as
    ``bidirectional`` (we have no per-sheet direction to infer)."""
    dir_by_sheet = {e.sheet: e.direction for e in declared.endpoints} if declared else {}
    ordered: list[str] = []
    if declared:
        ordered += [e.sheet for e in declared.endpoints if e.sheet in sheets]
    ordered += [s for s in sorted(sheets) if s not in ordered]
    return [SheetPin(sheet=s, direction=dir_by_sheet.get(s, "bidirectional")) for s in ordered]


def split_cross_sheet_connections(bom) -> list[str]:
    """Realize every connection on the sheet(s) its endpoints actually live on.

    A ``NetConnection`` carries a single ``sheet`` tag, but the wiring stage
    routinely lists endpoints whose parts are assigned (``BomPart.sheet``) to
    *other* sheets — a connector parked on a dedicated HEADER sheet but wired
    from the functional sheet's connection list (run_01's BNC ``J1``; run_30's
    2×10 GPIO header ``J2``). Two downstream failures follow, and they were the
    batch's #1 fab-blocker by breadth (`kicraft-erc-emitter-drops-label-stubs`):

      * ``route_sheet`` filters connections by ``c.sheet == sheet_name``, so the
        connector-holding sheet sees *zero* connections and draws no stub for
        any connector pin → every pin is ``pin_not_connected`` and its net's
        label ``label_dangling``. On the tagged sheet the foreign endpoints are
        silently dropped (their part isn't placed there).
      * :func:`reconcile_inter_sheet_nets` reads only ``c.sheet``, so a net that
        genuinely crosses sheets but sits in one single-sheet-tagged connection
        is never promoted to ``inter_sheet_nets`` — no hier label / sheet pin
        bridges the two sides even after the pins are drawn.

    Regroup each connection's endpoints by their part's sheet. A connection all
    on one sheet == its tag is untouched. Otherwise it is re-emitted as one
    per-sheet ``NetConnection`` (same ``net_name``): the connector sheet now
    draws a stub per pin, reconcile then sees the net on ≥2 sheets and promotes
    the signal ones (power/ground join globally through per-sheet power symbols,
    which reconcile leaves alone). Endpoints whose ref is unknown, or whose part
    sheet isn't a real sheet, stay on the original tag. §9.13 re-unifies the
    same-named halves (by name), so this never reads as a net merge. Mutates
    ``bom.connections`` in place; returns the changes made, for logging.
    """
    part_sheet = {p.ref: p.sheet for p in bom.parts}
    known_sheets = set(part_sheet.values())
    new_connections: list[NetConnection] = []
    changes: list[str] = []
    mutated = False
    for c in bom.connections:
        by_sheet: dict[str, list[PinEndpoint]] = defaultdict(list)
        for ep in c.endpoints:
            s = part_sheet.get(ep.ref, c.sheet)
            if s not in known_sheets:
                s = c.sheet
            by_sheet[s].append(ep)
        if len(by_sheet) == 1 and c.sheet in by_sheet:
            new_connections.append(c)
            continue
        mutated = True
        for s in sorted(by_sheet):
            new_connections.append(
                NetConnection(net_name=c.net_name, endpoints=by_sheet[s], sheet=s)
            )
        if len(by_sheet) > 1:
            changes.append(f"{c.net_name} {sorted(by_sheet)} (was tagged {c.sheet})")
        else:
            changes.append(f"{c.net_name} retagged {c.sheet}->{next(iter(by_sheet))}")
    if mutated:
        bom.connections = new_connections
    return changes


def reconcile_inter_sheet_nets(architecture, bom) -> list[str]:
    """Add wiring-proven crossings without weakening architecture contracts.

    Wiring may reveal a real crossing omitted by architecture, so a signal
    realized on two or more sheets is promoted deterministically. Declared
    crossings and endpoints are never removed: wiring owns connectivity, not
    architecture, and §9.14 must reject any declared endpoint it failed to
    realize.

    Mutates ``architecture.inter_sheet_nets`` in place and returns additions
    for logging.
    """
    known = {sheet.name for sheet in architecture.sheets}
    realized: dict[str, set[str]] = defaultdict(set)
    for connection in bom.connections:
        if connection.endpoints and connection.sheet in known:
            realized[connection.net_name].add(connection.sheet)

    declared = {net.name: net for net in architecture.inter_sheet_nets}
    changes: list[str] = []
    for name in sorted(realized):
        sheets = realized[name]
        if is_power_or_ground_name(name) or len(sheets) < 2:
            continue
        existing = declared.get(name)
        if existing is None:
            promoted = InterSheetNet(name=name, endpoints=_reconciled_endpoints(sheets, None))
            architecture.inter_sheet_nets.append(promoted)
            declared[name] = promoted
            changes.append(f"+{name} {sorted(sheets)} (realized, undeclared)")
            continue
        existing_sheets = {endpoint.sheet for endpoint in existing.endpoints}
        added_sheets = sheets - existing_sheets
        if not added_sheets:
            continue
        reconciled = {
            endpoint.sheet: endpoint for endpoint in _reconciled_endpoints(sheets, existing)
        }
        existing.endpoints.extend(reconciled[sheet] for sheet in sorted(added_sheets))
        changes.append(f"+{name} endpoints {sorted(added_sheets)} (realized, undeclared)")
    return changes


_PINLESS_MECHANICAL_SYMBOLS = frozenset(
    {
        "Mechanical:MountingHole",
        "Mechanical:Fiducial",
        "Mechanical:Heatsink",
    }
)


def pinless_mechanical_sheets(architecture, bom) -> set[str]:
    """Sheets populated only by KiCad no-pin mechanical symbols.

    Such a sheet cannot carry a hierarchical label, so an inter-sheet net
    endpoint there is unrealizable (a mounting hole is mechanical, not wired).
    The list is explicit: a ``*_Pad`` mounting hole DOES have a pin and must not
    be skipped.
    """
    by_sheet: dict[str, list] = defaultdict(list)
    for part in bom.parts:
        by_sheet[part.sheet].append(part)
    return {
        sheet.name
        for sheet in architecture.sheets
        if by_sheet.get(sheet.name)
        and all(str(part.symbol) in _PINLESS_MECHANICAL_SYMBOLS for part in by_sheet[sheet.name])
    }


def check_inter_sheet_nets_realized(architecture, bom) -> CheckResult:
    """§9.14 -- every SIGNAL inter-sheet net endpoint is realized by a
    same-named NetConnection in that sheet.

    The emitter draws a sheet pin on the parent for each signal (non-power)
    inter-sheet endpoint, and a matching hierarchical label inside the leaf
    only where a connection of that ``net_name`` wires a real pin in that
    sheet. An endpoint with no such connection leaves the sheet pin
    dangling -> KiCad ERC "sheet pin <NET> has no matching hierarchical
    label inside the sheet" (the PWM_H / PWM_L / COIL_OUT failures on the
    wireless charger).

    Power/ground inter-sheet nets are exempt: the emitter connects them via
    global power symbols in the leaves, not sheet pins (see
    ``emitter._emit_sheet_block``), so they never produce this ERC class;
    their per-pin coverage is enforced by §9.11 instead.
    """
    realized: dict[tuple[str, str], int] = defaultdict(int)
    for c in bom.connections:
        realized[(c.net_name, c.sheet)] += len(c.endpoints)
    pinless = pinless_mechanical_sheets(architecture, bom)
    bad: list[str] = []
    for net in architecture.inter_sheet_nets:
        if is_power_or_ground_name(net.name):
            continue
        for ep in net.endpoints:
            if ep.sheet in pinless:
                # A mechanical-only sheet has no pin to wire and no hierarchical
                # label to match; the endpoint is a modeling artifact, not a
                # missing connection.
                continue
            if realized.get((net.name, ep.sheet), 0) < 1:
                bad.append(
                    f"net {net.name!r} crosses into sheet {ep.sheet!r} but no "
                    f"connections[] entry wires it there (add net_name={net.name!r}, "
                    f"sheet={ep.sheet!r} with the pin that carries it)"
                )
    return CheckResult(
        name="9.14 inter-sheet net coverage",
        ok=not bad,
        message=(
            "every signal inter-sheet net is wired on both sides"
            if not bad
            else f"{len(bad)} inter-sheet endpoint(s) have a sheet pin but no hierarchical label"
        ),
        offenders=bad,
    )


def check_no_dangling_signal_nets(architecture, bom) -> CheckResult:
    """§9.15 -- every sheet-local SIGNAL net wires at least two distinct pins.

    The inverse of §9.14. §9.14 checks the forward direction (each *declared*
    inter-sheet net is realized on both sides); this catches the failure a weak
    wiring stage hits more often: a non-power net wired to a single pin that was
    never declared inter-sheet, so its label connects to nothing.

    That is exactly the SOIL_MOISTURE_BLE build failure -- the ESP32-S3's native
    USB D+/D- were split into four disjoint single-pin nets (USB_DP_POWER /
    USB_DN_POWER on the connector sheet, USB_DP_ESP32 / USB_DN_ESP32 on the MCU
    sheet), named inconsistently and absent from inter_sheet_nets. The emitter
    drew four hierarchical labels with nothing else on their nets -> four KiCad
    ERC "Label not connected to anything" errors that aborted the build with no
    per-stage signal (§9.12 ERC is slow, runs only after files are written, and
    does not say which pin or how to fix it).

    Exemptions, so the check flags only true orphans:
      - power/ground nets join globally via power symbols, so a lone pin still
        ties to the rail (their per-pin coverage is §9.11's job);
      - declared inter-sheet nets are owned by §9.14 -- they join across sheets,
        so a single local stub is correct (e.g. ANALOG_OUT: one pin on CAP
        SENSOR, one on ESP32).
    Everything else is sheet-local: a (net_name, sheet) wiring fewer than two
    distinct pins is a dangling label.

    Each offender carries the deterministic topology context built by
    :func:`_dangling_net_context` (pin function, proven series counterpart,
    same-sheet related-domain nets with translator-channel mates, declared
    inter-sheet names) so the wiring-correction pass sees candidate
    endpoints, not just the orphan. The appended text never changes the
    offender's identity: the lead clause's canonical ``REF.PIN`` stays the
    only pin token :func:`_offender_identity` can match.
    """
    inter_sheet_names = {n.name for n in architecture.inter_sheet_nets}
    local_pins: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in bom.connections:
        if is_power_or_ground_name(c.net_name) or c.net_name in inter_sheet_names:
            continue
        for ep in c.endpoints:
            local_pins[(c.net_name, c.sheet)].append(f"{ep.ref}.{ep.pin}")
    # Context indexes, built once per call (never a per-offender scan of
    # bom.connections).
    info, pin_count = _pin_info_by_ref(bom)
    ref_identities = {part.ref: f"{part.symbol} {part.value}" for part in bom.parts}
    pin_nets: dict[tuple[str, str], set[str]] = defaultdict(set)
    net_endpoints: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    sheet_net_bases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in bom.connections:
        for ep in c.endpoints:
            pin_nets[(ep.ref, ep.pin)].add(c.net_name)
            net_endpoints[c.net_name].add((c.sheet, ep.ref, ep.pin))
        if not is_power_or_ground_name(c.net_name):
            sheet_net_bases[(c.sheet, _net_domain_base(c.net_name))].add(c.net_name)
    inter_names = sorted(inter_sheet_names)
    bad: list[str] = []
    for (net, sheet), pins in sorted(local_pins.items()):
        if len(set(pins)) < 2:
            ref, _, pin = pins[0].rpartition(".")
            lead = (
                f"net {net!r} on sheet {sheet!r} wires only {pins[0]} and is "
                f"neither a power net nor a declared inter-sheet net, so it "
                f"connects to nothing (wire it to a second pin, mark it "
                f"no_connect, or declare an inter-sheet net to carry it to "
                f"another sheet)"
            )
            ctx = _dangling_net_context(
                net,
                sheet,
                ref,
                pin,
                info=info,
                pin_count=pin_count,
                ref_identities=ref_identities,
                pin_nets=pin_nets,
                net_endpoints=net_endpoints,
                sheet_net_bases=sheet_net_bases,
                inter_names=inter_names,
            )
            bad.append(f"{lead} -- {ctx}" if ctx else lead)
    return CheckResult(
        name="9.15 no dangling signal nets",
        ok=not bad,
        message=(
            "every sheet-local signal net wires >=2 pins"
            if not bad
            else f"{len(bad)} signal net(s) wire a single pin with nowhere to go"
        ),
        offenders=bad,
    )


# ---------- §9.16-§9.18 semantic wiring checks (pin function vs net role) ----------
#
# §9.10/§9.11 prove every endpoint pin EXISTS and is accounted for; they do not
# prove the model bound the right FUNCTION to the right net. Because a
# PinEndpoint addresses a pin by *number*, a weak wiring stage that ignores the
# pin-name table (extras.symbol_pinouts) can assign nets by geometric pin order
# and emit a netlist that is electrically legal -- ERC/DRC pass -- yet
# functionally wrong: reversed MCU power, a fuse shorted across its own
# terminals, an antenna feed tied to GND. These checks cross-read each part's
# symbol pin NAMES against the role of the net every pin lands on and fail the
# wiring commit with a precise, per-pin retry signal. All three are deliberately
# conservative -- they fire only on unambiguous contradictions -- so a
# correctly-wired design never trips them.

# Pin NAMES that denote a positive supply rail. Differential analog inputs
# (VIN+/VIN-/VINP/VINN) and bare numeric names ("0V", "5V" -- "0V" is the
# *ground* common on DC-DC modules) are intentionally excluded; only unambiguous
# supply names match, so the polarity check never trips on a correct connection.
_POS_SUPPLY_PIN_RE = re.compile(
    r"VDD(?!-)|VCC(?!-)"  # positive VDD/VCC + AVDD/DVDD/VDDA/VDDIO/VCCIO
    r"|^V(?:BAT|BUS|SYS|AA|IN|IO)$"  # exact supplies (VIN exact, not VIN-/VINP)
    r"|^V\+$|^VPLUS$",
    re.IGNORECASE,
)
# Pin NAMES that denote ground or the negative-most rail. "0V"/"0.0V" is the
# 0-volt common used by isolated DC-DC modules and is ground, not a rail.
_GND_PIN_RE = re.compile(
    r"GND|VSS"  # GND/AGND/DGND/PGND + VSS/AVSS/VSSA (substring)
    r"|^V(?:CC|DD)-$|^V-$|^VMINUS$|^VEE$|^0V$|^0\.0+V$",
    re.IGNORECASE,
)

# Negative supply net names are part of POWER_NET_PATTERNS because they must
# remain recognized as power globally, but they are not positive rails for the
# semantic polarity gate below.
_NEGATIVE_RAIL_NET_RE = re.compile(
    r"^(?:-\d+\.?\d*V|-\d+V\d+|VEE|VSS|VCC-|VDD-|VMINUS)$",
    re.IGNORECASE,
)

# Reference-designator prefixes of genuinely two-terminal parts: a 2-pin part of
# one of these classes with both pins on one net is shorted out / dead.
_TWO_TERMINAL_REF_PREFIXES = frozenset(
    {
        "R",
        "RV",
        "RT",
        "RP",
        "VR",  # resistors / pots / thermistors
        "C",  # capacitors
        "L",
        "FB",
        "FL",  # inductors / ferrite beads
        "D",
        "LED",
        "CR",
        "DZ",
        "TVS",  # diodes / LEDs / zeners / TVS
        "F",
        "FU",  # fuses
        "Y",
        "XTAL",  # 2-pin crystals / resonators
        "ANT",
        "AE",  # antennas
    }
)

_ANTENNA_REF_PREFIXES = frozenset({"ANT", "AE"})
# Antenna pin names that carry RF (must reach the feed line, never a rail/GND).
_RF_FEED_PIN_RE = re.compile(r"FEED|RF|^ANT", re.IGNORECASE)

_REF_ALPHA_RE = re.compile(r"^[A-Za-z]+")


def _ref_prefix(ref: str) -> str:
    m = _REF_ALPHA_RE.match(ref)
    return m.group(0).upper() if m else ""


def _net_is_ground(name: str) -> bool:
    s = name.lstrip("/")
    return any(p.search(s) for p in GND_NET_PATTERNS)


def _net_is_positive_rail(name: str) -> bool:
    s = name.lstrip("/")
    if _NEGATIVE_RAIL_NET_RE.fullmatch(s):
        return False
    return any(p.search(s) for p in POWER_NET_PATTERNS)


def _net_is_negative_rail(name: str) -> bool:
    return bool(_NEGATIVE_RAIL_NET_RE.fullmatch(name.lstrip("/")))


def _pin_info_by_ref(bom):
    """``({ref: {pin_number: {"name", "type"}}}, {ref: pin_count})``.

    Tolerant of unresolvable symbols (those are reported by §9.10/§9.11); such
    refs are simply absent, so the semantic checks skip them rather than
    double-reporting a missing symbol.
    """
    from .symbol_pinout import SymbolNotFoundError, lookup_pins

    info: dict[str, dict[str, dict]] = {}
    pin_count: dict[str, int] = {}
    for part in bom.parts:
        try:
            data = lookup_pins(part.symbol, all_units=True)
        except (SymbolNotFoundError, ValueError):
            continue
        info[part.ref] = {
            p["number"]: {
                "name": p.get("name") or "",
                "type": p.get("electrical_type") or "",
            }
            for p in data["pins"]
        }
        pin_count[part.ref] = len(data["pins"])
    return info, pin_count


def _nets_by_ref(bom):
    """``{ref: {pin_number: net_name}}`` over BOM.connections."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for c in bom.connections:
        for ep in c.endpoints:
            out[ep.ref][ep.pin] = c.net_name
    return out


# ---------- §9.15 topology-safe offender context (KC-VKUT5H A1) ----------
#
# A bare "net X wires only REF.PIN" sentence did not let the wiring stage fix
# its dangling far sides: on KC-VKUT5H attempts 1-2 left the USB series
# resistors' far ends single even with the generic series-path NOTE in
# stage_runtime._retry_feedback, because a NOTE can name the shape but not the
# candidate destination pins. The context appended per offender is derived
# ONLY from the frozen BOM/architecture (deterministic, never a guess) and is
# topology-safe by construction: name similarity locates related context, it
# NEVER authorizes a merge. `HUB75_C` / `HUB75_C_5V` are the two intentional
# sides of a level translator; `USB_D_P` / `USB_D_N` are distinct differential
# lines; `UART0` / `UART1` differ by a bare numeric suffix that is never
# stripped.
#
# Signature invariant (tests/test_stage_driver_retry.py pins it): the lead
# clause's canonical {pins[0]} is the ONLY token in an offender that matches
# _offender_identity's REF.PIN / "REF pin N" pin regex; every contextual pin
# here is written "pin N of REF", which that regex cannot match. Never emit a
# dotted pin or a "<REF> pin" adjacency in the appended context.

_GENERIC_PIN_NAME_RE = re.compile(r"^(pin_?\d+|passive|unnamed)$", re.I)
_CHANNEL_PIN_RE = re.compile(r"^([AB])(\d+)$")

# The ONLY domain suffixes stripped for related-net lookup, and only one
# occurrence of one of them. Bare numeric / one-letter suffixes are never
# stripped, so UART0/UART1, LED1/LED2, USB_D_P/USB_D_N stay distinct.
_NET_DOMAIN_SUFFIXES = (
    "_5V",
    "_3V3",
    "_MCU",
    "_POWER",
    "_ESP32",
    "_ISO",
    "_LV",
    "_HV",
)


def _net_domain_base(name: str) -> str:
    u = name.upper()
    for suf in _NET_DOMAIN_SUFFIXES:
        if u.endswith(suf) and len(u) > len(suf):
            return u[: -len(suf)]
    return u


def _pin_function(info, ref: str, pin: str):
    """A non-trivial pin function for display, or None.

    Empty, numeric-only, bare-`~`, and generic Pin_N / passive names carry no
    identifying signal, so they are not reported.
    """
    nm = ((info.get(ref) or {}).get(pin) or {}).get("name") or ""
    core = nm.strip().strip("~{} ")
    if not core or core.isdigit() or _GENERIC_PIN_NAME_RE.match(core):
        return None
    return nm.strip()


def _pin_label(ref: str, pin: str, func=None) -> str:
    """Identity-safe contextual pin rendering (see the invariant above)."""
    return f"pin {pin} of {ref}" + (f" ({func})" if func else "")


def _endpoint_labels(refs, info, limit: int = 4) -> str:
    """Render up to ``limit`` (ref, pin) pairs as sorted identity-safe labels."""
    picked = sorted(set(refs))[:limit]
    return ", ".join(_pin_label(r, p, _pin_function(info, r, p)) for r, p in picked)


def _dangling_net_context(
    net: str,
    sheet: str,
    ref: str,
    pin: str,
    *,
    info,
    pin_count,
    ref_identities,
    pin_nets,
    net_endpoints,
    sheet_net_bases,
    inter_names,
) -> str:
    """Deterministic topology context for one §9.15 dangling endpoint."""
    bits: list[str] = []
    func = _pin_function(info, ref, pin)
    if func:
        bits.append(f"the wired endpoint is {_pin_label(ref, pin, func)}")

    # -- series branch: proven two-terminal part with a proven single-net other
    #    terminal (§9.17's exact pin-count invariant; anything ambiguous is
    #    omitted rather than guessed).
    if _ref_prefix(ref) in _TWO_TERMINAL_REF_PREFIXES and pin_count.get(ref) == 2:
        others = [p for p in info.get(ref, {}) if p != pin]
        other_nets = pin_nets.get((ref, others[0]), set()) if len(others) == 1 else set()
        if len(others) == 1 and len(other_nets) == 1:
            other = others[0]
            onet = next(iter(other_nets))
            dests = {(r, p) for (_s, r, p) in net_endpoints.get(onet, ()) if r != ref}
            kept_dests = set()
            rejected_dests: list[tuple[str, str, str]] = []
            for candidate_ref, candidate_pin in dests:
                candidate_func = _pin_function(info, candidate_ref, candidate_pin)
                matched = _match_known_signal_assignment(
                    ref_identities.get(candidate_ref),
                    onet,
                    candidate_func,
                )
                if matched is not None and candidate_func is not None and not matched[2]:
                    rejected_dests.append((candidate_ref, candidate_pin, matched[1][1]))
                else:
                    # Fail open when identity/function resolution is incomplete,
                    # and retain connectors/passives outside a known family.
                    kept_dests.add((candidate_ref, candidate_pin))
            other_func = _pin_function(info, ref, other)
            frag = (
                f"{ref} is a two-terminal series part whose other terminal "
                f"({_pin_label(ref, other, other_func)}) sits on net {onet!r}"
            )
            if kept_dests:
                frag += (
                    f"; candidate endpoints on that net: "
                    f"{_endpoint_labels(kept_dests, info)}. Keep each endpoint on the "
                    "side required by architecture direction and resolved pin function; "
                    "add or move the missing endpoint so both nets are complete. Do not "
                    "assume which side is source or destination"
                )
            else:
                frag += (
                    f" and {onet!r} has no proven function-compatible non-part endpoint. "
                    "Restore one compatible non-part endpoint on each side; do not assume "
                    "which side is source or destination"
                )
            frag += (
                f", keeping the two terminals of {ref} on different nets -- "
                f"never merge {onet!r} with {net!r}, and never put both "
                f"terminals of {ref} on one net (§9.17)"
            )
            if rejected_dests:
                rejected_labels = ", ".join(
                    _pin_label(r, p, _pin_function(info, r, p))
                    for r, p, _want in sorted(rejected_dests)
                )
                required = ", ".join(sorted({want for _r, _p, want in rejected_dests}))
                frag += (
                    f"; required fixed function: {required}. Rejected wrong-function "
                    f"candidate(s) {rejected_labels} cannot carry either accepted "
                    f"name variant {onet!r} or {net!r}"
                )
            bits.append(frag)

    # -- related-domain branch: same sheet, same base after stripping one
    #    explicit suffix. Reported as related context only -- the wording
    #    forbids the merge a naive reader would take from it.
    base = _net_domain_base(net)
    related = [n for n in sorted(sheet_net_bases.get((sheet, base), ())) if n != net]
    for rel in related[:4]:
        eps = sorted({(r, p) for (_s, r, p) in net_endpoints.get(rel, ())})
        frag = (
            f"related net {rel!r} on the same sheet carries "
            f"{_endpoint_labels(eps, info) if eps else 'no resolved pin'}"
        )
        # 74x245-style channel permutations are exposed deterministically:
        # an A<n> endpoint names the net its same-ref B<n> mate sits on
        # (and vice versa).
        mates = []
        for r, p in eps[:4]:
            fm = _pin_function(info, r, p)
            cm = _CHANNEL_PIN_RE.match(fm or "")
            if not cm:
                continue
            comp = f"{'B' if cm.group(1) == 'A' else 'A'}{cm.group(2)}"
            for q, pdata in (info.get(r) or {}).items():
                if q == p:
                    continue
                qname = (pdata.get("name") or "").strip().strip("~{} ")
                if _CHANNEL_PIN_RE.match(qname) and qname == comp:
                    for mnet in sorted(pin_nets.get((r, q), ())):
                        mates.append(
                            f"{_pin_label(r, p, fm)} has its channel mate "
                            f"{_pin_label(r, q, comp)} on net {mnet!r}"
                        )
        if mates:
            frag += "; " + "; ".join(sorted(set(mates))[:4])
        frag += (
            f" -- attach the missing destination on the correct side or "
            f"repair the channel assignment; do NOT merge {net!r} with "
            f"{rel!r} across the resistor/buffer/isolator/level-shifter "
            f"that separates them"
        )
        bits.append(frag)

    if inter_names:
        bits.append(
            "declared inter-sheet net names: "
            + ", ".join(inter_names[:8])
            + " -- reuse an exact one only if this signal must reach another sheet"
        )
    return "; ".join(bits)


def check_power_pin_polarity(bom) -> CheckResult:
    """§9.16 -- a supply pin's NAME must agree with the polarity of its net.

    A pin named VDD/VCC/VBAT/... wired to a ground net, or a GND/VSS/V- pin
    wired to a positive rail, is the reversed-power mistake a wiring stage makes
    when it binds pins by position instead of by name (the 8-channel DAQ board
    tied the MCU's VDD pin to GND and its VSS pin to +3V3). ERC cannot see it --
    both ends are valid power nets. Fires only when BOTH the pin name and the net
    polarity are unambiguous AND opposite, so a correctly-wired rail never trips.
    """
    info, _ = _pin_info_by_ref(bom)
    bad: list[str] = []
    for c in bom.connections:
        net_gnd = _net_is_ground(c.net_name)
        net_pos = _net_is_positive_rail(c.net_name)
        net_negative = _net_is_negative_rail(c.net_name)
        if not (net_gnd or net_pos or net_negative):
            continue
        for ep in c.endpoints:
            pin = info.get(ep.ref, {}).get(ep.pin)
            if not pin:
                continue
            nm = pin["name"]
            pos_pin = bool(_POS_SUPPLY_PIN_RE.search(nm))
            gnd_pin = bool(_GND_PIN_RE.search(nm))
            if pos_pin == gnd_pin:  # neither, or ambiguously both -> skip
                continue
            if pos_pin and net_gnd:
                bad.append(
                    f"{ep.ref}.{ep.pin} (pin {nm!r}, a positive supply) is wired to "
                    f"ground net {c.net_name!r} -- power pins look reversed"
                )
            elif pos_pin and net_negative:
                bad.append(
                    f"{ep.ref}.{ep.pin} (pin {nm!r}, a positive supply) is wired to "
                    f"negative rail {c.net_name!r} -- power pins look reversed"
                )
            elif gnd_pin and net_pos:
                bad.append(
                    f"{ep.ref}.{ep.pin} (pin {nm!r}, a ground/negative pin) is wired to "
                    f"positive rail {c.net_name!r} -- power pins look reversed"
                )
    return CheckResult(
        name="9.16 power pin polarity",
        ok=not bad,
        message=(
            "supply pins agree with net polarity"
            if not bad
            else f"{len(bad)} supply pin(s) on the wrong-polarity net"
        ),
        offenders=bad,
    )


# ---------- §9.31 repeated-block coverage ----------

# Prefixes where an electrically-inert duplicate is essentially never
# intentional: connectors and human-interface parts. IC prefixes (U) are
# excluded on purpose -- an unused half of a dual op-amp is a legitimate spare.
_COVERAGE_REF_PREFIXES = frozenset({"J", "SW", "K", "LED", "D", "RV", "BT"})


def check_repeated_block_coverage(bom) -> CheckResult:
    """§9.31 — every instance of a repeated connector/HMI part must be wired.

    A brief that asks for N identical channels gets N identical parts; when
    only one is wired and the rest have every pin NC, the board silently ships
    with (N-1) dead channels (the four-jack audio buffer shipped fab-ready
    with 3 of 4 jacks electrically inert, batch 2026-07-17 run_28). ERC and
    §9.9 cannot see it: the NC declarations make the sheet "clean". Flag any
    part that (a) shares symbol+value+footprint with a WIRED sibling, and
    (b) itself has zero wired pins.
    """
    nets = _nets_by_ref(bom)
    groups: dict[tuple, list] = defaultdict(list)
    for part in bom.parts:
        if _ref_prefix(part.ref) not in _COVERAGE_REF_PREFIXES:
            continue
        groups[(part.symbol, part.value, part.footprint)].append(part)
    bad: list[str] = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(members) < 2:
            continue
        wired = [p for p in members if len(nets.get(p.ref, {})) >= 2]
        if not wired:
            continue  # the whole group is unwired -> §9.9/§9.11 territory
        for p in members:
            if not nets.get(p.ref):
                bad.append(
                    f"{p.ref} ({p.symbol} {p.value or ''}): no pin is wired "
                    f"while identical sibling(s) "
                    f"{', '.join(w.ref for w in wired)} are -- a declared "
                    f"channel is electrically inert (wire it or remove it "
                    f"from the BOM; declaring its pins NC hides, not fixes, "
                    f"the missing channel)"
                )
    return CheckResult(
        name="9.31 repeated-block coverage",
        ok=not bad,
        message=(
            "every repeated connector/HMI instance is wired"
            if not bad
            else f"{len(bad)} repeated part(s) electrically inert"
        ),
        offenders=bad,
    )


# ---------- §9.32 regulator feedback divider ----------

# Feedback reference voltages for common adjustable regulators, keyed by MPN
# prefix (longest match wins). Only families we are SURE of are listed -- a
# missing entry means "not checked", never a guess. (The judge model
# hallucinated 0.8 V for the TPS5430's 1.221 V reference and wrongly failed a
# correct 3.3 V design in the 2026-07-17 batch -- this table is the antidote.)
_REGULATOR_VREF: dict[str, float] = {
    "TPS5430": 1.221,
    "TPS54331": 0.8,
    "TPS54231": 0.8,
    "TPS54160": 0.8,
    "TPS562": 0.768,
    "LM2596": 1.23,
    "LM2576": 1.23,
    "LM2675": 1.21,
    "MP1584": 0.8,
    "MP2315": 0.811,
    "XL4015": 1.25,
    "MT3608": 0.6,
}

_GND_NET_NAMES = frozenset({"GND", "AGND", "PGND", "DGND", "0V", "GNDA"})

_R_VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kKmMrR]?)(?:\s*(?:Ω|ohm|Ohm|OHM))?\s*$")
_R_INFIX_RE = re.compile(r"^(\d+)([kKmMrR])(\d+)$")  # 4k7 style
# '3V3' / '1V25' (digits around V) or '3.3V' / '12V' / '+5V' (decimal + V).
_NET_VOLTAGE_RE = re.compile(
    r"(?:^|[^0-9.])(\d{1,2})V(\d{1,2})?(?:$|[^0-9])"
    r"|(?:^|[^0-9])(\d{1,2}(?:\.\d{1,2})?)V(?:$|[^0-9])"
)

# EIA/KiCad convention: lowercase m is milli (a 100m current-sense shunt),
# uppercase M is mega. Milli-ohm parts can never form a sane feedback divider,
# but mis-reading one as 10^8 ohms picks WRONG dividers.
_R_SCALE = {"k": 1e3, "K": 1e3, "m": 1e-3, "M": 1e6, "r": 1.0, "R": 1.0, "": 1.0}


def _resistance_ohms(value: str) -> float | None:
    s = (value or "").strip()
    m = _R_INFIX_RE.match(s)
    if m:
        whole, mult, frac = m.groups()
        base = float(f"{whole}.{frac}")
    else:
        m2 = _R_VALUE_RE.match(s)
        if not m2:
            return None
        base = float(m2.group(1))
        mult = m2.group(2)
    ohms = base * _R_SCALE[mult]
    return ohms if ohms > 0 else None


def check_typed_led_current_paths(architecture, bom) -> CheckResult:
    """§9.36 — simple typed passive LED channels need forward bias and a limiter.

    Trace only resistors and explicit net scope. Unrelated LEDs, negative-rail
    indicators and active current-driver circuits are not inferred from labels.
    """
    requirements = [
        row
        for row in architecture.requirements
        if row.family in {"led-current-resistor", "status-led"}
    ]
    if not requirements:
        return CheckResult("9.36 typed LED current paths", True)
    global_nets = {
        "GND",
        *architecture.power_nets,
        *(n.name for n in architecture.inter_sheet_nets),
    }

    def scoped(sheet, net):
        return ("" if net in global_nets else sheet, net)

    nets = _nets_by_ref(bom)
    graph = defaultdict(list)
    for part in bom.parts:
        if part.symbol not in {"Device:R", "Device:R_Small"}:
            continue
        pins = nets.get(part.ref, {})
        if "1" not in pins or "2" not in pins:
            continue
        left, right = (scoped(part.sheet, pins[pin]) for pin in ("1", "2"))
        positive = _resistance_ohms(part.value) is not None
        graph[left].append((right, positive))
        graph[right].append((left, positive))

    def reachable(start, without_limiter=False):
        seen, pending = {start}, [start]
        while pending:
            for other, positive in graph.get(pending.pop(), ()):
                if other not in seen and not (without_limiter and positive):
                    seen.add(other)
                    pending.append(other)
        return seen

    info, _ = _pin_info_by_ref(bom)
    bad = []
    for requirement in requirements:
        ground = requirement.ports.get("gnd")
        if not ground:
            continue
        ground_key = scoped(requirement.sheet, ground)
        supply_names = set(architecture.power_nets) | {requirement.ports.get("vdd", "")}
        supplies = {
            scoped(requirement.sheet, net)
            for net in supply_names
            if net and _net_is_positive_rail(net)
        }
        drives = {
            scoped(requirement.sheet, net)
            for key, net in requirement.ports.items()
            if not _net_is_negative_rail(net)
            and (key == "drive" or (net != ground and net not in supply_names))
        }
        for part in bom.parts:
            if part.sheet != requirement.sheet or not part.symbol.startswith("Device:LED"):
                continue
            terminals = {
                pin["name"]: number
                for number, pin in info.get(part.ref, {}).items()
                if pin["name"] in {"A", "K"}
            }
            pins = nets.get(part.ref, {})
            if len(terminals) != 2 or any(number not in pins for number in terminals.values()):
                continue  # Pin existence/coverage own unresolved terminal evidence.
            anode, cathode = (scoped(part.sheet, pins[terminals[name]]) for name in ("A", "K"))
            a_reach, k_reach = reachable(anode), reachable(cathode)
            source = bool(a_reach & drives) and ground_key in k_reach
            sink = bool(k_reach & drives) and bool(a_reach & supplies)
            reversed_path = (bool(k_reach & drives) and ground_key in a_reach) or (
                bool(a_reach & drives) and bool(k_reach & supplies)
            )
            identity = f"{part.ref}.{terminals['A']} (A), {part.ref}.{terminals['K']} (K)"
            if reversed_path and not (source or sink):
                bad.append(
                    f"{identity} on sheet {part.sheet!r}: reversed LED in "
                    f"{requirement.id!r}; positive current must enter A and leave K"
                )
            elif source or sink:
                a_unlimited = reachable(anode, without_limiter=True)
                k_unlimited = reachable(cathode, without_limiter=True)
                bypass = (bool(a_unlimited & drives) and ground_key in k_unlimited) or (
                    bool(k_unlimited & drives) and bool(a_unlimited & supplies)
                )
                if bypass:
                    bad.append(
                        f"{identity} on sheet {part.sheet!r}: {requirement.id!r} has "
                        "a drive path without a proven positive series resistance; "
                        "0-ohm links and unknown values do not establish current limiting"
                    )
    return CheckResult(
        "9.36 typed LED current paths",
        not bad,
        "typed passive LED paths agree with polarity and current limiting"
        if not bad
        else f"{len(bad)} invalid typed LED path(s)",
        bad,
    )


def _net_voltage(net_name: str) -> float | None:
    """Parse a rail voltage out of a net name ('3V3', '1V25', '+5V',
    'VOUT_12.5V'). Returns None rather than a wrong number on anything
    ambiguous."""
    m = _NET_VOLTAGE_RE.search((net_name or "").upper())
    if not m:
        return None
    if m.group(1) is not None:
        whole, frac = m.group(1), m.group(2)
        return float(f"{whole}.{frac}") if frac else float(whole)
    return float(m.group(3))


def regulator_vout_facts(parts: list[dict], connections: list[dict]) -> list[dict]:
    """Deterministic Vout computation for known adjustable regulators.

    Dict-based so both the §9.32 gate and the eval digest can call it (the
    latter reads raw state.json). For each part whose MPN matches
    ``_REGULATOR_VREF``, locate the classic divider -- a mid net joining one
    regulator pin, resistor A (other pin on a non-ground rail net) and
    resistor B (other pin on ground) -- and compute
    ``vout = vref * (1 + Ra/Rb)``. Returns one fact dict per UNAMBIGUOUS
    divider found: {ref, mpn, vref, r_top_ref, r_top, r_bot_ref, r_bot,
    vout, rail_net, rail_v, ok}; ok is None when the rail net names no
    parseable target voltage. Ambiguous or unrecognized topologies produce
    no fact (never a guess).
    """
    nets: dict[str, dict[str, str]] = defaultdict(dict)
    for c in connections:
        for ep in c.get("endpoints") or []:
            nets[str(ep.get("ref"))][str(ep.get("pin"))] = str(c.get("net_name"))
    r_ohms = {
        str(p.get("ref")): _resistance_ohms(str(p.get("value") or ""))
        for p in parts
        if str(p.get("ref", "")).startswith("R")
    }
    facts: list[dict] = []
    for p in parts:
        mpn = str(p.get("mpn") or p.get("value") or "").upper()
        vref = next(
            (
                v
                for prefix, v in sorted(_REGULATOR_VREF.items(), key=lambda kv: -len(kv[0]))
                if mpn.startswith(prefix)
            ),
            None,
        )
        if vref is None:
            continue
        ref = str(p.get("ref"))
        reg_nets = set(nets.get(ref, {}).values())
        candidates: list[dict] = []
        for mid in sorted(reg_nets):
            if mid.upper() in _GND_NET_NAMES:
                continue
            # Resistors with one pin on the mid net.
            on_mid = [r for r, pins in nets.items() if r in r_ohms and mid in pins.values()]
            for r_top in on_mid:
                for r_bot in on_mid:
                    if r_top == r_bot:
                        continue
                    if r_ohms.get(r_top) is None or r_ohms.get(r_bot) is None:
                        continue
                    top_other = [n for pin, n in nets[r_top].items() if n != mid]
                    bot_other = [n for pin, n in nets[r_bot].items() if n != mid]
                    if len(top_other) != 1 or len(bot_other) != 1:
                        continue
                    if bot_other[0].upper() not in _GND_NET_NAMES:
                        continue
                    if top_other[0].upper() in _GND_NET_NAMES:
                        continue
                    rail = top_other[0]
                    vout = vref * (1.0 + r_ohms[r_top] / r_ohms[r_bot])
                    candidates.append(
                        {
                            "ref": ref,
                            "mpn": str(p.get("mpn") or ""),
                            "vref": vref,
                            "r_top_ref": r_top,
                            "r_top": r_ohms[r_top],
                            "r_bot_ref": r_bot,
                            "r_bot": r_ohms[r_bot],
                            "vout": round(vout, 3),
                            "rail_net": rail,
                            "rail_v": _net_voltage(rail),
                        }
                    )
        if len(candidates) != 1:
            continue  # none found, or ambiguous -- never guess
        fact = candidates[0]
        fact["ok"] = (
            None
            if fact["rail_v"] is None
            else abs(fact["vout"] - fact["rail_v"]) <= 0.10 * fact["rail_v"]
        )
        facts.append(fact)
    return facts


def check_regulator_feedback_vout(bom) -> CheckResult:
    """§9.32 — an adjustable regulator's feedback divider must produce the
    rail its output net names. Deterministic: known Vref x the wired divider.
    Only flags an UNAMBIGUOUS divider whose computed Vout misses a parseable
    rail-net voltage by >10% -- everything uncertain passes silently."""
    parts = [{"ref": p.ref, "mpn": getattr(p, "mpn", None), "value": p.value} for p in bom.parts]
    connections = [
        {
            "net_name": c.net_name,
            "endpoints": [{"ref": ep.ref, "pin": ep.pin} for ep in c.endpoints],
        }
        for c in bom.connections
    ]
    facts = regulator_vout_facts(parts, connections)
    bad = [
        f"{f['ref']} ({f['mpn']}, Vref {f['vref']}V): divider "
        f"{f['r_top_ref']}/{f['r_bot_ref']} = "
        f"{f['r_top']:.0f}/{f['r_bot']:.0f} ohm gives "
        f"Vout {f['vout']}V but the output net {f['rail_net']!r} names "
        f"{f['rail_v']}V -- fix the divider (R_top ~= "
        f"{f['r_bot'] * (f['rail_v'] / f['vref'] - 1.0):.0f} ohm for "
        f"{f['rail_v']}V)"
        for f in facts
        if f["ok"] is False
    ]
    return CheckResult(
        name="9.32 regulator feedback divider",
        ok=not bad,
        message=(
            "feedback dividers match their named rails"
            if not bad
            else f"{len(bad)} regulator divider(s) produce the wrong voltage"
        ),
        offenders=bad,
    )


def check_two_terminal_self_short(bom) -> CheckResult:
    """§9.17 -- a two-terminal part with both pins on one net is shorted out.

    A fuse, resistor, inductor, diode, or antenna whose two terminals land on
    the SAME net does nothing (the +/-12V board wired both ends of its input
    fuse AND its reverse-polarity diode across VIN, silently disabling all input
    protection). ERC passes because the net is otherwise valid. Only 2-pin parts
    of known two-terminal classes are considered, and only when BOTH pins are
    actually wired (a pin in no_connect_pins is not a short).
    """
    _, pin_count = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
    for part in bom.parts:
        if pin_count.get(part.ref) != 2:
            continue
        if _ref_prefix(part.ref) not in _TWO_TERMINAL_REF_PREFIXES:
            continue
        wired = nets.get(part.ref, {})
        if len(wired) == 2 and len(set(wired.values())) == 1:
            bad.append(
                f"{part.ref} ({part.symbol}) has both terminals on net "
                f"{next(iter(set(wired.values())))!r} -- the part is shorted out "
                f"and does nothing"
            )
    return CheckResult(
        name="9.17 two-terminal self-short",
        ok=not bad,
        message=(
            "no two-terminal part shorts itself"
            if not bad
            else f"{len(bad)} two-terminal part(s) shorted across a single net"
        ),
        offenders=bad,
    )


def check_rf_feed_isolation(bom) -> CheckResult:
    """§9.18 -- an antenna's RF feed pin must not be tied to a rail or ground.

    A chip/PCB antenna whose feed pin lands on GND (or a power rail) cannot
    radiate (the nRF52 beacon shorted its antenna feed net to GND, so the
    radio's output never reached the antenna). ERC sees only a valid GND
    connection. Checks parts with an antenna refdes whose feed/RF-named pin sits
    on a ground or positive-rail net.
    """
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
    for part in bom.parts:
        if _ref_prefix(part.ref) not in _ANTENNA_REF_PREFIXES:
            continue
        for num, pin in info.get(part.ref, {}).items():
            if not _RF_FEED_PIN_RE.search(pin["name"]):
                continue
            net = nets.get(part.ref, {}).get(num)
            if net and (_net_is_ground(net) or _net_is_positive_rail(net)):
                bad.append(
                    f"{part.ref}.{num} (antenna feed pin {pin['name']!r}) is tied to "
                    f"{net!r} -- the antenna cannot radiate"
                )
    return CheckResult(
        name="9.18 rf feed isolation",
        ok=not bad,
        message=(
            "antenna feed pins reach the RF line"
            if not bad
            else f"{len(bad)} antenna feed pin(s) tied to a rail/ground"
        ),
        offenders=bad,
    )


# ---------- §9.19 single net per pin (Layer 2) ----------


def check_single_net_per_pin(bom) -> CheckResult:
    """§9.19 -- every part pin belongs to exactly one net.

    A pin (ref, number) listed in two NetConnections with DIFFERENT net_names
    shorts those nets together -- an ERC/DRC-invisible defect, because the
    emitter merges the two labels into one valid net. This is the wiring stage's
    most common functional short: the DRV8833 VM pin on both VBAT and VCP_VM
    (shorting the motor rail and removing the charge-pump cap); the nRF52
    matching cap pin on both ANT_FEED and GND (grounding the antenna); the CH224K
    zener pin on both VBUS and GND. A net_name repeated across sheets (an
    inter-sheet net wired on each side) is fine -- only DISTINCT names on one pin
    short, and a pin is only ever on one sheet, so >1 distinct name is always
    wrong.
    """
    nets_for_pin: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in bom.connections:
        for ep in c.endpoints:
            nets_for_pin[(ep.ref, ep.pin)].add(c.net_name)
    bad = [
        f"{ref}.{pin} is wired to {len(names)} different nets "
        f"({', '.join(sorted(names))}) -- this shorts them together"
        for (ref, pin), names in sorted(nets_for_pin.items())
        if len(names) > 1
    ]
    return CheckResult(
        name="9.19 single net per pin",
        ok=not bad,
        message=(
            "every pin belongs to one net"
            if not bad
            else f"{len(bad)} pin(s) wired to multiple nets (shorted)"
        ),
        offenders=bad,
    )


# ---------- §9.20 part-family wiring contracts (Layer 2) ----------
#
# A datasheet-keyed rulebook that asserts pin ROLES by name for the single net a
# pin lands on -- catching a functional pin bound to the wrong (but single) net,
# which §9.16 (cross-polarity) and §9.19 (multi-net short) do not see (e.g. a
# flash VCC scrambled onto a data net, or a CAN transceiver's RS pin strapped to
# the rail = standby). Each contract matches a part by a regex over
# "<symbol> <value>" and lists (pin-name regex, role). Conservative: a pin not in
# connections (no_connect) is skipped, and the net-class tests accept any
# power-ish / ground-ish *name* (not just canonical rails), so a filtered or
# locally-named rail never trips it. Append-only -- add a family by adding a row.

_PWR_NET_TOKENS = (
    "VDD",
    "VCC",
    "VBAT",
    "VBUS",
    "VSYS",
    "VIN",
    "VOUT",
    "VREG",
    "VPP",
    "3V3",
    "5V",
    "1V8",
    "2V5",
    "12V",
)


def _net_looks_power(name: str) -> bool:
    s = name.lstrip("/").upper()
    return s.startswith("+") or _net_is_positive_rail(name) or any(t in s for t in _PWR_NET_TOKENS)


def _net_looks_ground(name: str) -> bool:
    s = name.lstrip("/").upper()
    return _net_is_ground(name) or "GND" in s or s in ("VSS", "0V")


@dataclass(frozen=True)
class _FamilyContract:
    name: str
    match: re.Pattern
    rules: tuple  # ((pin-name re.Pattern, role:str), ...)


# Roles: "rail" (must be on a supply), "ground" (must be on ground), "signal"
# (data/clock/CS line -- must NOT be on a rail or ground), "not_rail" (must NOT
# be on a positive rail; e.g. CAN RS high = standby).
_FAMILY_CONTRACTS: tuple[_FamilyContract, ...] = (
    _FamilyContract(
        name="spi_flash",
        match=re.compile(r"w25q|gd25|mx25|en25|s25fl|is25|at25q", re.I),
        rules=(
            (re.compile(r"^~?\{?VCC\}?~?$|^VDD$", re.I), "rail"),
            (re.compile(r"^GND$|^VSS$", re.I), "ground"),
            # IO0/IO1 (DI/DO), CLK and CS are data/clock/select in BOTH SPI and
            # QSPI modes; WP/HOLD (IO2/IO3) are excluded -- they are legitimately
            # tied to VCC in plain SPI mode.
            (
                re.compile(r"(^|[^0-9])IO[01]([^0-9]|$)|^DI$|^DO$|/IO[01]$|^CLK$|^SCK$|CS", re.I),
                "signal",
            ),
        ),
    ),
    _FamilyContract(
        name="can_transceiver",
        match=re.compile(r"sn65hvd|mcp255\d|tja10|65hvd2", re.I),
        rules=(
            (re.compile(r"^VCC$|^VDD$", re.I), "rail"),
            (re.compile(r"^GND$", re.I), "ground"),
            (re.compile(r"^RS$|^STB$|^/STB$|^S$", re.I), "not_rail"),
        ),
    ),
)


# Known SIGNAL nets whose functional pin is fixed by silicon, not by design
# choice (KC-VKUT5H A2). The frozen candidates of that board bound native USB
# D+/D- to arbitrary GPIOs (IO11/IO12, later IO13/IO14): §9.15 can be cleared
# by moving those same wrong endpoints across the series resistors, so only a
# deterministic function-level gate prevents a §9.15-clean but non-functional
# commit. Rules are name-based (never physical pin numbers), so a symbol with
# different numbering but correct function names still passes.
@dataclass(frozen=True)
class _SignalAssignment:
    name: str
    family: re.Pattern  # matches "<symbol> <value>"
    signals: tuple  # ((net-name re.Pattern, required pin-function substring, role), ...)


_USB_DOMAIN_SUFFIX = r"(?:[-_]?(?:5V|3V3|MCU|POWER|ESP32|ISO|LV|HV))?"
# (family regex, D- pin function, D+ pin function); case-insensitive. These
# reviewed families are factory-native-programmable over their own USB pair.
_NATIVE_USB_FAMILIES: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"esp32[-_ ]?s3", re.I), "IO19", "IO20"),
    (re.compile(r"esp32[-_ ]?c3", re.I), "IO18", "IO19"),
    (re.compile(r"rp2040", re.I), "USB_DM", "USB_DP"),
)
_KNOWN_SIGNAL_ASSIGNMENTS: tuple[_SignalAssignment, ...] = tuple(
    _SignalAssignment(
        name=f"{re.sub(r'[^a-z0-9]', '', family.pattern.lower())}_native_usb",
        family=family,
        signals=(
            # Exact differential forms with ONE optional known domain suffix;
            # no loose substring matching (USB_P, USBD, USB_DPH are not D+).
            (re.compile(rf"^USB_D(?:\+|_?P){_USB_DOMAIN_SUFFIX}$", re.I), dp_pin, "D+"),
            (re.compile(rf"^USB_D(?:-|_?[NM]){_USB_DOMAIN_SUFFIX}$", re.I), dm_pin, "D-"),
        ),
    )
    for family, dm_pin, dp_pin in _NATIVE_USB_FAMILIES
)


def _match_known_signal_assignment(
    part_identity: str | None,
    net_name: str,
    pin_function: str | None,
) -> tuple[_SignalAssignment, tuple, bool] | None:
    """Return the fixed-signal rule matching ``part_identity``/``net_name``.

    The boolean reports whether the resolved pin function satisfies the rule.
    Missing identity or function information fails open at the caller.
    """
    if not part_identity:
        return None
    for assignment in _KNOWN_SIGNAL_ASSIGNMENTS:
        if not assignment.family.search(part_identity):
            continue
        for signal in assignment.signals:
            sig_re, want, _role = signal
            if sig_re.match(net_name):
                satisfies = pin_function is not None and want in pin_function.upper()
                return assignment, signal, satisfies
    return None


def _check_known_signal_assignments(part, info, nets) -> list[str]:
    """One _SignalAssignment pass for a single part (called from §9.20).

    Fires only when the net name unambiguously denotes the fixed-function
    signal AND the pin's resolvable function contradicts it; an unresolvable
    symbol, a missing USB-named net, or an ambiguous name fails open."""
    ident = f"{part.symbol} {part.value}"
    bad: list[str] = []
    pins = info.get(part.ref) or {}
    wired = nets.get(part.ref, {})
    for num, net in sorted(wired.items()):
        nm = (pins.get(num) or {}).get("name") or ""
        matched = _match_known_signal_assignment(ident, net, nm or None)
        if matched is None:
            continue
        assignment, signal, satisfies = matched
        sig_re, want, role = signal
        if satisfies or not nm.strip():
            continue
        # Name the concrete target so the correction is a move, not a
        # search: resolve the wanted function on THIS part's loaded symbol.
        # Only act on a unique match; report the target pin's current net.
        hits = sorted(q for q, pdata in pins.items() if want in (pdata.get("name") or "").upper())
        action = (
            f"Move this net's endpoint onto the pin whose function contains {want!r} "
            f"(native USB {role})"
        )
        target = ""
        if len(hits) == 1:
            q = hits[0]
            target = f" -- the correct endpoint is {_pin_label(part.ref, q, pins[q]['name'])}"
            cur = wired.get(q)
            if cur and cur != net:
                current_match = _match_known_signal_assignment(ident, cur, pins[q]["name"])
                if current_match is not None and current_match[1][0] is sig_re:
                    action = (
                        f"Keep the uniquely resolved required-function endpoint ({want}) "
                        f"on its existing USB net {cur!r}; remove "
                        f"{_pin_label(part.ref, num, nm)} from every accepted name "
                        f"for native USB {role}; do not merge {net!r} "
                        f"with {cur!r}; keep any proven series-part terminals on "
                        f"different nets. Connect the removed {nm!r} pin to a separately "
                        "named functional net only if another real endpoint requires "
                        "that function; otherwise mark it no_connect"
                    )
                    target += f", currently on accepted {role} net {cur!r}"
                else:
                    target += f", currently on net {cur!r} (swap the two)"
        bad.append(
            f"[{assignment.name}] {net!r} is a native USB "
            f"{role} data line (fixed silicon function), but it is wired "
            f"to {_pin_label(part.ref, num, nm)}. {action}{target}; "
            "do not substitute other GPIOs"
        )
    return bad


def check_family_wiring_contracts(bom) -> CheckResult:
    """§9.20 -- datasheet pin-role contracts for known part families.

    See the module comment above _FAMILY_CONTRACTS. Fires only on a wired pin
    that a family's datasheet says must (not) be on a rail/ground and is bound to
    a clearly-wrong net; correct and filtered rails pass. Also evaluates the
    fixed-function signal assignments (_KNOWN_SIGNAL_ASSIGNMENTS): a net that
    unambiguously names a silicon-fixed pin function (ESP32-S3 native USB
    D+/D-) must reach that exact functional pin.
    """
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
    for part in bom.parts:
        bad.extend(_check_known_signal_assignments(part, info, nets))
    for part in bom.parts:
        ident = f"{part.symbol} {part.value}"
        for contract in _FAMILY_CONTRACTS:
            if not contract.match.search(ident):
                continue
            wired = nets.get(part.ref, {})
            for num, pdata in info.get(part.ref, {}).items():
                nm = pdata["name"]
                net = wired.get(num)
                if net is None:
                    continue
                for rule_re, role in contract.rules:
                    if not rule_re.search(nm):
                        continue
                    problem = None
                    if role == "rail" and not _net_looks_power(net):
                        problem = "must be on a supply rail"
                    elif role == "ground" and not _net_looks_ground(net):
                        problem = "must be on ground"
                    elif role == "signal" and (_net_is_ground(net) or _net_is_positive_rail(net)):
                        problem = "is a data/clock/CS line but sits on power/ground"
                    elif role == "not_rail" and _net_is_positive_rail(net):
                        problem = "must not be on a positive rail (that selects standby/wrong mode)"
                    if problem:
                        bad.append(
                            f"[{contract.name}] {part.ref}.{num} (pin {nm!r}) {problem} "
                            f"-- wired to {net!r}"
                        )
                    break  # one rule per pin
    return CheckResult(
        name="9.20 part-family wiring contracts",
        ok=not bad,
        message=(
            "family pin roles satisfied"
            if not bad
            else f"{len(bad)} pin(s) violate a part-family wiring contract"
        ),
        offenders=bad,
    )


# ---------- §9.37 reviewed electrical realization ----------
#
# These checks intentionally consume only typed architecture claims, direct
# component terminals, and manufacturer-reviewed device facts.  A net name,
# a capacitor on a nearby net, or graph reachability through a control pin is
# not evidence that energy can reach a load.

_CAP_VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(p|n|u|µ|m)?(?:f|farad(?:s)?)?$", re.I)
_INDUCTANCE_VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(n|u|µ|m)?h$", re.I)
_CAP_SCALE = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "": 1.0}
_INDUCTANCE_SCALE = {"n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "": 1.0}


def _capacitance_farads(value: str) -> float | None:
    """Parse an explicitly unit-bearing capacitor value, never guessing units."""
    match = _CAP_VALUE_RE.match((value or "").strip())
    if match is None:
        return None
    magnitude, prefix = match.groups()
    return float(magnitude) * _CAP_SCALE[prefix.lower()]


def _inductance_henries(value: str) -> float | None:
    """Parse a typed inductance value; bare ``510`` is intentionally unknown."""
    match = _INDUCTANCE_VALUE_RE.match((value or "").strip())
    if match is None:
        return None
    magnitude, prefix = match.groups()
    return float(magnitude) * _INDUCTANCE_SCALE[prefix.lower()]


def _reviewed_fact_for_part(part) -> dict | None:
    """Return one exact reviewed identity; never infer one from a substring."""
    from kicraft.design.part_identity import reviewed_part

    identity = str(getattr(part, "mpn", None) or part.value or "").strip()
    if not identity:
        return None
    record = reviewed_part(identity)
    return vars(record) if record is not None else None


def _fact_number(fact: dict, *keys: str) -> float | None:
    for key in keys:
        value = fact.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed not in (float("inf"), float("-inf")):
            return parsed
    return None


def _fact_pin_name(fact: dict, key: str) -> str | None:
    pins = fact.get("pins") or fact.get("port_pins") or {}
    value = pins.get(key) if isinstance(pins, dict) else None
    if value is None:
        aliases = {"vin": "input", "ph": "switch"}
        value = pins.get(aliases[key]) if isinstance(pins, dict) and key in aliases else None
    if value is None:
        value = fact.get(f"{key}_pin")
    return str(value).upper() if value is not None else None


# A reviewed record names its voltage input consistently, not identically: the
# port key may be `vin`/`input`, and a converter's return-referenced input is
# `input_positive`. The limits carry the same three spellings.
_INPUT_PORT_KEYS = ("vin", "input", "input_positive")
_INPUT_MIN_KEYS = ("vin_min_v", "input_min_v", "input_voltage_min_v")
_INPUT_MAX_KEYS = ("vin_max_v", "input_max_v", "input_voltage_max_v")


def _fact_input_pin_name(fact: dict) -> str | None:
    """The reviewed record's voltage-input pin, under any published spelling."""
    for key in _INPUT_PORT_KEYS:
        name = _fact_pin_name(fact, key)
        if name:
            return name
    return None


def _reviewed_name(fact: dict) -> str | None:
    """The reviewed record's own identity, for stage feedback that names the device."""
    return fact.get("identity") or fact.get("mpn")


def _pin_number_named(info: dict, ref: str, pin_name: str) -> str | None:
    pins = info.get(ref) or {}
    if pin_name in pins:
        return pin_name
    hits = [number for number, pin in pins.items() if pin["name"].upper() == pin_name]
    return hits[0] if len(hits) == 1 else None


def _two_terminal_part_nets(part, nets) -> tuple[str, str] | None:
    wired = list((nets.get(part.ref) or {}).values())
    return (wired[0], wired[1]) if len(wired) == 2 and wired[0] != wired[1] else None


def check_reviewed_device_support_networks(bom) -> CheckResult:
    """§9.37 — prove direct mandatory support networks for reviewed devices.

    At present this covers a reviewed bootstrap specification.  The capacitor
    must directly span the actual BOOT and PH pin nets and meet the reviewed
    value; a capacitor to ground, a same-net BOOT/PH short, or a merely
    adjacent control network cannot establish bootstrap support.
    """
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
    for part in bom.parts:
        fact = _reviewed_fact_for_part(part)
        if fact is None:
            continue
        bootstrap = fact.get("bootstrap") or {}
        boot = (
            _fact_pin_name(fact, "boot") or str(bootstrap.get("positive_pin") or "").upper() or None
        )
        phase = (
            _fact_pin_name(fact, "ph") or str(bootstrap.get("negative_pin") or "").upper() or None
        )
        required_cap = _fact_number(fact, "bootstrap_capacitance_f")
        if required_cap is None:
            capacitance_uf = _fact_number(bootstrap, "capacitance_uf")
            required_cap = capacitance_uf * 1e-6 if capacitance_uf is not None else None
        if not (boot and phase and required_cap is not None):
            continue
        boot_number = _pin_number_named(info, part.ref, boot)
        phase_number = _pin_number_named(info, part.ref, phase)
        if boot_number is None or phase_number is None:
            bad.append(
                f"E_BOOTSTRAP_SUPPORT {part.ref}: reviewed {fact.get('mpn')!r} requires "
                f"distinct {boot}/ {phase} pin evidence, but the loaded symbol does not expose it"
            )
            continue
        boot_net = (nets.get(part.ref) or {}).get(boot_number)
        phase_net = (nets.get(part.ref) or {}).get(phase_number)
        if not boot_net or not phase_net:
            bad.append(
                f"E_BOOTSTRAP_SUPPORT {part.ref}.{boot_number}/{part.ref}.{phase_number}: "
                "BOOT and PH must both be wired to prove the required bootstrap loop"
            )
            continue
        if boot_net == phase_net:
            bad.append(
                f"E_BOOTSTRAP_SUPPORT {part.ref}: {boot} and {phase} share {boot_net!r}; "
                "they must be distinct nets bridged only by the bootstrap capacitor"
            )
            continue
        capacitors = []
        for candidate in bom.parts:
            if _ref_prefix(candidate.ref) != "C":
                continue
            pair = _two_terminal_part_nets(candidate, nets)
            if pair is not None and set(pair) == {boot_net, phase_net}:
                capacitors.append(candidate)
        if not capacitors:
            bad.append(
                f"E_BOOTSTRAP_SUPPORT {part.ref}: no capacitor directly spans "
                f"{boot} net {boot_net!r} and {phase} net {phase_net!r}"
            )
            continue
        tolerance = (
            _fact_number(fact, "bootstrap_capacitance_tolerance")
            or _fact_number(bootstrap, "capacitance_tolerance")
            or 0.20
        )
        if not any(
            (value := _capacitance_farads(candidate.value)) is not None
            and abs(value - required_cap) <= required_cap * tolerance
            for candidate in capacitors
        ):
            values = ", ".join(f"{candidate.ref}={candidate.value!r}" for candidate in capacitors)
            bad.append(
                f"E_BOOTSTRAP_SUPPORT {part.ref}: {values} span {boot}/{phase}, but "
                f"reviewed support requires {required_cap * 1e9:g}nF ±{tolerance * 100:g}%"
            )
    return CheckResult(
        "9.37 reviewed device support networks",
        not bad,
        "reviewed device support networks are directly realized"
        if not bad
        else f"{len(bad)} reviewed device support network(s) unproven",
        bad,
    )


def check_reviewed_input_operating_ranges(architecture, bom) -> CheckResult:
    """§9.38 — compare actual typed VIN rails with reviewed device ranges."""
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
    for part in bom.parts:
        fact = _reviewed_fact_for_part(part)
        if fact is None:
            continue
        limits = fact.get("operating_limits") or {}
        vin = _fact_input_pin_name(fact)
        vin_min = _fact_number(fact, *_INPUT_MIN_KEYS)
        vin_max = _fact_number(fact, *_INPUT_MAX_KEYS)
        if vin_min is None:
            vin_min = _fact_number(limits, *_INPUT_MIN_KEYS)
        if vin_max is None:
            vin_max = _fact_number(limits, *_INPUT_MAX_KEYS)
        if vin is None and vin_min is None and vin_max is None:
            # The reviewed record declares no voltage-input character at all (a
            # MOSFET's drain/source, a bare pass element, a holder's terminals):
            # there is no input rail to compare, so no range is missing.
            continue
        if not (vin and vin_min is not None and vin_max is not None):
            bad.append(
                f"E_INPUT_OPERATING_RANGE {part.ref}: reviewed {_reviewed_name(fact)!r} "
                "declares a voltage input but lacks a complete input pin/minimum/maximum "
                "operating-range record"
            )
            continue
        number = _pin_number_named(info, part.ref, vin)
        actual_net = (nets.get(part.ref) or {}).get(number or "")
        if actual_net is None:
            bad.append(
                f"E_INPUT_OPERATING_RANGE {part.ref}: reviewed {_reviewed_name(fact)!r} "
                f"has no wired {vin} pin; actual input voltage is unproven"
            )
            continue
        raw_voltage = architecture.rail_voltages.get(actual_net)
        try:
            voltage = float(raw_voltage)
        except (TypeError, ValueError):
            voltage = None
        if voltage is None:
            bad.append(
                f"E_INPUT_OPERATING_RANGE {part.ref}.{number}: actual VIN net {actual_net!r} "
                "has no typed architecture rail_voltages value"
            )
        elif not vin_min <= voltage <= vin_max:
            bad.append(
                f"E_INPUT_OPERATING_RANGE {part.ref}.{number}: {actual_net!r} is typed "
                f"{voltage:g}V, outside reviewed {_reviewed_name(fact)!r} VIN range "
                f"{vin_min:g}–{vin_max:g}V"
            )
    return CheckResult(
        "9.38 reviewed input operating ranges",
        not bad,
        "typed input rails are within reviewed operating ranges"
        if not bad
        else f"{len(bad)} reviewed operating-range violation(s)",
        bad,
    )


# The reference domain belongs to the port that *is* the return: a supply port
# already carries its rail, and the derivation refuses two nets on one port
# (`conflicting_port_binding`), so a rail-bound port can never also declare its
# domain. These are the published return ports of each side.
_INPUT_RETURN_PORT_KEYS = ("input_return", "input_negative", "gnd", "ground")
_OUTPUT_RETURN_PORT_KEYS = ("output_common", "output_return", "gnd_out")


def _requirement_reference_domain(
    requirement, port: str, *, returns: tuple[str, ...] = ()
) -> str | None:
    """The declared reference domain of one side of a requirement.

    An explicit domain on the checked port wins; otherwise the side's own return
    port carries it; otherwise the design's single ground is the reference. A
    design with no separate return port therefore keeps the common-return
    behaviour, and an isolated conversion must declare both sides explicitly.
    """
    claim = getattr(requirement, "declared_interface", None)
    if claim is not None:
        for key in (port, *returns):
            matches = [
                row.reference_domain
                for row in claim.ports
                if row.key == key and row.reference_domain
            ]
            if len(matches) == 1:
                return matches[0]
    gnd = requirement.ports.get("gnd")
    return gnd if gnd else None


def _reviewed_transfer_edges(bom, info, nets) -> dict[str, set[str]]:
    """Direct energy-transfer edges, excluding C/R/control connectivity."""
    graph: dict[str, set[str]] = defaultdict(set)
    for part in bom.parts:
        # A series inductor is an explicitly conductive power element.  A
        # capacitor and a resistor (including a PD VDD feed) are never accepted
        # as a source-to-load transfer witness.
        if _ref_prefix(part.ref) == "L":
            pair = _two_terminal_part_nets(part, nets)
            if pair is not None:
                graph[pair[0]].add(pair[1])
                graph[pair[1]].add(pair[0])
        fact = _reviewed_fact_for_part(part)
        if fact is None:
            continue
        transfer = fact.get("power_transfer") or {}
        if not isinstance(transfer, dict):
            continue
        paths = transfer.get("paths")
        if not isinstance(paths, (list, tuple)):
            paths = (transfer,)
        for path in paths:
            if not isinstance(path, dict):
                continue
            source_name = str(path.get("from_pin") or "").upper() or _fact_pin_name(fact, "vin")
            dest_name = str(path.get("to_pin") or "").upper() or _fact_pin_name(fact, "ph")
            source_number = _pin_number_named(info, part.ref, source_name) if source_name else None
            dest_number = _pin_number_named(info, part.ref, dest_name) if dest_name else None
            source_net = (nets.get(part.ref) or {}).get(source_number or "")
            dest_net = (nets.get(part.ref) or {}).get(dest_number or "")
            if source_net and dest_net and source_net != dest_net:
                graph[source_net].add(dest_net)
                graph[dest_net].add(source_net)
    return graph


def _transfer_reaches(graph: dict[str, set[str]], source: str, load: str) -> bool:
    pending, seen = [source], {source}
    while pending:
        node = pending.pop()
        if node == load:
            return True
        for neighbor in graph.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return False


def _requirement_input_key(requirement, bom) -> str | None:
    """The requirement's input port key, under the reviewer's own spelling.

    Conventional keys first.  A reviewed device may name its input differently
    (`input_positive` for a return-referenced converter), so that published key is
    accepted when the requirement binds it.
    """
    ports = requirement.ports
    canonical = next((key for key in ("input", "vin", "vbus") if ports.get(key)), None)
    if canonical:
        return canonical
    published = {
        key
        for part in bom.parts
        if getattr(part, "sheet", None) == requirement.sheet
        if (fact := _reviewed_fact_for_part(part)) is not None
        for key in _INPUT_PORT_KEYS
        if key in (fact.get("pins") or fact.get("port_pins") or {})
    }
    return next(
        (key for key in _INPUT_PORT_KEYS if key in published and ports.get(key)),
        None,
    )


def check_reviewed_power_transfer(architecture, bom) -> CheckResult:
    """§9.39 — conversion obligations require a reviewed input-to-output path."""
    info, _ = _pin_info_by_ref(bom)
    graph = _reviewed_transfer_edges(bom, info, _nets_by_ref(bom))
    bad: list[str] = []
    for requirement in architecture.requirements:
        family = requirement.family.casefold()
        has_power_conversion = any(
            obligation.kind == "conversion"
            and str(getattr(obligation, "input_kind", "")).casefold() in {"power", "voltage"}
            and str(getattr(obligation, "output_kind", "")).casefold() in {"power", "voltage"}
            for obligation in requirement.obligations
        )
        is_reviewed_pd = family in {"usb-pd-fixed-trigger", "usb-pd-selectable-trigger"}
        if not (
            has_power_conversion
            or getattr(requirement, "role", None) == "regulator"
            or is_reviewed_pd
        ):
            continue
        if _claims_constant_current_led(architecture, requirement) and _constant_current_led_ports(
            architecture, requirement
        ):
            # §9.41 owns this loop. Its regulated LED node is reached through the
            # load's own series elements - the sense resistor, the return
            # inductor, and the LED itself - which are deliberately not generic
            # source-to-load transfer witnesses.
            continue
        input_key = _requirement_input_key(requirement, bom)
        input_net = requirement.ports.get(input_key) if input_key else None
        outputs = [
            (key, net)
            for key, net in requirement.ports.items()
            if net
            and key not in _OUTPUT_RETURN_PORT_KEYS
            and (
                key
                in {"output", "vout", "positive", "negative", "positive_output", "negative_output"}
                or "output" in key
            )
        ]
        if is_reviewed_pd:
            outputs.extend(
                (f"{peer.id}:{key}", net)
                for peer in architecture.requirements
                if peer is not requirement
                for key, net in peer.ports.items()
                if net and (_net_looks_power(net) or net == input_net)
            )
            # A PD controller negotiates a source; it does not create a second
            # supply.  Every separately declared VBUS/VOUT-like power rail must
            # therefore have conductor/transfer evidence from its raw VBUS.
            outputs.extend(
                ("declared_power_rail", net)
                for net in architecture.power_nets
                if net != input_net
                and not _net_is_ground(net)
                and re.search(r"vbus|vout|power", net, re.I)
            )
        if not input_net or not outputs:
            bad.append(
                f"E_POWER_TRANSFER {requirement.id!r}: power conversion/PD claim needs "
                "an actual input/vin/vbus binding and one or more explicit output bindings"
            )
            continue
        isolated_transfer = any(
            isinstance((fact := _reviewed_fact_for_part(part)), dict)
            and isinstance((transfer := fact.get("power_transfer")), dict)
            and (
                bool(transfer.get("isolated"))
                or any(
                    isinstance(path, dict) and path.get("isolated")
                    for path in (transfer.get("paths") or ())
                )
            )
            for part in bom.parts
            if part.sheet == requirement.sheet
        )
        input_domain = _requirement_reference_domain(
            requirement, input_key, returns=_INPUT_RETURN_PORT_KEYS
        )
        for output_key, output_net in outputs:
            output_domain = _requirement_reference_domain(
                requirement, output_key, returns=_OUTPUT_RETURN_PORT_KEYS
            )
            if isolated_transfer:
                if input_domain is None or output_domain is None or input_domain == output_domain:
                    bad.append(
                        f"E_REFERENCE_DOMAIN {requirement.id!r}: reviewed isolated "
                        "conversion requires distinct explicit input/output reference domains"
                    )
                    continue
            elif (
                input_domain is not None
                and output_domain is not None
                and input_domain != output_domain
            ):
                bad.append(
                    f"E_REFERENCE_DOMAIN {requirement.id!r}: input reference {input_domain!r} "
                    f"and output reference {output_domain!r} are distinct; a conversion cannot "
                    "claim a common return without a reviewed isolated-domain model"
                )
                continue
            if not _transfer_reaches(graph, input_net, output_net):
                bad.append(
                    f"E_POWER_TRANSFER {requirement.id!r}: no reviewed source-to-load transfer "
                    f"from {input_net!r} to {output_net!r}; capacitors, control pins, and "
                    "unreviewed placeholders are not power paths"
                )
    return CheckResult(
        "9.39 reviewed source-to-load power transfer",
        not bad,
        "every typed conversion has a reviewed source-to-load path"
        if not bad
        else f"{len(bad)} conversion path/domain contract(s) unproven",
        bad,
    )


def _quantitative_obligation(requirement, *names: str):
    wanted = tuple(name.lower() for name in names)
    matches = [
        obligation
        for obligation in requirement.obligations
        if obligation.kind == "quantitative"
        and any(name in obligation.quantity.lower() for name in wanted)
    ]
    return matches[0] if len(matches) == 1 else None


def _quantity_in(value, unit: str, scales: dict[str, float]) -> float | None:
    if value is None:
        return None
    scale = scales.get((unit or "").strip().lower())
    if scale is None:
        return None
    return value * scale


def check_typed_passive_crossover_values(architecture, bom) -> CheckResult:
    """§9.40 — prove typed first-order passive crossover branches.

    A low-pass inductor must directly join the declared input and low output.
    A high-pass branch, when declared, may use direct parallel capacitors or a
    two-capacitor series chain between its declared endpoints.
    """
    bad: list[str] = []
    for requirement in architecture.requirements:
        if "crossover" not in requirement.family:
            continue
        frequency = _quantitative_obligation(requirement, "cutoff", "frequency")
        impedance = _quantitative_obligation(requirement, "impedance", "load")
        cutoff_hz = (
            _quantity_in(frequency.value, frequency.unit, {"hz": 1.0, "khz": 1e3})
            if frequency is not None and frequency.relation == "equal"
            else None
        )
        load_ohm = (
            _quantity_in(impedance.value, impedance.unit, {"ohm": 1.0, "ω": 1.0, "Ω": 1.0})
            if impedance is not None and impedance.relation == "equal"
            else None
        )
        if cutoff_hz is None or load_ohm is None or cutoff_hz <= 0 or load_ohm <= 0:
            bad.append(
                f"E_PASSIVE_CROSSOVER {requirement.id!r}: require typed equal cutoff "
                "frequency (Hz/kHz) and load impedance (ohm) obligations"
            )
            continue
        input_net = requirement.ports.get("input")
        low_net = (
            requirement.ports.get("low")
            or requirement.ports.get("low_out")
            or requirement.ports.get("lowpass_output")
            or requirement.ports.get("woofer")
        )
        if not input_net or not low_net:
            bad.append(
                f"E_PASSIVE_CROSSOVER {requirement.id!r}: supported low-pass topology "
                "needs explicit input and low/low_out/lowpass_output/woofer ports"
            )
            continue
        candidates = [
            part
            for part in bom.parts
            if part.sheet == requirement.sheet
            and _ref_prefix(part.ref) == "L"
            and (_two_terminal_part_nets(part, _nets_by_ref(bom)) is not None)
            and set(_two_terminal_part_nets(part, _nets_by_ref(bom)) or ()) == {input_net, low_net}
        ]
        expected = load_ohm / (2.0 * 3.141592653589793 * cutoff_hz)
        values = [(part, _inductance_henries(part.value)) for part in candidates]
        if not any(
            inductance is not None and abs(inductance - expected) <= expected * 0.20
            for _, inductance in values
        ):
            rendered = ", ".join(f"{part.ref}={part.value!r}" for part, _ in values) or "none"
            bad.append(
                f"E_PASSIVE_CROSSOVER {requirement.id!r}: low-pass L between "
                f"{input_net!r}/{low_net!r} is {rendered}; {load_ohm:g}ohm at "
                f"{cutoff_hz:g}Hz requires {expected * 1e6:.0f}uH"
            )
        high_net = (
            requirement.ports.get("high")
            or requirement.ports.get("high_out")
            or requirement.ports.get("highpass_output")
            or requirement.ports.get("tweeter")
        )
        if high_net:
            capacitors = [
                (part, pair, _capacitance_farads(part.value))
                for part in bom.parts
                if part.sheet == requirement.sheet
                and _ref_prefix(part.ref) == "C"
                and (pair := _two_terminal_part_nets(part, _nets_by_ref(bom))) is not None
            ]
            expected_cap = 1.0 / (2.0 * 3.141592653589793 * load_ohm * cutoff_hz)
            direct_parallel = [
                (part, value)
                for part, pair, value in capacitors
                if value is not None and set(pair) == {input_net, high_net}
            ]
            equivalent_caps = (
                [
                    (
                        tuple(part for part, _ in direct_parallel),
                        sum(value for _, value in direct_parallel),
                    )
                ]
                if direct_parallel
                else []
            )
            for first, first_pair, first_value in capacitors:
                if first_value is None or input_net not in first_pair:
                    continue
                middle = first_pair[1] if first_pair[0] == input_net else first_pair[0]
                for second, second_pair, second_value in capacitors:
                    if second is first or second_value is None or middle not in second_pair:
                        continue
                    other = second_pair[1] if second_pair[0] == middle else second_pair[0]
                    if other == high_net:
                        equivalent_caps.append(
                            ((first, second), 1.0 / (1.0 / first_value + 1.0 / second_value))
                        )
            if not any(
                abs(equivalent - expected_cap) <= expected_cap * 0.20
                for _, equivalent in equivalent_caps
            ):
                rendered = (
                    ", ".join(
                        "+".join(f"{part.ref}={part.value!r}" for part in parts)
                        for parts, _ in equivalent_caps
                    )
                    or "none"
                )
                bad.append(
                    f"E_PASSIVE_CROSSOVER {requirement.id!r}: high-pass capacitors "
                    f"from {input_net!r} to {high_net!r} are {rendered}; {load_ohm:g}ohm "
                    f"at {cutoff_hz:g}Hz requires {expected_cap * 1e6:.3g}uF equivalent"
                )
    return CheckResult(
        "9.40 typed passive crossover values",
        not bad,
        "typed passive crossover low-pass values match their topology"
        if not bad
        else f"{len(bad)} passive crossover value/topology violation(s)",
        bad,
    )


def _reviewed_high_side_led_loop_contract(fact: dict) -> dict | None:
    """Normalize the complete source-backed high-side LED-current loop contract.

    This is deliberately a narrow, fail-closed bridge from reviewed part
    metadata to both schematic and artifact checks.  A controller which merely
    advertises a sense resistor is not a loop model.
    """
    feedback = fact.get("current_feedback")
    transfer = fact.get("power_transfer")
    support = fact.get("support_network")
    limits = fact.get("operating_limits")
    if not all(isinstance(item, dict) for item in (feedback, transfer, support, limits)):
        return None
    if feedback.get("topology") != "high_side_sense_low_side_switch":
        return None
    sense_pin = _fact_pin_name(feedback, "sense")
    reference_pin = _fact_pin_name(feedback, "reference")
    switch_pin = _fact_pin_name(fact, "switch")
    ground_pin = _fact_pin_name(fact, "ground")
    sense_voltage = _fact_number(feedback, "sense_voltage_v")
    tolerance = _fact_number(feedback, "sense_tolerance")
    continuous_current = _fact_number(limits, "continuous_output_a", "continuous_current_a")
    decoupling = support.get("input_decoupling")
    catch_diode = support.get("catch_diode")
    if not isinstance(decoupling, dict) or not isinstance(catch_diode, dict):
        return None
    decoupling_positive = str(decoupling.get("positive_pin") or "").upper() or None
    decoupling_negative = str(decoupling.get("negative_pin") or "").upper() or None
    decoupling_uf = _fact_number(decoupling, "capacitance_min_uf")
    diode_anode = str(catch_diode.get("anode_pin") or "").upper() or None
    diode_cathode = str(catch_diode.get("cathode_pin") or "").upper() or None
    if (
        not all(
            (
                sense_pin,
                reference_pin,
                switch_pin,
                ground_pin,
                decoupling_positive,
                decoupling_negative,
                diode_anode,
                diode_cathode,
            )
        )
        or sense_voltage is None
        or sense_voltage <= 0
        or tolerance is None
        or not 0 < tolerance < 1
        or continuous_current is None
        or continuous_current <= 0
        or decoupling_uf is None
        or decoupling_uf <= 0
        or str(transfer.get("from_pin") or "").upper() != switch_pin
        or str(transfer.get("to_pin") or "").upper() != ground_pin
        or decoupling_positive != reference_pin
        or decoupling_negative != ground_pin
        or diode_anode != switch_pin
        or diode_cathode != reference_pin
    ):
        return None
    return {
        "topology": feedback["topology"],
        "sense_pin": sense_pin,
        "reference_pin": reference_pin,
        "switch_pin": switch_pin,
        "ground_pin": ground_pin,
        "sense_voltage_v": sense_voltage,
        "sense_tolerance": tolerance,
        "continuous_current_a": continuous_current,
        "input_decoupling": {
            "positive_pin": decoupling_positive,
            "negative_pin": decoupling_negative,
            "capacitance_min_f": decoupling_uf * 1e-6,
        },
        "catch_diode": {"anode_pin": diode_anode, "cathode_pin": diode_cathode},
    }


def _pins_named(info: dict, ref: str, name: str) -> list[str]:
    """Return every physical pin with one reviewed logical name."""
    wanted = name.upper()
    return [
        number
        for number, data in (info.get(ref) or {}).items()
        if str(data.get("name") or "").upper() == wanted
    ]


def _constant_current_led_ports(architecture, requirement) -> dict[str, str] | None:
    """Bind the driver and explicit LED-output requirement ports, never a symbol."""
    input_net = requirement.ports.get("input") or requirement.ports.get("vin")
    gnd_net = requirement.ports.get("gnd") or requirement.ports.get("ground")
    anode_net = requirement.ports.get("led_anode") or requirement.ports.get("set")
    cathode_net = requirement.ports.get("led_cathode")
    if cathode_net is None and anode_net is not None:
        outputs = [
            peer
            for peer in architecture.requirements
            if peer is not requirement
            and peer.sheet == requirement.sheet
            and str(getattr(peer, "role", "")).casefold() == "connector"
            and peer.ports.get("positive") == anode_net
            and peer.ports.get("negative")
        ]
        if len(outputs) == 1:
            cathode_net = outputs[0].ports["negative"]
    if not all((input_net, gnd_net, anode_net, cathode_net)):
        return None
    return {
        "input": input_net,
        "gnd": gnd_net,
        "led_anode": anode_net,
        "led_cathode": cathode_net,
    }


def _has_reviewed_local_led(part) -> bool:
    """Only exact physical LED evidence may make an on-board load an LED."""
    record = _reviewed_identity_for_bom_part(part)
    return record is not None and any(
        "led" in feature.casefold() for feature in getattr(record, "physical_features", ())
    )


def _reviewed_constant_current_led_loop_errors(
    requirement, bom, info, nets, *, ports: dict[str, str] | None = None
) -> list[str]:
    """Return all unproven predicates for one typed LED-current requirement."""
    target = _quantitative_obligation(requirement, "current")
    target_a = (
        _quantity_in(target.value, target.unit, {"a": 1.0, "ma": 1e-3})
        if target is not None and target.relation == "equal"
        else None
    )
    has_conversion = any(obligation.kind == "conversion" for obligation in requirement.obligations)
    if ports is None:
        from types import SimpleNamespace

        ports = _constant_current_led_ports(
            SimpleNamespace(requirements=[requirement]), requirement
        )
    if target_a is None or target_a <= 0 or not has_conversion or ports is None:
        needs = []
        if target_a is None or target_a <= 0:
            needs.append("a positive equal current obligation in A or mA")
        if not has_conversion:
            needs.append("a conversion obligation")
        if ports is None:
            needs.append("explicit input/gnd/led_anode/led_cathode requirement ports")
        return [f"E_LED_CURRENT_FEEDBACK {requirement.id!r}: needs {', '.join(needs)}"]
    supported = [
        (part, fact)
        for part in bom.parts
        if part.sheet == requirement.sheet
        if isinstance((fact := _reviewed_fact_for_part(part)), dict)
        if isinstance(fact.get("current_feedback"), dict)
        # Every reviewed record carries a (possibly empty) `current_feedback`
        # mapping; only one that declares its sense element models the loop. A
        # screw terminal or holder is not a current-feedback controller.
        and fact["current_feedback"].get("sense_pin")
    ]
    if len(supported) != 1:
        return [
            f"E_LED_CURRENT_FEEDBACK {requirement.id!r}: typed {target_a:g}A "
            "constant-current delivery needs exactly one reviewed controller feedback model"
        ]
    controller, fact = supported[0]
    contract = _reviewed_high_side_led_loop_contract(fact)
    if contract is None:
        return [
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: reviewed feedback metadata must "
            "completely define the high_side_sense_low_side_switch loop"
        ]
    bad: list[str] = []
    if target_a > contract["continuous_current_a"]:
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: requested {target_a:g}A exceeds "
            f"reviewed continuous output {contract['continuous_current_a']:g}A"
        )
    if len({ports["input"], ports["gnd"], ports["led_anode"], ports["led_cathode"]}) != 4:
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {requirement.id!r}: input, ground, and distinct "
            "LED anode/cathode requirement ports must be different nets"
        )
        return bad
    pin_nets = nets.get(controller.ref) or {}
    for label, pin_name, expected_net in (
        ("sense", contract["sense_pin"], ports["led_anode"]),
        ("source", contract["reference_pin"], ports["input"]),
    ):
        pins = _pins_named(info, controller.ref, pin_name)
        if len(pins) != 1 or pin_nets.get(pins[0]) != expected_net:
            bad.append(
                f"E_LED_CURRENT_FEEDBACK {controller.ref}: reviewed {label} pin {pin_name} "
                f"must be wired to {expected_net!r}"
            )
    switch_pins = _pins_named(info, controller.ref, contract["switch_pin"])
    ground_pins = _pins_named(info, controller.ref, contract["ground_pin"]) + _pins_named(
        info, controller.ref, "EP"
    )
    if not switch_pins:
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: reviewed switch pin inventory is absent"
        )
    switch_net = pin_nets.get(switch_pins[0]) if switch_pins else None
    if switch_pins and (
        not switch_net or any(pin_nets.get(pin) != switch_net for pin in switch_pins)
    ):
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: all parallel {contract['switch_pin']} "
            "pins must share one wired switch net"
        )
    if not ground_pins or any(pin_nets.get(pin) != ports["gnd"] for pin in ground_pins):
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: all reviewed GND/EP return pins "
            f"must be wired to {ports['gnd']!r}"
        )
    expected_r = contract["sense_voltage_v"] / target_a
    sense_resistors = [
        part
        for part in bom.parts
        if _ref_prefix(part.ref) == "R"
        and _two_terminal_part_nets(part, nets) is not None
        and set(_two_terminal_part_nets(part, nets) or ()) == {ports["led_anode"], ports["input"]}
    ]
    if not any(
        (value := _resistance_ohms(part.value)) is not None
        and abs(value - expected_r) <= expected_r * contract["sense_tolerance"]
        for part in sense_resistors
    ):
        values = ", ".join(f"{part.ref}={part.value!r}" for part in sense_resistors) or "none"
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: direct {contract['sense_pin']}/"
            f"{contract['reference_pin']} sense resistor is {values}; {target_a:g}A "
            f"requires {expected_r:g}ohm"
        )
    inductors = [
        part
        for part in bom.parts
        if _ref_prefix(part.ref) == "L"
        and _two_terminal_part_nets(part, nets) is not None
        and set(_two_terminal_part_nets(part, nets) or ()) == {ports["led_cathode"], switch_net}
    ]
    if not any(
        (value := _inductance_henries(part.value)) is not None and value > 0 for part in inductors
    ):
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {requirement.id!r}: LED cathode {ports['led_cathode']!r} "
            f"needs a positive-valued return inductor to reviewed switch net {switch_net!r}"
        )
    if not any(
        _ref_prefix(part.ref) == "D"
        and (anode := _pin_number_named(info, part.ref, "A")) is not None
        and (cathode := _pin_number_named(info, part.ref, "K")) is not None
        and (part_nets := nets.get(part.ref) or {}).get(anode) == switch_net
        and part_nets.get(cathode) == ports["input"]
        for part in bom.parts
    ):
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: reviewed catch diode must be anode "
            f"at {switch_net!r} and cathode at {ports['input']!r}"
        )
    decoupling = contract["input_decoupling"]
    if not any(
        _ref_prefix(part.ref) == "C"
        and (pair := _two_terminal_part_nets(part, nets)) is not None
        and set(pair) == {ports["input"], ports["gnd"]}
        and (value := _capacitance_farads(part.value)) is not None
        and value >= decoupling["capacitance_min_f"]
        for part in bom.parts
    ):
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {controller.ref}: VIN/GND input decoupling needs at "
            f"least {decoupling['capacitance_min_f'] * 1e6:g}uF"
        )
    local_loads = [
        part
        for part in bom.parts
        if part.sheet == requirement.sheet
        and (anode := _pin_number_named(info, part.ref, "A")) is not None
        and (cathode := _pin_number_named(info, part.ref, "K")) is not None
        and (part_nets := nets.get(part.ref) or {}).get(anode) == ports["led_anode"]
        and part_nets.get(cathode) == ports["led_cathode"]
    ]
    if local_loads and not all(_has_reviewed_local_led(part) for part in local_loads):
        refs = ", ".join(part.ref for part in local_loads)
        bad.append(
            f"E_LED_CURRENT_FEEDBACK {requirement.id!r}: {refs} bridge the requested "
            "LED terminals but lack exact reviewed LED physical identity"
        )
    return bad


def _claims_constant_current_led(architecture, requirement) -> bool:
    """Whether this requirement claims the reviewed constant-current LED loop.

    A named constant-current family is the claim itself.  Mild sheet prose
    ("LED driver, constant current") only counts when the requirement also states
    a typed current or conversion obligation, so a peer connector on the same
    sheet is never read as a driver.
    """
    if "constant-current" in requirement.family:
        return True
    topology = (getattr(architecture, "topologies", None) or {}).get(requirement.sheet, "")
    if "constant" not in topology.lower() or "current" not in topology.lower():
        return False
    return any(
        obligation.kind == "conversion"
        or (obligation.kind == "quantitative" and "current" in obligation.quantity.lower())
        for obligation in requirement.obligations
    )


def check_reviewed_constant_current_led_feedback(architecture, bom) -> CheckResult:
    """§9.41 — prove the complete reviewed constant-current LED power loop."""
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
    for requirement in architecture.requirements:
        if not _claims_constant_current_led(architecture, requirement):
            continue
        # Resolve external LED terminals from their explicit peer requirement
        # before invoking the reusable electrical predicate.
        ports = _constant_current_led_ports(architecture, requirement)
        if ports is None:
            bad.append(
                f"E_LED_CURRENT_FEEDBACK {requirement.id!r}: needs explicit input/gnd/"
                "led_anode/led_cathode requirement ports"
            )
            continue
        bad.extend(
            _reviewed_constant_current_led_loop_errors(requirement, bom, info, nets, ports=ports)
        )
    return CheckResult(
        "9.41 reviewed constant-current LED feedback",
        not bad,
        "typed constant-current LED power loops are completely realized"
        if not bad
        else f"{len(bad)} constant-current LED feedback contract(s) unproven",
        bad,
    )


def _reviewed_identity_for_bom_part(part):
    """Resolve physical evidence through the canonical fail-closed inventory."""
    from kicraft.design.part_identity import physical_inventory_record

    return physical_inventory_record(
        mpn=getattr(part, "mpn", None),
        symbol=getattr(part, "symbol", None),
        footprint=getattr(part, "footprint", None),
        datasheet=getattr(part, "datasheet", None),
        sourcing_note=getattr(part, "sourcing_note", None),
    )


def _declared_pin_selector(port) -> str | None:
    """The explicit pin selector a declared-interface port carries, else None."""
    return (
        getattr(port, "pin_selector", None)
        or getattr(port, "pin", None)
        or getattr(port, "pin_name", None)
    )


def _declared_port_pin(info, ref: str, port) -> str | None:
    """Resolve an explicit declared-interface selector, never port-name guessing."""
    selector = _declared_pin_selector(port)
    if selector is None:
        return None
    selector = str(selector)
    pins = info.get(ref) or {}
    if selector in pins:
        return selector
    hits = [number for number, data in pins.items() if data["name"] == selector]
    return hits[0] if len(hits) == 1 else None


def _part_is_deterministic_owned(bom, ref: str) -> bool:
    """Whether a recipe/lowerer expansion, not the model's group, owns ``ref``.

    Only those expansions contribute ``bom.connections`` (see
    ``_normalize_bom_stage_response``), so a declared interface can be proven
    from a pin/net comparison exactly when its implementing component is owned
    that way; the model-owned remainder is provable only once the wiring stage
    commits its own connections.
    """
    for manifest in bom.recipe_ownership or []:
        if ref in (getattr(manifest, "refs", None) or ()):
            return True
    for part in bom.parts:
        if part.ref == ref:
            return bool(getattr(part, "recipe_id", None)) or str(
                getattr(part, "resolution_source", "") or ""
            ) in {"recipe", "lowerer"}
    return False


def _part_names_exact_part(part, reviewed_record, exact_part: str | None) -> bool:
    """Whether a BOM part is the exact identity a requirement pinned."""
    if exact_part is None:
        return True
    if reviewed_record is not None:
        return reviewed_record.identity == exact_part.casefold()
    return str(getattr(part, "mpn", "") or "").casefold() == exact_part.casefold()


def _physical_requirement_parts(requirement, bom, reviewed, recipe_refs: set[str]):
    """Parts that can satisfy one requirement's physical obligations.

    An exact_part pins the requirement's principal component, not every
    supporting component a reviewed recipe emits.  Recipe ownership is the
    narrow exception: only refs emitted by the committed recipe selection for
    this requirement can supply its crystal, flash, connector, or other
    support-part obligation.  Model parts remain bound to exact_part evidence.
    """
    return [
        part
        for part in bom.parts
        if part.sheet == requirement.sheet
        and (
            part.ref in recipe_refs
            or _part_names_exact_part(part, reviewed.get(part.ref), requirement.exact_part)
        )
    ]


def _has_trusted_lowerer_topology_witness(requirement, bom, component_class: str) -> bool:
    """Prove one complete topology from its canonical lowerer-owned BOM parts."""
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.part_identity import lowerer_witnesses_physical_class

    owned = [
        part
        for part in bom.parts
        if getattr(part, "resolution_source", None) == "lowerer"
        and getattr(part, "lowering_requirement_id", None) == requirement.id
    ]
    lowerer_ids = {str(getattr(part, "resolution_id", "") or "") for part in owned}
    if len(lowerer_ids) != 1:
        return False
    (lowerer_id,) = lowerer_ids
    if not lowerer_witnesses_physical_class(lowerer_id, component_class):
        return False
    artifact = lower_requirement(requirement)
    if artifact is None or artifact.lowerer_id != lowerer_id:
        return False
    expected = {
        (group.role, index): group for group in artifact.groups for index in range(group.quantity)
    }
    seen: dict[tuple[str, int], object] = {}
    for part in owned:
        role = getattr(part, "lowering_role", None)
        index = getattr(part, "lowering_index", None)
        key = (role, index)
        group = expected.get(key)
        prefix = re.match(r"[A-Z]+", str(getattr(part, "ref", "") or ""))
        if (
            group is None
            or not isinstance(index, int)
            or prefix is None
            or key in seen
            or group.reference_prefix != prefix.group()
            or group.value != getattr(part, "value", None)
            or group.symbol != getattr(part, "symbol", None)
            or group.footprint != getattr(part, "footprint", None)
            or group.mpn != getattr(part, "mpn", None)
        ):
            return False
        seen[key] = part
    return set(seen) == set(expected)


def _part_implements_physical_class(part, reviewed_record, component_class: str) -> bool:
    """Whether one BOM part implements a demanded physical class.

    A reviewed record answers from its own family and reviewed features. A demanded class
    with no reviewed coverage anywhere falls back to real-part evidence — an exact MPN, a
    symbol whose pin inventory resolves, and a footprint (part_identity.
    resolved_part_evidence). The library cannot answer for a class it has never covered,
    and refusing such a demand would block exactly the new designs the pipeline exists to
    build; the part must still be real and resolvable, never a family label or a bare value.
    """
    from kicraft.design.part_identity import has_reviewed_coverage, resolved_part_evidence

    if not has_reviewed_coverage(component_class):
        return resolved_part_evidence(
            mpn=getattr(part, "mpn", None),
            symbol=getattr(part, "symbol", None),
            footprint=getattr(part, "footprint", None),
        )
    if reviewed_record is None:
        return False
    canonical = canonical_physical_features(component_class)
    return component_class == reviewed_record.family or bool(
        canonical & reviewed_record.physical_features
    )


def check_requirement_physical_realization(
    architecture, bom, *, declared_interface_scope: str = "all"
) -> CheckResult:
    """§9.42 — physical obligations and declared interfaces must be real BOM pins.

    A BOM group label or family-like value has no fulfillment authority. For a class the
    reviewed library covers, only an exact reviewed identity with its reviewed
    symbol/footprint pair can satisfy it; a class with no reviewed coverage at all is
    satisfied by a real resolved part instead (see
    :func:`_part_implements_physical_class`). The declared-interface half always needs an
    identity-matched component whose pin inventory resolves.

    ``declared_interface_scope`` splits the declared-interface half across the
    two stages that can prove it, because only recipe/lowerer expansions create
    BOM-stage connections: ``"recipe_owned"`` at BOM commit, ``"model_owned"``
    at wiring commit, ``"all"`` for build-time validation where both graphs are
    complete. A declared interface is never silently unchecked: each stage
    evaluates the half it can prove, with the same offenders.
    """
    from kicraft.design.part_identity import has_reviewed_coverage

    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    reviewed = {
        part.ref: record
        for part in bom.parts
        if (record := _reviewed_identity_for_bom_part(part)) is not None
    }
    recipe_refs_by_requirement: dict[str, set[str]] = defaultdict(set)
    for manifest in getattr(bom, "recipe_ownership", ()) or ():
        refs = {str(ref) for ref in (getattr(manifest, "refs", ()) or ())}
        for requirement_id in getattr(manifest, "requirement_ids", ()) or ():
            recipe_refs_by_requirement[str(requirement_id)].update(refs)
    requirements_by_id = {requirement.id: requirement for requirement in architecture.requirements}

    bad: list[str] = []
    aggregate_demands: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for requirement in architecture.requirements:
        requirement_parts = _physical_requirement_parts(
            requirement,
            bom,
            reviewed,
            recipe_refs_by_requirement.get(requirement.id, set()),
        )
        physical_classes = {
            obligation.component_class.casefold()
            for obligation in requirement.obligations
            if obligation.kind == "physical"
        }
        local_demands = {component_class: 1 for component_class in physical_classes}
        for obligation in requirement.obligations:
            if obligation.kind != "quantity":
                continue
            component_class = obligation.subject.casefold()
            if component_class in physical_classes:
                local_demands[component_class] = max(
                    local_demands[component_class], obligation.minimum
                )
        for component_class, minimum in local_demands.items():
            matching = (
                [requirement]
                if _has_trusted_lowerer_topology_witness(requirement, bom, component_class)
                else [
                    part
                    for part in requirement_parts
                    if _part_implements_physical_class(
                        part, reviewed.get(part.ref), component_class
                    )
                ]
            )
            if len(matching) < minimum:
                # "reviewed" only where the library could answer; a class it has never
                # covered is proven by a real resolved part instead.
                evidence = "reviewed" if has_reviewed_coverage(component_class) else "real"
                bad.append(
                    f"E_PHYSICAL_REALIZATION {requirement.id!r}: requires {minimum} {evidence} "
                    f"{component_class!r} physical part(s), found {len(matching)} with exact "
                    "MPN/symbol/footprint evidence"
                )
            aggregate_demands[(requirement.sheet, component_class)].append(
                (requirement.id, minimum)
            )
        claim = requirement.declared_interface
        if claim is None or not bom.connections:
            continue  # BOM commit proves identity; wiring owns pin/net evidence.
        interface_parts = [
            part
            for part in bom.parts
            if part.sheet == requirement.sheet
            and part.ref in info
            and bool(getattr(part, "mpn", None))
            and bool(part.symbol)
            and bool(part.footprint)
            and (
                requirement.exact_part is None
                or str(part.mpn).casefold() == requirement.exact_part.casefold()
            )
        ]
        deterministic_owned = any(
            _part_is_deterministic_owned(bom, part.ref) for part in interface_parts
        )
        if declared_interface_scope == "recipe_owned" and not deterministic_owned:
            continue  # the wiring stage proves the model-owned half
        if declared_interface_scope == "model_owned" and deterministic_owned:
            continue  # the BOM stage already proved the expansion-owned half
        # One owning hardware family per declared interface.  A bank of
        # identical instances (two USB-A power outputs behind two identical load
        # switches) is a single owned family: every instance carries the same
        # identity, so the claimed port-to-pin map has to hold on one of its
        # instances.  Claiming one interface with several different
        # symbol/footprint/mpn triples stays refused, as does an interface with
        # no matching instance at all.
        owners = {
            (part.symbol, part.footprint, (part.mpn or "").casefold()) for part in interface_parts
        }
        if not interface_parts or len(owners) != 1:
            bad.append(
                f"E_DECLARED_INTERFACE {requirement.id!r}: needs exactly one "
                "identity-matched BOM component with resolved pin inventory"
            )
            continue
        for port in claim.ports:
            expected_net = requirement.ports.get(port.key)
            if expected_net is None:
                # A declared port with no bound net is a pin-inventory claim (a
                # converter's switch/boot/feedback pin), not a wiring claim: the
                # design never bound that port to a top-level net, so a
                # net-graph comparison has nothing to compare against. Only the
                # ports the requirement actually binds are checked.
                continue
            found = {
                part.ref: (nets.get(part.ref) or {}).get(
                    _declared_port_pin(info, part.ref, port) or ""
                )
                for part in interface_parts
            }
            if expected_net in found.values():
                continue
            if len(found) == 1:
                (part_ref, actual_net) = next(iter(found.items()))
                bad.append(
                    f"E_DECLARED_INTERFACE {requirement.id!r}.{port.key}: expected "
                    f"{expected_net!r} on declared pin selector {_declared_pin_selector(port)!r} "
                    f"of {part_ref}, found {actual_net!r}"
                )
            else:
                bad.append(
                    f"E_DECLARED_INTERFACE {requirement.id!r}.{port.key}: expected "
                    f"{expected_net!r} on declared pin selector {_declared_pin_selector(port)!r} "
                    f"of one of {sorted(found)}, found {found}"
                )
    for (sheet, component_class), demand_rows in sorted(aggregate_demands.items()):
        demanded = sum(minimum for _, minimum in demand_rows)
        candidate_refs: set[str] = set()
        topology_witnesses = 0
        for requirement_id, _minimum in demand_rows:
            requirement = requirements_by_id.get(requirement_id)
            if requirement is None:
                continue
            if _has_trusted_lowerer_topology_witness(requirement, bom, component_class):
                topology_witnesses += 1
                continue
            candidate_refs.update(
                part.ref
                for part in _physical_requirement_parts(
                    requirement,
                    bom,
                    reviewed,
                    recipe_refs_by_requirement.get(requirement_id, set()),
                )
                if _part_implements_physical_class(part, reviewed.get(part.ref), component_class)
            )
        available = len(candidate_refs) + topology_witnesses
        if available < demanded:
            owners = ", ".join(
                f"{requirement_id}×{minimum}" for requirement_id, minimum in demand_rows
            )
            evidence = "reviewed" if has_reviewed_coverage(component_class) else "real"
            bad.append(
                f"E_PHYSICAL_REALIZATION {sheet!r}/{component_class!r}: {owners} demand "
                f"{demanded} distinct part(s), but only {available} exact {evidence} "
                "MPN/symbol/footprint realization(s) exist"
            )
    return CheckResult(
        "9.42 requirement physical/interface realization",
        not bad,
        "physical obligations and declared interfaces have exact realized evidence"
        if not bad
        else f"{len(bad)} physical/interface realization contract(s) unproven",
        bad,
    )


# ---------- §9.21 MCU first-flash / programming path (advisory) ----------
#
# A programmable MCU with no way to enter its bootloader/debug interface is the
# `unprogrammable_mcu` defect -- a true positive in every self-eval case: an
# ESP32 with IO0/GPIO0 hard-tied to +3V3 (cannot be pulled LOW into download
# mode), an RP2040 with no BOOTSEL button AND SWD left no-connect. The netlist is
# ERC/DRC clean, so only a role-aware check sees it, and it is immune to the model
# nondeterministically deleting the boot-strap resistors between runs.
#
# This is ADVISORY: cli_app surfaces a failure as a wiring open_question (a
# fab-readiness caveat), NEVER as a hard synthesis-check failure -- the per-family
# heuristic is med-high confidence and a board can be flashed by other means
# (pogo pins, pre-programmed parts). It fails OPEN (no flag) whenever the pinout
# is unresolvable or a programming affordance plausibly exists, so a sound design
# never trips it.

_ESP_FAMILY_RE = re.compile(r"esp32|esp8266|esp32c|esp32s", re.I)
_RP2040_FAMILY_RE = re.compile(r"rp2040", re.I)
_GENERIC_MCU_RE = re.compile(
    r"stm32|atmega|attiny|atsam|samd\d|samc\d|samr\d|nrf52|nrf51|nrf53|"
    r"gd32|msp430|efm32|max32|apollo\d|hc32|ch32|py32",
    re.I,
)
_BOOT0_PIN_RE = re.compile(r"^(IO0|GPIO0|BOOT0?)$", re.I)
_SWD_PIN_RE = re.compile(r"SWCLK|SWDIO|^SWD$|^TCK$|^TMS$|^TDI$|^TDO$|JTAG", re.I)
_UPDI_PIN_RE = re.compile(r"UPDI", re.I)
# A part whose symbol/value/sourcing_note names a programming interface --
# the physical access point a programmer clips or plugs onto.
_PROG_ACCESS_PART_RE = re.compile(
    r"updi|swd|swdio|jtag|icsp|\bisp\b|debug|prog|tag-?connect|tc2030"
    r"|test[ _-]?(point|pad)",
    re.I,
)
_USB_PART_RE = re.compile(r"usb", re.I)


def _ref_prefix(ref: str) -> str:
    m = re.match(r"[A-Za-z]+", ref or "")
    return m.group(0).upper() if m else ""


def _esp_boot_problem(pins, wired):
    """ESP32/8266: the IO0/GPIO0 download-mode strap must be drivable LOW. A bare
    hard-tie to a positive rail is the documented unprogrammable case."""
    boot = [num for num, p in pins.items() if _BOOT0_PIN_RE.search(p["name"])]
    if not boot:
        return None  # pinout doesn't expose IO0 -> can't judge (fail open)
    for num in boot:
        net = wired.get(num)
        if net is None:
            continue  # NC handled by the family default; an unwired strap is rare
        if _net_is_positive_rail(net):
            return (
                f"IO0/GPIO0 (download-mode strap) is hard-tied to rail {net!r}; it "
                "cannot be pulled LOW to enter the ROM bootloader (needs a boot "
                "button/strap to GND)"
            )
    return None


def _rp2040_boot_problem(pins, wired, nc, ref, bom):
    """Prove an RP2040 SWD or BOOTSEL path from the committed net graph."""
    if not bom.connections:
        return (
            "no programming graph is committed: SWDIO/SWCLK must reach one "
            "external interface or BOOTSEL must switch QSPI_CS to GND"
        )
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    access = _programming_access_parts(bom)
    access_refs = {
        part.ref
        for part in access
        if _PROG_ACCESS_PART_RE.search(f"{part.symbol} {part.value} {part.sourcing_note or ''}")
    }
    mcu_pins = info.get(ref, {})
    swdio = [
        number for number, pin in mcu_pins.items() if re.search(r"SWDIO|TMS", pin["name"], re.I)
    ]
    swclk = [
        number for number, pin in mcu_pins.items() if re.search(r"SWCLK|TCK", pin["name"], re.I)
    ]
    swdio_nets = {wired.get(number) for number in swdio if wired.get(number)}
    swclk_nets = {wired.get(number) for number in swclk if wired.get(number)}
    for access_ref in access_refs:
        access_nets = set(nets.get(access_ref, {}).values())
        has_signal_pair = bool(access_nets & swdio_nets) and bool(access_nets & swclk_nets)
        has_ground = any(_net_looks_ground(net) for net in access_nets)
        has_vtref = any(
            _net_is_positive_rail(net) or re.search(r"vtref|vref", net, re.I) for net in access_nets
        )
        if has_signal_pair and has_ground and has_vtref:
            return None

    cs_nets = {
        net
        for part in bom.parts
        if _RP2040_FAMILY_RE.search(f"{part.symbol} {part.value}")
        or re.search(r"qspi|flash|w25q", f"{part.symbol} {part.value}", re.I)
        for number, pin in info.get(part.ref, {}).items()
        if re.search(
            r"(?:qspi[_-]?)?(?:ss|cs)(?:_n)?$",
            re.sub(r"[^A-Za-z0-9]+", "_", pin["name"]).strip("_"),
            re.I,
        )
        for net in [nets.get(part.ref, {}).get(number)]
        if net
    }
    for part in bom.parts:
        if _ref_prefix(part.ref) not in {"SW", "S", "JP"}:
            continue
        switch_nets = set(nets.get(part.ref, {}).values())
        if switch_nets & cs_nets and any(_net_looks_ground(net) for net in switch_nets):
            return None
    return (
        "no programming path: SWDIO and SWCLK do not reach the same external "
        "interface with GND/VTref, and no BOOTSEL switch connects QSPI_CS to GND"
    )


def _generic_mcu_problem(pins, wired, nc, ref):
    """Other MCUs: flag only the unambiguous case -- the part exposes a SWD/JTAG
    debug interface and EVERY one of those pins is left unconnected (no debug
    header). Conservative: an MCU with no recognizable debug pins is not judged."""
    swd = [num for num, p in pins.items() if _SWD_PIN_RE.search(p["name"])]
    if not swd:
        return None
    if any(wired.get(num) is not None and (ref, num) not in nc for num in swd):
        return None
    return (
        "the SWD/JTAG debug interface is left unconnected and no programming "
        "header is provided -- the MCU cannot be flashed"
    )


def check_mcu_programming_path(bom) -> CheckResult:
    """§9.21 (advisory) -- assert a reachable first-flash path for each MCU.

    See the section comment. Returns offenders as ``"<ref> (<part>): <reason>"``;
    the caller turns each into a wiring open_question. Never raises; unresolvable
    pinouts are skipped.
    """
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    nc = {(ep.ref, ep.pin) for ep in (bom.no_connect_pins or [])}
    bad: list[str] = []
    for part in bom.parts:
        ident = f"{part.symbol} {part.value}".strip()
        pins = info.get(part.ref, {})
        wired = nets.get(part.ref, {})
        if _ESP_FAMILY_RE.search(ident):
            problem = _esp_boot_problem(pins, wired)
        elif _RP2040_FAMILY_RE.search(ident):
            problem = _rp2040_boot_problem(pins, wired, nc, part.ref, bom)
        elif _GENERIC_MCU_RE.search(ident):
            problem = _generic_mcu_problem(pins, wired, nc, part.ref)
        else:
            continue
        if problem:
            bad.append(f"{part.ref} ({ident}): {problem}")
    return CheckResult(
        name="9.21 MCU programming path",
        ok=not bad,
        message=(
            "every MCU has a first-flash path"
            if not bad
            else f"{len(bad)} MCU(s) have no guaranteed programming path"
        ),
        offenders=bad,
    )


# Connector / access-point refs are never the controller even when their value
# names one: J1 "ESP32-S3 UART PROGRAM" is a header, not a second MCU.
_MCU_ACCESS_REF_PREFIXES = frozenset({"J", "P", "CN", "CONN", "X", "H", "TP"})


def _is_mcu_part(part) -> bool:
    if _ref_prefix(part.ref) in _MCU_ACCESS_REF_PREFIXES:
        return False
    ident = f"{part.symbol} {part.value}"
    return bool(
        _ESP_FAMILY_RE.search(ident)
        or _RP2040_FAMILY_RE.search(ident)
        or _GENERIC_MCU_RE.search(ident)
    )


def _programming_access_parts(bom) -> list:
    """Parts a programmer can physically reach: any TP test pad, or a
    connector-class part (J/P/CN, plus H which some designs use for pin
    headers -- keyword-gated, so H mounting holes never match) whose identity
    names a programming interface or USB (native-USB flash / UART-bridge
    designs)."""
    out = []
    for p in bom.parts:
        pref = _ref_prefix(p.ref)
        if pref == "TP":
            out.append(p)
            continue
        if pref not in ("J", "P", "CN", "H"):
            continue
        ident = f"{p.symbol} {p.value} {p.sourcing_note or ''}"
        if _PROG_ACCESS_PART_RE.search(ident) or _USB_PART_RE.search(ident):
            out.append(p)
    return out


# Button/part identities for the family strap rules. Matched against
# "symbol value sourcing_note" like _PROG_ACCESS_PART_RE.
_STM32_FAMILY_RE = re.compile(r"stm32", re.I)
_BOOT_BUTTON_RE = re.compile(r"boot|io0\b|gpio0\b|io9\b|download|flash.?mode", re.I)
_RESET_BUTTON_RE = re.compile(r"reset|\brst\b|\ben\b|enable|\brun\b", re.I)
_USB_UART_BRIDGE_RE = re.compile(
    r"cp210\d|ch340|ch910\d|ft232|ftdi|pl2303|usb.?(?:to.?)?(?:uart|serial)|"
    r"uart.?bridge",
    re.I,
)


def _family_strap_gaps(bom, mcus, access) -> list[str]:
    """Bootloader-strap families need more than "a USB connector exists".

    Returns one offender string per MCU whose family requirement is unmet;
    families outside the two rules (and unrecognizable parts) are never
    judged. Part-presence only -- runs at BOM commit, where the model can
    still ADD the missing button/header (wiring-level strap analysis stays
    §9.21's job).
    """

    def _ident(p) -> str:
        return f"{p.symbol} {p.value} {getattr(p, 'sourcing_note', None) or ''}"

    buttons = [p for p in bom.parts if _ref_prefix(p.ref) in ("SW", "S")]
    tps = [p for p in bom.parts if _ref_prefix(p.ref) == "TP"]
    swd_access = [p for p in access if _PROG_ACCESS_PART_RE.search(_ident(p))]
    bridge = any(_USB_UART_BRIDGE_RE.search(_ident(p)) for p in bom.parts)
    gaps: list[str] = []
    for part in mcus:
        ident = f"{part.symbol} {part.value}".strip()
        if _RP2040_FAMILY_RE.search(ident):
            if not (swd_access or buttons or len(tps) >= 2):
                gaps.append(
                    f"{part.ref} ({ident}): RP2040 cannot re-enter its USB "
                    "bootloader without holding BOOTSEL at reset -- add a "
                    "BOOTSEL button/jumper (SW ref), an SWD header (name "
                    "'SWD' in its value), or TP pads on SWD/BOOTSEL"
                )
        elif _ESP_FAMILY_RE.search(ident):
            has_boot = any(_BOOT_BUTTON_RE.search(_ident(p)) for p in buttons)
            has_reset = any(_RESET_BUTTON_RE.search(_ident(p)) for p in buttons)
            if not (bridge or (has_boot and has_reset) or len(tps) >= 2):
                gaps.append(
                    f"{part.ref} ({ident}): entering ESP32 download mode "
                    "needs the BOOT strap plus a reset -- add BOOT and "
                    "EN/RESET buttons (SW refs, named so), or a USB-UART "
                    "bridge with DTR/RTS auto-reset, or TP pads on the "
                    "straps"
                )
        elif _STM32_FAMILY_RE.search(ident):
            # STM32's ROM bootloader (USB-DFU and UART alike) is only entered
            # with BOOT0 pulled HIGH at reset, so "has a USB connector" is not
            # a programming story on its own -- self-eval 2026-07-27 run_24
            # shipped an STM32F042 whose assumed path was native-USB DFU with
            # no BOOT0 access part anywhere in the BOM (and died in reconcile
            # asking for exactly that). One TP suffices: BOOT0 to a pad,
            # reset by power cycle.
            has_boot = any(_BOOT_BUTTON_RE.search(_ident(p)) for p in buttons)
            if not (swd_access or bridge or has_boot or tps):
                gaps.append(
                    f"{part.ref} ({ident}): entering the STM32 ROM bootloader "
                    "(USB-DFU/UART) needs BOOT0 pulled HIGH at reset and no "
                    "other programming story exists -- add an SWD header "
                    "(name 'SWD'/'DEBUG' in its value), a BOOT0 button/jumper "
                    "(SW ref, named BOOT), or a TP test pad on BOOT0"
                )
    return gaps


_USB_CONNECTOR_DM_PIN_RE = re.compile(r"^(?:DN|DM|D-|USB_?DM|USB_D-)\d*$", re.I)
_USB_CONNECTOR_DP_PIN_RE = re.compile(r"^(?:DP|D\+|USB_?DP|USB_D\+)\d*$", re.I)
_USB_HEADER_IDENTITY_RE = re.compile(r"conn_01x|pinheader|pin header|\bheader\b", re.I)
_NATIVE_USB_OFFENDER_PREFIX = "native_usb_programming_required"
# Reviewed USBLC6-2SC6 feed-through pairs: connector-side line to device-side
# line. 2 = GND and 5 = VBUS are rails and are never crossed.
_USBLC6_FEED_THROUGH = {"1": "3", "3": "1", "6": "4", "4": "6"}
_USB_SERIES_REF_PREFIXES = frozenset({"R", "RV", "RT", "RP"})


def _usb_connector_pins(info, ref) -> tuple[list[str], list[str]]:
    """(D- pin numbers, D+ pin numbers) resolved from the symbol's pin names."""
    pins = info.get(ref) or {}
    dm = sorted(
        num for num, pin in pins.items() if _USB_CONNECTOR_DM_PIN_RE.match(pin["name"] or "")
    )
    dp = sorted(
        num for num, pin in pins.items() if _USB_CONNECTOR_DP_PIN_RE.match(pin["name"] or "")
    )
    return dm, dp


def _usb_data_connector_parts(bom, info) -> list:
    """Socket parts that expose both a D- and a D+ pin (not a header/bridge).

    A UART bridge (CH340/CP210x/FT232/...) also carries D-/D+ pins on its host
    side, but it is not a USB *data connector*: nothing plugs into it, and its
    device side speaks UART. It must never satisfy a native-USB programming
    contract.
    """
    out = []
    for part in bom.parts:
        if _ref_prefix(part.ref) not in (_CONNECTOR_PREFIXES | {"H"}):
            continue
        ident = f"{part.symbol} {part.value} {part.sourcing_note or ''}"
        if not _USB_PART_RE.search(ident) or _USB_HEADER_IDENTITY_RE.search(ident):
            continue
        if _USB_UART_BRIDGE_RE.search(ident):
            continue
        if part.ref not in info:
            out.append(part)  # unresolvable symbol: accept the USB identity
            continue
        dm, dp = _usb_connector_pins(info, part.ref)
        if dm and dp:
            out.append(part)
    return out


def _usb_series_other_pin(part, pin, info, nets) -> str | None:
    """The far pin of a real two-terminal series resistor, else None."""
    if _ref_prefix(part.ref) not in _USB_SERIES_REF_PREFIXES:
        return None
    pins = info.get(part.ref) or {}
    if len(pins) != 2:
        return None
    others = [num for num in pins if num != pin]
    if len(others) != 1:
        return None
    wired = nets.get(part.ref, {})
    if not wired.get(pin) or not wired.get(others[0]):
        return None
    if wired[pin] == wired[others[0]]:
        return None
    return others[0]


def _usb_feed_through_other_pin(part, pin) -> str | None:
    ident = f"{part.symbol} {part.value} {part.mpn or ''}"
    if not re.search(r"usblc6", ident, re.I):
        return None
    return _USBLC6_FEED_THROUGH.get(pin)


def _usb_reachable_pins(bom, info, nets, by_ref, start_pins) -> set[tuple[str, str]]:
    """Reachability over same-net copper and the two permitted series elements."""
    net_pins: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for connection in bom.connections:
        for endpoint in connection.endpoints:
            net_pins[connection.net_name].append((endpoint.ref, endpoint.pin))
    reachable: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    stack = list(start_pins)
    while stack:
        ref, pin = stack.pop()
        if (ref, pin) in seen:
            continue
        seen.add((ref, pin))
        net = nets.get(ref, {}).get(pin)
        if not net:
            continue
        # Never bridge through a rail or ground: D-/D+ are signal conductors.
        if _net_looks_ground(net) or _net_is_positive_rail(net) or _net_looks_power(net):
            continue
        endpoints = net_pins.get(net, [])
        reachable.update(endpoints)
        for endpoint_ref, endpoint_pin in endpoints:
            part = by_ref.get(endpoint_ref)
            if part is None:
                continue
            other = _usb_series_other_pin(part, endpoint_pin, info, nets)
            if other is None:
                other = _usb_feed_through_other_pin(part, endpoint_pin)
            if other is not None:
                stack.append((endpoint_ref, other))
    return reachable


def _native_usb_programming_gaps(bom, mcus) -> list[str]:
    """§9.29 native-USB proof: each reviewed native MCU's D-/D+ must reach the
    correspondingly named pins of ONE physical USB data connector.

    Part-presence runs at BOM commit (before wiring); the reachability proof
    runs once connections exist. Net names never substitute for the physical
    pins, so renaming a net cannot fake a connection.
    """
    info, _pin_count = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    by_ref = {part.ref: part for part in bom.parts}
    connectors = _usb_data_connector_parts(bom, info)
    gaps: list[str] = []
    for part in mcus:
        ident = f"{part.symbol} {part.value}".strip()
        family = next((row for row in _NATIVE_USB_FAMILIES if row[0].search(ident)), None)
        if family is None:
            continue
        _family_re, dm_fn, dp_fn = family
        offender = f"{_NATIVE_USB_OFFENDER_PREFIX}:{part.ref}"
        if not bom.connections:
            if not connectors:
                gaps.append(
                    f"{offender} ({ident}): native USB programming needs one "
                    "physical USB data connector (a receptacle with real D-/D+ "
                    "pins); a header, UART bridge or power-only connector cannot "
                    "replace it"
                )
            continue
        pins = info.get(part.ref) or {}
        dm_pins = sorted(num for num, pin in pins.items() if dm_fn in (pin["name"] or "").upper())
        dp_pins = sorted(num for num, pin in pins.items() if dp_fn in (pin["name"] or "").upper())
        if not dm_pins or not dp_pins:
            gaps.append(
                f"{offender} ({ident}): the native USB {dm_fn}/{dp_fn} pins are "
                "unresolvable on this symbol; restore the reviewed device symbol"
            )
            continue
        wired = nets.get(part.ref, {})
        dm_net = next((wired[num] for num in dm_pins if wired.get(num)), None)
        dp_net = next((wired[num] for num in dp_pins if wired.get(num)), None)
        if dm_net is None or dp_net is None:
            gaps.append(
                f"{offender} ({ident}): {dm_fn}/{dp_fn} native USB data line is "
                "left open; both D- and D+ must reach the USB data connector"
            )
            continue
        if dm_net == dp_net:
            gaps.append(
                f"{offender} ({ident}): native USB D- and D+ share net "
                f"{dm_net!r}; the pair must stay distinct into the connector"
            )
            continue
        reachable_dm = _usb_reachable_pins(
            bom, info, nets, by_ref, [(part.ref, num) for num in dm_pins]
        )
        reachable_dp = _usb_reachable_pins(
            bom, info, nets, by_ref, [(part.ref, num) for num in dp_pins]
        )
        ok = False
        swapped = False
        for connector in connectors:
            connector_dm, connector_dp = _usb_connector_pins(info, connector.ref)
            if not (connector_dm and connector_dp):
                continue
            d_ok = any((connector.ref, num) in reachable_dm for num in connector_dm)
            p_ok = any((connector.ref, num) in reachable_dp for num in connector_dp)
            if d_ok and p_ok:
                ok = True
                break
            if any((connector.ref, num) in reachable_dm for num in connector_dp) or any(
                (connector.ref, num) in reachable_dp for num in connector_dm
            ):
                swapped = True
        if ok:
            continue
        reason = (
            "D- and D+ are swapped at the connector"
            if swapped
            else "no series path reaches a connector's corresponding pins (open "
            "line, wrong pin, or a path ending at a header/bridge)"
        )
        gaps.append(
            f"{offender} ({ident}): native USB D-/D+ do not both reach the "
            f"correspondingly named pins of one physical USB data connector -- {reason}"
        )
    return gaps


def check_mcu_programming_access(bom) -> CheckResult:
    """§9.29 (hard) -- an MCU board must be physically programmable.

    Two layers, matching what is statically decidable at each stage:

    * **Part presence** (works at BOM commit, before wiring): a BOM containing
      an MCU must also contain a programming-ACCESS part -- a programming
      header (UPDI/SWD/JTAG/ICSP, named as such), TP test pads, or a USB
      connector (native-USB or UART-bridge designs). KC-HN59RJ shipped an
      ATtiny412 whose UPDI had a pullup but no header or pad: electrically
      fine, physically unprogrammable -- and "pre-programmed" as a silent
      default is not an accepted answer; test pads cost nothing and satisfy
      even a "no connectors" brief.
    * **UPDI reachability** (runs once ``connections`` exist, at wiring
      commit): a UPDI-programmed MCU's UPDI pin must share a net with one of
      those access parts. Wired-to-a-pullup-only is the observed failure
      mode; conservative for other families (SWD heuristics stay §9.21).

    **Family strap/reset requirements** (part presence, BOM commit): "has a
    USB connector" is NOT a sufficient programming story for bootloader-strap
    families -- self-eval 2026-07-19 gated two boards at cap 50 on exactly
    this:

    * RP2040 (run_10): entering the ROM USB bootloader after first flash
      needs BOOTSEL held at reset -- without a BOOTSEL button/jumper or an
      SWD access part the board is one bad firmware away from a brick.
    * ESP32 family (run_30): entering download mode needs the BOOT strap +
      a reset, so the BOM must carry BOOT+EN/RESET buttons, or a USB-UART
      bridge (DTR/RTS auto-reset), or strap test pads.
    * STM32 (2026-07-27 run_24): the ROM bootloader (USB-DFU/UART) is only
      entered with BOOT0 HIGH at reset -- a BOM whose only story is native
      USB must also carry a BOOT0 affordance (button/jumper/TP) or SWD
      access.
    """
    mcus = [p for p in bom.parts if _is_mcu_part(p)]
    if not mcus:
        return CheckResult(name="9.29 MCU programming access", ok=True, message="no MCU in BOM")
    access = _programming_access_parts(bom)
    if not access:
        return CheckResult(
            name="9.29 MCU programming access",
            ok=False,
            message=(
                "the BOM has an MCU but NO programming-access part; add a "
                "programming header for the MCU's interface (3-pin UPDI header "
                "for ATtiny/AVR 0/1-series, 2x5 or 1x4 SWD header for "
                "STM32/nRF/RP2040 -- name the interface in the part's value) "
                "or, when the brief forbids connectors, TP test-pad parts on "
                "the programming pins; a USB connector also satisfies this for "
                "native-USB or UART-bridge designs"
            ),
            offenders=[f"{p.ref} ({p.symbol} {p.value})" for p in mcus],
        )
    bad: list[str] = []
    bad.extend(_family_strap_gaps(bom, mcus, access))
    bad.extend(_native_usb_programming_gaps(bom, mcus))
    if bom.connections:
        from collections import defaultdict as _dd

        access_refs = {p.ref for p in access}
        info, _ = _pin_info_by_ref(bom)
        nets = _nets_by_ref(bom)
        refs_on_net: dict[str, set[str]] = _dd(set)
        for c in bom.connections:
            for ep in c.endpoints:
                refs_on_net[c.net_name].add(ep.ref)
        for part in mcus:
            pins = info.get(part.ref, {})
            wired = nets.get(part.ref, {})
            updi = [n for n, p in pins.items() if _UPDI_PIN_RE.search(p["name"])]
            if updi:
                reachable = any(
                    wired.get(n) and (refs_on_net.get(wired[n], set()) & access_refs) for n in updi
                )
                if not reachable:
                    bad.append(
                        f"{part.ref} ({part.symbol} {part.value}): UPDI pin "
                        f"{'/'.join(updi)} does not reach any programming-access "
                        f"part ({', '.join(sorted(access_refs))}); wire the UPDI "
                        "net to a header pin or test pad (keeping the existing "
                        "pullup is fine)"
                    )
            if _RP2040_FAMILY_RE.search(f"{part.symbol} {part.value}"):
                problem = _rp2040_boot_problem(pins, wired, set(), part.ref, bom)
                if problem:
                    bad.append(f"{part.ref} ({part.symbol} {part.value}): {problem}")
    return CheckResult(
        name="9.29 MCU programming access",
        ok=not bad,
        message=(
            "every MCU has a physical programming path"
            if not bad
            else f"{len(bad)} MCU(s) missing a workable programming/recovery "
            f"path (strap buttons, debug access, or reachability)"
        ),
        offenders=bad,
    )


def mcu_programming_facts(bom) -> dict | None:
    """Deterministic programming-path facts for the eval digest (2026-07-27
    fix-plan P2.5).

    The judge re-derived programmability from a digest that never carried the
    §9.29/§9.21 verdicts and over-fired ``unprogrammable_mcu`` on boards those
    checks deliberately accept (a BOOTSEL button + USB is the RP2040 ROM UF2
    path; a UPDI TP pad satisfies a no-connectors brief). Handing the judge
    the computed verdict pre-empts the guess, exactly like
    ``regulator_vout_facts``. Returns ``None`` when the BOM has no MCU."""
    mcus = [p for p in bom.parts if _is_mcu_part(p)]
    if not mcus:
        return None
    access = _programming_access_parts(bom)
    acc = check_mcu_programming_access(bom)
    path = check_mcu_programming_path(bom)
    return {
        "mcus": [f"{p.ref} ({p.symbol} {p.value})".strip() for p in mcus],
        "access_parts": [f"{p.ref} ({(p.value or p.symbol).strip()})" for p in access],
        "access_ok": acc.ok,
        "access_problems": list(acc.offenders),
        "path_ok": path.ok,
        "path_problems": list(path.offenders),
    }


# ---------- §9.22 breakout / adapter intent (advisory) ----------
#
# A "breakout" or "adapter" board's whole job is to map one connector's pins onto
# another's, so at least one net must BRIDGE the two connectors. #11 fpc-breakout
# emitted 49 nets with NONE spanning both connectors -- J1 (FPC) and J2 (header)
# on mutually disconnected nets -- so the breakout did nothing. ERC/DRC are clean
# (every pin is on a legal net), so only an intent-aware check sees it.
#
# A DETECTOR, not a normalizer: the actual pin mapping is a synthesis-intent
# decision, not mechanically derivable. Advisory like §9.21 -- surfaced as a
# wiring open_question, never a hard fab gate -- and gated on a breakout/adapter
# brief with >=2 connectors, so a normal multi-connector board never trips it.

_BREAKOUT_RE = re.compile(
    r"break[- ]?out|breakout|adapter|adaptor|pass[- ]?through|fan[- ]?out", re.I
)
_CONNECTOR_PREFIXES = frozenset({"J", "P", "CN", "CONN", "X"})


def check_breakout_connectivity(intent, bom) -> CheckResult:
    """§9.22 (advisory) -- on a breakout/adapter brief, at least one net must
    bridge the two connectors. See the section comment."""
    name = "9.22 breakout connectivity"
    if intent is None or bom is None or not bom.connections:
        return CheckResult(name=name, ok=True, message="not applicable")
    text = " ".join([intent.goal or ""] + list(getattr(intent, "constraints", []) or []))
    if not _BREAKOUT_RE.search(text):
        return CheckResult(name=name, ok=True, message="not a breakout/adapter brief")
    conns = {p.ref for p in bom.parts if _ref_prefix(p.ref) in _CONNECTOR_PREFIXES}
    if len(conns) < 2:
        return CheckResult(name=name, ok=True, message="fewer than two connectors")
    bridging = sum(
        1 for c in bom.connections if len({ep.ref for ep in c.endpoints if ep.ref in conns}) >= 2
    )
    if bridging == 0:
        return CheckResult(
            name=name,
            ok=False,
            message="breakout/adapter brief but no net bridges the connectors",
            offenders=[
                f"connectors {sorted(conns)} share zero bridging nets -- the "
                "breakout's job (mapping one connector's pins to the other) is "
                "undone"
            ],
        )
    return CheckResult(name=name, ok=True, message=f"{bridging} net(s) bridge the connectors")


# ---------- §9.9 connectivity (Stage B) ----------


_LIB_ID_RE = re.compile(r'\(lib_id\s+"([^"]+)"')


@lru_cache(maxsize=512)
def _symbol_pin_count(symbol: str) -> int | None:
    """How many electrical pins a library symbol declares, or None if unresolved.

    None is deliberately NOT 0: a symbol this checkout cannot resolve has not
    been shown to be pinless, so it must not earn the exemption below."""
    from .symbol_pinout import lookup_pins

    try:
        return len(lookup_pins(symbol)["pins"])
    except Exception:  # noqa: BLE001 - any lookup failure means "unknown"
        return None


def check_connectivity(
    project_dir: Path, project_stem: str, *, board_fabricated: frozenset[str] = frozenset()
) -> CheckResult:
    """§9.9 — every leaf sheet with ≥2 component symbols must contain at
    least one ``(wire …)`` or at least one ``(symbol (lib_id "power:…")…)``
    instance. A leaf with components but zero electrical artifacts is a
    Stage-B regression (or a Stage-A pre-wiring snapshot, which is gated
    out by the caller).

    Two structural exemptions, both narrower than the rule they silence:

    ``board_fabricated`` lists the symbols of parts the BOM marks as board-fabricated
    (``assembly=False``): a leaf built only from those — the prototyping pad field — has no
    wires by design, because its pads ARE the user's own wiring surface. The exemption
    reads the BOM's own flag, never a name or a library prefix, so a leaf that is merely
    unwired is still reported.

    A leaf whose parts all declare **zero pins** (mounting holes and the like) is exempt
    for the same reason from the other direction: a part with no pin cannot be wired, so
    "0 wires" is that leaf's correct state rather than lost connectivity. The count comes
    from the symbol library, so it is a property of the part, not of its name."""
    bad: list[str] = []
    root_name = f"{project_stem}.kicad_sch"
    for sch in sorted(project_dir.glob("*.kicad_sch")):
        if sch.name == root_name:
            continue  # root has no components
        text = sch.read_text()
        components: list[str] = []
        power_symbols = 0
        for _offset, block in _iter_symbol_instance_blocks(text):
            lib_id_m = _LIB_ID_RE.search(block)
            if lib_id_m and lib_id_m.group(1).startswith("power:"):
                power_symbols += 1
            else:
                components.append(lib_id_m.group(1) if lib_id_m else "")
        if len(components) < 2:
            continue
        if components and all(symbol in board_fabricated for symbol in components):
            continue  # a bare board-fabricated field, wired by the user
        if components and all(_symbol_pin_count(symbol) == 0 for symbol in components):
            continue  # pinless mechanical parts: nothing to wire
        # Wire count (top-level only is fine — wires never appear inside
        # lib_symbols).
        wire_count = text.count("(wire")
        if wire_count == 0 and power_symbols == 0:
            bad.append(f"{sch.name}: {len(components)} components, 0 wires, 0 power symbols")
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
        return CheckResult(name="9.12 ERC", ok=False, message=f"{root_sch.name} missing")
    out_path = project_dir / f"{project_stem}_erc.rpt"
    try:
        proc = subprocess.run(
            [
                "kicad-cli",
                "sch",
                "erc",
                "--format",
                "json",
                "--output",
                str(out_path),
                str(root_sch),
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except FileNotFoundError:
        return CheckResult(
            name="9.12 ERC",
            ok=True,
            message="kicad-cli not available; ERC skipped",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="9.12 ERC",
            ok=False,
            message="kicad-cli timed out after 60s",
        )

    if not out_path.exists():
        return CheckResult(
            name="9.12 ERC",
            ok=False,
            message=(f"kicad-cli sch erc exit {proc.returncode}; no report at {out_path.name}"),
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
            name="9.12 ERC",
            ok=False,
            message=f"{len(error_lines)} ERC error(s)",
            offenders=error_lines[:20],
        )
    return CheckResult(name="9.12 ERC", ok=True, message="ERC clean (0 errors)")


# ---------- §9.13 netlist faithfulness (Stage B) ----------


def _extract_netlist_groups(netlist_text: str) -> list[set[tuple[str, str]]]:
    """Parse a kicadsexpr netlist into one (ref, pin) set per net.

    Paren-scans each ``(net ...)`` block (escape-aware) and collects its
    ``(node (ref "..") (pin "..") ...)`` entries. Power-symbol pseudo-refs
    (``#PWR..``, ``#FLG..``) are dropped.
    """
    groups: list[set[tuple[str, str]]] = []
    node_re = re.compile(r'\(node\s+\(ref\s+"([^"]+)"\)\s+\(pin\s+"([^"]+)"\)')
    i = 0
    n = len(netlist_text)
    while True:
        start = netlist_text.find("(net ", i)
        if start == -1:
            break
        depth = 0
        in_str = False
        j = start
        while j < n:
            c = netlist_text[j]
            if in_str:
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = netlist_text[start : j + 1]
        pins = {(ref, pin) for ref, pin in node_re.findall(block) if not ref.startswith("#")}
        if pins:
            groups.append(pins)
        i = j + 1
    return groups


def _compare_netlist_to_bom(
    extracted: list[set[tuple[str, str]]], bom
) -> tuple[list[str], list[str], list[str]]:
    """Compare extracted (ref, pin) net groups against ``bom.connections``.

    Returns ``(merges, splits, lost)``: human-readable merge descriptions for
    extracted nets containing pins of bom nets that share neither a name
    nor an endpoint; split descriptions for one bom net whose wired pins
    scatter across >1 extracted net; and ``ref.pin`` strings for wired pins
    absent from every extracted net. Same-named connections are expected to
    unify (local labels per sheet, power symbols / hier labels across sheets),
    as are connections sharing an endpoint — anything beyond that landing
    in one extracted net is a merge the design never asked for, and anything
    less (a bom net in several extracted nets) is a stub the emitter dropped.
    """
    bom_refs = {p.ref for p in bom.parts}
    ep_group: dict[tuple[str, str], str] = {}
    parent: dict[str, str] = {}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_name: dict[str, str] = {}
    for c in bom.connections:
        key = f"{c.sheet}//{c.net_name}"
        parent.setdefault(key, key)
        if c.net_name in by_name:
            union(key, by_name[c.net_name])
        else:
            by_name[c.net_name] = key
        for ep in c.endpoints:
            e = (ep.ref, str(ep.pin))
            if e in ep_group:
                union(key, ep_group[e])
            ep_group[e] = key

    # Library-backed sheets carry parts the BOM never wired; restrict both
    # directions of the comparison to endpoints bom.connections knows.
    wired = set(ep_group)
    seen: set[tuple[str, str]] = set()
    merges: list[str] = []
    # Which extracted-net indices each bom net (union-find root) lands in, and
    # the known pins seen for that root — for the cohesion (split) check below.
    group_indices: dict[str, set[int]] = defaultdict(set)
    group_pins: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for idx, net_pins in enumerate(extracted):
        known = {e for e in net_pins if e in wired and e[0] in bom_refs}
        seen |= known
        nets_here = {find(ep_group[e]) for e in known}
        if len(nets_here) > 1:
            names = sorted({g.split("//", 1)[1] for g in nets_here})
            sample = sorted(f"{r}.{p}" for r, p in known)[:6]
            merges.append(f"nets {names} merged at pins {sample}")
        for e in known:
            root = find(ep_group[e])
            group_indices[root].add(idx)
            group_pins[root].add(e)

    # Cohesion: a bom net whose wired pins scatter across >1 extracted net was
    # not fully joined by the emitter. A dropped stub (a connector pin on a
    # sheet with no realized connection, an undeclared cross-sheet net) leaves
    # the pin on its own singleton auto-net -- present, so not "lost", and one
    # bom net, so not a "merge" -- yet electrically orphaned (pin_not_connected
    # + label_dangling). This is the class §9.11/§9.13's older checks could not
    # see; it is caught structurally here against the KiCad-extracted netlist.
    splits: list[str] = []
    for root, idxs in group_indices.items():
        if len(idxs) > 1:
            name = root.split("//", 1)[1]
            sample = sorted(f"{r}.{p}" for r, p in group_pins[root])[:8]
            splits.append(f"net {name!r} split across {len(idxs)} nets at pins {sample}")

    lost = sorted(f"{r}.{p}" for (r, p) in wired - seen if r in bom_refs)
    return merges, sorted(splits), lost


def check_netlist_faithfulness(project_dir: Path, project_stem: str, bom) -> CheckResult:
    """§9.13 — the KiCad-extracted netlist matches ``bom.connections``.

    ERC misses two classes of wiring corruption this catches directly:

    - **lost pins** — a wired pin absent from the extracted netlist. Seen
      when an unescaped quote corrupted a child sheet (KiCad loads it as
      empty: every part on it vanishes from netlist AND board) and when a
      de-collision pass abandoned a pin's stub.
    - **silent net merges** — pins of two BOM nets landing in ONE extracted
      net with no shared endpoint to justify it. Seen when a slid label
      landed on a foreign stub (ISP_MISO≡ISP_MOSI): two labels on one wire
      is legal KiCad, so ERC stays quiet while MISO is shorted to MOSI.
    - **cohesion splits** — one BOM net whose wired pins scatter across
      several extracted nets: the emitter drew no stub for a pin (a connector
      pin on a sheet with no realized connection, an undeclared cross-sheet
      net), leaving it on its own singleton auto-net. This is the
      pin_not_connected + label_dangling class that slipped past §9.11 (which
      only checks bom.connections, already correct) — a model cannot fake it,
      since it is measured against the KiCad-extracted netlist of what was
      actually emitted.
    """
    root_sch = project_dir / f"{project_stem}.kicad_sch"
    if not root_sch.is_file():
        return CheckResult(
            name="9.13 netlist faithfulness",
            ok=False,
            message=f"{root_sch.name} missing",
        )
    out_path = project_dir / f"{project_stem}_netlist_check.net"
    try:
        subprocess.run(
            [
                "kicad-cli",
                "sch",
                "export",
                "netlist",
                "--format",
                "kicadsexpr",
                "--output",
                str(out_path),
                str(root_sch),
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except FileNotFoundError:
        return CheckResult(
            name="9.13 netlist faithfulness",
            ok=True,
            message="kicad-cli not available; netlist check skipped",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="9.13 netlist faithfulness",
            ok=False,
            message="kicad-cli timed out after 60s",
        )
    if not out_path.exists():
        return CheckResult(
            name="9.13 netlist faithfulness",
            ok=False,
            message="kicad-cli produced no netlist",
        )
    try:
        extracted = _extract_netlist_groups(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)

    merges, splits, lost = _compare_netlist_to_bom(extracted, bom)
    offenders = merges + splits + [f"pin missing from netlist: {e}" for e in lost]
    if offenders:
        return CheckResult(
            name="9.13 netlist faithfulness",
            ok=False,
            message=(
                f"{len(merges)} unexpected net merge(s), "
                f"{len(splits)} dropped-stub split(s), "
                f"{len(lost)} wired pin(s) lost"
            ),
            offenders=offenders[:20],
        )
    n_wired = len({(ep.ref, str(ep.pin)) for c in bom.connections for ep in c.endpoints})
    return CheckResult(
        name="9.13 netlist faithfulness",
        ok=True,
        message=f"netlist matches bom.connections ({n_wired} wired pins)",
    )


# ---------- aggregator ----------


def collect_validations(
    project_dir: Path,
    project_stem: str,
    bom=None,
) -> list[CheckResult]:
    """Run §9.1-§9.12 and return ALL results (does not raise).

    When ``bom`` is provided AND has a non-empty ``connections`` list,
    §9.10 (pin existence), §9.11 (net coverage), §9.9 (connectivity),
    §9.12 (ERC) and §9.13 (netlist faithfulness) also run. The latter
    three are Stage-B checks that only make sense once schematic wires +
    power symbols are being emitted.
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
        results.append(check_power_pin_polarity(bom))
        results.append(check_two_terminal_self_short(bom))
        results.append(check_repeated_block_coverage(bom))
        results.append(check_regulator_feedback_vout(bom))
        results.append(check_rf_feed_isolation(bom))
        results.append(check_single_net_per_pin(bom))
        results.append(check_family_wiring_contracts(bom))
        results.append(check_mcu_programming_access(bom))
        results.append(
            check_connectivity(
                project_dir,
                project_stem,
                board_fabricated=frozenset(part.symbol for part in bom.parts if not part.assembly),
            )
        )
        results.append(check_erc(project_dir, project_stem))
        results.append(check_netlist_faithfulness(project_dir, project_stem, bom))
    return results


def run_validations(
    project_dir: Path,
    project_stem: str,
    bom=None,
) -> list[CheckResult]:
    """Run §9.1-§9.12 (see ``collect_validations``) and raise
    ``SynthesisValidationError`` if any check failed."""
    results = collect_validations(project_dir, project_stem, bom=bom)
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
    return CheckResult(name="9.7 solve-subcircuits smoke", ok=True, message="exit 0")


# ---------------------------------------------------------------------------
# §9.23 — named-part substitution detection: when a resolved BOM part is a
# class substitution of what the brief named (e.g. "binding-post terminals"
# → screw-terminal-5mm-2p), surface it rather than silently committing.
# ---------------------------------------------------------------------------


def check_named_part_substitutions(intent, bom) -> CheckResult:
    """§9.23 (advisory) — detect BOM parts that silently substituted a named
    part from the intent.

    Each named part from ``intent.named_parts`` should appear (as a substring)
    in at least one BOM part's value, mpn, or sourcing_note. When a named part
    doesn't match anything, it was likely substituted with a different class
    of component — warn via open_question rather than committing silently.
    """
    name = "9.23 named-part substitution"
    if intent is None or bom is None:
        return CheckResult(name=name, ok=True, message="not applicable")
    named = list(getattr(intent, "named_parts", None) or [])
    if not named:
        return CheckResult(name=name, ok=True, message="no named parts in intent")
    parts_text = " ".join(
        f"{p.value} {p.mpn or ''} {p.sourcing_note or ''}" for p in (bom.parts or [])
    ).lower()
    offenders: list[str] = []
    for np in named:
        np_lower = np.lower()
        if np_lower not in parts_text:
            # Check a relaxed token-level match: every token of the named part
            # should appear somewhere in the parts corpus (e.g. "binding post"
            # tokens "binding" + "post" both appear, but "binding-post terminals"
            # vs "screw-terminal" — "terminals" appears but "binding" does not).
            tokens = [
                t for t in np_lower.replace("-", " ").replace("_", " ").split() if len(t) > 2
            ]  # skip short tokens like "a", "1u"
            missing = [t for t in tokens if t not in parts_text]
            if len(missing) >= len(tokens) // 2:
                offenders.append(
                    f"named part {np!r} not found in BOM values/notes (missing tokens: {missing})"
                )
    if offenders:
        return CheckResult(
            name=name,
            ok=False,
            message=f"{len(offenders)} named part(s) may have been substituted",
            offenders=offenders,
        )
    return CheckResult(name=name, ok=True, message="all named parts match BOM")


# ---------- §9.33 typed exact-part accountability (hard) ----------
#
# Exact identity is authoritative only when it crosses the architecture
# boundary in ``Requirement.exact_part``. Free-form functional-spec and
# architecture prose may explain a choice, but cannot silently create a new
# BOM identity contract.

_MPN_TOKEN_RE = re.compile(
    r"\b(?:ESP32[ _-][CS]\d|(?:ATtiny|ATmega)[ _-]?[0-9]{2,}|[A-Z]{2,4}[0-9]{2,})[A-Z0-9.\-]*",
    re.IGNORECASE,
)
_SHORT_MCU_FAMILY_RE = re.compile(r"^(?:ESP32|STM32|NRF51|NRF52|NRF53|NRF54)$", re.I)
_MPN_STOPWORD_RE = re.compile(
    r"^(?:USB|COM|GPIO|ADC|DAC|TIM|UART|USART|SPI|I2C|I2S|CAN|PWM|AIN|AOUT|"
    r"EXTI|REV|VER|LQFP|TQFP|QFN|DFN|SOIC|SOP|SSOP|TSSOP|MSOP|SOT|DIP|PDIP|"
    r"SOD|BGA|TO|IEC|ISO|AWG|PINS?|LEDS?|PCS|BITS?|MHZ|KHZ|HZ|MM|MIL|MA|MV|"
    r"VCC|VDD|VBAT|VBUS|VSYS|VOUT|VIN)[0-9]",
    re.IGNORECASE,
)

_NONCOMMITTAL_EXAMPLE_RE = re.compile(
    r"\((?:e\.?\s*g\.?|for example|such as)\b[^)]*\)",
    re.IGNORECASE,
)


def named_part_tokens(texts) -> dict[str, str]:
    """Return conservative normalized MPN/family tokens from arbitrary text."""
    out: dict[str, str] = {}
    for text in texts:
        text = _NONCOMMITTAL_EXAMPLE_RE.sub("", str(text))
        for match in _MPN_TOKEN_RE.finditer(text):
            token = match.group(0).rstrip(".-")
            if (
                len(token) < 6 and not _SHORT_MCU_FAMILY_RE.fullmatch(token)
            ) or _MPN_STOPWORD_RE.match(token):
                continue
            out.setdefault(token.lower(), token)
    return out


def spec_named_tokens(functional_spec, architecture) -> dict[str, str]:
    """Return typed architecture exact-part tokens keyed by lowercase form."""
    del functional_spec
    if architecture is None:
        return {}
    exact_parts = [
        requirement.exact_part
        for requirement in (getattr(architecture, "requirements", None) or [])
        if getattr(requirement, "exact_part", None)
    ]
    return named_part_tokens(exact_parts)


# Compatibility for callers that imported the former private helper.
_spec_named_tokens = spec_named_tokens


def check_spec_named_mpn_substitutions(functional_spec, architecture, bom) -> CheckResult:
    """§9.33 (hard) -- every spec-named MPN is either in the BOM or in the
    ``bom.substitutions`` ledger. See the section comment."""
    name = "9.33 spec-named part accountability"
    if bom is None or (functional_spec is None and architecture is None):
        return CheckResult(name=name, ok=True, message="not applicable")
    tokens = spec_named_tokens(functional_spec, architecture)
    if not tokens:
        return CheckResult(name=name, ok=True, message="no spec-named MPNs to account for")
    parts_text = " ".join(
        f"{p.value} {p.mpn or ''} {p.sourcing_note or ''}" for p in (bom.parts or [])
    ).lower()
    surfaced = " ".join(
        [f"{s.wanted} {s.got} {s.reason}" for s in (getattr(bom, "substitutions", None) or [])]
        + list(bom.assumptions or [])
    ).lower()
    offenders = [
        f"spec/architecture names {orig!r} but the BOM neither ships it nor "
        f"records a substitution for it"
        for low, orig in sorted(tokens.items())
        if low not in parts_text and low not in surfaced
    ]
    if offenders:
        return CheckResult(
            name=name,
            ok=False,
            message=(
                f"{len(offenders)} spec-named part(s) silently missing from "
                "the BOM -- either use the named part, or add a "
                'bom.substitutions entry {"wanted": "<named part>", "got": '
                '"<shipped part>", "reason": "<why>"} so the swap is '
                "surfaced, not silent"
            ),
            offenders=offenders,
        )
    return CheckResult(name=name, ok=True, message="all spec-named parts shipped or ledgered")


# ---------- §9.34 brief-stated mount type (hard) ----------
#
# 2026-07-27 run_20: the brief said "SMT I2C OLED" and the BOM shipped a
# through-hole OLED (footprint OLED-TH_..._P2.54), unsurfaced -- a
# silent_substitution cap. When the USER's own words pin a mount type to a
# part, a contradicting footprint must be ledgered. Narrow by construction:
# only fires on an explicit SMT/SMD/through-hole qualifier in the intent
# text, only for parts the qualified noun actually matches, and only when
# the footprint classifies unambiguously.

_MOUNT_ASK_RE = re.compile(
    r"\b(?P<mount>SMT|SMD|surface[- ]?mount(?:ed)?|through[- ]?hole|THT)\b"
    r"[ ,]*(?P<noun>(?:[A-Za-z0-9/.+-]+ ?){1,3})",
    re.IGNORECASE,
)
_TH_FOOTPRINT_RE = re.compile(
    r"THT|(?:^|[_:-])TH(?:[_-])|Through.?Hole|_DIP|DIP-|Axial|Radial|P2\.54",
    re.IGNORECASE,
)
_SMD_FOOTPRINT_RE = re.compile(
    r"SMD|SMT|(?:^|[_:-])(?:0402|0603|0805|1206|1210|2512)(?:[_-]|$)|SOIC|"
    r"QFN|LQFP|TQFP|SOT|SSOP|TSSOP|MSOP|BGA|WLCSP|PLCC",
    re.IGNORECASE,
)
_MOUNT_NOUN_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "for",
        "from",
        "into",
        "that",
        "this",
        "over",
        "under",
        "onto",
        "plus",
        "component",
        "components",
        "part",
        "parts",
        "device",
        "devices",
        "package",
        "packages",
        "only",
        "all",
        "version",
        "variant",
        "where",
        "possible",
        "preferred",
    }
)


def _mount_class(footprint: str) -> str | None:
    """ "th" / "smd" / None (unclassifiable -- never judged)."""
    th = bool(_TH_FOOTPRINT_RE.search(footprint or ""))
    smd = bool(_SMD_FOOTPRINT_RE.search(footprint or ""))
    if th == smd:
        return None
    return "th" if th else "smd"


def check_mount_type_consistency(intent, bom) -> CheckResult:
    """§9.34 (hard) -- a brief-stated SMT/through-hole qualifier must match
    the shipped footprint, or be ledgered. See the section comment."""
    name = "9.34 brief-stated mount type"
    if intent is None or bom is None:
        return CheckResult(name=name, ok=True, message="not applicable")
    text = " ".join(
        [getattr(intent, "goal", "") or ""]
        + list(getattr(intent, "constraints", None) or [])
        + list(getattr(intent, "named_parts", None) or [])
    )
    surfaced = " ".join(
        [f"{s.wanted} {s.got} {s.reason}" for s in (getattr(bom, "substitutions", None) or [])]
        + list(bom.assumptions or [])
    ).lower()
    offenders: list[str] = []
    for m in _MOUNT_ASK_RE.finditer(text):
        wanted = "smd" if m.group("mount")[:1].lower() == "s" else "th"
        nouns = [
            w.lower()
            for w in m.group("noun").split()
            if len(w) >= 3 and w.lower() not in _MOUNT_NOUN_STOPWORDS
        ]
        if not nouns:
            continue
        for p in bom.parts or []:
            ident = (f"{p.value} {p.symbol} {p.footprint} {p.sourcing_note or ''}").lower()
            hit = [n for n in nouns if n in ident]
            if not hit:
                continue
            got = _mount_class(p.footprint)
            if got is None or got == wanted:
                continue
            if any(n in surfaced for n in hit) or p.ref.lower() in surfaced:
                continue  # the deviation is on the record -- not silent
            offenders.append(
                f"{p.ref} ({p.value}): the brief asks for a "
                f"{'surface-mount' if wanted == 'smd' else 'through-hole'} "
                f"{' '.join(hit)} but footprint {p.footprint!r} is "
                f"{'through-hole' if got == 'th' else 'surface-mount'} -- "
                "use a matching footprint or record the deviation in "
                "bom.substitutions"
            )
    if offenders:
        return CheckResult(
            name=name,
            ok=False,
            message=(
                f"{len(offenders)} part(s) contradict a mount type the brief states explicitly"
            ),
            offenders=offenders,
        )
    return CheckResult(name=name, ok=True, message="no brief-stated mount-type contradictions")


# ---------- §9.24 opposite-edge connector conflict ----------

_OPPOSITE_EDGES = frozenset({frozenset({"top", "bottom"}), frozenset({"left", "right"})})


def check_sheet_connector_edge_conflicts(bom) -> CheckResult:
    """§9.24 — no sheet has edge-zoned connectors on opposite edges.

    A single rigid leaf can only satisfy one edge per axis.  Connectors
    zoned to opposite edges on one sheet guarantee one will strand inboard
    at compose time.  The synthesis stage auto-splits such sheets before
    they reach the BOM commit; this check is a safety net for any case
    the auto-split doesn't cover (e.g. sheet-name collisions).
    """
    ref_sheet: dict[str, str] = {}
    for p in bom.parts or []:
        if p.sheet and p.ref:
            ref_sheet[p.ref] = p.sheet

    sheet_edges: dict[str, set[str]] = defaultdict(set)
    for ref, zone in (bom.component_zones or {}).items():
        edge = zone.get("edge") if isinstance(zone, dict) else None
        sheet = ref_sheet.get(ref)
        if edge and sheet:
            sheet_edges[sheet].add(edge)

    offenders: list[str] = []
    for sheet, edges in sorted(sheet_edges.items()):
        for pair in _OPPOSITE_EDGES:
            if pair.issubset(edges):
                conflicting = sorted(
                    ref
                    for ref, zone in (bom.component_zones or {}).items()
                    if isinstance(zone, dict)
                    and zone.get("edge") in pair
                    and ref_sheet.get(ref) == sheet
                )
                offenders.append(
                    f"sheet {sheet!r} has connectors on opposite edges "
                    f"{sorted(pair)}: {', '.join(conflicting)}"
                )

    return CheckResult(
        name="9.24 no opposite-edge connectors on one sheet",
        ok=not offenders,
        message=(
            "every sheet has compatible edge zones"
            if not offenders
            else f"{len(offenders)} sheet(s) with opposite-edge connectors"
        ),
        offenders=offenders,
    )


# ---------- §9.25 capacitor symbol/footprint polarity consistency ----------
#
# KiCad's capacitor naming convention is unambiguous and machine-checkable:
#   symbol  Device:C*   -> NON-polarized (ceramic/film/etc.)
#   symbol  Device:CP*  -> POLARIZED (aluminium electrolytic / tantalum)
#   footprint  <lib>:C_*        -> NON-polarized
#   footprint  <lib>:CP_*       -> POLARIZED (has a + / cathode marking + a
#                                   physical orientation)
#   any Capacitor_Tantalum_*    -> POLARIZED
# A part whose symbol polarity disagrees with its footprint polarity is always
# wrong: a non-polarized ``Device:C`` on a polarized ``CP_Radial`` footprint (the
# KC-U2VAA8 speaker-crossover film caps -- the BOM stage picked an electrolytic
# can for a film cap) has no polarity to mark, and a ``Device:CP`` on a plain
# ``C_`` footprint loses the + marking. This is DRC/ERC-invisible (both are legal
# in isolation) but a real electrical/assembly defect, so it is gated at BOM
# commit where the model can still re-pick a matching footprint.


def _cap_symbol_polarity(symbol: str) -> str | None:
    """ "polarized" / "nonpolarized" for a KiCad capacitor symbol, else None.

    Classifies by the symbol NAME (the part after ``:``) using the C/CP
    convention. Returns None for anything that is not clearly a capacitor
    symbol, so the check never guesses on custom or unrelated symbols.
    """
    name = symbol.split(":", 1)[1] if ":" in symbol else symbol
    upper = name.upper()
    if upper.startswith("CP") or "POLAR" in upper:
        return "polarized"
    if name == "C" or name.startswith("C_"):
        return "nonpolarized"
    return None


def _cap_footprint_polarity(footprint: str) -> str | None:
    """ "polarized" / "nonpolarized" for a capacitor footprint, else None."""
    lib, _, name = footprint.partition(":")
    if "Tantalum" in lib:
        return "polarized"
    upper = name.upper()
    if upper.startswith("CP_") or "POLAR" in upper:
        return "polarized"
    if upper.startswith("C_"):
        return "nonpolarized"
    return None


def check_capacitor_polarity_consistency(bom) -> CheckResult:
    """§9.25 -- a capacitor's symbol polarity must match its footprint polarity.

    Fires only when BOTH the symbol and the footprint are unambiguously
    classified (KiCad C/CP naming, or a tantalum footprint) AND they disagree,
    so a correctly-paired part never trips and custom/odd names are skipped.
    """
    bad: list[str] = []
    for p in bom.parts or []:
        sym_pol = _cap_symbol_polarity(p.symbol or "")
        if sym_pol is None:
            continue  # not a recognized capacitor symbol
        fp_pol = _cap_footprint_polarity(p.footprint or "")
        if fp_pol is None:
            continue  # unrecognized footprint naming -- don't guess
        if sym_pol == fp_pol:
            continue
        if sym_pol == "nonpolarized":
            hint = (
                "a non-polarized cap must use a non-polarized (C_*) footprint, "
                "not a polarized CP_/tantalum one"
            )
        else:
            hint = (
                "a polarized cap needs a polarized (CP_*) footprint with a + "
                "marking, not a plain C_* one"
            )
        bad.append(
            f"{p.ref}: symbol {p.symbol!r} is {sym_pol} but footprint "
            f"{p.footprint!r} is {fp_pol} -- {hint}"
        )
    return CheckResult(
        name="9.25 capacitor polarity consistency",
        ok=not bad,
        message=(
            "capacitor symbol/footprint polarity agree"
            if not bad
            else f"{len(bad)} capacitor(s) with mismatched symbol/footprint polarity"
        ),
        offenders=bad,
    )


def _functional_block_sheets(
    functional_spec: FunctionalSpec, architecture
) -> tuple[dict[str, set[str]], list[str]]:
    """Resolve only explicit requirement ownership, retaining all owner sheets."""
    block_sheets: dict[str, set[str]] = {block.name: set() for block in functional_spec.blocks}
    sheet_names = {sheet.name for sheet in architecture.sheets}
    bad: list[str] = []
    # A `fabrication` row is a property of the board itself (a printed pad field, a thermal-via
    # field), never a user-visible function: the requirement derived from such a row implements no
    # functional block and no functional block implements it, so membership is not something it
    # can state. Requiring it would refuse every design whose intent carries the feature, so the
    # derived requirement is exempt -- and only that one: a requirement with no `fabrication` row
    # behind it still declares its block, so the gate cannot be cleared by leaving the field out.
    board_feature_ids = {
        row.original_obligation_id
        for row in (getattr(architecture, "obligations", None) or ())
        if row.kind == "fabrication"
    }
    # The same exemption, on the block side. A block whose every committed obligation is a
    # board-wide fact -- a `quantity`, `fabrication` or `negative` row, or a `quantitative` row
    # that measures the board outline or its stack-up (`obligation_requires_requirement_owner`,
    # the predicate the obligation-retention gate already uses) -- is realized by the board itself,
    # so no requirement can implement it: demanding one refuses a design the committed intent
    # legally declared (a brief's "two mounting holes" arrives as a `quantity` row with no part,
    # and the MOUNTING block then had no implementer at all -> every attempt refused). A block
    # carrying ANY obligation that needs a requirement owner is not exempt, so the gate still
    # refuses a block the architecture silently dropped; an id the architecture does not carry is
    # treated the same way (unprovable, not exempt).
    ownership_rows = {
        row.original_obligation_id: row
        for row in (getattr(architecture, "obligations", None) or ())
    }
    board_wide_blocks: set[str] = set()
    for block in functional_spec.blocks:
        ids = list(block.obligation_ids or ())
        if not ids or any(oid not in ownership_rows for oid in ids):
            continue
        if all(
            not obligation_requires_requirement_owner(ownership_rows[oid]) for oid in ids
        ):
            board_wide_blocks.add(block.name)
    if not architecture.requirements:
        bad.append("architecture has no implementation requirements")
    for requirement in architecture.requirements:
        if not requirement.functional_blocks and requirement.id not in board_feature_ids:
            bad.append(f"requirement {requirement.id!r} has no functional_blocks membership")
        if requirement.sheet not in sheet_names:
            bad.append(
                f"requirement {requirement.id!r} references unknown sheet {requirement.sheet!r}"
            )
        for name in requirement.functional_blocks:
            if name not in block_sheets:
                bad.append(f"requirement {requirement.id!r} references unknown block {name!r}")
            elif requirement.sheet in sheet_names:
                block_sheets[name].add(requirement.sheet)
    for name, sheets in block_sheets.items():
        if not sheets and name not in board_wide_blocks:
            bad.append(f"functional block {name!r} has no implementation requirement on a sheet")
    return block_sheets, bad


def check_every_block_has_sheet(functional_spec: FunctionalSpec, architecture) -> CheckResult:
    """Require explicit, complete block membership without prescribing sheet layout.

    Several requirements may implement one block; a composite requirement may
    implement several blocks, and all of them may share a sheet.
    """
    block_sheets, bad = _functional_block_sheets(functional_spec, architecture)
    return CheckResult(
        name="block-sheet mapping",
        ok=not bad,
        message=(
            f"all {len(block_sheets)} functional block(s) have explicit implementation requirements"
            if not bad
            else "; ".join(bad)
        ),
        offenders=bad,
    )


def check_fs_connections_mapped(functional_spec: FunctionalSpec, architecture) -> CheckResult:
    """Require explicit endpoint ownership and a sheet crossing for each signal.

    Power/ground may use global nets, but still require mapped block endpoints.
    Every declared net endpoint, including power/ground, must bind its exact net
    name through a requirement on that sheet, regardless of implementation owner.
    A shared bus may cover several pairs, and multiple owners allow a connection
    through any of the block's implementing sheets.
    """
    block_sheets, bad = _functional_block_sheets(functional_spec, architecture)
    bound_nets: dict[str, set[str]] = defaultdict(set)
    for requirement in architecture.requirements:
        bound_nets[requirement.sheet].update(requirement.ports.values())
    for net in architecture.inter_sheet_nets:
        for endpoint in net.endpoints:
            if net.name not in bound_nets.get(endpoint.sheet, set()):
                bad.append(
                    f"inter-sheet net {net.name!r} endpoint on sheet {endpoint.sheet!r} "
                    "has no requirement.ports value bound to that exact net name"
                )
    endpoint_sets = [
        {endpoint.sheet for endpoint in net.endpoints} for net in architecture.inter_sheet_nets
    ]
    for conn in functional_spec.connections:
        from_sheets = block_sheets.get(conn.from_block, set())
        to_sheets = block_sheets.get(conn.to_block, set())
        if not from_sheets or not to_sheets:
            bad.append(
                f"connection {conn.from_block!r}→{conn.to_block!r} "
                f"({conn.signal_type}) has an unknown or unmapped block endpoint"
            )
            continue
        if conn.signal_type in ("power", "ground") or from_sheets & to_sheets:
            continue
        covered = any(
            from_sheets & endpoints and to_sheets & endpoints for endpoints in endpoint_sets
        )
        if not covered:
            bad.append(
                f"connection {conn.from_block!r}→{conn.to_block!r} "
                f"({conn.signal_type}) crosses sheets but has no inter_sheet_net"
            )
    return CheckResult(
        name="fs-connection mapping",
        ok=not bad,
        message=(
            "every functional connection has mapped endpoints and declared sheet crossings"
            if not bad
            else f"{len(bad)} functional connection mapping defect(s)"
        ),
        offenders=bad,
    )
