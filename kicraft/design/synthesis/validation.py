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
)


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
    """Align ``architecture.inter_sheet_nets`` with the crossings the wiring
    stage actually realized, so the wiring stage is never handed a cross-sheet
    contract it has no power to edit.

    The wiring stage emits only ``connections``/``no_connect_pins`` — it cannot
    add or remove an ``inter_sheet_nets`` entry (those freeze at the architecture
    stage). Two failure modes follow directly, and each made real boards
    unbuildable (KC-WFFXZ3's ESP32 auto-reset; the proto-shield PROTO AREA):

      * A net the wiring stage legitimately wires across two sheets but that
        architecture never declared is flagged dangling by §9.15 on each side
        (it exempts only *declared* inter-sheet nets), and the emitter draws two
        disconnected local labels so the net would not even connect. The model
        cannot escape this — it cannot declare the net. **ADD** every signal net
        realized on >=2 sheets to ``inter_sheet_nets`` (e.g. EN/IO0 to the MCU).

      * A signal net architecture declared as crossing but that wiring only ever
        wires on one sheet — its real consumers live there (DTR/RTS at the
        auto-reset transistors) — leaves the other sheet's pin with no label, so
        §9.14 can never pass because nothing on that sheet consumes the net.
        **DROP** it; it is not actually inter-sheet.

    Net effect: a signal net is inter-sheet iff it is wired (under one name) on
    >=2 sheets. Power/ground inter-sheet nets are preserved verbatim — they join
    globally through power symbols, not per-pin connections, so realization does
    not apply (§9.11 owns their per-pin coverage).

    Because it only ever rewrites a contract the wiring stage failed to satisfy,
    it is a no-op on any design that already passes §9.14/§9.15: such a design
    has every cross-sheet net declared with all endpoints realized, and no
    undeclared net wired across sheets. Inconsistent-name dangles (the
    SOIL_MOISTURE_BLE USB_DP_POWER/USB_DP_ESP32 split) stay caught — each name
    is single-sheet, so nothing is promoted and §9.15 still fires.

    Mutates ``architecture.inter_sheet_nets`` in place; returns the changes made,
    for logging.
    """
    known = {s.name for s in architecture.sheets}
    realized: dict[str, set[str]] = defaultdict(set)
    for c in bom.connections:
        if c.endpoints and c.sheet in known:
            realized[c.net_name].add(c.sheet)

    declared = {n.name: n for n in architecture.inter_sheet_nets}
    power = [n for n in architecture.inter_sheet_nets if is_power_or_ground_name(n.name)]
    kept: set[str] = {n.name for n in power}

    signal: list[InterSheetNet] = []
    changes: list[str] = []
    for name in sorted(realized):
        if is_power_or_ground_name(name):
            continue
        sheets = realized[name]
        if len(sheets) < 2:
            continue
        signal.append(
            InterSheetNet(name=name, endpoints=_reconciled_endpoints(sheets, declared.get(name)))
        )
        kept.add(name)
        if name not in declared:
            changes.append(f"+{name} {sorted(sheets)} (realized, undeclared)")
        elif {e.sheet for e in declared[name].endpoints} != sheets:
            was = sorted(e.sheet for e in declared[name].endpoints)
            changes.append(f"~{name} {was} -> {sorted(sheets)}")

    for n in architecture.inter_sheet_nets:
        if not is_power_or_ground_name(n.name) and n.name not in kept:
            changes.append(
                f"-{n.name} (declared {sorted(e.sheet for e in n.endpoints)}, "
                f"realized {sorted(realized.get(n.name, set()))})"
            )

    if changes:
        architecture.inter_sheet_nets = power + signal
    return changes


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
    bad: list[str] = []
    for net in architecture.inter_sheet_nets:
        if is_power_or_ground_name(net.name):
            continue
        for ep in net.endpoints:
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
    "_5V", "_3V3", "_MCU", "_POWER", "_ESP32", "_ISO", "_LV", "_HV",
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
    return ", ".join(
        _pin_label(r, p, _pin_function(info, r, p)) for r, p in picked
    )


def _dangling_net_context(
    net: str,
    sheet: str,
    ref: str,
    pin: str,
    *,
    info,
    pin_count,
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
            other_func = _pin_function(info, ref, other)
            frag = (
                f"{ref} is a two-terminal series part whose other terminal "
                f"({_pin_label(ref, other, other_func)}) sits on net {onet!r}"
            )
            if dests:
                frag += (
                    f"; candidate endpoints on that net: "
                    f"{_endpoint_labels(dests, info)}. Move the intended "
                    f"load/destination endpoint from {onet!r} onto {net!r}"
                )
            else:
                frag += (
                    f" and {onet!r} wires no other pin here either. Complete "
                    f"the series path from its real source onto {net!r}"
                )
            frag += (
                f", keeping the two terminals of {ref} on different nets -- "
                f"never merge {onet!r} with {net!r}, and never put both "
                f"terminals of {ref} on one net (§9.17)"
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


def check_family_wiring_contracts(bom) -> CheckResult:
    """§9.20 -- datasheet pin-role contracts for known part families.

    See the module comment above _FAMILY_CONTRACTS. Fires only on a wired pin
    that a family's datasheet says must (not) be on a rail/ground and is bound to
    a clearly-wrong net; correct and filtered rails pass.
    """
    info, _ = _pin_info_by_ref(bom)
    nets = _nets_by_ref(bom)
    bad: list[str] = []
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


def _is_mcu_part(part) -> bool:
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
            bad.append(f"{sch.name}: {non_power_components} components, 0 wires, 0 power symbols")
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
        results.append(check_connectivity(project_dir, project_stem))
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


# ---------- §9.33 spec-named part accountability (hard) ----------
#
# The 2026-07-27 self-eval batch capped 6 runs on silent_substitution: an
# architecture-named RECOM RP12-2412DA (~500 mA/rail) silently became a
# WRA2412S-3WR2 (~125 mA/rail); a symbol-field STM32F103C8T6 shipped as a
# CBT6. §9.23 above only covers intent.named_parts (advisory); the
# spec/architecture free text is where the model itself commits to example
# parts, and a BOM that quietly walks away from them is the gate's exact
# condition. HARD at BOM commit: the model still has retries, and the fix is
# always writable -- use the named part, or record the swap in
# ``bom.substitutions`` (that ledger is the surfacing the gate demands; the
# substitution itself is often a fine engineering call).
#
# The token regex is deliberately conservative (>= 2 letters, >= 2 digits,
# total >= 6, protocol/package/pin-name stoplist): a missed family name like
# "LM317" costs nothing (no enforcement), while a false positive would cost a
# commit retry.

_MPN_TOKEN_RE = re.compile(r"\b[A-Za-z]{2,4}[0-9]{2,}[A-Za-z0-9.\-]*")
_MPN_STOPWORD_RE = re.compile(
    r"^(?:USB|COM|GPIO|ADC|DAC|TIM|UART|USART|SPI|I2C|I2S|CAN|PWM|AIN|AOUT|"
    r"EXTI|REV|VER|LQFP|TQFP|QFN|DFN|SOIC|SOP|SSOP|TSSOP|MSOP|SOT|DIP|PDIP|"
    r"SOD|BGA|TO|IEC|ISO|AWG)[0-9]",
    re.IGNORECASE,
)


def named_part_tokens(texts) -> dict[str, str]:
    """Return conservative normalized MPN/family tokens from arbitrary text."""
    out: dict[str, str] = {}
    for text in texts:
        for match in _MPN_TOKEN_RE.finditer(str(text)):
            token = match.group(0).rstrip(".-")
            if len(token) < 6 or _MPN_STOPWORD_RE.match(token):
                continue
            out.setdefault(token.lower(), token)
    return out


def spec_named_tokens(functional_spec, architecture) -> dict[str, str]:
    """MPN-like tokens the spec/architecture free text commits to, keyed by
    lowercase form (original casing kept for messages)."""
    texts: list[str] = []
    if architecture is not None:
        texts += list(getattr(architecture, "assumptions", None) or [])
        topo = getattr(architecture, "topologies", None) or {}
        texts += [f"{k}: {v}" for k, v in topo.items()]
        texts += [s.function or "" for s in (architecture.sheets or [])]
    if functional_spec is not None:
        texts += list(getattr(functional_spec, "assumptions", None) or [])
        texts += [b.purpose or "" for b in (functional_spec.blocks or [])]
    return named_part_tokens(texts)


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


def check_every_block_has_sheet(functional_spec: FunctionalSpec, architecture) -> CheckResult:
    """Architecture-stage gate: the architecture must have enough sheets to
    cover every functional_spec block (with count expansion).

    Blocks may be merged onto a sheet named for the IC domain (e.g. "POWER
    PATH" covers CHARGER + BOOST), so we don't require a 1:1 block→sheet
    mapping. We only flag when the architecture has zero sheets or far too
    few for the block count. Individual dropped blocks are caught at BOM
    commit by ``check_sheets_have_parts``.
    """
    total_block_instances = sum(max(1, b.count) for b in functional_spec.blocks)
    n_sheets = len(architecture.sheets)
    bad: list[str] = []
    if n_sheets == 0 and total_block_instances > 0:
        bad.append(
            f"functional_spec has {total_block_instances} block instance(s) "
            f"but the architecture has zero sheets"
        )
    return CheckResult(
        name="block-sheet mapping",
        ok=not bad,
        message=(
            f"{n_sheets} sheet(s) for {total_block_instances} block instance(s)"
            if not bad
            else bad[0]
        ),
        offenders=bad,
    )


def check_fs_connections_mapped(functional_spec: FunctionalSpec, architecture) -> CheckResult:
    """Architecture-stage gate: every non-power/ground functional_spec connection
    is either intra-sheet (both blocks on the same sheet) or declared in
    ``architecture.inter_sheet_nets``.

    Catches the historical DTR/RTS→ESP32 and RESET/D0→PROTO cases where a
    cross-sheet signal was declared in the functional_spec but never surfaced
    as an inter-sheet net, leaving a hierarchical label dangling at synthesis.
    """
    # Build block → set of sheet names mapping (heuristic: block name appears
    # in sheet name, case-insensitive). Blocks not matching any sheet are
    # "unmapped" — their connections can't be verified and are left to the
    # wiring-stage safety net.
    sheet_names_upper = [s.name.upper() for s in architecture.sheets]

    def _sheets_for_block(block_name: str) -> set[str]:
        bu = block_name.upper()
        return {sn for sn in sheet_names_upper if bu in sn}

    # Build a map: inter_sheet_net name → set of endpoint sheet names (uppercased)
    isn_by_sheets: dict[frozenset[str], list[str]] = {}
    for net in architecture.inter_sheet_nets:
        endpoint_sheets = frozenset(ep.sheet.upper() for ep in net.endpoints)
        isn_by_sheets.setdefault(endpoint_sheets, []).append(net.name)

    bad: list[str] = []
    for conn in functional_spec.connections:
        # Power/ground nets are global (power symbols in leaves, not sheet pins)
        if conn.signal_type in ("power", "ground"):
            continue
        from_sheets = _sheets_for_block(conn.from_block)
        to_sheets = _sheets_for_block(conn.to_block)
        # If both blocks map to the same sheet, the connection is intra-sheet
        if from_sheets and to_sheets and (from_sheets & to_sheets):
            continue
        # If either block is unmapped, we can't verify — skip (advisory only)
        if not from_sheets or not to_sheets:
            continue
        # Cross-sheet: check if any inter_sheet_net connects these sheets.
        # Subset, not equality: a shared bus declared once across 3+ sheets
        # (e.g. I2C_SDA over MCU/SENSOR/DISPLAY) covers every pairwise
        # functional-spec connection between its endpoints; requiring an exact
        # 2-sheet match would bounce that correct architecture forever.
        cross_pairs = {frozenset({f, t}) for f in from_sheets for t in to_sheets if f != t}
        covered = any(
            pair <= endpoint_sheets for pair in cross_pairs for endpoint_sheets in isn_by_sheets
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
            "every cross-sheet connection is declared in inter_sheet_nets"
            if not bad
            else f"{len(bad)} connection(s) not mapped to inter_sheet_nets"
        ),
        offenders=bad,
    )
