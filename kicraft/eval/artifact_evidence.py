"""Extract acceptance facts from persisted KiCad and stage artifacts.

This is evaluation-only.  It never reads a claimed acceptance status from a run:
all fact values come from state/BOM, the saved board, or named output files.
Unknown electrical/physical relations deliberately remain unverified.
"""
from __future__ import annotations

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.symbol_pinout import SymbolNotFoundError, lookup_pins
from .electrical_artifact_evidence import extract_electrical_facts
from .geometry_artifact_evidence import extract_geometry_facts
import json
from collections.abc import Mapping
from collections import Counter
from pathlib import Path
import zipfile
from typing import Any

from .acceptance_contracts import ORIGINAL_CORPUS_VERSION, contract_for
from .design_acceptance import _check_facts, _count, new_acceptance_evidence


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _existing(root: Path, *paths: Path) -> list[dict[str, str]]:
    return [{"kind": "artifact", "path": _relative(root, path)} for path in paths if path.is_file()]




# Every class name declared by an acceptance contract must be derivable from a
# reviewed part's physical features, and the counter below always records every
# known class (0 when absent) so an exact-zero obligation is a real observation
# rather than a missing key.
_FEATURE_TO_CLASS = {
    "bnc-connector": "bnc_connector", "trim-potentiometer": "trim_potentiometer",
    "usb-c-receptacle": "usb_c_receptacle", "coin-cell-holder": "coin_cell_holder",
    "audio-jack-3-5mm": "audio_jack_3p5mm", "fpc-connector": "fpc_0p5mm_24",
    "air-core-inductor": "air_core_inductor", "binding-post": "binding_post",
    "buck-converter": "reviewed_5v_3a_buck",
    "microcontroller": "microcontroller", "led-0805": "led_0805",
    "screw-terminal": "screw_terminal",
    "speaker-crossover-capacitor": "film_capacitor",
    "momentary-button": "button", "spst-switch": "button",
    "sma-connector": "sma_connector", "qwiic-i2c-connector": "qwiic_connector",
    "i2c-oled-display": "smt_i2c_oled",
    "rotary-encoder": "through_hole_rotary_encoder",
    "stacking-header": "arduino_stacking_header",
    "edge-connector": "edge_connector",
    "adc": "ADS1115", "motor-driver": "drv8833", "relay": "through_hole_relay",
    "smt-3v3-regulator": "smt_3v3_regulator", "usb-a-receptacle": "usb_a_connector",
    "attiny402": "ATtiny402", "attiny1614": "ATtiny1614",
}

# "No active part is fitted" is only observable when every reviewed active
# device feature contributes to that counter.  These are additional classes
# derived from a feature; a feature keeps its own class too.
_ACTIVE_FEATURES = frozenset({
    "rs485-transceiver", "can-transceiver", "thermocouple-converter",
    "darlington-array", "pwm-driver", "io-expander", "quad-operational-amplifier",
    "optocoupler", "digital-isolator", "current-limited-power-switch",
    "isolated-dc-dc-converter", "dual-output-dc-dc-converter", "p-channel-mosfet",
    "highside-switch", "constant-current-led-driver", "voltage-regulator",
    "wifi-module", "bluetooth-le-soc", "environmental-sensor", "microcontroller",
    "buck-converter", "adc", "motor-driver",
})


def _physical_record(part: dict[str, Any]):
    return physical_inventory_record(
        mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"),
        datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"),
    )


def _classify_part(part: dict[str, Any]) -> set[str]:
    record = _physical_record(part)
    if record is None:
        return set()
    classes = {_FEATURE_TO_CLASS[feature] for feature in record.physical_features if feature in _FEATURE_TO_CLASS}
    if record.physical_features & _ACTIVE_FEATURES:
        classes.add("active_device")
    return classes


def _identity(part: dict[str, Any]) -> str | None:
    record = _physical_record(part)
    return record.family.upper() if record is not None else None



def _standard_pin_contacts(part: dict[str, Any]) -> tuple[str, ...]:
    """Resolve a symbol/footprint contract independently from stock policy."""
    try:
        pins = lookup_pins(str(part.get("symbol") or ""), all_units=True).get("pins") or []
    except (SymbolNotFoundError, ValueError, OSError, KeyError):
        return ()
    return tuple(str(pin["number"]) for pin in pins if isinstance(pin, Mapping) and pin.get("number") is not None)

def _state_facts(state: dict[str, Any]) -> dict[str, Any]:
    bom = state.get("bom") if isinstance(state.get("bom"), dict) else {}
    parts = [part for part in bom.get("parts") or [] if isinstance(part, dict)]
    return {
        "part_classes": {},
        "bom_present": isinstance(state.get("bom"), dict) and isinstance(bom.get("parts"), list),
        "bom_parts": parts,
        "all_bom_parts": parts,
        "part_identities": [],
        "part_inventory": [],
        "net_paths": {},
        "channel_counts": {},
        "geometry_counts": {},
        "gates": {},
        "artifacts": [],
        "assumptions": (state.get("architecture") or {}).get("assumptions") or [],
    }


def _board_facts(board_path: Path, facts: dict[str, Any]) -> Any | None:
    """Add pad/net/geometry observations from the saved KiCad board when readable."""
    try:
        import pcbnew  # KiCad is optional for unbuilt/partial runs.
        board = pcbnew.LoadBoard(str(board_path))
    except Exception:
        return None
    if board is None:
        return None
    board_footprints: dict[str, str] = {}
    pads: list[dict[str, str]] = []
    net_members: dict[str, list[str]] = {}
    mounting_holes = 0
    for fp in board.GetFootprints():
        ref = fp.GetReferenceAsString()
        # ``str(GetFPID())`` yields a SWIG proxy repr under KiCad 9, and a saved
        # board may carry no library nickname, so record "nickname:item" only
        # when a nickname is present and the bare item name otherwise.  Reviewed
        # pairs are compared item-exact with the nickname required only when the
        # board actually carries one (see :func:`_reviewed_footprint_agrees`).
        try:
            footprint_item = str(fp.GetFPID().GetLibItemName())
            footprint_library = str(fp.GetFPID().GetLibNickname())
        except Exception:
            footprint_item, footprint_library = "", ""
        board_footprints[ref] = (
            f"{footprint_library}:{footprint_item}" if footprint_library and footprint_item
            else footprint_item or str(fp.GetFPID())
        )
        for pad in fp.Pads():
            number = pad.GetNumber()
            net = pad.GetNetname()
            pad_id = f"{ref}.{number}"
            if number:
                pads.append({"symbol_pin": pad_id, "footprint_pad": pad_id})
            if net and number:
                net_members.setdefault(net, []).append(pad_id)
            drill = pad.GetDrillSize()
            fpid = str(fp.GetFPID().GetLibItemName()).lower()
            if ref.upper().startswith("H") or "mountinghole" in fpid or "mounting_hole" in fpid:
                if drill.x > 0 or drill.y > 0:
                    mounting_holes += 1
    facts["board_pad_observations"] = pads
    facts["board_footprints"] = board_footprints
    facts["board_net_members"] = net_members
    facts["board_net_member_counts"] = {net: len(members) for net, members in net_members.items()}
    facts["geometry_counts"]["mounting_hole"] = mounting_holes
    # pin_mapping obligation remains unverified until a cross-library extractor
    # produces the actual symbol-to-pad relation.
    return board


def _bom_parts_digest(parts: list[dict[str, Any]]) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reviewed_footprint_agrees(reviewed: str, recorded: str) -> bool:
    """Whether a recorded board footprint is the reviewed library pair.

    The library item name must agree exactly; the nickname must agree only when
    the board carries one, because a saved board can drop it.  A prefix or family
    match is never enough: the reviewed pair is the land pattern the record's
    symbol and contacts were reviewed against.
    """
    reviewed_library, _, reviewed_item = str(reviewed or "").partition(":")
    recorded_library, separator, recorded_item = str(recorded or "").partition(":")
    if not separator:  # the board carried no library nickname
        recorded_library, recorded_item = "", recorded_library
    if not reviewed_item or not recorded_item:
        return False
    return recorded_item == reviewed_item and (not recorded_library or recorded_library == reviewed_library)


def _reconcile_inventory(facts: dict[str, Any]) -> None:
    """Reconcile board pads with semantic records and stock-independent pin maps."""
    board_footprints = facts.get("board_footprints")
    if not isinstance(board_footprints, dict):
        return
    classes: Counter[str] = Counter({name: 0 for name in set(_FEATURE_TO_CLASS.values()) | {"active_device"}})
    identities: list[dict[str, str]] = []
    inventory: list[dict[str, Any]] = []
    unclassified: list[str] = []
    pin_mapping: list[dict[str, str]] = []
    unmapped: list[str] = []
    source_unverified: list[str] = []
    receipt = facts.get("sourcing_validation")
    receipt_rows = receipt.get("parts") if isinstance(receipt, Mapping) else []
    receipt_by_ref = {
        str(row.get("ref")): row for row in receipt_rows or []
        if isinstance(row, Mapping)
    }
    receipt_digest_ok = (
        isinstance(receipt, Mapping)
        and receipt.get("schema_version") == 1
        and receipt.get("parts_digest") == _bom_parts_digest(
            [part for part in facts.get("all_bom_parts", []) if isinstance(part, dict)]
        )
    )
    for part in facts.pop("bom_parts", []):
        if not isinstance(part, dict):
            continue
        record = _physical_record(part)
        ref = str(part.get("ref") or "")
        contacts = record.contacts if record is not None else _standard_pin_contacts(part)
        expected_footprint = record.footprint if record is not None else str(part.get("footprint") or "")
        if not contacts or not _reviewed_footprint_agrees(expected_footprint, str(board_footprints.get(ref) or "")):
            unclassified.append(ref or "<unreferenced>")
            continue
        if record is not None:
            classes.update(_classify_part(part))
            identities.append({
                "identity": record.identity,
                "package": record.package,
                "footprint": record.footprint,
            })
        actual_pads = {
            item["footprint_pad"].rsplit(".", 1)[-1]
            for item in facts.get("board_pad_observations", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("footprint_pad"), str)
            and item["footprint_pad"].startswith(f"{ref}.")
        }
        if not set(contacts) <= actual_pads:
            unmapped.append(ref)
        else:
            pin_mapping.extend(
                {"symbol_pin": f"{ref}.{contact}", "footprint_pad": f"{ref}.{contact}"}
                for contact in contacts
            )
        row = receipt_by_ref.get(ref) if receipt_digest_ok else None
        if row is None:
            source_unverified.append(ref)
        elif row.get("verdict") == "pass":
            inventory.append({
                "ref": ref, "mpn": row.get("catalog_mpn"), "manufacturer": row.get("manufacturer"),
                "source_url": f"https://www.lcsc.com/product-detail/{row.get('exact_lcsc')}",
                "package": row.get("package"), "rated_limits": {"assembly_stock": row.get("assembly_stock"), "retail_stock": row.get("retail_stock")},
            })
        elif row.get("verdict") != "excluded_not_orderable":
            source_unverified.append(ref)
    facts["part_classes"] = dict(classes)
    facts["part_inventory"] = inventory
    facts["part_identities"] = identities
    facts["unclassified_bom_refs"] = unclassified
    facts["pin_mapping"] = pin_mapping
    facts["unmapped_bom_refs"] = unmapped
    facts["unqualified_source_refs"] = source_unverified

def _merge_fact_delta(facts: dict[str, Any], delta: Mapping[str, Any]) -> None:
    """Merge independently extracted facts without replacing sibling namespaces."""
    for key, value in delta.items():
        current = facts.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_fact_delta(current, value)
        else:
            facts[key] = value


def _verified_export_paths(rundir: Path, archives: list[Path]) -> list[str]:
    """Accept only run-local archives containing Gerbers and a drill export."""
    verified: list[str] = []
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as contents:
                names = [name.lower() for name in contents.namelist()]
        except (OSError, zipfile.BadZipFile):
            continue
        if (
            any(name.endswith(".gbrjob") for name in names)
            and any(name.endswith((".drl", ".xln")) for name in names)
            and any(name.endswith((".gtl", ".gbl", ".gbr")) for name in names)
        ):
            verified.append(_relative(rundir, archive))
    return verified


def _specialized_fact_deltas(rundir: Path, state: dict[str, Any], board: Any, contract: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Extract electrical and geometry facts from one already-loaded board."""
    return (
        extract_electrical_facts(rundir, state, board, contract),
        extract_geometry_facts(rundir, state, board, contract),
    )


_GEOMETRY_GATE_KEYS = frozenset({
    "geometry", "castellations", "thermal_geometry", "uno_shield_geometry",
    "prototyping_area", "servo_headers_edge", "audio_jacks_edge",
    "ws2812b_even_circle", "star_point_led_placement", "rf_antenna_geometry",
})
_GEOMETRY_KINDS = frozenset({"outline", "geometry_count"})


def _applicable_gates(contract: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, bool]:
    """Which build gates this delivered design actually owes.

    Programming applies only when programmable hardware is present in the
    delivered BOM (never merely because the contract mentions the gate).
    Geometry applies when the contract declares a geometry-sensitive obligation,
    because that is the brief asking for measured delivered geometry.
    """
    obligations = [ob for ob in (contract.get("obligations") or []) if isinstance(ob, Mapping)]
    checks = [ob.get("check") or {} for ob in obligations]
    # Only a *declared* gate obligation counts; the common
    # ``<gate>-when-applicable`` entry is the applicability mechanism itself and
    # must not make every brief owe that gate.
    gates = {check.get("gate") for check in checks if check.get("kind") == "gate"}
    microcontrollers = (facts.get("part_classes") or {}).get("microcontroller")
    return {
        "programming": isinstance(microcontrollers, int) and microcontrollers > 0,
        "geometry": bool({check.get("kind") for check in checks} & _GEOMETRY_KINDS)
        or bool(gates & _GEOMETRY_GATE_KEYS),
    }


def _json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _manufacturing_verdicts(facts: dict[str, Any], rundir: Path, artifacts: list[dict[str, str]]) -> None:
    """Publish the build's structured manufacturing verdicts, never prose.

    ERC comes from the synthesis check summary, DRC from the routed-board
    verification gate.  Both are optional: when a build wrote no such file the
    verdict stays absent, which the gate reads as unverified rather than passing.
    """
    gate_path = rundir / ".kicraft" / "build_gate.json"
    gate = _json_artifact(gate_path)
    if gate is not None:
        facts["build_gate"] = gate
        artifacts.extend(_existing(rundir, gate_path))
    check_path = rundir / ".kicraft" / "synthesis_check.json"
    check = _json_artifact(check_path)
    if check is not None:
        facts["synthesis_check"] = check
    if gate is None and check is None:
        return
    gates: dict[str, str] = {}
    erc = [
        row for row in (check or {}).get("checks") or []
        if isinstance(row, Mapping) and str(row.get("name", "")).casefold().endswith("erc")
    ]
    if erc:
        gates["erc"] = "pass" if all(row.get("ok") is True for row in erc) else "fail"
    if gate is not None:
        blocking = ("shorts", "unconnected", "courtyard", "keepout")
        gates["drc"] = (
            "pass"
            if gate.get("fab_acceptable") is True
            and all((gate.get(key) or 0) == 0 for key in blocking)
            else "fail"
        )
    # The contract's gate checks read `facts["gates"]`, so the manufacturing
    # verdicts join the geometry/electrical gates the extractors publish.
    facts.setdefault("gates", {}).update(gates)


def extract_artifact_facts(rundir: Path, contract: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return facts plus paths to artifacts from which they were extracted."""
    state_path = rundir / ".kicraft" / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state = state if isinstance(state, dict) else {}
    facts = _state_facts(state)
    facts["board_loaded"] = False
    artifacts = _existing(rundir, state_path)
    receipt_path = rundir / ".kicraft" / "sourcing_validation.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        receipt = None
    if isinstance(receipt, dict):
        facts["sourcing_validation"] = receipt
        artifacts.extend(_existing(rundir, receipt_path))
    generated = rundir / "generated"
    boards = [path for path in generated.glob("*/*.kicad_pcb") if ".experiments" not in path.parts] if generated.is_dir() else []
    board = None
    if boards:
        board = _board_facts(boards[0], facts)
        artifacts.extend(_existing(rundir, boards[0]))
    # Board-backed obligations are only evaluable when the saved board actually
    # loaded; recording the flag here is what lets them pass or fail honestly.
    facts["board_loaded"] = board is not None
    if board is not None and contract is not None:
        for delta in _specialized_fact_deltas(rundir, state, board, contract):
            if not isinstance(delta, Mapping):
                raise TypeError("specialized artifact extractor must return a mapping")
            _merge_fact_delta(facts, delta)
    _reconcile_inventory(facts)
    archives = [path for path in generated.glob("*/*.zip") if path.is_file()] if generated.is_dir() else []
    artifacts.extend(_existing(rundir, *archives))
    facts["fab_archive_paths"] = [_relative(rundir, path) for path in archives]
    facts["verified_export_paths"] = _verified_export_paths(rundir, archives)
    if contract is not None:
        facts["applicable_gates"] = _applicable_gates(contract, facts)
    _manufacturing_verdicts(facts, rundir, artifacts)
    facts.pop("bom_parts", None)
    facts["artifacts"] = facts["verified_export_paths"]
    return facts, artifacts


def _result_status(check: dict[str, Any], facts: dict[str, Any]) -> str:
    """Only direct extracted facts produce a pass/fail; absent coverage is unverified."""
    kind = check.get("kind")
    if kind == "part_class_exact" and check.get("value") == 0 and facts.get("unclassified_bom_refs"):
        return "unverified"
    if kind in {"part_class_count", "part_class_exact", "part_classes", "part_identity", "part_identities"}:
        if not facts.get("bom_present") or facts.get("board_loaded") is not True:
            return "unverified"
        return "pass" if _check_facts(check, facts) is None else "fail"
    if kind == "geometry_count":
        if facts.get("board_loaded") is not True:
            return "unverified"
    elif kind == "gate":
        gates = facts.get("gates")
        if not isinstance(gates, Mapping) or check.get("gate") not in gates:
            return "unverified"
    elif kind == "net_paths":
        paths = facts.get("net_paths")
        required = check.get("paths") or []
        if not isinstance(paths, Mapping) or any(path not in paths for path in required):
            return "unverified"
    elif kind == "connector_map":
        maps = facts.get("connector_maps")
        if not isinstance(maps, Mapping) or check.get("connector") not in maps:
            return "unverified"
    elif kind == "channel_count":
        if _count(facts, "channel_counts", str(check.get("channel"))) is None:
            return "unverified"
    elif kind == "numeric_range":
        if check.get("fact") not in facts:
            return "unverified"
    elif kind == "outline":
        if not isinstance(facts.get("outline"), Mapping):
            return "unverified"
    elif kind == "pin_mapping":
        if not facts.get("pin_mapping") or facts.get("unclassified_bom_refs") or facts.get("unmapped_bom_refs"):
            return "unverified"
    elif kind == "part_inventory":
        if not facts.get("part_inventory") or facts.get("unclassified_bom_refs") or facts.get("unqualified_source_refs"):
            return "unverified"
    elif kind == "artifacts":
        if not facts.get("verified_export_paths"):
            return "unverified"
    elif kind == "set_members":
        if check.get("fact") not in facts:
            return "unverified"
    elif kind == "applicable_gate":
        applicable = facts.get("applicable_gates")
        gate = check.get("gate")
        if not isinstance(applicable, Mapping) or not isinstance(applicable.get(gate), bool):
            return "unverified"
        # A gate this design does not owe is satisfied by construction; an owed
        # gate is only pass/fail when its report was actually produced.
        if not applicable[gate]:
            return "pass"
        gates = facts.get("gates")
        if not isinstance(gates, Mapping) or gate not in gates:
            return "unverified"
    else:
        return "unverified"
    return "pass" if _check_facts(check, facts) is None else "fail"


def generate_artifact_evidence(rundir: Path, slug: str, *, contract_version: str = ORIGINAL_CORPUS_VERSION, build_rc: int | None = None, design_committed: bool = False) -> dict[str, Any]:
    """Generate an honest evidence record from artifacts already on disk.

    ``build_rc`` participates in the fabrication verdict: only an rc0 build whose
    archived exports verify and whose structured ERC/DRC verdicts pass is
    reported as fabricated.
    """
    contract = contract_for(slug, contract_version)
    facts, artifacts = extract_artifact_facts(rundir, contract)
    evidence = new_acceptance_evidence(slug, contract_version=contract_version)
    evidence["artifact_facts"] = facts
    for obligation in contract["obligations"]:
        evidence["obligations"][obligation["id"]] = {
            "status": _result_status(obligation["check"], facts),
            "evidence": artifacts,
        }
    evidence["generation"] = {"status": "pass" if design_committed else "fail", "evidence": artifacts}
    build_gates = {
        gate: verdict for gate, verdict in (facts.get("gates") or {}).items()
        if gate in {"erc", "drc"}
    }
    evidence["common_gates"] = dict(build_gates)
    fab_ok = (
        build_rc == 0
        and bool(facts.get("verified_export_paths"))
        and all(build_gates.get(gate) == "pass" for gate in ("erc", "drc"))
    )
    evidence["fabrication"] = {
        "status": "pass" if fab_ok else "unverified",
        "artifact_paths": list(facts.get("verified_export_paths") or []),
        "reason": (
            "rc0 build with a verified Gerber/drill archive and passing ERC/DRC verdicts."
            if fab_ok
            else "Board/archive presence is not an ERC/DRC/connectivity/manufacturing verdict."
        ),
    }
    # Software fulfillment is exactly the plan's definition: a fabricated board
    # whose every mandatory obligation was positively verified by these facts.
    fulfilled = fab_ok and all(
        item["status"] == "pass" for item in evidence["obligations"].values()
    )
    evidence["software_fulfillment"] = {
        "status": "pass" if fulfilled else "unverified",
        "reason": (
            "fabrication passed and every mandatory obligation is positively verified."
            if fulfilled
            else "; ".join(
                f"{key}: {item['status']}"
                for key, item in evidence["obligations"].items()
                if item["status"] != "pass"
            ) or "fabrication is not positively verified"
        ),
    }
    evidence["physical_validation"] = {"status": "unverified"}
    return evidence
