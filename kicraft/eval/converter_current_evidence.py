"""Conditional component-stage evidence for explicitly reviewed converters."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.validation import _capacitance_farads


_MODULE = "k7803m-1000r3"
_INPUT_CAP = "cl31b106kbhnnne"
_OUTPUT_CAP = "emk316bb7226ml-t"


def _identity(part: Mapping[str, Any]) -> str | None:
    record = physical_inventory_record(
        mpn=str(part.get("mpn") or "") or None,
        symbol=str(part.get("symbol") or "") or None,
        footprint=str(part.get("footprint") or "") or None,
        datasheet=str(part.get("datasheet") or "") or None,
        sourcing_note=str(part.get("sourcing_note") or "") or None,
    )
    return record.identity if record is not None else None


def _connected_cap(
    parts: list[dict[str, Any]], pad_nets: Mapping[tuple[str, str], str],
    mpn: str, net: str, ground: str,
) -> bool:
    for part in parts:
        ref = str(part.get("ref") or "")
        if (
            str(part.get("mpn") or "").casefold() == mpn
            and str(part.get("symbol") or "") == "Device:C"
            and str(part.get("footprint") or "") == "Capacitor_SMD:C_1206_3216Metric"
            and {pad_nets.get((ref, "1")), pad_nets.get((ref, "2"))} == {net, ground}
        ):
            return True
    return False


def qualified_module_output_net(
    converter: Mapping[str, Any],
    parts: list[dict[str, Any]],
    pad_nets: Mapping[tuple[str, str], str],
) -> str | None:
    """Prove the exact module and its two manufacturer-recommended bypasses."""
    if _identity(converter) != _MODULE:
        return None
    ref = str(converter.get("ref") or "")
    input_net = pad_nets.get((ref, "1"))
    ground_net = pad_nets.get((ref, "2"))
    output_net = pad_nets.get((ref, "3"))
    if not (
        all(isinstance(net, str) and net for net in (input_net, ground_net, output_net))
        and len({input_net, ground_net, output_net}) == 3
        and _connected_cap(parts, pad_nets, _INPUT_CAP, input_net, ground_net)
        and _connected_cap(parts, pad_nets, _OUTPUT_CAP, output_net, ground_net)
    ):
        return None
    capacitance = 0.0
    for part in parts:
        if part.get("symbol") not in {"Device:C", "Device:C_Small", "Device:C_Polarized", "Device:C_Polarized_Small"}:
            continue
        cap_ref = str(part.get("ref") or "")
        if {pad_nets.get((cap_ref, "1")), pad_nets.get((cap_ref, "2"))} != {output_net, ground_net}:
            continue
        value = _capacitance_farads(str(part.get("value") or ""))
        if value is None:
            return None
        capacitance += value
    if capacitance > 2200e-6:
        return None
    return output_net


def qualified_converter_power_stage_currents(
    parts: list[dict[str, Any]],
    names: Mapping[tuple[str, str], str],
    pad_nets: Mapping[tuple[str, str], str],
    observed_outputs: list[Mapping[str, Any]],
    supply_voltage: Callable[[str, str], float | None],
) -> list[dict[str, Any]]:
    """Return conditional component-stage current evidence, never board capacity.

    YLPTEC K7803M-1000R3, manufacturer datasheet pp. 1--5:
    https://jlcpcb.com/api/file/downloadByFileSystemAccessId/8603362912006254592
    The exact integrated 3.3 V converter is rated 1 A at 6--36 V; its natural
    convection graph holds 100% through 61 C, dropping to about 60% at 85 C.
    This profile caps evidence at 0.5 A and VIN at 6 <= VIN < 30 V: at 30 V
    and above the vendor requires a 33 uF low-ESR electrolytic not checked here.
    It requires the vendor-recommended nominal 10 uF/50 V input and 22 uF
    output ceramics on the delivered nets. Effective MLCC capacitance is not
    claimed; module support placement, actual voltage/transients, ambient,
    natural convection, and total output capacitance <=2200 uF remain conditions.
    Neither module transients nor input-source, connector, trace or board thermal
    capacity are inferred from this component-level envelope.

    No TPS54331/FXL current profile is emitted: FXL0630-150-M Rev. 7 pp. 14,
    27 specifies 2.6 A and 3.5 A in MAX columns and 3 A and 4 A as TYP test
    points, not guaranteed minimum heating or saturation currents. The old
    3 A XAL profile also did not describe the delivered power stage.
    """
    qualified: list[dict[str, Any]] = []
    for output in observed_outputs:
        ref = output.get("ref")
        input_net = output.get("input_net")
        output_net = output.get("output_net")
        ground_net = output.get("ground_net")
        voltage = output.get("voltage")
        if not (
            all(isinstance(item, str) and item for item in (ref, input_net, output_net, ground_net))
            and len({input_net, output_net, ground_net}) == 3
            and isinstance(voltage, (int, float)) and not isinstance(voltage, bool)
            and math.isfinite(voltage) and math.isclose(voltage, 3.3, abs_tol=0.165, rel_tol=0.0)
        ):
            continue
        matches = [part for part in parts if part.get("ref") == ref]
        if (
            len(matches) != 1
            or qualified_module_output_net(matches[0], parts, pad_nets) != output_net
        ):
            continue
        if any(pad_nets.get((ref, pin)) != net for pin, net in (
            ("1", input_net), ("2", ground_net), ("3", output_net),
        )):
            continue
        input_voltage = supply_voltage(input_net, ground_net)
        if not (
            isinstance(input_voltage, (int, float)) and not isinstance(input_voltage, bool)
            and math.isfinite(input_voltage) and 6.0 <= input_voltage < 30.0
        ):
            continue
        qualified.append({
            "ref": ref, "output_net": output_net, "voltage": float(voltage),
            "input_voltage_v": float(input_voltage), "power_stage_current_a": 0.5,
        })
    return qualified
