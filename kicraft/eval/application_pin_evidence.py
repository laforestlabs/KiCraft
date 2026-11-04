"""Package contacts with reviewed application capabilities, independent of net labels."""
from __future__ import annotations

from collections import defaultdict
import re

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.recipes import registered_recipes


def reviewed_application_pins(capability: str) -> dict[str, set[str]]:
    """Return physical pin numbers, excluding reserved and input-only outputs."""
    result: dict[str, set[str]] = defaultdict(set)
    for registered in registered_recipes():
        definition = registered.definition
        if definition.maturity != "production":
            continue
        records = {
            group.role: physical_inventory_record(
                mpn=group.mpn, symbol=group.symbol, footprint=group.footprint
            )
            for group in definition.parts
        }
        for pin in definition.allocatable_pins:
            if (
                capability not in pin.capabilities
                or pin.reserved
                or (capability == "output" and pin.input_only)
            ):
                continue
            record = records.get(pin.role)
            if record is not None:
                result[record.identity].add(pin.pin)
        # Some reviewed cores publish fixed GPIO ports rather than allocatable
        # pins. Only their explicit output-capable public ports qualify.
        if capability == "output":
            outputs = {
                port.name for port in definition.ports
                if re.fullmatch(r"gpio\d+", port.name, re.I)
                and port.direction in {"output", "bidirectional"}
            }
            for pin in definition.pins:
                record = records.get(pin.role)
                if (
                    pin.net in outputs and record is not None
                    and "microcontroller" in record.physical_features
                ):
                    result[record.identity].add(pin.pin)
    return dict(result)
