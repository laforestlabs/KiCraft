"""Typed wiring patch contract, constraints, and pure patch application."""
from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from kicraft.design import models
from .stage_contracts import WiringSlotResponse, _json_response_format

class _AddEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_endpoint"]
    ref: str
    pin: str
    sheet: str
    net: str
    expected_net: None = Field(...)


class _RemoveEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["remove_endpoint"]
    ref: str
    pin: str
    expected_net: str


class _SetPinNet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["set_pin_net"]
    ref: str
    pin: str
    sheet: str
    expected_net: str | None = Field(...)
    net: str


class _AddConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_connection"]
    connection: models.NetConnection
    expected_absent: Literal[True]


class _RemoveConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["remove_connection"]
    sheet: str
    net: str
    expected_endpoints: list[models.PinEndpoint]


class _MarkNoConnect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["mark_no_connect"]
    endpoint: models.PinEndpoint
    expected_net: None = Field(...)
    expected_no_connect: Literal[False]


class _UnmarkNoConnect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["unmark_no_connect"]
    endpoint: models.PinEndpoint
    expected_no_connect: Literal[True]


WiringPatchOperation = Annotated[
    (
        _AddEndpoint
        | _RemoveEndpoint
        | _SetPinNet
        | _AddConnection
        | _RemoveConnection
        | _MarkNoConnect
        | _UnmarkNoConnect
    ),
    Field(discriminator="op"),
]
_WIRING_PATCH_OPERATION = TypeAdapter(WiringPatchOperation)


class WiringPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operations: list[WiringPatchOperation] = Field(min_length=1, max_length=24)

def _endpoint_key(endpoint: dict | models.PinEndpoint) -> tuple[str, str]:
    if isinstance(endpoint, models.PinEndpoint):
        return endpoint.ref, endpoint.pin
    return str(endpoint["ref"]), str(endpoint["pin"])


def _canonical_wiring(candidate: dict) -> dict:
    connections = []
    for connection in candidate.get("connections") or []:
        row = dict(connection)
        row["endpoints"] = sorted(
            [dict(endpoint) for endpoint in row.get("endpoints") or []],
            key=lambda endpoint: (endpoint["ref"], endpoint["pin"]),
        )
        connections.append(row)
    return {
        "connections": sorted(
            connections, key=lambda connection: (connection["sheet"], connection["net_name"])
        ),
        "no_connect_pins": sorted(
            [dict(endpoint) for endpoint in candidate.get("no_connect_pins") or []],
            key=lambda endpoint: (endpoint["ref"], endpoint["pin"]),
        ),
    }


def apply_wiring_patch(
    candidate: dict,
    patch_payload: dict,
    *,
    allowed_refs: set[str],
    allowed_pins: dict[str, set[str]] | None = None,
    allowed_nets: set[str] | None = None,
) -> tuple[dict, int]:
    """Apply a typed patch to a copy; stale preconditions fail without mutation."""
    current = WiringSlotResponse.model_validate(candidate).model_dump(exclude_none=True)
    patch = WiringPatchResponse.model_validate(patch_payload)
    assignments: dict[tuple[str, str], set[tuple[str, str]]] = {}
    connection_by_key: dict[tuple[str, str], dict] = {}
    for connection in current["connections"]:
        ckey = (connection["sheet"], connection["net_name"])
        if ckey in connection_by_key:
            raise ValueError(f"duplicate connection {ckey}")
        connection_by_key[ckey] = connection
        for endpoint in connection["endpoints"]:
            key = _endpoint_key(endpoint)
            assignments.setdefault(key, set()).add(ckey)
    no_connect = {_endpoint_key(endpoint) for endpoint in current["no_connect_pins"]}

    def validate_endpoint(key: tuple[str, str]) -> None:
        ref, pin = key
        if ref not in allowed_refs:
            raise ValueError(f"unknown reference {ref!r}")
        if allowed_pins is not None and pin not in allowed_pins.get(ref, set()):
            raise ValueError(f"unknown pin {ref}.{pin}")

    def validate_net(net: str) -> None:
        if allowed_nets is not None and net not in allowed_nets:
            raise ValueError(f"unknown net {net!r}")

    for raw_operation in patch.operations:
        operation = _WIRING_PATCH_OPERATION.validate_python(raw_operation)
        op = operation.op
        if op in {"add_endpoint", "remove_endpoint", "set_pin_net"}:
            key = (operation.ref, operation.pin)
            validate_endpoint(key)
            actual = assignments.get(key, set())
            expected_net = operation.expected_net
            matching = {ckey for ckey in actual if ckey[1] == expected_net}
            if expected_net is None:
                precondition_ok = not actual
            else:
                precondition_ok = len(matching) == 1 and (
                    op == "remove_endpoint" or len(actual) == 1
                )
            if not precondition_ok:
                found = sorted(ckey[1] for ckey in actual)
                raise ValueError(
                    f"stale precondition for {key}: expected {expected_net!r}, found {found!r}"
                )
            if op in {"remove_endpoint", "set_pin_net"}:
                actual_connection = next(iter(matching))
                connection = connection_by_key[actual_connection]
                connection["endpoints"] = [
                    endpoint
                    for endpoint in connection["endpoints"]
                    if _endpoint_key(endpoint) != key
                ]
                assignments[key].remove(actual_connection)
                if not assignments[key]:
                    del assignments[key]
                if not connection["endpoints"]:
                    current["connections"].remove(connection)
                    del connection_by_key[actual_connection]
            if op in {"add_endpoint", "set_pin_net"}:
                if key in no_connect:
                    raise ValueError(f"endpoint {key} is marked no-connect")
                validate_net(operation.net)
                ckey = (operation.sheet, operation.net)
                connection = connection_by_key.get(ckey)
                if connection is None:
                    raise ValueError(f"unknown connection {ckey}; add it explicitly first")
                connection["endpoints"].append({"ref": key[0], "pin": key[1]})
                assignments.setdefault(key, set()).add(ckey)
        elif op == "add_connection":
            connection = operation.connection.model_dump(exclude_none=True)
            ckey = (connection["sheet"], connection["net_name"])
            validate_net(connection["net_name"])
            if ckey in connection_by_key:
                raise ValueError(f"connection {ckey} already exists")
            for endpoint in connection["endpoints"]:
                key = _endpoint_key(endpoint)
                validate_endpoint(key)
                if assignments.get(key) or key in no_connect:
                    raise ValueError(f"endpoint {key} is already assigned")
            current["connections"].append(connection)
            connection_by_key[ckey] = connection
            for endpoint in connection["endpoints"]:
                assignments.setdefault(_endpoint_key(endpoint), set()).add(ckey)
        elif op == "remove_connection":
            ckey = (operation.sheet, operation.net)
            connection = connection_by_key.get(ckey)
            if connection is None:
                raise ValueError(f"connection {ckey} does not exist")
            expected = {_endpoint_key(endpoint) for endpoint in operation.expected_endpoints}
            actual_endpoints = {_endpoint_key(endpoint) for endpoint in connection["endpoints"]}
            if expected != actual_endpoints:
                raise ValueError(f"stale endpoint set for connection {ckey}")
            current["connections"].remove(connection)
            del connection_by_key[ckey]
            for key in actual_endpoints:
                assignments[key].discard(ckey)
                if not assignments[key]:
                    del assignments[key]
        elif op == "mark_no_connect":
            key = _endpoint_key(operation.endpoint)
            validate_endpoint(key)
            if assignments.get(key) or key in no_connect:
                raise ValueError(f"stale no-connect precondition for {key}")
            current["no_connect_pins"].append({"ref": key[0], "pin": key[1]})
            no_connect.add(key)
        elif op == "unmark_no_connect":
            key = _endpoint_key(operation.endpoint)
            validate_endpoint(key)
            if key not in no_connect:
                raise ValueError(f"stale no-connect precondition for {key}")
            current["no_connect_pins"] = [
                endpoint
                for endpoint in current["no_connect_pins"]
                if _endpoint_key(endpoint) != key
            ]
            no_connect.remove(key)
    duplicates = {key: sorted(value) for key, value in assignments.items() if len(value) > 1}
    if duplicates:
        raise ValueError(f"patch left duplicate endpoint assignments: {duplicates}")
    overlap = no_connect & assignments.keys()
    if overlap:
        raise ValueError(f"patch left net/no-connect overlap: {sorted(overlap)}")

    canonical = _canonical_wiring(current)
    WiringSlotResponse.model_validate(canonical)
    return canonical, len(patch.operations)

def wiring_patch_response_format() -> dict:
    return _json_response_format("kicraft_wiring_patch_v1", WiringPatchResponse.model_json_schema())


class PatchApplicationError(ValueError):
    pass

def wiring_patch_constraints(
    prompt_state: dict, extras: dict, candidate: dict, rejection: dict
) -> tuple[set[str], dict[str, set[str]], set[str]]:
    parts = (prompt_state.get("bom") or {}).get("parts") or []
    allowed_refs = {str(part.get("ref")) for part in parts if part.get("ref")}
    pinouts = extras.get("symbol_pinouts") or {}
    allowed_pins: dict[str, set[str]] = {}
    for part in parts:
        ref, symbol = str(part.get("ref") or ""), str(part.get("symbol") or "")
        pins = (pinouts.get(symbol) or {}).get("pins") or []
        allowed_pins[ref] = {
            str(pin.get("number")) for pin in pins if isinstance(pin, dict) and pin.get("number")
        }
    architecture = prompt_state.get("architecture") or {}
    allowed_nets = {str(net) for net in architecture.get("power_nets") or []}
    for net in architecture.get("inter_sheet_nets") or []:
        if isinstance(net, dict) and net.get("name"):
            allowed_nets.add(str(net["name"]))
    allowed_nets.update(
        str(connection.get("net_name"))
        for connection in candidate.get("connections") or []
        if connection.get("net_name")
    )
    feedback = json.dumps(rejection, ensure_ascii=False)
    allowed_nets.update(
        match.group(1)
        for match in re.finditer(r"\bnet\s+['\"]([^'\"]+)['\"]", feedback, re.IGNORECASE)
    )
    return allowed_refs, allowed_pins, allowed_nets


def wiring_patch_messages(
    candidate: dict, rejection: dict, prompt_state: dict, extras: dict, *, clean_slate: bool
) -> list[dict]:
    offender_text = json.dumps(
        {
            "errors": rejection.get("errors") or [],
            "offenders": rejection.get("offenders") or [],
        },
        ensure_ascii=False,
    )
    known_refs = {
        str(part.get("ref"))
        for part in ((prompt_state.get("bom") or {}).get("parts") or [])
        if part.get("ref")
    }
    refs = {
        match.group(1).upper()
        for match in re.finditer(r"\b([A-Za-z]+[0-9]+[A-Za-z0-9_-]*)\b", offender_text)
        if match.group(1).upper() in known_refs
    }
    named_nets = {
        match.group(1)
        for match in re.finditer(r"\bnet\s+['\"]([^'\"]+)['\"]", offender_text, re.IGNORECASE)
    }
    parts = [
        part
        for part in ((prompt_state.get("bom") or {}).get("parts") or [])
        if part.get("ref") in refs
    ]
    symbols = {part.get("symbol") for part in parts}
    pinouts = {
        symbol: info
        for symbol, info in (extras.get("symbol_pinouts") or {}).items()
        if symbol in symbols
    }
    relevant_connections = [
        connection
        for connection in candidate.get("connections") or []
        if connection.get("net_name") in named_nets
        or any(endpoint.get("ref") in refs for endpoint in connection.get("endpoints") or [])
    ]
    context = {
        "rejection": {
            "errors": rejection.get("errors") or [],
            "offenders": rejection.get("offenders") or [],
        },
        "offender_parts": parts,
        "offender_pinouts": pinouts,
        "valid_offender_refs": sorted(refs),
        "architecture_nets": {
            "power_nets": (prompt_state.get("architecture") or {}).get("power_nets") or [],
            "inter_sheet_nets": (
                (prompt_state.get("architecture") or {}).get("inter_sheet_nets") or []
            ),
        },
        "current_connections_touching_offenders": relevant_connections,
        "existing_connection_keys": sorted(
            [
                [connection.get("sheet"), connection.get("net_name")]
                for connection in candidate.get("connections") or []
            ]
        ),
        "current_no_connects_touching_offenders": [
            endpoint
            for endpoint in candidate.get("no_connect_pins") or []
            if endpoint.get("ref") in refs
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Repair only rejected endpoints with the typed operations in the schema. "
                "Use add_endpoint/remove_endpoint/set_pin_net for an existing connection. "
                "Use add_connection only when its [sheet,net] key is absent, and remove it "
                "before re-adding the same key. Architecture sheet endpoint names are not "
                "component refs; only valid_offender_refs may appear as endpoint.ref. "
                "Every operation must state its expected current value. Never change an "
                "unrelated endpoint or invent a net."
            ),
        },
        {
            "role": "user",
            "content": (
                ("This is the one clean-slate escape correction. " if clean_slate else "")
                + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            ),
        },
    ]
