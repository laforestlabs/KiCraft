"""Pure helpers for invalidating stale design-stage state."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

DESIGN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


def downstream_stages(stage: str) -> tuple[str, ...]:
    """Return the canonical design stages strictly after ``stage``."""
    if stage not in DESIGN_STAGES:
        return ()
    return DESIGN_STAGES[DESIGN_STAGES.index(stage) + 1 :]


def _retain_compiler_wiring(bom: MutableMapping[str, Any]) -> None:
    """Keep only pins published by this new BOM's compiler ownership.

    Ownership is a claim here, not an exemption from validation: the commit
    gates still reconstruct and verify every retained graph before saving it.
    Authored wiring from the preceding BOM is invalidated.
    """
    claims = list(bom.get("lowerer_ownership") or [])
    for manifest in bom.get("recipe_ownership") or []:
        if isinstance(manifest, Mapping):
            claims.extend(manifest.get("pins") or [])
    owned = {
        (row.get("ref"), str(row["pin"]), row.get("net"))
        for row in claims
        if isinstance(row, Mapping) and row.get("ref") and row.get("pin") is not None
    }
    connections = []
    for connection in bom.get("connections") or []:
        if not isinstance(connection, Mapping):
            continue
        endpoints = [
            endpoint for endpoint in connection.get("endpoints") or []
            if isinstance(endpoint, Mapping)
            and (endpoint.get("ref"), str(endpoint.get("pin")), connection.get("net_name")) in owned
        ]
        if endpoints:
            connections.append({**connection, "endpoints": endpoints})
    bom["connections"] = connections
    bom["no_connect_pins"] = [
        endpoint for endpoint in bom.get("no_connect_pins") or []
        if isinstance(endpoint, Mapping)
        and (endpoint.get("ref"), str(endpoint.get("pin")), None) in owned
    ]


def invalidate_downstream(
    state: MutableMapping[str, Any], stage: str
) -> tuple[str, ...]:
    """Remove every state field made stale by accepting ``stage``."""
    invalidated = downstream_stages(stage)
    for downstream in invalidated:
        if downstream == "wiring":
            bom = state.get("bom")
            if isinstance(bom, MutableMapping):
                if stage == "bom":
                    _retain_compiler_wiring(bom)
                else:
                    bom["connections"] = []
                    bom["no_connect_pins"] = []
        else:
            state[downstream] = None

        statuses = state.get("stage_status")
        if isinstance(statuses, MutableMapping):
            statuses.pop(downstream, None)

    state["open_questions"] = [
        question
        for question in (state.get("open_questions") or [])
        if not isinstance(question, MutableMapping)
        or question.get("stage") not in invalidated
    ]
    return invalidated
