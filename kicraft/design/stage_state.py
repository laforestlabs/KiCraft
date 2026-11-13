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


def _compiler_owned_pins(bom: Mapping[str, Any]) -> set[tuple]:
    claims = list(bom.get("lowerer_ownership") or [])
    for manifest in bom.get("recipe_ownership") or []:
        if isinstance(manifest, Mapping):
            claims.extend(manifest.get("pins") or [])
    return {
        (row.get("ref"), str(row["pin"]), row.get("net"))
        for row in claims
        if isinstance(row, Mapping) and row.get("ref") and row.get("pin") is not None
    }


def legacy_recipe_resume_stage(state: Mapping[str, Any]) -> str | None:
    """Earliest stage that must be regenerated for an obsolete recipe graph.

    Inspect raw state before model parsing: an ambiguous saved selection cannot
    pass the architecture validator until that architecture has been replaced.
    """
    from .recipe_versions import (
        is_legacy_recipe_id,
        legacy_recipe_requires_new_architecture,
    )

    architecture = state.get("architecture")
    selections = architecture.get("recipe_selections") or [] if isinstance(architecture, Mapping) else []
    legacy_selections = [
        selection for selection in selections
        if isinstance(selection, Mapping) and is_legacy_recipe_id(selection.get("recipe"))
    ]
    if legacy_selections:
        from .recipes import migrate_recipe_selection

        for selection in legacy_selections:
            try:
                migrate_recipe_selection(selection)
            except (TypeError, ValueError):
                return "architecture"
    bom = state.get("bom")
    if not isinstance(bom, Mapping):
        return None
    owned_ids = [
        row.get("recipe") for row in bom.get("recipe_ownership") or []
        if isinstance(row, Mapping)
    ] + [
        row.get("recipe_id") for row in bom.get("parts") or []
        if isinstance(row, Mapping)
    ]
    if any(legacy_recipe_requires_new_architecture(value) for value in owned_ids):
        return "architecture"
    if legacy_selections or any(
        isinstance(manifest, Mapping) and is_legacy_recipe_id(manifest.get("recipe"))
        for manifest in bom.get("recipe_ownership") or []
    ) or any(
        isinstance(part, Mapping) and is_legacy_recipe_id(part.get("recipe_id"))
        for part in bom.get("parts") or []
    ):
        return "bom"
    return None


def stage_done(stage: str, state: Mapping[str, Any]) -> bool:
    """Return whether a stage has a durable committed contribution.

    New states explicitly mark wiring complete or invalidated: a BOM can
    contain compiler-owned connections before the wiring stage runs. A legacy
    successful status is trusted; without one, an entirely compiler-owned graph
    is ambiguous and must be wired again rather than falsely reported complete.
    """
    legacy_stage = legacy_recipe_resume_stage(state) if stage in {"architecture", "bom", "wiring"} else None
    if legacy_stage == "architecture" or (legacy_stage == "bom" and stage != "architecture"):
        return False
    if stage != "wiring":
        return state.get(stage) is not None
    bom = state.get("bom")
    if not isinstance(bom, Mapping):
        return False
    has_wiring = bool(bom.get("connections")) or bool(bom.get("no_connect_pins"))
    status = (state.get("stage_status") or {}).get("wiring")
    if isinstance(status, Mapping) and status.get("ok") is False:
        return False
    marker = state.get("wiring_complete")
    if marker is not None:
        return marker is True and has_wiring
    if not has_wiring or (isinstance(status, Mapping) and status.get("ok") is True):
        return has_wiring
    owned = _compiler_owned_pins(bom)
    if owned:
        assigned = set()
        for connection in bom.get("connections") or []:
            if not isinstance(connection, Mapping):
                continue
            assigned.update(
                (endpoint.get("ref"), str(endpoint.get("pin")), connection.get("net_name"))
                for endpoint in connection.get("endpoints") or []
                if isinstance(endpoint, Mapping)
            )
        assigned.update(
            (endpoint.get("ref"), str(endpoint.get("pin")), None)
            for endpoint in bom.get("no_connect_pins") or []
            if isinstance(endpoint, Mapping)
        )
        if assigned and assigned <= owned:
            return False
    return has_wiring


def _retain_compiler_wiring(bom: MutableMapping[str, Any]) -> None:
    """Keep only pins published by this new BOM's compiler ownership.

    Ownership is a claim here, not an exemption from validation: the commit
    gates still reconstruct and verify every retained graph before saving it.
    Authored wiring from the preceding BOM is invalidated.
    """
    owned = _compiler_owned_pins(bom)
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
            state["wiring_complete"] = False
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
