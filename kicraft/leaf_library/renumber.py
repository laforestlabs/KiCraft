"""Refdes renumber algorithm for installing a leaf into a project.

A leaf carries its own refs (``U1``, ``C1``, ``C2``, ...) that conflict
with whatever the host project already uses. On import we generate a
``{leaf_ref -> project_ref}`` map so each leaf component lands in the
next free slot of its letter class. Multi-instance imports advance the
allocator's per-class counter so two copies of the same leaf get two
non-overlapping ranges.
"""

from __future__ import annotations

import re
from collections import defaultdict

_REF_RE = re.compile(r"^([A-Z]+)([0-9]+)$")


def parse_ref(ref: str) -> tuple[str, int]:
    """Split ``"U7"`` into ``("U", 7)``.

    Raises ``ValueError`` on suffix forms (``"U1A"``) or other shapes
    not allowed in the leaf manifest. Used as a sort key by
    :func:`renumber_leaf` so the allocation order is deterministic.
    """
    m = _REF_RE.match(ref)
    if not m:
        raise ValueError(
            f"ref {ref!r} must match ^[A-Z]+[0-9]+$ (no suffix forms in v1)"
        )
    return m.group(1), int(m.group(2))


def renumber_leaf(
    leaf_refs: list[str],
    project_refs: list[str],
) -> dict[str, str]:
    """Return ``{leaf_ref -> project_ref}`` allocating from the next free
    slot per letter class.

    ``project_refs`` is the union of every ref already present in the
    project (schematic + PCB + autoplacer.json) before this import.
    For multi-instance reuse, the caller invokes this function once per
    instance, including prior instances' allocated refs in
    ``project_refs`` for the next call.

    The allocation order is deterministic across runs: leaf refs are
    sorted by ``(class, number)`` before being assigned slots.
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for r in project_refs:
        try:
            cls, num = parse_ref(r)
        except ValueError:
            # Some project refs may be #PWR or other power-flag forms
            # not following the strict pattern; ignore for slot purposes.
            continue
        by_class[cls].append(num)

    next_in_class: dict[str, int] = {
        cls: max(nums) + 1 for cls, nums in by_class.items()
    }

    ref_map: dict[str, str] = {}
    for leaf_ref in sorted(leaf_refs, key=parse_ref):
        cls, _ = parse_ref(leaf_ref)
        n = next_in_class.get(cls, 1)
        ref_map[leaf_ref] = f"{cls}{n}"
        next_in_class[cls] = n + 1
    return ref_map


def apply_ref_map(
    container: object, ref_map: dict[str, str]
) -> object:
    """Apply ``ref_map`` to every ref-shaped key/value inside a nested
    container of dicts / lists / strings.

    Used to renumber an in-memory ``autoplacer_fragment.json`` payload
    (ic_groups, group_labels, thermal_refs, signal_flow_order,
    component_zones, etc.) before merging into the project's autoplacer
    JSON. Strings that don't match ``^[A-Z]+[0-9]+$`` (e.g. sheet names
    or labels) pass through untouched.
    """
    if isinstance(container, dict):
        out_dict: dict[object, object] = {}
        for k, v in container.items():
            new_k = ref_map.get(k, k) if isinstance(k, str) else k
            out_dict[new_k] = apply_ref_map(v, ref_map)
        return out_dict
    if isinstance(container, list):
        return [apply_ref_map(item, ref_map) for item in container]
    if isinstance(container, str):
        return ref_map.get(container, container)
    return container


__all__ = [
    "apply_ref_map",
    "parse_ref",
    "renumber_leaf",
]
