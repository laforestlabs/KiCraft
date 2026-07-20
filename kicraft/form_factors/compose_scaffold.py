"""Compose-side scaffold: fixed outline + locked connector components.

The compose half of "replace & rewire". Given a validated standard template, this
builds the placement primitives compose needs: the exact board outline and a set
of connector :class:`Component`s frozen (``locked=True``) at their fixed board
positions. The parent solver then auto-places every leaf/local around them, and
the outline is pinned to the standard rect instead of grown from content.

Kept out of ``compose_subcircuits`` so it is unit-testable in isolation and so
the (autoplacer-type) imports don't leak into the pure ``scaffold`` module.
``resolve_scaffold`` is the gate: it returns a scaffold ONLY when the project
cfg both requests enforcement and carries a *validated* standard -- so the
compose fork stays dormant everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point

from . import FormFactorTemplate, get_template

# Pad copper size for a 2.54 mm THT header pin (matches the stock PinSocket).
_PAD_MM = 1.7
_PAD_MARGIN_MM = 0.9  # courtyard slack folded into the locked component bbox


@dataclass(slots=True)
class FormFactorScaffold:
    template: FormFactorTemplate
    components: dict[str, Component]

    @property
    def width_mm(self) -> float:
        return self.template.board_width_mm

    @property
    def height_mm(self) -> float:
        return self.template.board_height_mm

    @property
    def outline(self) -> tuple[Point, Point]:
        return (Point(0.0, 0.0), Point(self.width_mm, self.height_mm))


def _stamp_rotation_deg(conn) -> float:
    """Footprint rotation that makes the KiCad single-row header stamp with its
    pins along the template's pin axis.

    A KiCad ``PinHeader/PinSocket_1xNN_..._Vertical`` footprint at rotation 0 has
    its pins advancing +Y (top-left frame). The template states each header's pin
    axis in the BOARD frame: ``axis="x"`` (Arduino's horizontal edge headers)
    needs a +90 deg turn to send the pins +X; ``axis="y"`` is already aligned.
    The stamp positions the real seed footprint by ref at this rotation, so its
    pads land on the template pin coordinates. Any extra per-connector turn the
    datum specifies is added on top.
    """
    base = 90.0 if conn.axis == "x" else 0.0
    return (base + conn.rotation_deg) % 360.0


def build_scaffold(
    template: FormFactorTemplate,
    *,
    ref_start: int = 1,
    role_to_ref: dict[str, str] | None = None,
) -> FormFactorScaffold:
    """Locked connector components at the template's fixed positions.

    Board-local top-left frame (the template's frame == compose's outline/stamp
    frame). ``pos`` is the connector's pin-1 centre and ``rotation`` orients the
    real footprint so its pads land on the template pins (see
    :func:`_stamp_rotation_deg`); the pads carried here are already in board
    coordinates for the solver's net/keepout reasoning. ``NC`` pins get an empty
    net.

    Refs: when ``role_to_ref`` is given (the synthesis half's actual BOM refs,
    keyed by connector role) those refs are used so the scaffold locks the SAME
    parts the schematic emitted; otherwise refs default to ``J{ref_start..}`` in
    connector order.
    """
    comps: dict[str, Component] = {}
    ref_n = ref_start
    for conn in template.fixed_connectors:
        pin_xy = conn.pin_positions()  # [(net, x, y), ...]
        xs = [x for _n, x, _y in pin_xy]
        ys = [y for _n, _x, y in pin_xy]
        half = _PAD_MM / 2.0 + _PAD_MARGIN_MM
        width_mm = (max(xs) - min(xs)) + 2 * half
        height_mm = (max(ys) - min(ys)) + 2 * half
        ref = (role_to_ref or {}).get(conn.role) or f"J{ref_n}"
        pads = [
            Pad(
                ref=ref,
                pad_id=str(i),
                pos=Point(x, y),
                net="" if net == "NC" else net,
                layer=Layer.FRONT,
                size_mm=Point(_PAD_MM, _PAD_MM),
            )
            for i, (net, x, y) in enumerate(pin_xy, start=1)
        ]
        comps[ref] = Component(
            ref=ref,
            value=f"{template.display_name} {conn.role}",
            pos=Point(conn.x_mm, conn.y_mm),
            rotation=_stamp_rotation_deg(conn),
            layer=Layer.FRONT,
            width_mm=width_mm,
            height_mm=height_mm,
            # pos is PIN 1, not the header centre -- without body_center,
            # Component.bbox() falls back to pos and the solver reasons about
            # a courtyard box up to ~11mm off the real header (Arduino
            # digital_high: centre x=30.2 vs pin-1 x=18.8). Every other
            # locked-obstacle construction site sets this (2026-07-19 §3.9).
            body_center=Point(
                (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            ),
            pads=pads,
            locked=True,
            kind="connector",
        )
        ref_n += 1
    return FormFactorScaffold(template=template, components=comps)


def resolve_scaffold(cfg: dict, *, ref_start: int = 1) -> FormFactorScaffold | None:
    """The gate. Return a scaffold only when the project cfg BOTH sets
    ``form_factor_enforce`` and carries a validated ``form_factor_standard``
    block (or a resolvable validated template key). Otherwise ``None`` -- the
    compose fork is a no-op, so nothing changes for non-shield boards or until
    the enforcement flag is turned on together with the synthesis half.

    The scaffold's refs come from the cfg's ``form_factor_standard.header_refs``
    (role -> BOM ref, emitted by the synthesis half) so compose locks the exact
    parts the reconcile added; absent that, refs default to ``J{ref_start..}``.
    """
    if not cfg.get("form_factor_enforce"):
        return None
    ffs = cfg.get("form_factor_standard")
    key = ffs.get("key") if isinstance(ffs, dict) else (ffs if isinstance(ffs, str) else None)
    if isinstance(ffs, dict) and not ffs.get("validated"):
        return None
    template = get_template(key)
    if template is None or not template.validated:
        return None
    role_to_ref = ffs.get("header_refs") if isinstance(ffs, dict) else None
    return build_scaffold(template, ref_start=ref_start, role_to_ref=role_to_ref)


__all__ = ["FormFactorScaffold", "build_scaffold", "resolve_scaffold"]
