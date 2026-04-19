"""Configurable acceptance gate system for leaf subcircuits.

This module consolidates acceptance logic that was previously scattered across
``solve_subcircuits.py`` and ``freerouting_runner.py``.  Each gate evaluates a
single aspect of the routed leaf result and produces a pass/fail verdict with
structured detail.  The top-level :func:`evaluate_leaf_acceptance` function
runs every gate in sequence and returns a single :class:`LeafAcceptanceResult`
that callers can inspect programmatically.

Pure Python -- no pcbnew dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LeafAcceptanceConfig",
    "LeafAcceptanceResult",
    "evaluate_leaf_acceptance",
    "acceptance_config_from_dict",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LeafAcceptanceConfig:
    """Knobs that control which gates must pass for a leaf to be accepted.

    Each flag/threshold corresponds to one of the gates evaluated by
    :func:`evaluate_leaf_acceptance`.  The defaults match the current
    *lenient* behaviour where only fatal problems (shorts, missing board,
    Python exceptions) cause rejection.

    Attributes
    ----------
    require_drc_clean:
        When ``True``, *any* DRC violation (including clearance) causes
        rejection.  When ``False`` (default), clearance violations are
        evaluated separately via *max_clearance_violations*.
    allow_footprint_internal_clearance:
        When ``True`` (default), clearance violations that are flagged as
        footprint-internal (e.g. dense USB-C pads) are subtracted from the
        clearance count before the *max_clearance_violations* gate runs.
    require_anchor_completeness:
        When ``True``, every required interface port must have an anchor.
        When ``False`` (default), missing anchors produce a note but do not
        block acceptance.
    max_shorts:
        Maximum number of DRC shorts allowed.  The default is ``0`` (any
        short causes rejection).
    max_clearance_violations:
        If set, reject when the (adjusted) clearance violation count exceeds
        this threshold.  ``None`` (default) means clearance violations alone
        never cause rejection.
    require_routed_board:
        When ``True`` (default), the routed board file must exist on disk.
    require_no_python_exception:
        When ``True`` (default), any Python exception during routing or
        validation causes rejection.
    """

    require_drc_clean: bool = False
    allow_footprint_internal_clearance: bool = True
    require_anchor_completeness: bool = False
    max_shorts: int = 0
    max_clearance_violations: int | None = None
    require_routed_board: bool = True
    require_no_python_exception: bool = True


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LeafAcceptanceResult:
    """Structured verdict from :func:`evaluate_leaf_acceptance`.

    Attributes
    ----------
    accepted:
        ``True`` when every evaluated gate passed.
    rejection_reasons:
        Human-readable strings describing each failed gate.  Empty when
        *accepted* is ``True``.
    gate_results:
        Per-gate detail keyed by gate name.  Each value is a dict with at
        least ``{"passed": bool}`` and optional extra context.
    drc_summary:
        Condensed DRC numbers extracted from the validation dict.
    anchor_summary:
        Condensed anchor/port numbers extracted from the anchor validation
        dict.
    notes:
        Informational messages that do not affect the verdict (e.g.
        "footprint-internal clearance violations ignored").
    """

    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    gate_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    drc_summary: dict[str, Any] = field(default_factory=dict)
    anchor_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------
# Each ``_gate_*`` function receives the full context and returns
# ``(passed: bool, detail: dict)``.


def _gate_board_exists(
    validation: dict[str, Any],
    _anchor: dict[str, Any],
    _cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: the routed board file must exist on disk."""
    exists = bool(validation.get("board_exists", False))
    return exists, {
        "passed": exists,
        "board_path": validation.get("board_path", ""),
    }


def _gate_no_python_exception(
    validation: dict[str, Any],
    _anchor: dict[str, Any],
    cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: no Python exception was raised during routing/validation."""
    had_exception = bool(validation.get("python_exception", False))
    if not cfg.require_no_python_exception:
        return True, {
            "passed": True,
            "python_exception": had_exception,
            "skipped": True,
            "reason": "require_no_python_exception is False",
        }
    passed = not had_exception
    return passed, {"passed": passed, "python_exception": had_exception}


def _gate_no_shorts(
    validation: dict[str, Any],
    _anchor: dict[str, Any],
    cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: DRC shorts must not exceed *max_shorts*."""
    drc = validation.get("drc", {})
    shorts = int(drc.get("shorts", 0))
    passed = shorts <= cfg.max_shorts
    return passed, {
        "passed": passed,
        "shorts": shorts,
        "max_shorts": cfg.max_shorts,
    }


def _gate_no_illegal_geometry(
    validation: dict[str, Any],
    _anchor: dict[str, Any],
    _cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: no malformed or obviously illegal routed geometry."""
    malformed = bool(validation.get("malformed_board_geometry", False))
    illegal = bool(validation.get("obviously_illegal_routed_geometry", False))
    passed = not malformed and not illegal
    return passed, {
        "passed": passed,
        "malformed_board_geometry": malformed,
        "obviously_illegal_routed_geometry": illegal,
    }


def _gate_drc_clearance(
    validation: dict[str, Any],
    _anchor: dict[str, Any],
    cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: DRC clearance violations within configured tolerance.

    When *require_drc_clean* is ``True``, any violation fails.
    Otherwise, the clearance count is optionally reduced by footprint-
    internal violations and compared against *max_clearance_violations*.
    """
    drc = validation.get("drc", {})
    raw_clearance = int(drc.get("clearance", 0))
    footprint_internal = int(validation.get("footprint_internal_clearance_count", 0))

    adjusted_clearance = raw_clearance
    if cfg.allow_footprint_internal_clearance:
        adjusted_clearance = max(0, raw_clearance - footprint_internal)

    detail: dict[str, Any] = {
        "raw_clearance_violations": raw_clearance,
        "footprint_internal_clearance_count": footprint_internal,
        "adjusted_clearance_violations": adjusted_clearance,
        "allow_footprint_internal_clearance": cfg.allow_footprint_internal_clearance,
    }

    if cfg.require_drc_clean:
        passed = adjusted_clearance == 0
        detail["mode"] = "strict"
        detail["passed"] = passed
        return passed, detail

    if cfg.max_clearance_violations is not None:
        passed = adjusted_clearance <= cfg.max_clearance_violations
        detail["mode"] = "threshold"
        detail["max_clearance_violations"] = cfg.max_clearance_violations
        detail["passed"] = passed
        return passed, detail

    # No clearance constraint configured -- always pass.
    detail["mode"] = "unconstrained"
    detail["passed"] = True
    return True, detail


def _gate_anchor_completeness(
    _validation: dict[str, Any],
    anchor_validation: dict[str, Any],
    cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: all required interface ports have anchors.

    Uses the *anchor_validation* dict produced by
    ``subcircuit_artifacts.build_anchor_validation``.
    """
    all_required = bool(
        anchor_validation.get(
            "all_required_ports_anchored",
            anchor_validation.get("all_required_anchored", True),
        )
    )
    missing = list(
        anchor_validation.get(
            "missing_required_ports",
            anchor_validation.get("missing_required", []),
        )
    )

    detail: dict[str, Any] = {
        "all_required_ports_anchored": all_required,
        "missing_required_ports": missing,
    }

    if not cfg.require_anchor_completeness:
        detail["passed"] = True
        detail["skipped"] = True
        detail["reason"] = "require_anchor_completeness is False"
        return True, detail

    passed = all_required
    detail["passed"] = passed
    return passed, detail


def _gate_routed_board(
    validation: dict[str, Any],
    _anchor: dict[str, Any],
    cfg: LeafAcceptanceConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate: a routed board artifact must exist.

    This is subtly different from *board_exists* -- it checks whether the
    board file is present and usable.  When *require_routed_board* is
    ``False`` the gate is skipped entirely.
    """
    if not cfg.require_routed_board:
        return True, {
            "passed": True,
            "skipped": True,
            "reason": "require_routed_board is False",
        }

    track_summary = validation.get("track_summary", {})
    traces = int(track_summary.get("traces", 0))
    vias = int(track_summary.get("vias", 0))
    board_exists = bool(validation.get("board_exists", False))

    # A board with zero traces AND zero vias may be acceptable for
    # subcircuits that have no internal nets (pure passthrough).  So we
    # only hard-fail if the board file itself is missing.
    passed = board_exists
    return passed, {
        "passed": passed,
        "board_exists": board_exists,
        "traces": traces,
        "vias": vias,
    }


# Ordered list of (gate_name, gate_function) pairs.  Order determines
# evaluation sequence.  All gates are *always* evaluated so that
# ``gate_results`` is complete.
_GATES: list[tuple[str, Any]] = [
    ("board_exists", _gate_board_exists),
    ("no_python_exception", _gate_no_python_exception),
    ("no_shorts", _gate_no_shorts),
    ("no_illegal_geometry", _gate_no_illegal_geometry),
    ("drc_clearance", _gate_drc_clearance),
    ("anchor_completeness", _gate_anchor_completeness),
    ("routed_board", _gate_routed_board),
]


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_leaf_acceptance(
    validation: dict[str, Any],
    anchor_validation: dict[str, Any],
    config: LeafAcceptanceConfig | None = None,
) -> LeafAcceptanceResult:
    """Run every acceptance gate and return a structured verdict.

    Parameters
    ----------
    validation:
        The dict returned by
        ``freerouting_runner.validate_routed_board``.
    anchor_validation:
        The dict returned by
        ``subcircuit_artifacts.build_anchor_validation``.
    config:
        Optional configuration overrides.  When ``None``, the default
        (lenient) :class:`LeafAcceptanceConfig` is used.

    Returns
    -------
    LeafAcceptanceResult
        Fully populated result with per-gate detail.
    """
    if config is None:
        config = LeafAcceptanceConfig()

    result = LeafAcceptanceResult()
    all_passed = True

    for gate_name, gate_fn in _GATES:
        passed, detail = gate_fn(validation, anchor_validation, config)
        result.gate_results[gate_name] = detail
        if not passed:
            all_passed = False
            result.rejection_reasons.append(gate_name)

    # -- Build condensed summaries -------------------------------------------
    drc = validation.get("drc", {})
    result.drc_summary = {
        "shorts": int(drc.get("shorts", 0)),
        "clearance": int(drc.get("clearance", 0)),
        "footprint_internal_clearance_count": int(
            validation.get("footprint_internal_clearance_count", 0)
        ),
        "timed_out": bool(drc.get("timed_out", False)),
        "missing_cli": bool(drc.get("missing_cli", False)),
    }

    result.anchor_summary = {
        "required_port_count": int(
            anchor_validation.get("required_port_count", 0)
        ),
        "anchored_count": int(
            anchor_validation.get(
                "anchor_count",
                anchor_validation.get("anchored_count", 0),
            )
        ),
        "all_required_anchored": bool(
            anchor_validation.get(
                "all_required_ports_anchored",
                anchor_validation.get("all_required_anchored", True),
            )
        ),
        "missing_required": list(
            anchor_validation.get(
                "missing_required_ports",
                anchor_validation.get("missing_required", []),
            )
        ),
    }

    # -- Informational notes -------------------------------------------------
    fp_internal = int(validation.get("footprint_internal_clearance_count", 0))
    if fp_internal > 0 and config.allow_footprint_internal_clearance:
        result.notes.append(
            f"Ignored {fp_internal} footprint-internal clearance violation(s)"
        )

    if (
        not result.anchor_summary["all_required_anchored"]
        and not config.require_anchor_completeness
    ):
        missing = result.anchor_summary["missing_required"]
        result.notes.append(
            f"Missing required anchors (not enforced): {missing}"
        )

    track_summary = validation.get("track_summary", {})
    traces = int(track_summary.get("traces", 0))
    vias = int(track_summary.get("vias", 0))
    if traces == 0 and vias == 0 and validation.get("board_exists", False):
        result.notes.append("Board exists but contains no routed copper")

    result.accepted = all_passed
    return result


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def acceptance_config_from_dict(cfg: dict[str, Any]) -> LeafAcceptanceConfig:
    """Build a :class:`LeafAcceptanceConfig` from a flat project-config dict.

    Recognised keys (all optional -- missing keys fall back to the dataclass
    defaults):

    - ``leaf_acceptance_require_drc_clean`` (bool)
    - ``leaf_acceptance_allow_footprint_internal_clearance`` (bool)
    - ``leaf_acceptance_require_anchor_completeness`` (bool)
    - ``leaf_acceptance_max_shorts`` (int)
    - ``leaf_acceptance_max_clearance_violations`` (int or None)
    - ``leaf_acceptance_require_routed_board`` (bool)
    - ``leaf_acceptance_require_no_python_exception`` (bool)

    Parameters
    ----------
    cfg:
        A flat dict, typically loaded from a JSON project config file.

    Returns
    -------
    LeafAcceptanceConfig
    """
    kwargs: dict[str, Any] = {}

    _BOOL_KEYS = [
        (
            "leaf_acceptance_require_drc_clean",
            "require_drc_clean",
        ),
        (
            "leaf_acceptance_allow_footprint_internal_clearance",
            "allow_footprint_internal_clearance",
        ),
        (
            "leaf_acceptance_require_anchor_completeness",
            "require_anchor_completeness",
        ),
        (
            "leaf_acceptance_require_routed_board",
            "require_routed_board",
        ),
        (
            "leaf_acceptance_require_no_python_exception",
            "require_no_python_exception",
        ),
    ]
    for cfg_key, attr_name in _BOOL_KEYS:
        if cfg_key in cfg:
            kwargs[attr_name] = bool(cfg[cfg_key])

    if "leaf_acceptance_max_shorts" in cfg:
        kwargs["max_shorts"] = int(cfg["leaf_acceptance_max_shorts"])

    if "leaf_acceptance_max_clearance_violations" in cfg:
        raw = cfg["leaf_acceptance_max_clearance_violations"]
        kwargs["max_clearance_violations"] = None if raw is None else int(raw)

    return LeafAcceptanceConfig(**kwargs)
