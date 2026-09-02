"""Pure, versioned semantic diagnostics for schema-valid stage candidates."""

from __future__ import annotations

import re
from collections.abc import Iterable

from kicraft.design.models import StageDiagnostic
from kicraft.design.synthesis.validation import named_part_tokens

DETECTOR_VERSION = 1

_EXPLICIT_FACT_RE = re.compile(
    r"\b(?:qfn|bga|lqfp|tqfp|soic|usb(?:-c)?|i2c|spi|qspi|uart|gpio|swd|jtag|"
    r"castellat(?:ed|ion)|through[- ]hole|surface[- ]mount|\d+(?:\.\d+)?\s*"
    r"(?:v|mv|a|ma|hz|khz|mhz|ghz|mm|mil|pins?|channels?|pieces?|pcs?))\b",
    re.IGNORECASE,
)
_TOPOLOGY_RE = re.compile(r"\b(ldo|buck|boost|flyback|charge pump|esd protection)\b", re.I)
_NONFUNCTIONAL_RE = re.compile(
    r"\b(?:ground|gnd|rail|mounting holes?|decoupling|crystal|castellated pads?)\b", re.I
)
_BIDIRECTIONAL_RE = re.compile(r"(?:^|[_\s-])(usb|gpio|i2c|qspi)(?:$|[_\s-])", re.I)
_SUPPORT_RE = re.compile(r"\b(crystal|clock|decoupl|pull[- ]?up|castellat|passive)\b", re.I)
_POWER_RE = re.compile(r"\b(power|vbus|vcc|vdd|3v3|5v|1v1|ldo|regulat)\b", re.I)


def _diag(code: str, severity: str, message: str, evidence: Iterable[str] = (), *, attempt=None):
    return StageDiagnostic(
        code=code,
        severity=severity,
        message=message,
        evidence=sorted({str(item).strip().lower() for item in evidence if str(item).strip()}),
        detector_version=DETECTOR_VERSION,
        attempt=attempt,
    )


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _text(value) -> str:
    if isinstance(value, dict):
        return " ".join(_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    return str(value or "")


def _intent(brief: str, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    expected = named_part_tokens([brief])
    supplied = {_norm_token(part) for part in candidate.get("named_parts") or []}
    omitted = [token for token in expected.values() if _norm_token(token) not in supplied]
    if omitted:
        diagnostics.append(
            _diag(
                "intent_named_part_omitted",
                "advisory",
                "Explicit part or family tokens from the brief were not classified.",
                omitted,
            )
        )
    facts = [m.group(0) for m in _EXPLICIT_FACT_RE.finditer(brief)]
    if not candidate.get("constraints") and facts:
        severity = "repair_required" if len({_norm_token(f) for f in facts}) >= 2 else "advisory"
        diagnostics.append(
            _diag(
                "intent_constraints_empty",
                severity,
                "The brief contains explicit constraints but constraints is empty.",
                facts,
            )
        )
    goal = re.sub(r"\s+", " ", str(candidate.get("goal") or "")).strip().lower()
    source = re.sub(r"\s+", " ", brief).strip().lower()
    copied = bool(source and (goal == source or (len(source) >= 40 and goal in source)))
    if (
        copied
        and (facts or expected)
        and not candidate.get("constraints")
        and not candidate.get("named_parts")
    ):
        diagnostics.append(
            _diag(
                "intent_unclassified_copy",
                "advisory",
                "The goal copies the brief without classifying explicit content.",
            )
        )
    return diagnostics


def _functional_spec(brief: str, upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    allowed = _text([brief, upstream.get("intent", {})]).lower()
    assumptions = " ".join(candidate.get("assumptions") or []).lower()
    for block in candidate.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        block_text = _text(block)
        introduced = sorted({m.group(1).lower() for m in _TOPOLOGY_RE.finditer(block_text)})
        premature = [term for term in introduced if term not in allowed]
        if premature:
            diagnostics.append(
                _diag(
                    "functional_spec_premature_topology",
                    "repair_required",
                    "Functional specification introduced an unrequested implementation topology.",
                    premature,
                )
            )
        if _NONFUNCTIONAL_RE.search(str(block.get("name", ""))) and not re.search(
            r"\b(interface|process|power conversion|sensor|actuat)", block_text, re.I
        ):
            diagnostics.append(
                _diag(
                    "functional_spec_nonfunctional_block",
                    "repair_required",
                    "A component-level support item or net was emitted as a functional block.",
                    [block.get("name", "")],
                )
            )
        purpose = str(block.get("purpose") or "").lower()
        additions = [
            term
            for term in ("esd protection", "ldo", "buck", "boost")
            if term in purpose and term not in allowed
        ]
        if additions and not all(term in assumptions for term in additions):
            diagnostics.append(
                _diag(
                    "functional_spec_unrecorded_assumption",
                    "repair_required",
                    "An introduced default was not recorded in assumptions.",
                    additions,
                )
            )
    return diagnostics


def _architecture(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    fs = upstream.get("functional_spec") or {}
    blocks = {str(b.get("name")): b for b in fs.get("blocks") or [] if isinstance(b, dict)}
    sheets = candidate.get("sheets") or []
    for sheet in sheets:
        if not isinstance(sheet, dict):
            continue
        name = str(sheet.get("name") or "")
        function = str(sheet.get("function") or "")
        block = blocks.get(name.replace(" ", "_"))
        if (
            block
            and block.get("category") == "power"
            and re.search(r"^(?:POWER|GROUND|[0-9.]+V)", name)
        ) or (
            _POWER_RE.search(name)
            and not re.search(r"regulat|convert|supply|input", function, re.I)
        ):
            diagnostics.append(
                _diag(
                    "architecture_power_block_as_sheet",
                    "repair_required",
                    "A power net or distribution-only block was emitted as a physical sheet.",
                    [name],
                )
            )
        if _SUPPORT_RE.search(f"{name} {function}"):
            diagnostics.append(
                _diag(
                    "architecture_fragmented_physical_domain",
                    "repair_required",
                    "A trivial support or board feature was split from the IC domain it supports.",
                    [name],
                )
            )
    for net in candidate.get("inter_sheet_nets") or []:
        if not isinstance(net, dict):
            continue
        name = str(net.get("name") or "")
        endpoints = net.get("endpoints") or []
        directions = {str(ep.get("direction") or "") for ep in endpoints if isinstance(ep, dict)}
        if _BIDIRECTIONAL_RE.search(name) and directions - {"bidirectional", "passive"}:
            diagnostics.append(
                _diag(
                    "architecture_wrong_signal_direction",
                    "repair_required",
                    "A known bidirectional protocol or GPIO net was declared one-way.",
                    [name],
                )
            )
        if _POWER_RE.search(name):
            endpoint_names = {
                str(ep.get("sheet") or "") for ep in endpoints if isinstance(ep, dict)
            }
            expected = {
                str(sheet.get("name"))
                for sheet in sheets
                if isinstance(sheet, dict)
                and _POWER_RE.search(f"{sheet.get('name', '')} {sheet.get('function', '')}")
            }
            missing = sorted(expected - endpoint_names)
            if missing:
                diagnostics.append(
                    _diag(
                        "architecture_missing_power_endpoint",
                        "repair_required",
                        "A power producer or consumer is absent from the rail endpoints.",
                        [name, *missing],
                    )
                )
    if candidate.get("mcu_present") and not re.search(
        r"\b(?:swd|jtag|updi|icsp|bootsel|program|debug)\b", _text(candidate), re.I
    ):
        diagnostics.append(
            _diag(
                "architecture_programming_decision_incomplete",
                "repair_required",
                "MCU architecture lacks an explicit programming or recovery choice.",
            )
        )
    return diagnostics


def _bom(candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    placeholders = [
        str(p.get("ref"))
        for p in candidate.get("parts") or []
        if isinstance(p, dict)
        and re.search(r"PinHeader_1x01|vertical.*header", _text(p), re.I)
        and re.search(r"castellat", _text(p), re.I)
    ]
    if placeholders:
        diagnostics.append(
            _diag(
                "bom_castellation_placeholder",
                "fab_gate",
                "Board-fabricated castellations were represented as assembly headers.",
                placeholders,
            )
        )
    return diagnostics


def _wiring(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    bom = dict(upstream.get("bom") or {})
    bom.update(candidate)
    parts = {str(p.get("ref")): p for p in bom.get("parts") or [] if isinstance(p, dict)}
    nets: dict[str, set[str]] = {}
    for row in bom.get("connections") or []:
        if not isinstance(row, dict):
            continue
        nets.setdefault(str(row.get("net_name") or ""), set()).update(
            str(ep.get("ref")) for ep in row.get("endpoints") or [] if isinstance(ep, dict)
        )
    bootsel_nets = [refs for name, refs in nets.items() if "bootsel" in name.lower()]
    if bootsel_nets and all(not any(ref.startswith("U") for ref in refs) for refs in bootsel_nets):
        diagnostics.append(
            _diag(
                "wiring_bootsel_unreachable",
                "fab_gate",
                "BOOTSEL switching does not reach the MCU or QSPI chip-select graph.",
            )
        )
    nc = {
        (str(ep.get("ref")), str(ep.get("pin")))
        for ep in bom.get("no_connect_pins") or []
        if isinstance(ep, dict)
    }
    testens = [
        f"{ref}.{pin}"
        for ref, pin in nc
        if "rp2040" in _text(parts.get(ref, {})).lower() and pin == "19"
    ]
    if testens:
        diagnostics.append(
            _diag(
                "wiring_special_pin_no_connect",
                "fab_gate",
                "A required family special pin was marked no-connect.",
                testens,
            )
        )
    return diagnostics


def diagnose_stage(
    stage: str, *, brief: str, upstream_state: dict, candidate: dict
) -> list[StageDiagnostic]:
    """Diagnose a schema-valid candidate without mutating it or durable state."""
    if stage == "intent":
        findings = _intent(brief, candidate)
    elif stage == "functional_spec":
        findings = _functional_spec(brief, upstream_state, candidate)
    elif stage == "architecture":
        findings = _architecture(upstream_state, candidate)
    elif stage == "bom":
        findings = _bom(candidate)
    elif stage == "wiring":
        findings = _wiring(upstream_state, candidate)
    else:
        findings = []
    return sorted(findings, key=lambda finding: (finding.severity, finding.code, finding.evidence))
