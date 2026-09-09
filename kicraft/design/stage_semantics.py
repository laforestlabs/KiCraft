"""Pure, versioned semantic diagnostics for schema-valid stage candidates."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable

from kicraft.design.models import Architecture, BOM, StageDiagnostic
from kicraft.design.synthesis.validation import named_part_tokens

DETECTOR_VERSION = 1

_EXPLICIT_FACT_RE = re.compile(
    r"\b(?:qfn|bga|lqfp|tqfp|soic|usb(?:-c)?|i2c|spi|qspi|uart|gpio|swd|jtag|"
    r"castellat(?:ed|ion)|through[- ]hole|surface[- ]mount|\d+(?:\.\d+)?\s*"
    r"(?:v|mv|a|ma|hz|khz|mhz|ghz|mm|mil|pins?|channels?|pieces?|pcs?))\b",
    re.IGNORECASE,
)
_TOPOLOGY_RE = re.compile(
    r"\b(ldo|buck|boost|flyback|charge pump|esd protection|direct pwm|pwm|"
    r"dac(?:-driven)?|(?:dedicated |audio |high-power )?amplif(?:ier|ied)|"
    r"analog audio|single-wire|single data-line|driven directly|ws2812(?:-style)?|"
    r"\d+\s*x\s*\d+|5v (?:power|supply) rail|5v supply to (?:the )?(?:esp32|mcu)|"
    r"powered directly)\b",
    re.I,
)
_NONFUNCTIONAL_RE = re.compile(
    r"\b(?:ground|gnd|rail|power[_ -]distribution|mounting holes?|decoupling|"
    r"crystal|castellated pads?)\b",
    re.I,
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


def complete_intent_classification(brief: str, candidate: dict) -> dict:
    """Fill omitted intent classifications from exact user text, without invention."""
    completed = dict(candidate)
    expected = named_part_tokens([brief])
    supplied = {_norm_token(part) for part in completed.get("named_parts") or []}
    missing_parts = [token for token in expected.values() if _norm_token(token) not in supplied]
    if missing_parts:
        completed["named_parts"] = [*(completed.get("named_parts") or []), *missing_parts]

    facts = [m.group(0) for m in _EXPLICIT_FACT_RE.finditer(brief)]
    if not completed.get("constraints") and (facts or expected):
        requirements = [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", brief)
            if sentence.strip()
        ]
        completed["constraints"] = requirements
    return completed


def normalize_project_stem(value: str) -> str:
    """Keep at most three complete UPPER_SNAKE_CASE words within 32 characters."""
    words = [word for word in re.split(r"[^A-Z0-9]+", str(value).upper()) if word]
    selected: list[str] = []
    for word in words[:3]:
        candidate = "_".join([*selected, word])
        if len(candidate) > 32:
            break
        selected.append(word)
    if selected:
        return "_".join(selected)
    return words[0][:32] if words else "PROJECT"


def _intent(brief: str, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    expected = named_part_tokens([brief])
    supplied = {_norm_token(part) for part in candidate.get("named_parts") or []}
    omitted = [token for token in expected.values() if _norm_token(token) not in supplied]
    if omitted:
        diagnostics.append(
            _diag(
                "intent_named_part_omitted",
                "repair_required",
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


def _mislabeled_functional_defaults(
    brief: str, upstream: dict, assumption_rows: list[str]
) -> list[str]:
    intent = upstream.get("intent", {})
    explicit_facts = {
        _norm_token(match.group(0))
        for match in _EXPLICIT_FACT_RE.finditer(_text([brief, intent.get("constraints", [])]))
    }
    named_parts = {
        _norm_token(part)
        for part in [*named_part_tokens([brief]).values(), *(intent.get("named_parts") or [])]
    }
    stopwords = {"the", "and", "for", "from", "with", "load", "loads"}
    answer_token_sets = []
    for answer in upstream.get("_stage_answers", []):
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", str(answer.get("answer", "")).lower())
            if len(token) > 2 and token not in stopwords
        }
        if tokens:
            answer_token_sets.append(tokens)

    mislabeled = []
    for assumption in assumption_rows:
        normalized = _norm_token(assumption)
        matched_facts = {fact for fact in explicit_facts if fact and fact in normalized}
        matched_part = any(part and part in normalized for part in named_parts)
        assumption_tokens = set(re.findall(r"[a-z0-9]+", assumption.lower())) - stopwords
        matched_answer = any(
            len(tokens & assumption_tokens) >= 2
            and len(tokens & assumption_tokens) * 5 >= len(tokens) * 3
            for tokens in answer_token_sets
        )
        if matched_part or matched_answer or len(matched_facts) >= 2:
            mislabeled.append(assumption)
    return mislabeled


def remove_mislabeled_functional_defaults(brief: str, upstream: dict, candidate: dict) -> dict:
    """Remove assumptions that merely relabel user requirements as defaults."""
    completed = dict(candidate)
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    mislabeled = set(_mislabeled_functional_defaults(brief, upstream, assumptions))
    if mislabeled:
        completed["assumptions"] = [
            assumption for assumption in assumptions if assumption not in mislabeled
        ]
    return completed


def remove_mislabeled_architecture_defaults(upstream: dict, candidate: dict) -> dict:
    """Remove architecture assumptions that merely repeat a stage answer."""
    answer_values = [
        _norm_token(answer.get("answer", ""))
        for answer in upstream.get("_stage_answers", [])
        if isinstance(answer, dict) and str(answer.get("answer", "")).strip()
    ]
    completed = dict(candidate)
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    completed["assumptions"] = [
        assumption
        for assumption in assumptions
        if not any(value and value in _norm_token(assumption) for value in answer_values)
    ]
    return completed


def complete_unsourced_external_rails(
    candidate: dict,
    diagnostics: list[StageDiagnostic],
) -> dict:
    """Default an otherwise unsourced low-voltage rail to a simple external input."""
    implicated = {
        str(evidence).lower()
        for diagnostic in diagnostics
        if diagnostic.code == "architecture_rail_source_unspecified"
        for evidence in diagnostic.evidence
    }
    rails = sorted(
        str(rail)
        for rail in (candidate.get("rail_voltages") or {})
        if str(rail).lower() in implicated
    )
    if not rails:
        return candidate
    completed = copy.deepcopy(candidate)
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    for rail in rails:
        note = f"{rail} is supplied externally through the power input (defaulted)"
        if note not in assumptions:
            assumptions.append(note)
    completed["assumptions"] = assumptions
    sheets = list(completed.get("sheets") or [])
    if not any(
        isinstance(sheet, dict) and sheet.get("name") == "POWER INPUT" for sheet in sheets
    ):
        sheets.append(
            {
                "name": "POWER INPUT",
                "stem": "POWER_INPUT",
                "function": f"Two-pin external {'/'.join(rails)} and GND power input",
                "from_library": None,
                "library_instance": None,
                "replication_group": None,
                "replication_instance": None,
            }
        )
    completed["sheets"] = sheets
    topologies = dict(completed.get("topologies") or {})
    topologies.setdefault(
        "POWER INPUT",
        f"2-pin header for external {'/'.join(rails)} and GND",
    )
    completed["topologies"] = topologies
    return completed


def _functional_spec(brief: str, upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    intent = upstream.get("intent", {})
    allowed = _text([brief, intent]).lower()
    assumption_rows = [str(item) for item in candidate.get("assumptions") or []]
    assumptions = " ".join(assumption_rows).lower()

    introduced = sorted({m.group(1).lower() for m in _TOPOLOGY_RE.finditer(_text(candidate))})
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

    mislabeled_defaults = _mislabeled_functional_defaults(brief, upstream, assumption_rows)
    if mislabeled_defaults:
        diagnostics.append(
            _diag(
                "functional_spec_explicit_fact_defaulted",
                "repair_required",
                "An explicit user requirement was incorrectly labeled as a default.",
                mislabeled_defaults,
            )
        )

    blocks_by_name = {
        str(block.get("name")): block
        for block in candidate.get("blocks") or []
        if isinstance(block, dict)
    }
    external_power_assumptions = []
    for connection in candidate.get("connections") or []:
        if not isinstance(connection, dict) or connection.get("signal_type") != "power":
            continue
        target = str(connection.get("to_block") or "")
        target_text = target.replace("_", " ")
        block = blocks_by_name.get(target) or {}
        if block.get("category") != "drive" or not re.search(
            r"\b(?:hub75|display|addressable[_ -]?led|led[_ -]?(?:string|strip)|motor|heater)\b",
            target_text,
            re.I,
        ):
            continue
        target_terms = (
            r"hub75|display|panel"
            if re.search(r"hub75|display", target_text, re.I)
            else r"addressable led|led string|led strip|leds"
        )
        answer_text = _text(upstream.get("_stage_answers", [])).lower()
        answered_board_power = (
            "board supplies power to both" in answer_text or "power both from board" in answer_text
        )
        explicitly_powered = (
            re.search(
                rf"\bpower(?:s|ed|ing)?\b[^.]{{0,40}}\b(?:{target_terms})\b|"
                rf"\b(?:{target_terms})\b[^.]*\bpowered\s+(?:by|from)\b",
                brief,
                re.I,
            )
            or answered_board_power
        )
        if not explicitly_powered:
            external_power_assumptions.append(target)
    if external_power_assumptions:
        diagnostics.append(
            _diag(
                "functional_spec_external_load_power_assumed",
                "repair_required",
                "The board was made responsible for external-load power without user direction.",
                external_power_assumptions,
            )
        )
    connections = [
        connection
        for connection in candidate.get("connections") or []
        if isinstance(connection, dict)
    ]
    incoming_power = {
        str(connection.get("to_block") or "")
        for connection in connections
        if connection.get("signal_type") == "power"
    }
    drive_blocks = {
        name for name, block in blocks_by_name.items() if block.get("category") == "drive"
    }
    missing_drive_power = sorted(drive_blocks - incoming_power)
    if missing_drive_power:
        diagnostics.append(
            _diag(
                "functional_spec_drive_missing_power",
                "repair_required",
                "A driven output has no incoming power flow.",
                missing_drive_power,
            )
        )

    ground_connections = [
        connection for connection in connections if connection.get("signal_type") == "ground"
    ]
    if ground_connections:
        ground_targets = {
            str(connection.get("to_block") or "") for connection in ground_connections
        }
        expected_ground = {
            name for name, block in blocks_by_name.items() if block.get("category") != "power"
        }
        missing_ground = sorted(expected_ground - ground_targets)
        if missing_ground:
            diagnostics.append(
                _diag(
                    "functional_spec_partial_ground_flow",
                    "repair_required",
                    "Ground flows were listed for only some powered functions.",
                    missing_ground,
                )
            )

    for block in candidate.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        block_text = _text(block)
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
        physical_power_domain = re.search(
            r"\b(?:ldo|buck|regulat(?:or|ion)?|convert(?:er|ing)?|supply|input|sink|controller|protection)\b",
            f"{name} {function}",
            re.I,
        )
        looks_like_power_only = bool(block and block.get("category") == "power") or bool(
            _POWER_RE.search(name)
        )
        if looks_like_power_only and not physical_power_domain:
            diagnostics.append(
                _diag(
                    "architecture_power_block_as_sheet",
                    "repair_required",
                    "A power net or distribution-only block was emitted as a physical sheet.",
                    [name],
                )
            )
        support_text = f"{name} {function}"
        if re.search(
            r"\b(?:crystal|decoupl|pull[- ]?up|passive support)\b|"
            r"\bclock (?:source|generator|oscillator)\b",
            support_text,
            re.I,
        ):
            diagnostics.append(
                _diag(
                    "architecture_fragmented_physical_domain",
                    "repair_required",
                    "A trivial support or board feature was split from the IC domain it supports.",
                    [name],
                )
            )
    candidate_text = _text(candidate)
    intent_text = _text(upstream.get("intent", {}))
    rail_voltages = candidate.get("rail_voltages") or {}
    has_3v3_rail = any(abs(float(voltage) - 3.3) <= 0.05 for voltage in rail_voltages.values())
    if re.search(r"esp32[- ]?s3", intent_text, re.I) and not has_3v3_rail:
        diagnostics.append(
            _diag(
                "architecture_mcu_supply_rail_missing",
                "repair_required",
                "ESP32-S3 architecture is missing its required 3.3V supply rail.",
                ["esp32-s3", "3.3v"],
            )
        )
    if re.search(r"\busb[- ]?c?\s*pd\b", intent_text, re.I) and (
        re.search(r"\bno pd (?:negotiation )?ic\b", candidate_text, re.I)
        or re.search(r"\bpd\b[^.]{0,60}\bvia\b[^.]{0,30}\bcc resistors?\b", candidate_text, re.I)
    ):
        diagnostics.append(
            _diag(
                "architecture_usb_pd_without_controller",
                "repair_required",
                "A requested USB-PD input was implemented as a non-PD CC-resistor sink.",
                ["usb pd", "cc resistors"],
            )
        )
    esp32_s3_present = bool(re.search(r"esp32[- ]?s3", intent_text, re.I))
    unsupported_esp32_audio = bool(
        re.search(r"\bdac\b", candidate_text, re.I)
        or (
            re.search(r"\banalog audio\b|\banalog (?:audio )?signal\b", candidate_text, re.I)
            and not re.search(r"\b(?:pwm|i2s)\b", candidate_text, re.I)
        )
    )
    if esp32_s3_present and unsupported_esp32_audio:
        diagnostics.append(
            _diag(
                "architecture_unsupported_esp32s3_dac",
                "repair_required",
                "ESP32-S3 cannot directly produce the claimed analog audio; use I2S or filtered PWM.",
                ["esp32-s3", "analog audio"],
            )
        )
    extras = upstream.get("_stage_extras", {})
    if re.search(r"\bcore defaults?\b", candidate_text, re.I) and not extras.get(
        "core_defaults_block"
    ):
        diagnostics.append(
            _diag(
                "architecture_unavailable_core_default",
                "repair_required",
                "Architecture cited a core default that was not supplied to the stage.",
                ["core defaults"],
            )
        )
    answer_text = _text(upstream.get("_stage_answers", []))
    functional_connections = (upstream.get("functional_spec") or {}).get("connections") or []
    functional_power_targets = {
        str(connection.get("to_block") or "").lower()
        for connection in functional_connections
        if isinstance(connection, dict) and connection.get("signal_type") == "power"
    }
    functional_powers_external = any(
        "hub75" in target or "display" in target for target in functional_power_targets
    ) and any("led" in target for target in functional_power_targets)
    board_powers_external = functional_powers_external or bool(
        re.search(
            r"board supplies power to both|power both from board",
            answer_text,
            re.I,
        )
    )
    current_context = _text([candidate, upstream.get("_stage_answers", [])])
    has_5v_load_budget = re.search(
        r"(?:5v|vbus|hub75|led string|external load)[^.;]{0,80}"
        r"\d+(?:\.\d+)?\s*(?:a|ma)\b",
        current_context,
        re.I,
    )
    if board_powers_external and not has_5v_load_budget:
        diagnostics.append(
            _diag(
                "architecture_external_load_current_unspecified",
                "repair_required",
                "Board-powered external loads have no maximum 5V current budget.",
                ["hub75", "led string", "5v"],
            )
        )
    external_load_currents = [
        float(match.group(1))
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*a\b",
            answer_text,
            re.I,
        )
    ]
    if board_powers_external and external_load_currents:
        external_load_power_w = 5.0 * max(external_load_currents)
        source_power_profiles: list[tuple[float, float, float, str]] = []
        for name, description in (candidate.get("topologies") or {}).items():
            source_text = f"{name} {description}"
            source_topology = bool(
                re.search(r"\b(?:usb|pd|input|source)\b", str(name), re.I)
                or re.search(
                    r"\busb(?:-c)?\b|\bpd\b[^.;]{0,40}\bcontract\b",
                    str(description),
                    re.I,
                )
            )
            if not source_topology:
                continue

            contract_match = None
            for pattern in (
                r"(\d+(?:\.\d+)?)\s*v(?:dc)?[^.;]{0,40}?"
                r"(\d+(?:\.\d+)?)\s*a\b[^.;]{0,30}\bcontract\b",
                r"(\d+(?:\.\d+)?)\s*v(?:dc)?[^.;]{0,40}\bcontract\b"
                r"[^.;]{0,40}?(\d+(?:\.\d+)?)\s*a\b",
            ):
                contract_match = re.search(pattern, source_text, re.I)
                if contract_match:
                    break
            source_match = contract_match or re.search(
                r"(\d+(?:\.\d+)?)\s*v(?:dc)?[^.;]{0,80}?"
                r"(\d+(?:\.\d+)?)\s*a\b",
                source_text,
                re.I,
            )
            if source_match:
                source_voltage_v = float(source_match.group(1))
                source_current_a = float(source_match.group(2))
                source_power_profiles.append(
                    (
                        source_voltage_v * source_current_a,
                        source_voltage_v,
                        source_current_a,
                        source_text,
                    )
                )
        if not source_power_profiles:
            diagnostics.append(
                _diag(
                    "architecture_external_load_source_capacity_unspecified",
                    "repair_required",
                    "The external-load budget has no explicit input-source power capacity.",
                    [f"external loads: {external_load_power_w:g}w"],
                )
            )
        else:
            source_power_w, _, _, source_text = max(source_power_profiles)
            if source_power_w <= external_load_power_w:
                diagnostics.append(
                    _diag(
                        "architecture_external_load_source_has_no_headroom",
                        "repair_required",
                        "Input-source capacity must exceed the external-load budget so the board and conversion losses are also powered.",
                        [
                            f"external loads: {external_load_power_w:g}w",
                            f"input source: {source_power_w:g}w",
                            source_text,
                        ],
                    )
                )
            overcurrent_profiles = [
                (source_current_a, source_text)
                for _, _, source_current_a, source_text in source_power_profiles
                if source_current_a > 5.0 and re.search(r"\b(?:usb|pd)\b", source_text, re.I)
            ]
            if overcurrent_profiles:
                diagnostics.append(
                    _diag(
                        "architecture_usb_pd_current_exceeds_standard",
                        "repair_required",
                        "A USB-PD contract cannot supply more than 5 A; use a higher-voltage contract and convert down for a 5 V high-current load.",
                        [
                            f"{source_current_a:g}a: {source_text}"
                            for source_current_a, source_text in overcurrent_profiles
                        ],
                    )
                )
            if max(external_load_currents) >= 5.0:
                converter_currents = [
                    float(match.group(1))
                    for name, description in (candidate.get("topologies") or {}).items()
                    if "5v" in str(name).lower()
                    and re.search(r"\b(?:buck|convert)", str(description), re.I)
                    for match in re.finditer(
                        r"(\d+(?:\.\d+)?)\s*a\b",
                        str(description),
                        re.I,
                    )
                ]
                if not converter_currents:
                    diagnostics.append(
                        _diag(
                            "architecture_5v_converter_capacity_unspecified",
                            "repair_required",
                            "The 5 V converter has no explicit output-current rating.",
                            [f"external loads: {max(external_load_currents):g}a"],
                        )
                    )
                elif max(converter_currents) <= max(external_load_currents):
                    diagnostics.append(
                        _diag(
                            "architecture_5v_converter_has_no_headroom",
                            "repair_required",
                            "The regulated 5 V converter must exceed the external-load current budget so onboard loads are also powered.",
                            [
                                f"external loads: {max(external_load_currents):g}a",
                                f"5v converter: {max(converter_currents):g}a",
                            ],
                        )
                    )
                if converter_currents:
                    converter_power_w = 5.0 * max(converter_currents)
                    unused_power_w = source_power_w - converter_power_w
                    if source_power_w >= 2.0 * converter_power_w and unused_power_w >= 30.0:
                        diagnostics.append(
                            _diag(
                                "architecture_input_power_grossly_overprovisioned",
                                "advisory",
                                "Input-source capacity is grossly larger than the regulated 5 V converter capacity; right-size the contract or name the load that needs the margin.",
                                [
                                    f"input source: {source_power_w:g}w",
                                    f"5v converter: {converter_power_w:g}w",
                                    f"unused capacity: {unused_power_w:g}w",
                                ],
                            )
                        )

    inter_sheet_nets = [
        net for net in candidate.get("inter_sheet_nets") or [] if isinstance(net, dict)
    ]
    topology_descriptions = [
        str(description) for description in (candidate.get("topologies") or {}).values()
    ]
    topology_descriptions.extend(
        str(sheet.get("function") or "") for sheet in sheets if isinstance(sheet, dict)
    )
    relation_terms = re.compile(
        r"\b(?:fuse|switch|net[- ]?tie|filter|ideal diode|converter|regulator|"
        r"buck|boost|power path)\b",
        re.I,
    )
    voltage_groups: dict[float, list[str]] = {}
    for rail_name, voltage in rail_voltages.items():
        voltage_groups.setdefault(round(float(voltage), 3), []).append(str(rail_name))
    for voltage, rail_names in voltage_groups.items():
        for index, left_name in enumerate(rail_names):
            for right_name in rail_names[index + 1 :]:
                left_endpoints = {
                    str(endpoint.get("sheet") or "")
                    for net in inter_sheet_nets
                    if str(net.get("name") or "") == left_name
                    for endpoint in net.get("endpoints") or []
                    if isinstance(endpoint, dict)
                }
                right_endpoints = {
                    str(endpoint.get("sheet") or "")
                    for net in inter_sheet_nets
                    if str(net.get("name") or "") == right_name
                    for endpoint in net.get("endpoints") or []
                    if isinstance(endpoint, dict)
                }
                common_endpoints = left_endpoints & right_endpoints
                left_token = _norm_token(left_name)
                right_token = _norm_token(right_name)
                relationship_defined = any(
                    left_token in _norm_token(description)
                    and right_token in _norm_token(description)
                    and relation_terms.search(description)
                    for description in topology_descriptions
                )
                if not relationship_defined:
                    diagnostics.append(
                        _diag(
                            "architecture_duplicate_voltage_rails_unrelated",
                            "repair_required",
                            "Same-voltage rails need one canonical net or an explicit component relationship.",
                            [
                                f"{left_name}/{right_name}: {voltage:g}v",
                                *common_endpoints,
                            ],
                        )
                    )

    for rail, voltage in (candidate.get("rail_voltages") or {}).items():
        if abs(float(voltage) - 3.3) > 0.05:
            continue
        rail_name = str(rail)
        rail_token = re.escape(_norm_token(rail_name))
        rail_pattern = rf"(?:3v3|33v|{rail_token})"
        source_pattern = (
            r"ldo|regulat|buck|convert|externallysupplied|suppliedexternally|"
            r"externallyprovided|providedexternally|externalsource|suppliedvia|"
            r"externalpowerinput|powerinput|inputrail"
        )
        normalized_candidate = _norm_token(candidate_text)
        has_source = re.search(
            rf"(?:{source_pattern}).{{0,60}}{rail_pattern}|"
            rf"{rail_pattern}.{{0,60}}(?:{source_pattern})",
            normalized_candidate,
            re.I,
        )
        if not has_source:
            diagnostics.append(
                _diag(
                    "architecture_rail_source_unspecified",
                    "repair_required",
                    "A declared 3.3V rail has no regulator, converter, or external input source.",
                    [rail_name],
                )
            )
    if re.search(r"esp32[- ]?s3", intent_text, re.I) and has_3v3_rail:
        regulator_terms = re.compile(r"\b(?:ldo|regulat|buck|convert)", re.I)
        regulator_topology_text = " ".join(
            f"{name} {description}"
            for name, description in (candidate.get("topologies") or {}).items()
            if re.search(r"(?:3v3|3\.3v|3 v3)", f"{name} {description}", re.I)
        )
        regulator_sheets = [
            sheet
            for sheet in sheets
            if isinstance(sheet, dict)
            and regulator_terms.search(f"{sheet.get('name', '')} {sheet.get('function', '')}")
            and re.search(
                r"(?:3v3|3\.3v|3 v3)",
                f"{sheet.get('name', '')} {sheet.get('function', '')}",
                re.I,
            )
            and not re.search(
                r"\b(?:mcu|esp32)\b",
                f"{sheet.get('name', '')} {sheet.get('stem', '')}",
                re.I,
            )
        ]
        has_sized_source = bool(
            regulator_terms.search(regulator_topology_text)
            and re.search(
                r"\b(?:1(?:\.0+)?|[2-9](?:\.\d+)?)\s*a\b",
                regulator_topology_text,
                re.I,
            )
            and regulator_sheets
        )

        if not has_sized_source:
            diagnostics.append(
                _diag(
                    "architecture_mcu_regulator_incomplete",
                    "repair_required",
                    "ESP32-S3 needs an explicit >=1A 3.3V regulator topology in its own IC sheet.",
                    ["esp32-s3", "+3v3", ">=1a", "separate regulator sheet"],
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
            net_token = _norm_token(name)
            expected = {
                str(sheet.get("name"))
                for sheet in sheets
                if isinstance(sheet, dict)
                and net_token
                and net_token in _norm_token(f"{sheet.get('name', '')} {sheet.get('function', '')}")
            }
            missing = sorted(expected - endpoint_names)
            if missing:
                diagnostics.append(
                    _diag(
                        "architecture_missing_power_endpoint",
                        "repair_required",
                        "A sheet that explicitly names this rail is absent from its endpoints.",
                        [name, *missing],
                    )
                )
    if candidate.get("mcu_present") and not re.search(
        r"\b(?:swd|jtag|updi|icsp|bootsel|boot|flash|program|debug|reset|native usb)\b",
        _text(candidate),
        re.I,
    ):
        diagnostics.append(
            _diag(
                "architecture_programming_decision_incomplete",
                "repair_required",
                "MCU architecture lacks an explicit programming or recovery choice.",
            )
        )
    return diagnostics


def _bom(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
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
    architecture = upstream.get("architecture") or {}
    parts_by_sheet: dict[str, list[dict]] = {}
    for part in candidate.get("parts") or []:
        if isinstance(part, dict):
            parts_by_sheet.setdefault(str(part.get("sheet") or ""), []).append(part)
    ic_role = re.compile(
        r"\b(?:controller|mcu|regulator|converter|buck|boost|amplifier|"
        r"level shifter|sensor|bridge|driver|hub)\b",
        re.I,
    )
    unsupported_roles: list[str] = []
    for sheet in architecture.get("sheets") or []:
        if not isinstance(sheet, dict):
            continue
        sheet_name = str(sheet.get("name") or "")
        role_text = f"{sheet_name} {sheet.get('function', '')}"
        if not ic_role.search(role_text):
            continue
        sheet_parts = parts_by_sheet.get(sheet_name, [])
        if not any(str(part.get("ref") or "").startswith("U") for part in sheet_parts):
            unsupported_roles.append(sheet_name)
    if unsupported_roles:
        diagnostics.append(
            _diag(
                "bom_architecture_role_unsupported",
                "repair_required",
                "An architecture IC role has no corresponding U-reference implementation on its sheet.",
                sorted(unsupported_roles),
            )
        )
    return diagnostics


def _shared_wiring_gate_diagnostics(
    upstream: dict,
    candidate: dict,
) -> list[StageDiagnostic]:
    """Run the same pure graph gates used by final commit before provider retry."""
    architecture_payload = upstream.get("architecture")
    bom_payload = upstream.get("bom")
    if not isinstance(architecture_payload, dict) or not isinstance(bom_payload, dict):
        return []
    try:
        architecture = Architecture.model_validate(architecture_payload)
        bom = BOM.model_validate(
            {
                **bom_payload,
                "connections": candidate.get("connections") or [],
                "no_connect_pins": candidate.get("no_connect_pins") or [],
            }
        )
    except (TypeError, ValueError):
        return []
    from kicraft.design.synthesis.validation import (
        check_inter_sheet_nets_realized,
        check_mcu_programming_access,
        check_net_coverage,
        check_no_dangling_signal_nets,
    )

    checks = (
        ("wiring_gate_9_11", check_net_coverage(bom)),
        ("wiring_gate_9_14", check_inter_sheet_nets_realized(architecture, bom)),
        ("wiring_gate_9_15", check_no_dangling_signal_nets(architecture, bom)),
        ("wiring_gate_9_29", check_mcu_programming_access(bom)),
    )
    return [
        _diag(code, "fab_gate", result.message, list(result.offenders))
        for code, result in checks
        if not result.ok
    ]


def _wiring(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = _shared_wiring_gate_diagnostics(
        upstream,
        candidate,
    )
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
        findings = _bom(upstream_state, candidate)
    elif stage == "wiring":
        findings = _wiring(upstream_state, candidate)
    else:
        findings = []
    return sorted(findings, key=lambda finding: (finding.severity, finding.code, finding.evidence))
