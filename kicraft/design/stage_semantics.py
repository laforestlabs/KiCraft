"""Pure, versioned semantic diagnostics for schema-valid stage candidates."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from functools import lru_cache

from kicraft.design.models import (
    Architecture,
    BOM,
    StageDiagnostic,
    is_board_stackup_obligation,
    is_power_or_ground_name,
    obligation_requires_requirement_owner,
)
from kicraft.design.synthesis.board_features import (
    PROTOTYPING_AREA_FEATURE,
    has_prototyping_area,
    prototyping_area_requested,
)
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
# A mechanical board feature or a bare net is not a functional block. `crystal` is deliberately
# absent: it is a functional timing component a brief asks for ("an 8 MHz crystal"), not a
# board feature, and listing it refused a shipped STM32 dev board whose crystal block was
# exactly what the brief demanded (replay 2026-09-26).
_NONFUNCTIONAL_RE = re.compile(
    r"\b(?:ground|gnd|rail|power[_ -]distribution|mounting holes?|decoupling|"
    r"castellated pads?)\b",
    re.I,
)
_POWER_RE = re.compile(r"\b(power|vbus|vcc|vdd|3v3|5v|1v1|ldo|regulat)\b", re.I)
# The one code that asks for a fact only the user has (next-steps plan §4 B2).
EXTERNAL_LOAD_CURRENT_CODE = "architecture_external_load_current_unspecified"
EXTERNAL_OUTPUT_ALLOCATION_CODE = "architecture_external_output_allocation_unspecified"
_EXTERNAL_LOAD_PATTERN = re.compile(
    r"\b(?:hub\s*75|(?:addressable\s+)?led\s+(?:strip|string)|"
    r"ws2812(?:[-\w]*)?|external\s+(?:5\s*v\s+)?(?:loads?|panel|display))\b",
    re.I,
)
_CURRENT_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(a|ma)\b", re.I)


def _text_fragments(value) -> list[str]:
    """Keep authored fields separate when looking for a load-current statement."""
    if isinstance(value, dict):
        return [
            fragment
            for item in value.values()
            for fragment in _text_fragments(item)
        ]
    if isinstance(value, list):
        return [fragment for item in value for fragment in _text_fragments(item)]
    return [str(value or "")]


def external_load_budget_stated(text) -> bool:
    """Does one authored statement give an external display or LED load current?

    A regulator's own ``VBUS to 3.3 V at 1 A`` rating is not a budget for a
    HUB75 panel or LED string.  Do not flatten independently authored fields:
    doing so makes the converter rating appear to qualify a later load name.
    """
    for fragment in _text_fragments(text):
        for clause in re.split(r"(?<!\d)\.|\.(?!\d)|[;\n]+", fragment):
            load_spans = [match.span() for match in _EXTERNAL_LOAD_PATTERN.finditer(clause)]
            current_spans = [match.span() for match in _CURRENT_PATTERN.finditer(clause)]
            if any(
                abs(current_start - load_end) <= 80
                or abs(load_start - current_end) <= 80
                for load_start, load_end in load_spans
                for current_start, current_end in current_spans
            ):
                return True
    return False


def _external_load_budget_answered(answers) -> bool:
    """Accept a current only when it answers the external-load current question."""
    for answer in answers or []:
        if not isinstance(answer, dict):
            continue
        question = str(answer.get("text") or "")
        if not (
            "current" in question.casefold()
            and re.search(r"\b(?:external|load|display|led|hub75)\b", question, re.I)
        ):
            continue
        if _CURRENT_PATTERN.search(str(answer.get("answer") or "")):
            return True
    return False


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


def _committed_topology_text(candidate: dict) -> str:
    """The fields where a spec *commits* to a technology, never the prose that describes behaviour.

    A block's ``purpose`` and a connection's ``description`` say what a function does. Naming a
    technology there ("accept the amplifier input", "the analog audio circuitry") describes the
    world; it does not commit the board, and the whole-candidate scan refused valid designs over
    it — replay 2026-09-26: 72 of 73 firings were on boards that shipped, the live cases being a
    passive crossover fed by an external amplifier and a spec describing its signal domain. The
    commitment is the block *name* the writer undertakes to realize (`LDO_3V3`, `BUCK_SUPPLY`) and
    the typed obligation class, so only those are scanned. Underscores become spaces so a name
    like `LDO_3V3` matches ``\\bldo\\b``.
    """
    parts: list[str] = []
    for block in candidate.get("blocks") or []:
        if isinstance(block, dict):
            parts.append(str(block.get("name") or "").replace("_", " "))
            parts.append(str(block.get("category") or ""))
    for obligation in candidate.get("obligations") or []:
        if isinstance(obligation, dict):
            parts.extend(
                str(obligation.get(key) or "") for key in ("kind", "component_class", "feature")
            )
    return " ".join(parts)


def complete_intent_classification(brief: str, candidate: dict) -> dict:
    """Fill omitted intent classifications from exact user text, without invention."""
    from kicraft.design.part_identity import non_part_obligation_row

    completed = dict(candidate)
    expected = named_part_tokens([brief])
    supplied = {_norm_token(part) for part in completed.get("named_parts") or []}
    missing_parts = []
    for token in expected.values():
        identity = _norm_token(token)
        if identity not in supplied:
            missing_parts.append(token)
            supplied.add(identity)
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

    # A stated stack-up is a machine-readable board fact the builder reads to set the copper
    # layer count; prose in `constraints` reaches no gate. The brief's own words are the
    # evidence, so no default is recorded -- unlike the power-entry fallback, nothing is
    # invented here. An existing row retains its ID, but cannot contradict the original brief.
    layers = requested_board_copper_layers(brief)
    if layers is not None:
        obligations = list(completed.get("obligations") or [])
        stackup_rows = [
            index for index, row in enumerate(obligations) if is_board_stackup_obligation(row)
        ]
        if stackup_rows:
            for index in stackup_rows:
                row = obligations[index]
                if row.get("value") != layers or row.get("relation") != "equal":
                    obligations[index] = {
                        **row, "relation": "equal", "value": float(layers),
                        "minimum": None, "maximum": None,
                    }
        else:
            obligations.append(_board_stackup_obligation(layers))
        completed["obligations"] = obligations

    # A driver for a named off-board load owns the interface, not the load's LEDs.
    # Only retype an explicitly described strip class when the brief describes a
    # driver/interface for that strip and identifies a reviewed interface protocol.
    from kicraft.design.recipes.registry import registered_recipes

    protocols = {
        protocol
        for registered in registered_recipes()
        for protocol in registered.definition.external_interface_protocols
        if re.search(rf"\b{re.escape(protocol)}\b", brief, re.I)
    }
    external_strip = bool(
        re.search(r"\b(?:driver|interface|controller)\s+(?:board\s+)?for\b[^.;]*\b(?:strip|string)\b", brief, re.I)
        and not re.search(r"\b(?:on[ -]?board|board[ -]?mounted)\b[^.;]*\b(?:leds?|strip|string)\b", brief, re.I)
    )
    if external_strip and len(protocols) == 1:
        protocol = next(iter(protocols))
        completed["obligations"] = [
            {
                "kind": "external_interface",
                "original_obligation_id": row["original_obligation_id"],
                "protocol": protocol,
            }
            if isinstance(row, dict)
            and row.get("kind") == "physical"
            and re.search(r"\b(?:strip|string)\b", str(row.get("component_class") or "").replace("-", " "), re.I)
            and row.get("original_obligation_id")
            else row
            for row in completed.get("obligations") or []
        ]
    obligations = list(completed.get("obligations") or [])
    jumper_rows = [
        row for row in obligations
        if isinstance(row, dict)
        and row.get("kind") == "physical"
        and str(row.get("component_class") or "").casefold() == "jumper"
    ]
    if (
        len(jumper_rows) == 1
        and re.search(r"\b(?:enable[ -]+jumper|jumper[ -]+enable)\b", brief, re.I)
        and not any(
            isinstance(row, dict) and row.get("kind") == "adjustability"
            and str(row.get("mechanism") or "").casefold() == "jumper"
            and "enable" in str(row.get("parameter") or "").casefold()
            for row in obligations
        )
    ):
        identifier = f"{jumper_rows[0].get('original_obligation_id', '')}-control"
        if len(identifier) <= 64 and not any(
            isinstance(row, dict) and row.get("original_obligation_id") == identifier
            for row in obligations
        ):
            completed["obligations"] = [
                *obligations,
                {"kind": "adjustability", "original_obligation_id": identifier,
                 "parameter": "enable", "mechanism": "jumper"},
            ]
    normalized_obligations = []
    for obligation in completed.get("obligations") or []:
        # A demanded class that states a board fact (an outline or shape, a mechanical hole, the
        # board's own copper, its edge plating) or the ABSENCE of a class is not a part demand:
        # `non_part_obligation_row` retypes it to the row the typed vocabulary has for it, so no
        # later gate asks for a BOM line no part can ever be.
        retyped = non_part_obligation_row(obligation)
        normalized_obligations.append(retyped if retyped is not None else obligation)
    if normalized_obligations != completed.get("obligations"):
        completed["obligations"] = normalized_obligations
    return completed


#: A class whose wording negates what it names ("no-microcontroller") still contains a reviewed
#: class as a strict token subset, so the superset relation inverts there.
_CLASS_NEGATION_RE = re.compile(r"\b(?:no|not|without|non|none)\b", re.IGNORECASE)


def complete_class_spellings(candidate: dict) -> dict:
    """Spell a physical obligation the way the reviewed library does, before diagnosis.

    "A physical obligation spells a reviewed part class differently" is what the intent check
    reports, and its own evidence already names the repair (``-> reviewed class: flash-memory``).
    Asking the writer to make that rename costs a repair round, and on 154 archived runs the
    class then reached the parts stage unresolved. The rename is deterministic -- the reviewed
    class is a strict subset of the demanded class's tokens and has a reviewed carrier -- and it
    is recorded as a defaulted assumption, so the operator sees the reading.

    Deliberately skipped, because the rename would invert or distort the demand:

    * a negated class (``no-microcontroller``): the token-superset relation inverts there;
    * a class that is not a part at all, or that names the off-board power source itself: those
      have their own repairs (record it as a constraint; demand the mate class) which only the
      writer can choose;
    * a variant whose reviewed class has no carrier: renaming onto a second dead class is worse
      than the honest failure.
    """
    from kicraft.design.part_identity import (
        class_is_not_a_part,
        off_board_source_class,
        package_descriptor_class,
        realizable_physical_features,
        reviewed_class_variants,
    )

    obligations = candidate.get("obligations") or []
    renamed: list[tuple[int, str, str]] = []
    for index, row in enumerate(obligations):
        if not isinstance(row, dict) or row.get("kind") != "physical":
            continue
        demanded = str(row.get("component_class") or "").strip()
        if not demanded or realizable_physical_features(demanded):
            continue
        if _CLASS_NEGATION_RE.search(demanded.replace("-", " ").replace("_", " ")):
            continue
        if (
            class_is_not_a_part(demanded)
            or off_board_source_class(demanded)
            or package_descriptor_class(demanded)
        ):
            continue
        target = next(
            (
                name
                for name in reviewed_class_variants(demanded)
                if realizable_physical_features(name)
            ),
            None,
        )
        if target is not None:
            renamed.append((index, demanded, target))
    if not renamed:
        return candidate

    completed = copy.deepcopy(candidate)
    completed["obligations"] = list(obligations)
    notes: list[str] = []
    for index, demanded, target in renamed:
        completed["obligations"][index] = {
            **completed["obligations"][index],
            "component_class": target,
        }
        notes.append(f"part class {demanded!r} read as reviewed class {target!r} (defaulted)")
    # A count row that bound the old spelling must follow it. `buttons = 3` binds to
    # `rotary-encoder-push-button` on the shared `button` token; leaving the subject behind would
    # make the count dangle the moment the class is renamed, which is a repair round bought for
    # nothing (replay 2026-09-26: two runs).
    from kicraft.design.part_identity import quantity_subject_binds

    for index, row in enumerate(obligations):
        if not isinstance(row, dict) or row.get("kind") != "quantity":
            continue
        subject = str(row.get("subject") or "")
        matches = [
            (demanded, target)
            for _index, demanded, target in renamed
            if subject and quantity_subject_binds(subject, demanded)
        ]
        if len(matches) != 1:
            continue
        demanded, target = matches[0]
        completed["obligations"][index] = {**completed["obligations"][index], "subject": target}
        notes.append(
            f"count {subject!r} follows part class {demanded!r} read as {target!r} (defaulted)"
        )
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    completed["assumptions"] = [
        *assumptions,
        *[note for note in notes if note not in assumptions],
    ]
    return completed


#: The brief's spelled-out layer counts. "four-layer stack-up" and "4 layers" both read here;
#: the generator's own trailing sentences use the first spelling.
_STACKUP_COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "hundred": 100,
}

#: A count bound to the layer noun, so an unrelated number ("under 100 x 100 mm") never reads
#: as a stack-up. `two-layer`, `4 layer`, `four copper layers` all match.
_STACKUP_COUNT_RE = re.compile(
    r"\b(?P<count>\d+(?:\.\d+)?|"
    + "|".join(_STACKUP_COUNT_WORDS)
    + r")[\s-]+(?:copper[\s-]+)?layers?\b",
    re.IGNORECASE,
)


def requested_board_copper_layers(text: str) -> int | float | None:
    """Read the stated count, retaining unsupported counts for an honest refusal."""
    for match in _STACKUP_COUNT_RE.finditer(text or ""):
        raw = match.group("count").casefold()
        if raw in _STACKUP_COUNT_WORDS:
            return _STACKUP_COUNT_WORDS[raw]
        return float(raw)
    return None


def _board_stackup_obligation(layers: int | float) -> dict:
    return {
        "kind": "quantitative",
        "original_obligation_id": "board-copper-layers",
        "quantity": "PCB copper layers",
        "relation": "equal",
        "value": float(layers),
        "unit": "layers",
    }


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


def _names_board_field(name: object) -> bool:
    """Whether a block name names the prototyping pad field, whatever separators it uses."""
    return prototyping_area_requested(str(name or "").replace("_", " ")) is not None


def _block_is_board_realized(block: dict, intent: dict) -> bool:
    """Whether every obligation a block names is a board-wide fact the printed board realizes.

    Mirrors the architecture-side exemption in ``validation._functional_block_sheets``: a block
    whose only rows are ``quantity``/``fabrication``/``negative`` facts, or a ``quantitative`` row
    that measures the board outline or its stack-up, is realized by the board itself and needs no
    implementing sheet. A block naming no obligation, or any obligation the committed intent does
    not carry, is not evidence of that exemption.
    """
    ids = [str(identifier) for identifier in block.get("obligation_ids") or []]
    if not ids:
        return False
    rows = {
        str(row.get("original_obligation_id")): row
        for row in intent.get("obligations") or []
        if isinstance(row, dict)
    }
    return all(
        identifier in rows and not obligation_requires_requirement_owner(rows[identifier])
        for identifier in ids
    )


def _omitted_board_features(brief: str, candidate: dict) -> list[StageDiagnostic]:
    """Board features the brief asks for that no row records.

    A prototyping area is the whole point of a prototyping shield and it fits no other
    field: it is not a component class, owns no pin and draws no net, so if the intent
    drops it there is nothing for any later stage to build. Recorded as a `fabrication`
    row, it survives to the sheet and the pad field the realization stages derive from it.
    """
    phrase = prototyping_area_requested(
        _text([brief, candidate.get("goal"), candidate.get("constraints")])
    )
    if phrase is None or has_prototyping_area(candidate.get("obligations")):
        return []
    return [
        _diag(
            "intent_prototyping_area_omitted",
            "repair_required",
            "The brief asks for a prototyping area; record it as a board feature.",
            [f"{phrase} -> obligations: kind fabrication, feature {PROTOTYPING_AREA_FEATURE}"],
        )
    ]


#: The reading a package or footprint descriptor gets, whether it carries the noun
#: ("qfn-56-package") or is nothing but the package itself ("lqfp-48", which the package pattern
#: in `part_identity.package_descriptor_class` proves).
_PACKAGE_DESCRIPTOR_READING = (
    "a package or footprint descriptor, which is an attribute of a part",
    "name the package on the part that carries it (or keep it in constraints); a package "
    "descriptor is never a class",
)

#: What a not-a-part token proves about a demanded class, and the repair it implies, as
#: ``(tokens, reading, repair)``. The first group whose tokens the class carries names the
#: reading, so a package descriptor is never reported as printed copper and a region label is
#: never reported as an interface. The repair is the concrete move the writer must make, because
#: a refusal the writer cannot act on only spends a repair round.
_NOT_A_PART_READINGS: tuple[tuple[frozenset[str], str, str], ...] = (
    (frozenset({"package", "footprint"}), *_PACKAGE_DESCRIPTOR_READING),
    (
        frozenset({"section", "region"}),
        "a region or section label, which says where parts go",
        "keep the label in constraints or the goal and demand the parts it names",
    ),
    (
        frozenset({"connection"}),
        "a net-level connection, which the wiring owns",
        "keep it in constraints: the wiring stage draws it and no BOM line implements it",
    ),
    (
        frozenset({"isolation", "barrier"}),
        "a dielectric or clearance property of the build",
        "record it as a constraint on the build, not as a part",
    ),
    (
        frozenset({"switchable"}),
        "a network named by its function rather than by the parts that build it",
        "demand the parts that build it (its resistor and its switch), or keep the function in "
        "constraints",
    ),
    (
        frozenset({"outline", "shape", "layout"}),
        "the board outline or layout",
        "record it as a `fabrication` row, or keep it in constraints",
    ),
    (
        frozenset({"interface", "bus", "protocol", "format"}),
        "an interface, bus or board format",
        "record it as a constraint",
    ),
    (
        frozenset({"pour", "plane", "via", "vias"}),
        "printed copper the board itself carries",
        "record it as a `fabrication` row",
    ),
    (
        frozenset({"net", "netlist"}),
        "a net or netlist fact",
        "keep it in constraints: the wiring stage owns it",
    ),
)


def _not_a_part_reading(not_a_part: tuple[str, ...]) -> tuple[str, str]:
    """The reading and repair a not-a-part token match proves."""
    matched = set(not_a_part)
    for tokens, reading, repair in _NOT_A_PART_READINGS:
        if matched & tokens:
            return reading, repair
    return "a board or wiring fact", "move the fact out of `obligations` into constraints"


def _unrealizable_obligation_classes(obligations) -> list[StageDiagnostic]:
    """Physical obligations whose class is a not-a-part fact or a puzzled variant name.

    Four cases, each safe to reject at the stage that writes them:

    * the class carries a token no physical class can have (an interface, bus, board
      format, package style, region label, net-level connection, dielectric property,
      printed-board feature, or a switchable network named by its function), so it
      belongs in `constraints`, a `fabrication` row, or a `negative` row. The evidence
      names WHICH reading it is and the concrete repair;
    * the class states the absence of a class (`no-microcontroller`), which no BOM line
      implements: the repair is a `negative` row with that `absent_class`;
    * the class names the power source itself (a battery, a cell, a pack), which no placed
      part can implement: the board carries the mate, so the demand has no satisfying group
      however many times the unit is re-driven;
    * the class is a longer spelling of a reviewed class, so the reviewed name is the
      repair.

    A class with no reviewed coverage and no such relation — a genuinely new part category
    such as `gps-module` or `air-quality-sensor` — is deliberately NOT flagged. The
    reviewed library can only answer for the classes it covers; refusing a new category
    here, or renaming it to a reviewed neighbour, would block or silently distort exactly
    the novel designs the pipeline exists to build.

    ``coin-cell-holder`` and ``battery-connector`` are the mate classes and stay unflagged
    for the same reason: they name something the board does place. `mounting-hole` and
    `thermal-pad` stay unflagged because a reviewed record carries each of them.

    A class this module's completion already retypes (a board outline, a mechanical hole,
    printed copper, castellations, or a leading negation — see
    :func:`kicraft.design.part_identity.non_part_obligation_row`) is not reported here when
    it carries no not-a-part token: the compiler has already read it, so asking the writer to
    restate it would buy a repair round for nothing. The pipeline diagnoses the completed
    candidate, so those rows are typed before this check ever sees them.
    """
    from kicraft.design.part_identity import (
        absent_class_of_negation,
        class_is_not_a_part,
        negation_names_a_reviewed_class,
        off_board_source_class,
        package_descriptor_class,
        realizable_physical_features,
        reviewed_class_variants,
    )

    diagnostics: list[StageDiagnostic] = []
    seen: set[str] = set()
    for obligation in obligations or []:
        if not isinstance(obligation, dict) or obligation.get("kind") != "physical":
            continue
        component_class = str(obligation.get("component_class") or "").strip().casefold()
        if not component_class or component_class in seen:
            continue
        if realizable_physical_features(component_class):
            continue
        seen.add(component_class)
        not_a_part = class_is_not_a_part(component_class)
        variants = reviewed_class_variants(component_class)
        if not_a_part or package_descriptor_class(component_class):
            reading, repair = (
                _not_a_part_reading(not_a_part)
                if not_a_part
                else _PACKAGE_DESCRIPTOR_READING
            )
            marker = ", ".join(not_a_part) if not_a_part else "package"
            diagnostics.append(
                _diag(
                    "intent_obligation_class_unrealizable",
                    "repair_required",
                    f"A physical obligation names {reading} rather than a part class.",
                    [f"{component_class} -> not a part class ({marker}): {repair}"],
                )
            )
            continue
        absent = absent_class_of_negation(component_class)
        if absent is not None:
            repair = f"record it as kind `negative` with absent_class {absent}"
            if not negation_names_a_reviewed_class(absent):
                # The pipeline leaves this wording to the writer, because the negated rest is no
                # class it reads: a "not gate" is a part. Both repairs must travel.
                repair += (
                    ", or restate the class plainly if the negation word is part of its own "
                    f"name (the pipeline read {absent!r} as the class being denied)"
                )
            diagnostics.append(
                _diag(
                    "intent_obligation_class_unrealizable",
                    "repair_required",
                    "A physical obligation states the absence of a class; no BOM line implements "
                    "an absence.",
                    [f"{component_class} -> {repair}"],
                )
            )
            continue
        if _CLASS_NEGATION_RE.search(component_class.replace("-", " ").replace("_", " ")):
            diagnostics.append(
                _diag(
                    "intent_obligation_class_unrealizable",
                    "repair_required",
                    "A physical obligation states the absence of a class inside a larger demand; "
                    "no BOM line implements an absence.",
                    [
                        f"{component_class} -> split the row: record the absence as kind "
                        "`negative` with its `absent_class`, and keep the part or function it "
                        "also names as its own demand"
                    ],
                )
            )
            continue
        if off_board_source_class(component_class):
            diagnostics.append(
                _diag(
                    "intent_obligation_class_unrealizable",
                    "repair_required",
                    "A physical obligation names the power source itself; the board carries the "
                    "mate that the source plugs into, never the source.",
                    [
                        f"{component_class} -> off-board power source "
                        f"({', '.join(off_board_source_class(component_class))}): demand the "
                        "connector/holder class that mates with it, or drop the obligation and "
                        "keep the source in the goal/named_parts"
                    ],
                )
            )
        elif variants:
            diagnostics.append(
                _diag(
                    "intent_obligation_class_unrealizable",
                    "repair_required",
                    "A physical obligation spells a reviewed part class differently.",
                    [f"{component_class} -> reviewed class: {', '.join(variants)}"],
                )
            )
    return diagnostics


def _quantity_subject_unbound(candidate: dict) -> list[StageDiagnostic]:
    """Counts of a part class this slot does not carry.

    A count binds to the class it names (:func:`quantity_class_for`), so a slot that demands
    "two JST-XH connectors" as a count but records no `jst-xh-connector` obligation loses the
    demand entirely: nothing requires the part. The writer restores it -- spell the class, or
    drop the row if the brief's count was a property of one part.

    Only a subject that *reads as a part class* (its head noun is one the reviewed vocabulary
    uses) is refused. A property count ("pins on the 0.1 inch header") or a design fact
    ("relay channels") is left to the writer: it was never a part count, and demanding a
    rewrite would churn designs that are already correct.
    """
    from kicraft.design.part_identity import (
        PROPERTY_PREPOSITIONS,
        class_key,
        quantity_class_for,
        reviewed_class_heads,
    )

    obligations = [row for row in candidate.get("obligations") or [] if isinstance(row, dict)]
    classes = [
        str(row.get("component_class") or "")
        for row in obligations
        if row.get("kind") == "physical"
    ]
    heads = reviewed_class_heads()
    diagnostics: list[StageDiagnostic] = []
    for row in obligations:
        if row.get("kind") != "quantity":
            continue
        subject = str(row.get("subject") or "").strip()
        if not subject or quantity_class_for(subject, classes) is not None:
            continue
        if set(re.findall(r"[a-z0-9]+", subject.casefold())) & PROPERTY_PREPOSITIONS:
            continue
        key = class_key(subject)
        if not key or key.split("-")[-1] not in heads:
            continue  # a design fact or a property count, not a part class
        diagnostics.append(
            _diag(
                "intent_quantity_subject_unbound",
                "repair_required",
                "A quantity row counts a part class this slot does not carry, so no gate "
                "can enforce the count.",
                [
                    f"{subject} = {row.get('minimum')} -> add the physical obligation for "
                    f"the class it counts (spell the class, e.g. {key}), or delete the row "
                    "if the brief's count was a property of one part"
                ],
            )
        )
    return diagnostics


@lru_cache(maxsize=1)
def _reviewed_class_heads() -> tuple[tuple[str, str], ...]:
    """`(head word, class name)` for every reviewed class a brief may demand by name.

    Deterministic by construction: the exact class name wins over a compound that happens to end in
    the same word ("fuse" over "input-fuse"), and the rest are read in sorted order, so the
    diagnostic's evidence never depends on set iteration order.
    """
    from kicraft.design.part_identity import reviewed_feature_vocabulary

    best: dict[str, str] = {}
    for name in sorted(reviewed_feature_vocabulary()):
        words = re.split(r"[-_ ]+", str(name).casefold())
        if not 0 < len(words) <= 3:
            continue
        head = words[-1]
        current = best.get(head)
        if current is None or (len(words) == 1 and len(re.split(r"[-_ ]+", current)) > 1):
            best[head] = str(name)
    return tuple(sorted(best.items()))


def _omitted_demanded_classes(brief: str, candidate: dict) -> list[StageDiagnostic]:
    """A class the brief brands onto the design that no obligation records.

    Live cohort 2026-09-30, project 1108: the brief asked for "two fused outputs" and the committed
    intent recorded no fuse at all -- only a count of screw terminals -- so nothing downstream could
    place one. That board built, exported a fab package, and would have counted as a success while
    carrying no fuse. No later obligation gate can see it, because the whole chain starts here.

    Read only from the *participle* form (`fuse` -> "fused"), never the bare noun or the plural.
    Measured over every board in the live store that shipped: the bare and plural forms fire on 30
    designs that are correct as built ("screw terminals", "film capacitors", "servo headers" -- each
    already recorded under another class spelling), while the participle form fires on exactly one,
    and that one is the non-compliant board above. The refusal is repairable: record the class the
    brief names, or say in assumptions why the design does not carry it.
    """
    recorded = " ".join(
        str(row.get("component_class") or row.get("subject") or "")
        for row in candidate.get("obligations") or []
        if isinstance(row, dict)
    ).casefold()
    recorded_words = {word for word in re.split(r"[\s\-_]+", recorded) if word}
    text = str(brief or "").casefold()
    offenders = [
        f"{form!r} -> {class_name}"
        for head, class_name in _reviewed_class_heads()
        if head not in recorded_words
        for form in (head + "ed", head + "d")
        if re.search(rf"\b{re.escape(form)}\b", text)
    ]
    if not offenders:
        return []
    return [
        _diag(
            "intent_demanded_class_omitted",
            "repair_required",
            "The brief brands a part class onto the design (a participle such as 'fused') and no "
            "obligation records that class, so no later stage can place it. Record the physical "
            "obligation for the class it names, or state in assumptions why the design does not "
            "carry it.",
            sorted(set(offenders)),
        )
    ]


def _intent(brief: str, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    diagnostics.extend(_omitted_demanded_classes(brief, candidate))
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
    diagnostics.extend(_unrealizable_obligation_classes(candidate.get("obligations")))
    diagnostics.extend(_quantity_subject_unbound(candidate))
    diagnostics.extend(_omitted_board_features(brief, candidate))
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


#: Reviewed classes that can carry a board's supply in. Deliberately broad: any connector
#: the writer already named counts as an entry path, so the default below only ever fills a
#: supply path nobody described.
_POWER_ENTRY_FEATURES = frozenset(
    {
        "barrel-jack",
        "barrel-jack-connector",
        "screw-terminal",
        "terminal-block",
        "screw-clamp-terminal",
        "binding-post",
        "power-connector",
        "wire-to-board-connector",
        "usb-c-receptacle",
        "usb-a-receptacle",
        "usb-connector",
        "header",
        "pin-header",
        "pin-socket",
        "stacking-header",
        "jst-ph",
        "battery-holder",
        "battery-connector",
        "coin-cell-holder",
    }
)

#: Words that name a way power enters the board. Only these suppress the default: a signal
#: connector the brief names (the "two JST-XH connectors" of a motor driver) is not a supply
#: path, so `jst` and bare `connector` are deliberately absent.
_POWER_ENTRY_WORDS = (
    "usb",
    "barrel",
    "jack",
    "screw",
    "terminal",
    "binding post",
    "battery",
    "batteries",
    "pack",
    "cell",
    "holder",
    "header",
    "socket",
    "plug",
    "receptacle",
    "dc-in",
    "psu",
    "wall wart",
)

#: The carrier a DC supply the brief never routes is defaulted to: one reviewed 2-position
#: screw terminal, the corpus's field-wiring DC entry (the generator's own supply values
#: name "24 V DC screw terminal" the same way).
DEFAULT_POWER_ENTRY_CLASS = "screw-terminal"

_VOLTAGE_NEAR_INPUT_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*V(?:olt)?s?\s*(?:DC)?\s*(?:input|supply)"
    r"|(?:input|supply)[^.;]{0,20}?(\d+(?:\.\d+)?)\s*V)",
    re.IGNORECASE,
)


def _dc_supply_voltage(brief: str, candidate: dict) -> str | None:
    """The board's stated DC supply voltage, as an assumption should read it.

    The typed `quantitative` row wins -- the writer already parsed it -- and the brief is
    the fallback, where the voltage must sit next to input/supply wording so a signal
    rail ("3.3 V logic") is not mistaken for the supply.
    """
    for row in candidate.get("obligations") or []:
        if not isinstance(row, dict) or row.get("kind") != "quantitative":
            continue
        quantity = str(row.get("quantity") or "").casefold()
        unit = str(row.get("unit") or "").strip()
        if row.get("value") is None or not unit or "v" not in unit.casefold():
            continue
        if not any(word in quantity for word in ("input", "supply", "rail")):
            continue
        value = f"{row['value']:g}"
        return f"{value} {unit}" if unit.casefold().startswith("v") else f"{value} V"
    match = _VOLTAGE_NEAR_INPUT_RE.search(brief)
    if match is None:
        return None
    return f"{match.group(1) or match.group(2)} V"


def complete_unavailable_part_classes(
    stage: str, candidate: dict, semantic_state: dict
) -> dict:
    """Answer a demanded part class the library cannot, by researching a real part once.

    A brief can demand a physical class no vendored record carries -- a series Schottky diode, a
    varistor, a part nobody has needed before. Every gate that later checks the demand refuses
    (the architecture audit, the parts work-unit obligation check, §9.42) because nothing can
    satisfy it, and no stage is allowed to invent a part. The owner's rule is that such a demand is
    **researched and added, not refused**, so it is answered here, before diagnosis: the class is
    searched in the offline catalog, the best in-stock single-device candidate is vendored into the
    machine-wide parts library and recorded as reviewed, and the assumption says so in plain words.
    From then on the class is covered -- for this project and every later one.

    Soft by construction: an unavailable catalog or a failed fetch changes nothing and the stage's
    own refusal stands, rather than a wrong part being smuggled in. Bounded to two classes per
    call, so one draft cannot become an unbounded fetching run.
    """
    if stage not in {"architecture", "bom"}:
        return candidate
    from kicraft.design.part_identity import reviewed_parts_for_feature
    from kicraft.design.part_research import research_uncovered_classes

    demanded: list[str] = []
    sources = [
        candidate,
        semantic_state.get("intent") or {},
        semantic_state.get("architecture") or {},
        semantic_state.get("functional_spec") or {},
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for row in source.get("obligations") or []:
            if isinstance(row, dict) and row.get("kind") == "physical":
                demanded.append(str(row.get("component_class") or ""))

    unanswered = [
        component_class
        for component_class in demanded
        if component_class.strip() and not reviewed_parts_for_feature(component_class)
    ]
    if not unanswered:
        return candidate

    researched = research_uncovered_classes(unanswered, limit=2)
    if not researched:
        return candidate

    completed = dict(candidate)
    completed["assumptions"] = [
        *(candidate.get("assumptions") or []),
        *(result.as_record_note() for result in researched),
    ]
    return completed


def complete_unstated_power_input(brief: str, candidate: dict) -> dict:
    """Default the entry path for a DC supply the brief states but never routes.

    A brief can name a supply voltage and no way for it to reach the board ("an 18 V DC
    input"); the deliverable still needs a real connector, and parking on a question the
    live product would auto-answer is not this stage's job. The default is one reviewed
    2-position screw terminal, recorded as an assumption so the user can see and override
    it -- the same "safety net" shape as the board-outline default.

    Idempotent, and silent whenever the supply path is already described: an entry class in
    the obligations, an entry word in the brief's own words, or an off-board source
    (battery/pack/cell) that must instead be answered by its mate class.
    """
    from kicraft.design.part_identity import canonical_physical_features

    obligations = [row for row in candidate.get("obligations") or [] if isinstance(row, dict)]
    if any(
        canonical_physical_features(str(row.get("component_class") or "")) & _POWER_ENTRY_FEATURES
        for row in obligations
        if row.get("kind") == "physical"
    ):
        return candidate
    words = _text(
        [
            brief,
            candidate.get("goal"),
            candidate.get("constraints"),
            candidate.get("named_parts"),
            candidate.get("assumptions"),
        ]
    ).casefold()
    if any(word in words for word in _POWER_ENTRY_WORDS):
        return candidate
    voltage = _dc_supply_voltage(brief, candidate)
    if voltage is None:
        return candidate

    completed = copy.deepcopy(candidate)
    completed["obligations"] = [
        *obligations,
        {
            "kind": "physical",
            "original_obligation_id": "power_input_connector",
            "component_class": DEFAULT_POWER_ENTRY_CLASS,
        },
    ]
    note = (
        f"Power input: 2-position {DEFAULT_POWER_ENTRY_CLASS.replace('-', ' ')} "
        f"for the {voltage} supply (defaulted)"
    )
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    if note not in assumptions:
        assumptions.append(note)
    completed["assumptions"] = assumptions
    return completed


_LOAD_POWER_STOPWORDS = {"block", "drive", "output", "the", "of", "and"}


def _defaulted_load_power_disclosed(assumption_rows, target_text: str) -> bool:
    """Whether a recorded default already discloses who powers this external load.

    The contract forbids *silently* making the board responsible for an external load's
    power. ``functional_spec`` is an auto-default stage, so under the production policy the
    response schema carries no ``questions`` array and the prompt says: apply the sensible
    default and record it in ``assumptions`` ending ``(defaulted)``. A row that names the
    load (or "external loads"/"both loads") and states that the board supplies its power is
    that disclosure. Refusing it would leave the stage unsatisfiable: the only other exit
    the checker offers -- a blocking question -- is disabled by the same policy.
    """
    words = {
        word
        for word in re.findall(r"[a-z0-9]+", str(target_text).casefold())
        if word not in _LOAD_POWER_STOPWORDS
    }
    for row in assumption_rows:
        text = str(row).casefold()
        if "(defaulted)" not in text:
            continue
        if not re.search(
            r"\b(?:suppl(?:y|ies|ied)|power(?:s|ed|ing)?|provid(?:e|es|ed)|feed(?:s|ed)?)\b", text
        ):
            continue
        if not (
            re.search(r"\b(?:board|on-?board)\b", text)
            or re.search(r"\b(?:both|the|external) loads?\b", text)
        ):
            continue
        if re.search(r"\b(?:both|external) loads?\b", text) or (
            words & set(re.findall(r"[a-z0-9]+", text))
        ):
            return True
    return False


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


def remove_board_feature_blocks(candidate: dict) -> dict:
    """Drop functional blocks that model a board feature, with every connection they carry.

    A prototyping pad field is not a functional block: it names no component function and
    carries no signal, so a block for it can only be wired through nets no requirement can
    own — the architecture stage then refuses with "crosses sheets but has no
    inter_sheet_net" and the design stops there (seen on the proto-shield r2 run, twice,
    after the semantic repair round the functional-spec stage still commits with).

    Removing it is deterministic and lossless: the pad field itself survives as the intent's
    `fabrication` row, from which the sheet and the pad grid are derived. The semantic
    diagnostic still reports the block, so the model's mistake stays visible.
    """
    from kicraft.design.synthesis.board_features import prototyping_area_requested

    blocks = candidate.get("blocks") or []
    board_feature_names = {
        str(block.get("name") or "")
        for block in blocks
        if isinstance(block, dict)
        and prototyping_area_requested(str(block.get("name") or "").replace("_", " "))
    }
    if not board_feature_names:
        return candidate
    completed = dict(candidate)
    completed["blocks"] = [
        block
        for block in blocks
        if not isinstance(block, dict) or str(block.get("name") or "") not in board_feature_names
    ]
    completed["connections"] = [
        connection
        for connection in candidate.get("connections") or []
        if isinstance(connection, dict)
        and str(connection.get("from_block") or "") not in board_feature_names
        and str(connection.get("to_block") or "") not in board_feature_names
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
    if not any(isinstance(sheet, dict) and sheet.get("name") == "POWER INPUT" for sheet in sheets):
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


def _passive_speaker_requested(brief: str, upstream: dict) -> bool:
    intent = upstream.get("intent") or {}
    text = (brief or str(intent.get("goal") or "")).lower()
    if not re.search(r"\b(?:speaker|loudspeaker)s?\b", text):
        return False
    # A crossover or a line-level feed to another amplifier has no onboard speaker-power duty.
    return not bool(re.search(
        r"\b(?:passive\s+(?:[\w-]+\s+){0,4}crossover|externally[- ]amplified|external\s+amplifier|"
        r"powered\s+speaker|active\s+speaker|line[- ]level)\b", text
    ))


def _speaker_functional_diagnostics(brief: str, upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    if not _passive_speaker_requested(brief, upstream):
        return []
    blocks = [row for row in candidate.get("blocks") or [] if isinstance(row, dict)]
    speaker_blocks = [row for row in blocks if re.search(r"speaker", _text(row), re.I)]
    if any(re.search(r"\b(?:driv\w*|amplif\w*|power(?:ed|ing)?)\b",
                     str(row.get("purpose") or ""), re.I)
           for row in speaker_blocks):
        return []
    return [_diag(
        "functional_spec_passive_speaker_drive_omitted", "repair_required",
        "A passive speaker output must provide speaker power drive, not just a logic-level signaling header. Default an 8 ohm passive speaker if unspecified.",
        [str(row.get("name") or "") for row in speaker_blocks],
    )]


def _speaker_architecture_diagnostics(brief: str, upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    if not _passive_speaker_requested(brief, upstream):
        return []
    requirements = [row for row in candidate.get("requirements") or [] if isinstance(row, dict)]
    for row in requirements:
        text = " ".join(str(row.get(key) or "") for key in ("family", "exact_part", "recipe")).lower()
        if re.search(r"pam8302|pwm.passive.speaker.driver", text):
            source = brief or str((upstream.get("intent") or {}).get("goal") or "")
            # Do not mistake a nearby panel, resistor or other load's rating for the speaker.
            clauses = re.split(r"[;\n]|\.(?!\d)", source.lower())
            ratings = []
            for clause in clauses:
                speakers = list(re.finditer(r"\b(?:loud)?speakers?\b", clause))
                for rating in re.finditer(
                    r"(?<![\w.])(\d+(?:\.\d+)?)[\s-]*(ohms?|ω|w(?:atts?)?)\b", clause
                ):
                    for speaker in speakers:
                        start, end = sorted((rating.end(), speaker.start()))
                        between = clause[start:end]
                        if len(between) <= 35 and not re.search(
                            r"\b(?:resistor|led|panel|display|supply|rail|motor)\b", between
                        ):
                            ratings.append(rating.groups())
                            break
            for stage in ("intent", "functional_spec"):
                for obligation in (upstream.get(stage) or {}).get("obligations") or []:
                    if (
                        obligation.get("kind") == "quantitative"
                        and re.search(r"\b(?:loud)?speaker\b", str(obligation.get("quantity") or ""), re.I)
                        and obligation.get("value") is not None
                        and str(obligation.get("unit") or "").lower() in {"ohm", "ohms", "ω", "w", "watt", "watts"}
                    ):
                        ratings.append((str(obligation["value"]), str(obligation["unit"]).lower()))
            unsupported = [
                f"{number} {unit}" for number, unit in ratings
                if (unit.startswith("ohm") or unit == "ω") and float(number) != 8
                or unit.startswith("w") and float(number) < 2
            ]
            if unsupported:
                return [_diag(
                    "architecture_unsupported_speaker_load", "repair_required",
                    "The reviewed PWM speaker circuit supports an 8 ohm speaker rated at least 2 W; preserve the explicitly requested load and choose a supported design rather than silently changing it.",
                    unsupported,
                )]
            return []
        if re.search(r"speaker.driver|audio.power|audio.amplif", text):
            return []
    return [_diag(
        "architecture_passive_speaker_drive_omitted", "repair_required",
        "A GPIO or signaling header cannot drive a passive speaker. Use the reviewed pam8302a-pwm-speaker@1 recipe, with its owned two-pin bridge output, 5 V supply and filtered PWM input.",
        [str(row.get("id") or "") for row in requirements if re.search(r"speaker", _text(row), re.I)],
    )]


def _functional_spec(brief: str, upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
    diagnostics.extend(_speaker_functional_diagnostics(brief, upstream, candidate))
    intent = upstream.get("intent", {})
    allowed = _text([brief, intent]).lower()
    assumption_rows = [str(item) for item in candidate.get("assumptions") or []]
    assumptions = " ".join(assumption_rows).lower()

    introduced = sorted(
        {m.group(1).lower() for m in _TOPOLOGY_RE.finditer(_committed_topology_text(candidate))}
    )
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
        # The vocabulary that proves the brief assigned this power duty must cover the load the
        # connection names. It only ever offered LED/display terms, so a brief that explicitly
        # powers its motors (or heater) could never satisfy the rule and was refused anyway --
        # the check's own `motor|heater` branch could not be discharged (replay 2026-09-26).
        if re.search(r"hub75|display", target_text, re.I):
            target_terms = r"hub75|display|panel"
        elif re.search(r"motors?", target_text, re.I):
            target_terms = r"motors?"
        elif re.search(r"heaters?", target_text, re.I):
            target_terms = r"heaters?|heating"
        else:
            target_terms = r"addressable led|led string|led strip|leds"
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
        if not (
            explicitly_powered
            or _defaulted_load_power_disclosed(assumption_rows, target_text)
        ):
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
    requested_functions = _text(
        [brief, intent.get("goal"), intent.get("constraints")]
    ).lower()
    if re.search(
        r"(?<!not )(?<!non-)\bbattery[- ]powered\b|"
        r"(?<!not )\bpowered\s+(?:from|by)\s+(?:an?\s+)?batter(?:y|ies)\b",
        requested_functions,
    ):
        battery_sources = {
            name
            for name, block in blocks_by_name.items()
            if block.get("category") == "power"
            and re.search(
                r"\bbatter(?:y|ies)\b|\bli[- ]?ion\b|\blipo\b",
                _text([name.replace("_", " "), block.get("purpose")]),
                re.I,
            )
        }
        reachable = set(battery_sources)
        pending = list(battery_sources)
        power_edges: dict[str, list[str]] = {}
        for connection in connections:
            if connection.get("signal_type") == "power":
                power_edges.setdefault(str(connection.get("from_block")), []).append(
                    str(connection.get("to_block"))
                )
        while pending:
            for target in power_edges.get(pending.pop(), []):
                if target in blocks_by_name and target not in reachable:
                    reachable.add(target)
                    pending.append(target)
        denied_battery_operation = [
            row for row in assumption_rows
            if re.search(
                r"\bno\s+battery\s+power\s+path\b|"
                r"\bbattery\s+(?:operation|power(?:\s+path)?)\s+"
                r"(?:is\s+)?(?:omitted|excluded|not\s+included)\b",
                row, re.I,
            )
        ]
        if not any(
            blocks_by_name[name].get("category") in {"sense", "process", "drive", "interface"}
            for name in reachable - battery_sources
        ) or denied_battery_operation:
            diagnostics.append(
                _diag(
                    "functional_spec_battery_operation_omitted",
                    "repair_required",
                    "Requested battery operation needs a battery power-source function and "
                    "a power flow to the functions it supplies. Unspecified chemistry or "
                    "charging does not justify omitting battery operation. Assumptions must "
                    "not contradict that required power path.",
                    [*sorted(battery_sources), *denied_battery_operation],
                )
            )
    if re.search(r"\bambient[- ]light[- ]+(?:node|sensor|channels?)\b", requested_functions):
        light_sources = {
            name
            for name, block in blocks_by_name.items()
            if block.get("category") == "sense"
            and re.search(
                r"\blight\b|\bphoto(?:transistor|diode)\b|\blux\b",
                _text([name.replace("_", " "), block.get("purpose")]),
                re.I,
            )
        }
        if not any(
            connection.get("from_block") in light_sources
            and connection.get("signal_type") in {"analog", "digital", "other"}
            and blocks_by_name.get(str(connection.get("to_block")), {}).get("category")
            in {"process", "interface", "drive"}
            for connection in connections
        ):
            diagnostics.append(
                _diag(
                    "functional_spec_ambient_light_acquisition_omitted",
                    "repair_required",
                    "An ambient-light node, sensor or channel requires a light-acquisition "
                    "function with a measurement flow to processing or an output interface; "
                    "a processor described as an ambient-light node does not acquire light.",
                    sorted(light_sources),
                )
            )
    # A connection that starts and ends at the same block states no flow between blocks.
    # Live run 6 (KC-KAHKR7, seed 23) died on one and the only feedback the model got was the
    # commit gate's bare `self-loop connection: 'DISPLAY_DRIVE' → 'DISPLAY_DRIVE'`, which
    # carries no diagnostic code: the model could not tell a semantic defect it owned from a
    # schema problem, and triage could not see it as a named refusal. Name it here, on the
    # same surface as every other functional-spec defect, with the offending pair as evidence.
    self_loops = [
        f"{connection.get('from_block')!r} -> {connection.get('to_block')!r}"
        for connection in connections
        if connection.get("from_block")
        and connection.get("from_block") == connection.get("to_block")
    ]
    if self_loops:
        diagnostics.append(
            _diag(
                "functional_spec_self_loop",
                "repair_required",
                "A functional connection starts and ends at the same block, so it states no "
                "flow between blocks; name the block the signal actually moves to.",
                self_loops,
            )
        )
    # A block whose every committed obligation is a board-wide fact is realized by the printed
    # board itself, so no architecture sheet can ever implement it -- the architecture's own
    # block/sheet mapping gate exempts such a block from needing a requirement. A *connection*
    # naming it is therefore unmappable for good, and the spec is frozen by the time the
    # architecture gate sees it. Live cohort 2026-09-30, brief 3: the functional spec committed a
    # PCB_STACKUP block (obligation `pcb-copper-layers`, a board-wide stack-up measurement) plus a
    # PCB_STACKUP -> POWER_INPUT connection, and every architecture draft was refused with
    # "connection 'PCB_STACKUP'->'POWER_INPUT' ... has an unknown or unmapped block endpoint"
    # until the stage exhausted its attempts. Refuse it here, where a redraft can drop the
    # connection and leave the board-wide row board-wide.
    board_realized = {
        name
        for name, block in blocks_by_name.items()
        if _block_is_board_realized(block, intent)
    }
    unmappable_connections = sorted(
        f"{connection.get('from_block')!r} -> {connection.get('to_block')!r}"
        for connection in connections
        if connection.get("from_block") in board_realized
        or connection.get("to_block") in board_realized
    )
    if unmappable_connections:
        diagnostics.append(
            _diag(
                "functional_spec_board_realized_block_connected",
                "repair_required",
                "A block realized by the printed board itself (its only obligations are board-wide "
                "facts) has no sheet, so a connection naming it can never be mapped to one; remove "
                "the connection and leave the board-wide fact at the board level.",
                unmappable_connections,
            )
        )
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

    # A board feature is not a function: a prototyping pad field names no component function,
    # owns no pin and carries no signal of its own, so it is not a functional block and no
    # connection may touch it. Its sheet and its bare pad grid are derived downstream from the
    # intent's `fabrication` row; a block for it here asks the architecture for a part that
    # cannot exist -- and would then need a bound port on a sheet whose pads carry no net at all,
    # which the block/connection mapping gate and the pad-field lowerer both refuse.
    board_field_blocks = [
        f"{block.get('name')}: remove this block — a board feature is not a functional block; "
        "the sheet and pad field are derived from the intent's fabrication obligation"
        for block in candidate.get("blocks") or []
        if isinstance(block, dict) and _names_board_field(block.get("name"))
    ]
    if board_field_blocks:
        diagnostics.append(
            _diag(
                "functional_spec_board_feature_block",
                "repair_required",
                "A board feature is not a functional block; remove it and let the derived sheet "
                "own the pad field.",
                board_field_blocks,
            )
        )

    ground_connections = [
        connection for connection in connections if connection.get("signal_type") == "ground"
    ]
    if ground_connections:
        # A block is grounded when it appears at *either* end of a ground connection, not only
        # as the target: a design whose ground reference originates at its input connector
        # (the BNC/screw-terminal case, recorded as a common-ground assumption) lists that
        # block as the source, and demanding it also be its own sink refused the valid design.
        ground_members = {
            str(connection.get(field) or "")
            for connection in ground_connections
            for field in ("from_block", "to_block")
        }
        expected_ground = {
            name
            for name, block in blocks_by_name.items()
            # A `mechanical` block is a board feature (mounting holes, a logo): it draws no
            # current and owns no return, so demanding a ground flow refused a shipped
            # USB-UART bridge whose only "ungrounded" block was MOUNTING_FEATURES.
            if block.get("category") not in {"power", "mechanical"}
        }
        missing_ground = sorted(expected_ground - ground_members)
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


@lru_cache(maxsize=2)
def _onboard_relay_coil_rating(voltage: float) -> tuple[str, float] | None:
    """Read the selected canonical construction's reviewed nominal coil rating."""
    from kicraft.design.actuator_circuits import RELAY_OUTPUT_LOWERER
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.part_identity import physical_inventory_record

    # Select the actual lowerer's supply variant; the contact rails are not
    # substitutes for its coil supply. Unsupported parameters are rejected by
    # the construction validator, not assigned an invented relay rating here.
    ports = {"vmotor": "SUPPLY", "gnd": "GND"}
    for index in (1, 2):
        ports[f"control_{index}"] = f"CONTROL_{index}"
        for contact in ("common", "normally_open", "normally_closed"):
            ports[f"relay_{index}_{contact}"] = f"CONTACT_{index}_{contact}"
    artifact = RELAY_OUTPUT_LOWERER.build(CircuitRequirement(
        id="relay_rating_witness", sheet="RELAY", role="driver",
        family="relay-output", parameters={"channel_count": 2, "motor_voltage_v": voltage}, ports=ports,
    ))
    if artifact is None:
        return None
    group = next(group for group in artifact.groups if group.role == "relay")
    record = physical_inventory_record(
        mpn=group.mpn, symbol=group.symbol, footprint=group.footprint
    )
    if record is None or not isinstance(record.operating_limits.get("coil_voltage_v"), (int, float)):
        raise RuntimeError("Canonical on-board relay has no reviewed coil voltage")
    return record.identity, float(record.operating_limits["coil_voltage_v"])


def _onboard_relay_supply_diagnostics(requirement: dict, rails: dict) -> list[StageDiagnostic]:
    from kicraft.design.actuator_circuits import RELAY_OUTPUT_LOWERER

    if requirement.get("family") not in RELAY_OUTPUT_LOWERER.families:
        return []
    net = (requirement.get("ports") or {}).get("vmotor")
    voltage = rails.get(net)
    if type(voltage) not in (int, float):
        return []
    try:
        selected_voltage = float((requirement.get("parameters") or {}).get("motor_voltage_v", 5.0))
    except (TypeError, ValueError):
        return []
    selected = _onboard_relay_coil_rating(selected_voltage)
    if selected is None:
        return []
    identity, rating = selected
    if abs(voltage - rating) <= 0.05:
        return []
    evidence = [
        f"requirement {requirement.get('id')!r} on sheet {requirement.get('sheet')!r}",
        f"ports.vmotor={net!r}; declared rail voltage={voltage!r} V",
        f"canonical on-board relay {identity!r} nominal coil voltage={rating:g} V",
    ]
    return [_diag(
        "architecture_relay_coil_supply_mismatch",
        "repair_required",
        "The declared relay coil supply conflicts with the canonical on-board relay rating. "
        "Repair the actual supply binding and its physical source while preserving the declared "
        "requirements and voltages; isolated contact nets are separate from the coil supply. "
        + "; ".join(evidence),
        evidence,
    )]


def _linear_regulator_dissipation_diagnostics(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    """Reject a proven excess over a reviewed linear regulator's absolute dissipation limit."""
    from kicraft.design.part_identity import reviewed_part

    rails = _architecture_rail_voltages(candidate)
    source_rows = {
        str(row.get("original_obligation_id")): row
        for slot in ("intent", "functional_spec")
        for row in (upstream.get(slot) or {}).get("obligations") or []
        if isinstance(row, dict) and row.get("original_obligation_id")
    }
    diagnostics = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict) or requirement.get("role") != "regulator":
            continue
        record = reviewed_part(str(requirement.get("exact_part") or ""))
        if record is None or record.family != "linear-regulator":
            continue
        limit = record.operating_limits.get("dissipated_power_limit_w")
        if not isinstance(limit, (int, float)) or limit <= 0:
            continue
        ports = requirement.get("ports") or {}
        params = requirement.get("parameters") or {}
        input_values = [rails.get(ports.get("input") or ports.get("vin")), params.get("input_voltage")]
        output_values = [rails.get(ports.get("output") or ports.get("vout")), params.get("output_voltage")]
        owned = [source_rows[key] for key in _requirement_obligation_ids(requirement) if key in source_rows]
        owned.extend(row for row in requirement.get("obligations") or [] if isinstance(row, dict))
        input_net = ports.get("input") or ports.get("vin")
        source_voltage_rows = []
        for source in candidate.get("requirements") or []:
            if (
                not isinstance(source, dict)
                or source.get("role") not in {"power_input", "connector"}
                or not input_net
                or input_net not in (source.get("ports") or {}).values()
            ):
                continue
            source_voltage_rows.extend(
                source_rows[key] for key in _requirement_obligation_ids(source) if key in source_rows
            )
            source_voltage_rows.extend(
                row for row in source.get("obligations") or [] if isinstance(row, dict)
            )
        for row in source_voltage_rows:
            quantity = str(row.get("quantity") or "").casefold()
            unit = str(row.get("unit") or "").casefold().strip()
            if (
                row.get("kind") != "quantitative"
                or "voltage" not in quantity
                or not re.search(r"\b(?:input|source|supply|battery|pack|rail)\b", quantity)
                or unit not in {"v", "mv"}
            ):
                continue
            input_values.extend(
                value / (1000 if unit == "mv" else 1)
                for value in (row.get("value"), row.get("minimum"), row.get("maximum"))
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        current_values = []
        for row in owned:
            if row.get("kind") != "quantitative":
                continue
            quantity = str(row.get("quantity") or "").casefold()
            unit = str(row.get("unit") or "").casefold().strip()
            values = [value for value in (row.get("value"), row.get("minimum"), row.get("maximum"))
                      if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if "current" in quantity and unit in {"a", "ma"}:
                current_values.extend(value / (1000 if unit == "ma" else 1) for value in values)
            if "input" in quantity and "voltage" in quantity and unit in {"v", "mv"}:
                input_values.extend(value / (1000 if unit == "mv" else 1) for value in values)
            if "output" in quantity and "voltage" in quantity and unit in {"v", "mv"}:
                output_values.extend(value / (1000 if unit == "mv" else 1) for value in values)
        inputs = [float(value) for value in input_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        outputs = [float(value) for value in output_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not inputs or not outputs or not current_values:
            continue
        vin, vout, current = max(inputs), min(outputs), max(current_values)
        loss = max(0.0, vin - vout) * current
        if loss > float(limit) + 1e-9:
            evidence = (
                f"requirement {requirement.get('id')!r}: ({vin:g} V - {vout:g} V) × "
                f"{current:g} A = {loss:g} W; {record.identity} absolute limit {limit:g} W"
            )
            diagnostics.append(_diag(
                "architecture_linear_regulator_dissipation_exceeded", "repair_required",
                "The required linear-regulator loss exceeds its reviewed absolute dissipation limit. "
                "A value below that limit alone would not establish thermal qualification.",
                [evidence],
            ))
    return diagnostics


def architecture_power_requirement_diagnostics(
    upstream: dict, candidate: dict
) -> list[StageDiagnostic]:
    """Reject power functions with no independently owned physical implementation."""
    requirements = [row for row in candidate.get("requirements") or [] if isinstance(row, dict)]
    blocks = {
        str(block.get("name")): str(block.get("purpose") or "")
        for block in (upstream.get("functional_spec") or {}).get("blocks") or []
        if isinstance(block, dict)
    }
    rails = candidate.get("rail_voltages") or {}
    declared_rails = _architecture_rail_voltages(candidate)
    conversions = _conversion_source_obligations(upstream)
    diagnostics = _linear_regulator_dissipation_diagnostics(upstream, candidate)
    for requirement in requirements:
        diagnostics.extend(_onboard_relay_supply_diagnostics(requirement, declared_rails))
        family = _norm_token(requirement.get("family") or "")
        generic_power = family in {
            "powerinput",
            "powerconversion",
            "powerdistribution",
            "directbatteryrail",
        }
        distribution = family in {"powerdistribution", "directbatteryrail"}
        power_conversions = [
            identifier
            for identifier in sorted(_requirement_obligation_ids(requirement) & conversions.keys())
            if _is_power_conversion(conversions[identifier])
        ]
        if not generic_power and requirement.get("role") not in {"power_input", "regulator"}:
            continue
        ports = requirement.get("ports") or {}
        port_aliases = {_norm_token(key): net for key, net in ports.items()}
        input_net = port_aliases.get("input") or port_aliases.get("vin")
        output_net = port_aliases.get("output") or port_aliases.get("vout")
        output_nets = [output_net] if output_net else [
            net for key, net in port_aliases.items()
            if re.fullmatch(r"(?:output|vout)[1-9][0-9]*", key)
        ]
        ground_net = port_aliases.get("gnd") or port_aliases.get("ground")
        parameters = requirement.get("parameters") or {}
        input_voltage = parameters.get("input_voltage", rails.get(input_net))
        output_voltage = parameters.get("output_voltage", rails.get(output_net))
        voltage_change = (
            type(input_voltage) in (int, float)
            and type(output_voltage) in (int, float)
            and abs(input_voltage - output_voltage) > 0.05
        )
        # This checks declared ownership, not hardware capability. A same-voltage
        # path may be protection or a fused distribution, not an invented
        # regulator. Different voltages still need a physical regulator.
        non_converter_family = bool(re.search(
            r"(?:connector|socket|terminal|protection|distribution|batteryrail|"
            r"powersink|powerinput|fuse)",
            family,
        ))
        physical_converter = (
            requirement.get("role") == "regulator"
            and bool(family or requirement.get("exact_part"))
            and not generic_power
            and not non_converter_family
        )
        physical_path = (
            bool(family or requirement.get("exact_part"))
            and not generic_power
            and not re.search(
                r"(?:connector|socket|terminal|batteryrail|powersink|powerinput)",
                family,
            )
        )
        typed_voltage_change = any(
            len(inputs := _TYPED_VOLTAGE_RE.findall(str(conversions[identifier].get("input_kind") or ""))) == 1
            and (output := _typed_output_voltage(conversions[identifier])) is not None
            and abs(float(inputs[0]) - output) > 0.05
            for identifier in power_conversions
        )
        owns_path = physical_converter if voltage_change or typed_voltage_change else physical_path
        distinct_ports = (
            input_net and ground_net and output_nets
            and len({input_net, ground_net, *output_nets}) == len(output_nets) + 2
        )
        incomplete_conversion = (
            bool(power_conversions) and (not owns_path or not distinct_ports)
        ) or (
            voltage_change and (generic_power or not distinct_ports)
        )
        if not distribution and not incomplete_conversion:
            continue
        owner = (
            f"requirement {requirement.get('id')!r} on sheet {requirement.get('sheet')!r} "
            f"(role={requirement.get('role')!r}, family={requirement.get('family')!r})"
        )
        evidence = [
            owner,
            *(f"ports.{key}={net!r}" for key, net in sorted(ports.items())),
            *(
                f"functional block {name!r}: {blocks.get(name, '<purpose unavailable>')}"
                for name in requirement.get("functional_blocks") or []
            ),
        ]
        if power_conversions:
            evidence.extend(
                f"committed power conversion {identifier!r}: "
                f"input_kind={conversions[identifier].get('input_kind')!r}; "
                f"output_kind={conversions[identifier].get('output_kind')!r}"
                for identifier in power_conversions
            )
        if voltage_change:
            evidence.extend(
                [
                    f"input_voltage={input_voltage!r}",
                    f"output_voltage={output_voltage!r}",
                    f"input net={input_net!r}; output net={output_net!r}; ground net={ground_net!r}",
                ]
            )
        if incomplete_conversion:
            repair = (
                "A committed power path needs a typed physical owner with distinct "
                "input/output/GND ports before BOM. Different voltages require a regulator; "
                "an unchanged-voltage path may belong to the actual protection or fused "
                "distribution circuit. A source connector alone does not implement that path. "
                "Preserve declared voltages, exact identities and functional ownership; "
                "reuse the existing physical owner when it implements the full function."
            )
            code = "architecture_unowned_power_conversion"
        else:
            for sibling in requirements:
                if sibling is requirement or sibling.get("sheet") != requirement.get("sheet"):
                    continue
                shared = set(ports.values()) & set((sibling.get("ports") or {}).values())
                if shared:
                    evidence.append(
                        f"same-sheet requirement {sibling.get('id')!r} "
                        f"({sibling.get('family')!r}) shares nets {sorted(shared)!r}"
                    )
            repair = (
                "A distribution-only requirement cannot own a separate nonempty BOM unit. "
                "Specify the actual conditioning/filter/protection circuit and its physical "
                "port bindings. If only shared wiring is intended, assign that function to "
                "an existing physical owner only when it implements the full committed "
                "functional purpose. Preserve every functional block and net; do not "
                "duplicate a source/holder, erase conditioning, or emit an empty BOM."
            )
            code = "architecture_unowned_power_support"
        diagnostics.append(
            _diag(code, "repair_required", repair + " " + "; ".join(evidence), evidence)
        )
    return diagnostics


def _rail_producers(candidate: dict, rails: dict) -> list[dict]:
    """Every requirement that generates a declared rail, with the recipe's reviewed rating.

    The fact the ESP32-S3 3.3V check needs is the *part's*, not the model's prose:
    the requirement's recipe port named ``output``/``vout`` bound to a declared
    rail means that part drives it, and the current is the datasheet figure the
    recipe reviews (`RecipeDefinition.rated_output_current_a`). ``None`` there
    means the registry holds no rating for that part, and the caller treats it as
    unproven rather than as a number. This reads no topology text: how the model
    phrased the converter no longer decides the check.
    """
    from kicraft.design.recipes import get_recipe

    resolutions = {
        str(row.get("requirement_id")): str(row.get("recipe"))
        for row in candidate.get("recipe_resolution") or []
        if isinstance(row, dict) and row.get("requirement_id") and row.get("recipe")
    }
    rows = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        ports = {_norm_token(key): net for key, net in (requirement.get("ports") or {}).items()}
        rail = ports.get("output") or ports.get("vout")
        if not isinstance(rail, str) or rail not in rails:
            continue
        recipe = resolutions.get(str(requirement.get("id")))
        rows.append(
            {
                "rail": rail,
                "sheet": str(requirement.get("sheet") or ""),
                "requirement_id": str(requirement.get("id") or ""),
                "rated_output_current_a": (
                    get_recipe(recipe).rated_output_current_a if recipe else None
                ),
            }
        )
    return rows


def _architecture_rail_voltages(candidate: dict) -> dict[str, float]:
    """Declared rail voltages, in either candidate shape.

    The model states ``power.rails[name].voltage``; the architecture response contract derives
    the slot before anything diagnoses it, and the derived shape states the same numbers as
    ``rail_voltages``. Reading only the first made this family of checks dead in production:
    a live walkthrough draft (2026-09-25) put the 18 V input straight on a DRV8833's ``vm``
    (reviewed maximum 10.8 V) and reported zero diagnostics.
    """
    voltages: dict[str, float] = {}
    for name, row in ((candidate.get("power") or {}).get("rails") or {}).items():
        try:
            voltages[str(name)] = float((row or {}).get("voltage"))
        except (AttributeError, TypeError, ValueError):
            continue
    for name, volts in (candidate.get("rail_voltages") or {}).items():
        try:
            voltages.setdefault(str(name), float(volts))
        except (TypeError, ValueError):
            continue
    return voltages


def _requirement_supply_rails(requirement: dict) -> list[str]:
    """Every rail a requirement is powered from, in either candidate shape.

    The model writes ``supply``; the derived shape binds the family's own supply ports
    (``vm``, ``vdd``, ``input`` …) to their nets in ``ports``. Both are the same fact.
    """
    rails: list[str] = []
    declared = requirement.get("supply")
    if isinstance(declared, str) and declared:
        rails.append(declared)
    from kicraft.design.architecture_intent import _supply_port_name

    for port, net in (requirement.get("ports") or {}).items():
        if (str(port).casefold() in {"input", "input_positive"} or
                _supply_port_name(str(port).casefold())) and str(net) not in rails:
            rails.append(str(net))
    return rails


def _architecture_supply_over_rating(candidate: dict) -> list[StageDiagnostic]:
    """A rail a requirement is powered from that exceeds every supply rating its part publishes.

    The reviewed records carry the ratings (the DRV8833's ``motor_supply_max_v`` 10.8 V against
    its ``vm`` pin), so an 18 V rail landing on that pin is a fault the architecture states
    outright -- and the writer can fix it here by making the rail a regulated one. The
    build-time range check only sees this after the fact, and used to skip the motor domain
    entirely; a refusal at this stage is what lets the design step the rail down.
    """
    from kicraft.design.part_identity import reviewed_part, reviewed_supply_voltage_limits

    rails = _architecture_rail_voltages(candidate)
    diagnostics: list[StageDiagnostic] = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        exact = str(requirement.get("exact_part") or "").strip()
        record = reviewed_part(exact) if exact else None
        if record is None:
            continue
        rated = [row for row in reviewed_supply_voltage_limits(record) if row[2] is not None]
        if not rated:
            continue
        allowed = max(row[2] for row in rated)
        over = [
            (rail_name, rails[rail_name])
            for rail_name in _requirement_supply_rails(requirement)
            if rails.get(rail_name) is not None and rails[rail_name] > allowed
        ]
        if not over:
            continue
        rail_name, voltage = max(over, key=lambda pair: pair[1])
        label, _low, worst_row = min(rated, key=lambda row: row[2])
        worst = worst_row if len(rated) == 1 else allowed
        diagnostics.append(
            _diag(
                "architecture_supply_exceeds_part_rating",
                "repair_required",
                "A requirement is powered from a rail above every supply rating its reviewed part "
                "publishes.",
                [
                    f"requirement {requirement.get('id')!r} "
                    f"(family={requirement.get('family')!r}, exact_part={exact!r}) "
                    f"supply={rail_name!r} ({voltage:g}V); reviewed {record.identity!r} "
                    f"allows {label} {worst:g}V — add a regulator that steps this rail down into "
                    "range and keep the load drive on the regulated rail, or choose a part rated "
                    f"for {voltage:g}V",
                ],
            )
        )
    return diagnostics

def _architecture_motor_clamp_standoff(candidate: dict) -> list[StageDiagnostic]:
    """The native DRV recipe's clamp narrows the IC's otherwise wider VM range."""
    rails = _architecture_rail_voltages(candidate)
    offenders = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        exact = str(requirement.get("exact_part") or "").casefold()
        if exact and exact != "drv8833pwpr":
            continue
        if not exact and str(requirement.get("family") or "").casefold() != "dual-dc-motor-driver":
            continue
        vm = (requirement.get("ports") or {}).get("vm") or requirement.get("supply")
        voltage = rails.get(str(vm))
        if voltage is not None and voltage > 5.0:
            offenders.append(f"requirement {requirement.get('id')!r} VM {vm!r} is "
                             f"{voltage:g} V; the native SMBJ5.0A standoff is 5 V")
    return [_diag("architecture_motor_clamp_standoff", "repair_required",
                  "The native DRV8833 clamp cannot support this nominal motor rail.",
                  offenders)] if offenders else []



def complete_usb_socket_rail(candidate: dict) -> dict:
    """Declare the rail a native-USB socket exposes, and keep it resolvable.

    Sending ``usb_dm``/``usb_dp`` to an ``edge:`` peer makes the compiler write the socket, and
    that socket carries VBUS. Two defects follow from that, and both are the writer's to state
    rather than the compiler's to guess: no ~5 V rail at all (``usb_connector_supply_unknown``,
    which then reports the MCU's USB pins as unwired), or a rail sourced from a requirement the
    draft never declared — the socket is compiler-created, so ``<socket>.vbus`` cannot resolve
    (``unknown_signal_requirement``, the one refusal repeated in every round of the seed-37
    walkthrough's last draft). The rail is the host's, so the honest statement is
    ``from: null``; the unsourced-rail completion and the design assumptions then say where it
    comes from.
    """
    rails = (candidate.get("power") or {}).get("rails")
    if not isinstance(rails, dict):
        return candidate
    port_keys = {
        str(reference).rsplit(".", 1)[-1].casefold()
        for signal in candidate.get("signals") or []
        if isinstance(signal, dict)
        for reference in [signal.get("from"), *(signal.get("to") or [])]
        if isinstance(reference, str)
    }
    if not port_keys & {"usb_dm", "usb_dp"}:
        return candidate

    def _is_socket_rail(name: str, rail: dict) -> bool:
        voltage = rail.get("voltage")
        if isinstance(voltage, (int, float)) and abs(float(voltage) - 5.0) <= 0.5:
            return True
        return str(name).strip().upper() in {"VBUS", "+5V"}

    declared = {
        str(row.get("id")) for row in candidate.get("requirements") or [] if isinstance(row, dict)
    }
    socket_rails = {
        name: rail
        for name, rail in rails.items()
        if isinstance(rail, dict) and _is_socket_rail(str(name), rail)
    }
    unresolvable = sorted(
        name
        for name, rail in socket_rails.items()
        if isinstance(rail.get("from"), str)
        and str(rail["from"]).partition(".")[0] not in declared
    )
    if socket_rails and not unresolvable:
        return candidate

    completed = copy.deepcopy(candidate)
    updated = dict(completed["power"]["rails"])
    for name in unresolvable:
        updated[name] = {**updated[name], "from": None}
    if not socket_rails:
        updated["VBUS"] = {"voltage": 5.0, "from": None}
    completed["power"] = {**completed["power"], "rails": updated}
    note = "VBUS: the USB socket's 5 V host rail, declared for the USB data connector (defaulted)"
    assumptions = [str(item) for item in completed.get("assumptions") or []]
    if note not in assumptions:
        assumptions.append(note)
    completed["assumptions"] = assumptions
    return completed


def _architecture_derivation_diagnostics(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    """The design-level contracts the derivation enforces, as repairable diagnostics.

    ``derive_architecture`` refuses a payload that breaks a design contract — a USB edge with no
    5 V rail, a lowerer whose declared contacts cannot be numbered, a named part with no owning
    requirement. Those refusals used to surface only at the commit step, where the loop spends its
    single from-scratch retry on them and then fails the stage (four live drafts, no candidate).
    Reported here they join the ordinary repair path instead: the correction keeps every other
    part of the candidate and carries the whole defect list at once, which is what a draft with
    several independent structural faults needs.
    """
    from kicraft.design.architecture_intent import ArchitectureIntentError, derive_architecture
    from pydantic import ValidationError

    try:
        derive_architecture(candidate, upstream.get("functional_spec"))
    except ValidationError:
        # An incomplete payload is the response-schema lane's business, not this one.
        return []
    except ArchitectureIntentError as exc:
        rows = list(getattr(exc, "diagnostics", None) or [])
        if not rows:
            return [
                _diag(
                    "architecture_derivation_refused",
                    "repair_required",
                    str(exc),
                    [],
                )
            ]
        diagnostics: list[StageDiagnostic] = []
        for row in rows:
            payload = row.model_dump(exclude_none=True) if hasattr(row, "model_dump") else dict(row)
            diagnostics.append(
                _diag(
                    str(payload.get("code") or "architecture_derivation_refused"),
                    str(payload.get("severity") or "repair_required"),
                    str(payload.get("message") or str(exc)),
                    [str(item) for item in payload.get("evidence") or []],
                )
            )
        return diagnostics
    return []


#: Roles that a rail powering a drive part must not be shared with: these are the logic loads
#: whose supply the drive would then steal headroom from.
_LOGIC_ROLES = frozenset(
    {"mcu_core", "sensor", "bus_interface", "programming", "analog_block", "user_io"}
)

#: Supply-port names that a part *generates* rather than consumes.
_GENERATOR_PORTS = ("output", "vout", "sw")


#: Words that mean "the load this board drives", for finding claims about its supply.
_LOAD_WORDS = r"(?:motor|actuator|solenoid|heater|load)"


def _load_part_tokens(requirement: dict, exact: str) -> list[str]:
    """The tokens that identify this part in the writer's own prose."""
    tokens = {
        str(requirement.get("id") or "").strip().casefold(),
        str(requirement.get("family") or "").strip().casefold(),
        str(exact or "").strip().casefold(),
    }
    for name in requirement.get("functional_blocks") or []:
        tokens.add(str(name).replace("-", " ").replace("_", " ").casefold())
    return sorted(token for token in tokens if token)


def _mentions_load_part(text: str, tokens: list[str], exact: str) -> bool:
    """Whether this field names the part, by its id/family/block or by its order code.

    The writer names a part three ways in prose -- the requirement id, the block it implements,
    and the order code -- and shortens the order code ("DRV8833" for "DRV8833PWPR"). A short
    form that prefixes the code it belongs to is the same part.
    """
    lowered = str(text).casefold()
    if any(token in lowered for token in tokens):
        return True
    order_code = str(exact or "").casefold()
    if not order_code:
        return False
    return any(
        order_code.startswith(word) for word in re.findall(r"[a-z0-9]{4,}", lowered)
    )


def _voltage_pattern(voltage: float) -> str:
    return rf"\b{re.escape(f'{float(voltage):g}')}\s*v\b"


def _rewrite_load_supply_claim(
    text: str,
    tokens: list[str],
    exact: str,
    wrong_rail: str,
    wrong_voltage: float,
    new_rail: str,
    new_voltage: float,
    *,
    force: bool = False,
) -> str:
    """Point a claim about a load's supply at the regulated rail it now has.

    ``force`` skips the part-name gate for narrative fields: a sheet function that says
    "Drive the actuators from the 18 V motor supply" never names the part, but it makes the
    same claim about the same rail.
    """
    if not text:
        return text
    row = str(text)
    if not force and not any(token in row.casefold() for token in tokens):
        return row
    changed_before = row
    row = _rewrite_rail_claims(row, tokens, exact, wrong_rail, new_rail)
    row = _rewrite_voltage_claims(row, tokens, exact, wrong_voltage, new_voltage)
    if (row != changed_before or _mentions_load_part(row, tokens, exact)) and (
        new_rail.casefold() not in row.casefold()
    ):
        # The part's supply is now a rail the sentence never named ("fed from the 18 V input"
        # becomes "fed from the 10 V input", which reads as if the board input changed). Name
        # the rail the design actually uses rather than leaving the reader to infer it.
        row = f"{row} (fed from the regulated {new_rail} rail)"
    return row


def _claim_clause(row: str, start: int, end: int) -> str:
    """The statement a claim sits in, so a rewrite stays inside it.

    Two separators, because both join independent statements: ``;`` and ``and`` -- "18 V DC
    input feeding the DRV8833 motor supply directly **and** a 3.3 V buck regulator" claims two
    different things, and judging them together let the converter mention excuse the load claim
    (live draft, 2026-09-25).
    """
    left = row.rfind(";", 0, start)
    right = row.find(";", end)
    clause_left = left + 1
    clause = row[clause_left : right if right != -1 else len(row)]
    offset = start - clause_left
    position = 0
    for piece in re.split(r" and ", clause, flags=re.I):
        if position <= offset < position + len(piece):
            return piece
        position += len(piece) + len(" and ")
    return clause


def _drives_the_load(clause: str) -> bool:
    """Whether this clause states that something drives or supplies a load.

    Only such a clause carries a supply claim to correct: "Drive two actuator outputs from the
    18 V input" does, while "Run actuator control and provide H-bridge control signals from the
    18 V input" describes signals the MCU produces and "18 V input with buck conversion to 3.3 V"
    describes the board input. Correcting either of those would falsify a true statement.
    """
    return bool(
        re.search(_LOAD_WORDS, clause, re.I)
        and re.search(
            r"\b(?:drive[sdn]?|driven|feed(?:s|ing)?|fed|power(?:s|ed|ing)?|"
            r"suppl(?:y|ies|ied|ying))\b",
            clause,
            re.I,
        )
    )


def _rewrite_rail_claims(
    row: str, tokens: list[str], exact: str, wrong_rail: str, new_rail: str
) -> str:
    """Name the regulated rail wherever a clause about this load named the wrong one.

    Only clauses that mention the part or a load are touched: a statement about the board input
    ("the +18V input") stays exactly as the writer wrote it.
    """
    # Non-word guards: "VIN" must not rewrite the "vin" inside "driving".
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(str(wrong_rail))}(?![A-Za-z0-9_])", re.I
    )
    out: list[str] = []
    last = 0
    for match in pattern.finditer(row):
        tail = row[match.end() : match.end() + 12]
        clause = _claim_clause(row, match.start(), match.end())
        names_part = _mentions_load_part(clause, tokens, exact)
        if not names_part and not _drives_the_load(clause):
            continue

        if names_part and re.match(r"\s*(?:dc\s+)?input\b", tail, re.I) and re.search(
            r"\b(?:convert(?:s|ed|ing)?|step(?:s|ped)?\s+down|regulat(?:e|es|ed|or|ion)?|buck)\b",
            clause,
            re.I,
        ):
            continue
        input_phrase = re.match(r"\s*((?:dc\s+)?input)\b", tail, re.I)
        if input_phrase and (names_part or _drives_the_load(clause)):
            # "…from the 18 V input" is a claim about the load's supply that names the board
            # input: both facts belong in the statement, so the rail is named and the input is
            # kept, rather than the input's own voltage being falsified.
            replaced = (
                f"the regulated {new_rail} rail (stepped down from the "
                f"{row[match.start():match.end()]}{input_phrase.group(0)})"
            )
            out.append(row[last : match.start()])
            out.append(replaced)
            last = match.end() + len(input_phrase.group(0))
            continue
        out.append(row[last : match.start()])
        out.append(new_rail)
        last = match.end()
    out.append(row[last:])
    return "".join(out)


def _rewrite_voltage_claims(
    row: str, tokens: list[str], exact: str, wrong_voltage: float, new_voltage: float
) -> str:
    """Rewrite the voltage of a clause about this load's supply, never the board input.

    Live walkthrough (2026-09-25): the first version replaced every occurrence in a field, which
    turned "18 V input converted to 3.3 V ... DRV8833 supplied from the 18 V motor rail" into
    "10 V input converted to 3.3 V" -- the input rail's own statement, falsified. A clause about
    the input ("the 18 V input", "18 V DC input") is left alone; the load's clause is rewritten.
    """
    out: list[str] = []
    last = 0
    for match in re.finditer(_voltage_pattern(wrong_voltage), row, re.I):
        tail = row[match.end() : match.end() + 12]
        clause = _claim_clause(row, match.start(), match.end())
        names_part = _mentions_load_part(clause, tokens, exact)
        if not names_part and not _drives_the_load(clause):
            continue
        # "Convert the 18 V input to a 3.3 V rail for the ESP32-C3 and the DRV8833" states the
        # conversion, not the part's supply: the part is fed by the converter's *output*. A
        # clause with a conversion verb and an input-rail mention is describing the board input.
        if names_part and re.match(r"\s*(?:dc\s+)?input\b", tail, re.I) and re.search(
            r"\b(?:convert(?:s|ed|ing)?|step(?:s|ped)?\s+down|regulat(?:e|es|ed|or|ion)?|buck)\b",
            clause,
            re.I,
        ):
            continue
        out.append(row[last : match.start()])
        out.append(f"{new_voltage:g} V")
        last = match.end()
    out.append(row[last:])
    return "".join(out)


def _rewrite_load_narrative(
    text: str,
    tokens: list[str],
    exact: str,
    wrong_rail: str,
    wrong_voltage: float,
    new_rail: str,
    new_voltage: float,
) -> str:
    """Correct a narrative field (a sheet function, a topology line) that fed the load.

    The field is rewritten when it names the part *or* when it claims the load's supply at the
    input voltage; either way it states where the load's power comes from, and after the fix
    that is the regulated rail. Fields about the input itself name no load and are left alone.
    """
    row = str(text or "")
    lowered = row.casefold()
    names_part = _mentions_load_part(row, tokens, exact)
    feeds_load = bool(re.search(_LOAD_WORDS, lowered, re.I)) and bool(
        re.search(_voltage_pattern(wrong_voltage), row, re.I)
    )
    if not (names_part or feeds_load):
        return row
    return _rewrite_load_supply_claim(
        row, tokens, exact, wrong_rail, wrong_voltage, new_rail, new_voltage, force=True
    )


def _corrected_load_supply_rows(
    rows,
    tokens: list[str],
    exact: str,
    wrong_rail: str,
    wrong_voltage: float,
    new_rail: str,
    new_voltage: float,
) -> list[str]:
    """Rewrite the part's own supply claims; drop a row that feeds the load at the input.

    A row like "the 18 V DC input is treated as the actuator supply" mixes the input voltage
    with the load's supply and cannot be corrected by substitution; left in place it states the
    opposite of the binding, which is how a candidate ends up asking a reviewer to accept an
    18 V motor supply that the ports say is 10 V. The pipeline's own disclosure replaces it.
    """
    out: list[str] = []
    for original in rows:
        row = str(original)
        lowered = row.casefold()
        named = any(token in lowered for token in tokens)
        claims_input = bool(re.search(_voltage_pattern(wrong_voltage), row, re.I))
        if named and claims_input:
            out.append(
                _rewrite_load_supply_claim(
                    row, tokens, exact, wrong_rail, wrong_voltage, new_rail, new_voltage
                )
            )
            continue
        if not named and claims_input and re.search(_LOAD_WORDS, lowered, re.I):
            continue
        out.append(row)
    return out


# Component-stage qualification: fixed output and half-open input envelope.
_REVIEWED_RAIL_FAMILIES: tuple[tuple[str, float, float, float], ...] = (
    ("ylptec-k7803m-1000r3", 3.3, 6.0, 30.0),
)


def _selected_lowerer_realizes_class(requirement: dict, component_class: str) -> bool:
    """Prove the class from this exact construction, including optional support parts."""
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.part_identity import (
        lowerer_witnesses_physical_class,
        physical_inventory_record,
        reviewed_record_realizes_class,
    )

    try:
        artifact = lower_requirement(requirement)
    except ValueError:
        return False
    if artifact is None:
        return False
    if lowerer_witnesses_physical_class(artifact.lowerer_id, component_class):
        return True
    return any(
        reviewed_record_realizes_class(
            physical_inventory_record(mpn=group.mpn, symbol=group.symbol, footprint=group.footprint),
            component_class,
        )
        for group in artifact.groups
    )


def _selected_recipe_realizes_class(candidate: dict, requirement: dict, component_class: str) -> bool:
    """Whether the recipe selected for this requirement emits a reviewed part of the class.

    A recipe family that constructs the class out of its own reviewed parts is a realization, not
    a substitute -- the same standard :func:`_selected_lowerer_realizes_class` applies to a
    lowerer. Live walkthrough (usb-pd-trigger, 2026-09-29): the ``pd_trigger`` requirement
    claimed a ``switch`` and the selected ``ch224k-pd-selectable@1`` emits the reviewed
    SS13D07VG4 selector, but the audit knew only the raw carrier list and refused the writer's
    own binding family. Only the recipe the architecture actually selected for this requirement
    counts, and only that recipe's own exact MPN/symbol/footprint parts, resolved through the
    same fail-closed inventory the parts stage uses.
    """
    from kicraft.design.part_identity import (
        physical_inventory_record,
        reviewed_record_realizes_class,
    )
    from kicraft.design.recipes.registry import get_recipe

    requirement_id = str(requirement.get("id") or "")
    if not requirement_id:
        return False
    for selection in candidate.get("recipe_selections") or []:
        if not isinstance(selection, dict):
            continue
        if requirement_id not in {str(row) for row in (selection.get("requirement_ids") or ())}:
            continue
        try:
            definition = get_recipe(str(selection.get("recipe") or ""))
        except ValueError:
            continue
        for part in definition.parts:
            record = physical_inventory_record(
                mpn=part.mpn, symbol=part.symbol, footprint=part.footprint
            )
            if reviewed_record_realizes_class(record, component_class):
                return True
    return False


def _architecture_obligation_family_mismatch(candidate: dict) -> list[StageDiagnostic]:
    """A requirement's family cannot implement a part class that requirement claims.

    Live walkthrough (2026-09-25): the architecture gave the two JST-XH connector requirements the
    generic lowerer family ``pin-header`` while they carried the ``jst-xh-connector`` obligation.
    The parts stage may not reopen a requirement family ("Architecture is binding"), so the BOM
    could only fail: four repair rounds, then
    "work unit bom-r001 invalid: missing-requirement-implementation=['motor_a']", unit repair
    exhausted. The family is still the writer's to choose *here*, so it is refused here.

    A class with no reviewed carrier is legitimate and is left alone: the intent contract says a
    class the library does not cover yet may be named plainly, and the parts step resolves it.
    """
    from kicraft.design.part_identity import (
        reviewed_part,
        reviewed_parts_for_feature,
        reviewed_record_realizes_class,
    )

    diagnostics: list[StageDiagnostic] = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        family = str(requirement.get("family") or "")
        exact_part = str(requirement.get("exact_part") or "").strip()
        exact = exact_part.casefold()
        # The requirement's own pinned exact part is the library's evidence: a reviewed record
        # whose reviewed family or physical features (through the demanded-class alias map)
        # implement the class realizes it, whatever the lowerer family is called. Live
        # walkthrough (stepper-a4988, 2026-09-29): family `a4988-stepper-driver` with exact part
        # A4988SETTR-T and a `motor-driver` obligation was refused here and pushed toward the
        # DRV8833's family, though the pinned reviewed part IS a motor driver.
        named = reviewed_part(exact_part) if exact_part else None
        for obligation in requirement.get("obligations") or []:
            if not isinstance(obligation, dict) or obligation.get("kind") != "physical":
                continue
            component_class = str(obligation.get("component_class") or "")
            if not component_class:
                continue
            if reviewed_record_realizes_class(named, component_class):
                continue
            if _selected_recipe_realizes_class(candidate, requirement, component_class):
                continue
            carriers = reviewed_parts_for_feature(component_class)
            if not carriers:
                continue
            if family in {part.family for part in carriers} or exact in {
                part.identity for part in carriers
            }:
                continue
            if _selected_lowerer_realizes_class(requirement, component_class):
                continue
            diagnostics.append(
                _diag(
                    "architecture_obligation_family_mismatch",
                    "repair_required",
                    "A requirement's family cannot implement the part class it claims.",
                    [
                        f"requirement {requirement.get('id')!r} claims {component_class!r} but its "
                        f"family {family!r} cannot realize it "
                        f"(exact_part={requirement.get('exact_part')!r}); the reviewed carrier(s) "
                        f"are {sorted(part.identity for part in carriers)} in family "
                        f"{sorted({part.family for part in carriers})} — name that family (with "
                        "`declared_ports` for the carrier's interface) or that exact part"
                    ],
                )
            )
    return diagnostics


def _logic_rails(candidate: dict) -> set[str]:
    """The rails that power the logic this board is controlled by (MCU, sensors, buses)."""
    from kicraft.design.architecture_intent import _supply_port_name

    rails: set[str] = set()
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        if str(requirement.get("role") or "") not in _LOGIC_ROLES:
            continue
        for port, net in (requirement.get("ports") or {}).items():
            port_key = str(port).casefold()
            if port_key in _GENERATOR_PORTS or not _supply_port_name(port_key):
                continue
            rails.add(str(net))
    return rails


def _family_output_voltages() -> dict[str, set[float]]:
    """The output voltages each registered recipe family actually carries."""
    from kicraft.design.recipes import registered_recipes

    voltages: dict[str, set[float]] = {}
    for registered in registered_recipes():
        definition = getattr(registered, "definition", registered)
        family = str(getattr(definition, "family", "") or "").strip().casefold()
        default = (getattr(definition, "parameter_defaults", None) or {}).get("output_voltage")
        if family and isinstance(default, (int, float)):
            voltages.setdefault(family, set()).add(float(default))
    return voltages


def _retarget_unbuildable_regulators(candidate: dict) -> list[str]:
    """Select a reviewed family at the already-requested voltage when identity is not fixed.

    A family can have no reviewed recipe instance at the voltage the requirement already
    specifies.  A different family may be used only when it has an instance at that same
    voltage and the requirement did not name an exact part.  An unavailable requested voltage,
    or a protected exact identity, is a contract refusal rather than permission to rewrite the
    requested rail or output parameter.
    """
    retargeted: list[str] = []
    rails = _architecture_rail_voltages(candidate)
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict) or str(requirement.get("role") or "") != "regulator":
            continue
        parameters = requirement.get("parameters") or {}
        target = parameters.get("output_voltage")
        if not isinstance(target, (int, float)):
            # The derived shape states the same fact as the rail this part's output port feeds.
            for port, net in (requirement.get("ports") or {}).items():
                port_key = str(port).casefold()
                if port_key.startswith(("output", "vout")) or port_key == "out":
                    target = rails.get(str(net))
                    break
        if not isinstance(target, (int, float)):
            continue
        family = str(requirement.get("family") or "")
        if any(family == row[0] and float(target) == row[1] for row in _REVIEWED_RAIL_FAMILIES):
            continue
        family_key = family.strip().casefold()
        carried = _family_output_voltages().get(family_key)
        if carried is None:
            # Not a recipe family this pipeline manages: whatever is wrong with it is the
            # checkers' business, not this pass's. Replacing a deliberately weak regulator
            # would erase the very defect the stage exists to report.
            continue
        if any(abs(voltage - float(target)) <= 0.05 for voltage in carried):
            continue
        inputs = _requirement_supply_rails(requirement)
        if len(inputs) != 1 or inputs[0] not in rails:
            continue
        input_voltage = rails[inputs[0]]
        reviewed = next(
            (row for row in _REVIEWED_RAIL_FAMILIES
             if row[1] == float(target) and row[2] <= input_voltage < row[3]),
            None,
        )
        if reviewed is None or any(
            str(requirement.get(key) or "").strip()
            for key in ("exact_part", "manufacturer", "exact_manufacturer")
        ):
            continue
        requirement["family"] = reviewed[0]
        retargeted.append(str(requirement.get("id") or ""))
    return retargeted


def complete_over_rated_supply(candidate: dict) -> dict:
    """Give an out-of-range load its own rail, then resolve the added converter."""
    return _resolve_added_requirements(_complete_load_supply_rails(candidate))


def _resolve_added_requirements(completed: dict) -> dict:
    """Run the recipe resolver without weakening a protected exact identity."""
    try:
        from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

        return apply_architecture_recipe_resolution(completed).model_dump(exclude_none=True)
    except Exception:  # a resolution problem is the stage's to report, not this pass's
        return completed


def _generated_regulator_requirement_id(source: object, requirements: list[object]) -> str:
    """Return a schema-valid, collision-free identifier for a compiler-added regulator."""
    stem = re.sub(r"[^a-z0-9_-]+", "_", str(source).lower()).strip("_-")
    if not stem:
        stem = "rail"
    elif not stem[0].isalpha():
        stem = f"rail_{stem}"
    suffix = "_regulator"
    base = f"{stem[: 64 - len(suffix)].rstrip('_-')}{suffix}"
    used = {
        str(requirement.get("id"))
        for requirement in requirements
        if isinstance(requirement, dict) and requirement.get("id") is not None
    }
    if base not in used:
        return base
    index = 2
    while True:
        collision_suffix = f"_{index}"
        identifier = f"{base[: 64 - len(collision_suffix)].rstrip('_-')}{collision_suffix}"
        if identifier not in used:
            return identifier
        index += 1

def _reviewed_external_usb_vbus_rails(candidate: dict) -> set[str]:
    """Return VBUS nets that reviewed USB interfaces receive from the host.

    A USB receptacle's VBUS is a board entry, not a load merely because it has a
    consumer-side port.  USB-UART recipes that integrate their own reviewed
    receptacle have the same ownership.  The determination follows reviewed
    interface identities and recipe contents, never a rail's spelling.
    """
    from kicraft.design.part_identity import (
        physical_inventory_record,
        reviewed_inventory,
        reviewed_part,
    )
    from kicraft.design.recipes import registered_recipes

    usb_receptacle_families = {
        part.family
        for part in reviewed_inventory()
        if part.family.startswith("usb-") and part.family.endswith("-receptacle")
    }
    integrated_bridge_families: dict[str, set[str]] = {}
    for registered in registered_recipes():
        definition = getattr(registered, "definition", registered)
        if not any(port.name == "vbus" for port in definition.ports):
            continue
        has_receptacle = any(
            (record := physical_inventory_record(
                mpn=part.mpn, symbol=part.symbol, footprint=part.footprint
            )) is not None
            and record.family in usb_receptacle_families
            for part in definition.parts
            if part.role == "connector"
        )
        family = str(definition.family or "").casefold()
        if has_receptacle and family:
            integrated_bridge_families.setdefault(family, set()).add(
                str(definition.exact_part or "").casefold()
            )

    rails: set[str] = set()
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        ports = requirement.get("ports") or {}
        vbus = ports.get("vbus")
        if not isinstance(vbus, str) or not vbus:
            continue
        family = str(requirement.get("family") or "").casefold()
        exact = str(requirement.get("exact_part") or "").casefold()
        reviewed = reviewed_part(exact) if exact else None
        direct_receptacle = (
            str(requirement.get("role") or "").casefold() == "connector"
            and (family in usb_receptacle_families or (
                reviewed is not None and reviewed.family in usb_receptacle_families
            ))
        )
        integrated_exact = integrated_bridge_families.get(family)
        integrated_bridge = integrated_exact is not None and (
            not exact or exact in integrated_exact
        )
        if direct_receptacle or integrated_bridge:
            rails.add(vbus)
    return rails


def _declared_external_input_rails(candidate: dict) -> set[str]:
    """Return explicitly owned board-entry rails, without inferring consumer ownership."""
    entries = _reviewed_external_usb_vbus_rails(candidate)
    for name, row in ((candidate.get("power") or {}).get("rails") or {}).items():
        if isinstance(row, dict) and "from" in row and row.get("from") is None:
            entries.add(str(name))
    voltages = candidate.get("rail_voltages") or {}
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict) or str(requirement.get("role") or "") != "power_input":
            continue
        ports = requirement.get("ports") or {}
        named_entries = {
            str(net)
            for port, net in ports.items()
            if str(port).casefold() in {"positive", "input", "input_positive", "vin", "vbus"}
        }
        if named_entries:
            entries.update(named_entries)
            continue
        # A reviewed generic input connector can expose numbered contacts rather
        # than a semantic ``positive`` port. Its declared non-ground rail is
        # still explicit provenance; arbitrary consumer ports are not.
        for net in ports.values():
            try:
                if abs(float(voltages.get(str(net)))) > 0.05:
                    entries.add(str(net))
            except (TypeError, ValueError):
                continue
    return entries



def _declared_upstream_supply_source(candidate: dict, rail: str) -> str | None:
    """Return the one explicit board-entry source upstream of ``rail``.

    A regulated load may only be moved to a new rail when its existing supply has a
    concrete ancestry.  In particular, a USB VBUS rail elsewhere on the board is
    not evidence that a battery-powered load may use VBUS.  ``None`` intentionally
    covers a missing producer, a branch, and a cycle; those cases need an
    architecture repair rather than an invented converter input.
    """
    upstream: dict[str, set[str]] = {}
    for name, row in ((candidate.get("power") or {}).get("rails") or {}).items():
        if isinstance(row, dict) and isinstance(row.get("from"), str) and row["from"]:
            upstream.setdefault(str(name), set()).add(str(row["from"]))
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict) or str(requirement.get("role") or "") != "regulator":
            continue
        ports = requirement.get("ports") or {}
        outputs = {
            str(net)
            for port, net in ports.items()
            if str(port).casefold().startswith(("output", "vout")) or str(port).casefold() == "out"
        }
        inputs = {
            str(net)
            for port, net in ports.items()
            if str(port).casefold() in {"input", "input_positive", "vin", "vin_positive"}
        }
        if len(outputs) != 1 or len(inputs) != 1:
            continue
        upstream.setdefault(next(iter(outputs)), set()).update(inputs)

    board_entries = _declared_external_input_rails(candidate)

    current = str(rail)
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        sources = upstream.get(current)
        if not sources:
            return current if current in board_entries else None
        if len(sources) != 1:
            return None
        current = next(iter(sources))
    return None


def _underpowered_supply_rails(candidate: dict, rails: dict[str, float]) -> set[str]:
    """Return rails below a reviewed load's minimum supply voltage."""
    from kicraft.design.part_identity import reviewed_part, reviewed_supply_voltage_limits

    underpowered: set[str] = set()
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        exact = str(requirement.get("exact_part") or "").strip()
        record = reviewed_part(exact) if exact else None
        if record is None:
            continue
        minima = [
            float(low)
            for _label, low, _high in reviewed_supply_voltage_limits(record)
            if low is not None
        ]
        if not minima:
            continue
        # Records can publish alternative operating modes (for example CH340's
        # 3.3 V and 5 V modes). Below every minimum is unambiguously underpowered;
        # below only one mode's minimum is not.
        required = min(minima)
        underpowered.update(
            rail
            for rail in _requirement_supply_rails(requirement)
            if rails.get(rail) is not None and rails[rail] < required
        )
    return underpowered


def _complete_load_supply_rails(candidate: dict) -> dict:
    """Give a reviewed load outside its supply range its own regulated rail.

    The compiler may add a reviewed converter only when the existing load supply
    has one declared upstream board-entry source.  That preserves a battery
    load's power ancestry instead of silently moving it to an unrelated USB
    rail.  Nothing is invented about the load: its current, voltage, exact part,
    and original conversion obligations stay as the writer stated them.
    """
    from kicraft.design.part_identity import (
        reviewed_part,
        reviewed_parts_for_feature,
        reviewed_supply_voltage_limits,
    )

    retargeted = _retarget_unbuildable_regulators(candidate)
    if retargeted:
        candidate["assumptions"] = [
            *(candidate.get("assumptions") or []),
            (
                "regulator requirement(s) "
                + ", ".join(sorted(retargeted))
                + " moved to the reviewed converter family registered at the already-requested "
                "output voltage, because the named family has no instance there (defaulted)"
            ),
        ]
    rails = _architecture_rail_voltages(candidate)
    if not rails:
        return candidate
    underpowered_rails = _underpowered_supply_rails(candidate, rails)
    requirements = candidate.get("requirements") or []
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            continue
        role = str(requirement.get("role") or "")
        exact = str(requirement.get("exact_part") or "").strip()
        record = reviewed_part(exact) if exact else None
        if record is None:
            continue
        limits = [
            (str(label), low, high)
            for label, low, high in reviewed_supply_voltage_limits(record)
            if high is not None
        ]
        if not limits:
            continue
        allowed = max(high for _label, _low, high in limits)
        minimum = min(
            (float(low) for _label, low, _high in limits if low is not None),
            default=None,
        )
        logic_rails = _logic_rails(candidate)
        supplied = _requirement_supply_rails(requirement)
        if len(supplied) != 1 or supplied[0] not in rails:
            continue
        over = [rail for rail in supplied if rails[rail] > allowed]
        under = [
            rail for rail in supplied if minimum is not None and rails[rail] < minimum
        ]
        # A drive or analog load riding the logic rail needs its own supply even when its
        # voltage limit permits that rail; other roles are adapted only for an actual
        # reviewed over/undervoltage violation.
        on_logic = (
            [rail for rail in supplied if rail in logic_rails]
            if role in {"driver", "analog_block"}
            else []
        )
        if not over and not under and not on_logic:
            continue
        if under:
            upstream_sources = {
                _declared_upstream_supply_source(candidate, rail) for rail in under
            }
            if len(upstream_sources) != 1 or None in upstream_sources:
                continue
            source_rail = next(iter(upstream_sources))
        elif over:
            if len(over) != 1:
                continue
            source_rail = over[0]
        else:
            sources = [
                rail
                for rail, volts in rails.items()
                if rail not in logic_rails and rail not in on_logic and abs(float(volts)) > 0.0
            ]
            if len(sources) != 1:
                continue
            source_rail = sources[0]
        if source_rail not in rails:
            continue
        if not under and any(
            any(
                high is not None and high >= rails[source_rail]
                for _label, _low, high in reviewed_supply_voltage_limits(alternative)
            )
            for alternative in reviewed_parts_for_feature(str(requirement.get("family") or ""))
        ):
            continue
        if under:
            label, low, _high = max(limits, key=lambda row: float(row[1] or 0.0))
        else:
            label, low, _high = min(limits, key=lambda row: row[2])
        # Select only a construction qualified for both input and load voltages.
        reviewed = next(
            (
                (family, volts)
                for family, volts, vin_min, vin_max in _REVIEWED_RAIL_FAMILIES
                if float(low or 0.0) <= volts <= allowed
                and vin_min <= rails[source_rail] < vin_max
            ),
            None,
        )
        if reviewed is None:
            continue
        converter_family, voltage = reviewed
        new_rail = f"{str(requirement.get('id') or 'load').upper()}_RAIL"
        converter_id = _generated_regulator_requirement_id(
            requirement.get("id") or "load", requirements
        )
        sheet_name = re.sub(r"[^A-Z0-9]+", " ", str(requirement.get("id") or "load").upper()).strip() + " REGULATOR"
        completed = dict(candidate)
        completed["requirements"] = [dict(row) if isinstance(row, dict) else row for row in requirements]
        offending = {*over, *under, *on_logic}
        completed["requirements"][index]["ports"] = {
            key: (new_rail if str(value) in offending else value)
            for key, value in (requirement.get("ports") or {}).items()
        }
        if requirement.get("supply") in offending:
            completed["requirements"][index]["supply"] = new_rail
        completed.setdefault("sheets", [])
        completed["sheets"] = [dict(row) for row in completed.get("sheets") or []]
        completed["sheets"].append(
            {
                "name": sheet_name,
                "stem": sheet_name.replace(" ", "_"),
                "role": "regulator",
                "function": (
                    f"Step the {source_rail} input down to {voltage:g} V for the "
                    f"{requirement.get('id')} load supply."
                ),
            }
        )
        completed["requirements"].append(
            {
                "id": converter_id,
                "sheet": sheet_name,
                "role": "regulator",
                "family": converter_family,
                "parameters": {"output_voltage": voltage},
                "ports": {"input": source_rail, "output": new_rail, "gnd": "GND"},
                "functional_blocks": list(requirement.get("functional_blocks") or []),
            }
        )
        rail_voltages = dict(completed.get("rail_voltages") or {})
        rail_voltages[new_rail] = voltage
        completed["rail_voltages"] = rail_voltages
        if isinstance(completed.get("power_nets"), list):
            completed["power_nets"] = [*completed["power_nets"], new_rail]
        # The writer's own statements are now false: they say this part runs on the
        # under- or over-rated rail. A candidate whose prose contradicts its bindings is
        # not reviewable, so the claims are corrected before the disclosure below states
        # the whole picture in one row.
        tokens = _load_part_tokens(requirement, exact)
        # The rail the part was wrongly on is the source rail for an over-rated part,
        # the logic rail for a deliberately separated drive, and its existing rail for
        # an under-rated part whose upstream source feeds the added converter.
        wrong_rail = (
            source_rail if over else (under[0] if under else (on_logic[0] if on_logic else source_rail))
        )
        wrong_voltage = float(rails[wrong_rail])
        completed["assumptions"] = _corrected_load_supply_rows(
            completed.get("assumptions") or [],
            tokens,
            exact,
            wrong_rail,
            wrong_voltage,
            new_rail,
            voltage,
        )
        completed["topologies"] = {
            key: _rewrite_load_narrative(
                value, tokens, exact, wrong_rail, wrong_voltage, new_rail, voltage
            )
            for key, value in (completed.get("topologies") or {}).items()
        }
        completed["sheets"] = [
            (
                {
                    **row,
                    "function": _rewrite_load_narrative(
                        row.get("function") or "",
                        tokens,
                        exact,
                        wrong_rail,
                        wrong_voltage,
                        new_rail,
                        voltage,
                    ),
                }
                if isinstance(row, dict)
                else row
            )
            for row in completed.get("sheets") or []
        ]
        refresh_touched_rail_nets(completed, {wrong_rail, new_rail, source_rail})
        if under:
            disclosure = (
                f"{exact} runs from a regulated {voltage:g} V rail ({new_rail}) because its "
                f"reviewed {label} minimum is {minimum:g} V, while {wrong_rail} is "
                f"{wrong_voltage:g} V; {converter_id} steps the declared upstream "
                f"{source_rail} source down, and the existing {wrong_rail} rail remains "
                "unchanged (defaulted)"
            )
        else:
            disclosure = (
                f"{exact} runs from a regulated {voltage:g} V rail ({new_rail}) because its "
                f"reviewed {label} limit is {allowed:g} V and no reviewed part for "
                f"{requirement.get('family')!r} is rated for {source_rail} "
                f"({rails[source_rail]:g} V); {converter_id} steps {source_rail} down, and the "
                f"{source_rail} input stays the board input only (defaulted)"
            )
        completed["assumptions"] = [
            *(completed.get("assumptions") or []),
            disclosure,
        ]
        # Resolve the added requirement with the same recipe pass as the existing slot.
        return completed
    # A declared load rail without a generator may be completed only from the one
    # explicit board-entry source.  Consumer rails never establish source ownership.
    generators = _rail_generators(candidate)
    external_input_rails = _declared_external_input_rails(candidate)
    for rail_name, rail_voltage in rails.items():
        if (
            rail_name in underpowered_rails
            or rail_name in generators
            or rail_name in external_input_rails
            or _rail_needs_no_generator(rail_name, rail_voltage)
        ):
            continue
        consumers = [
            requirement
            for requirement in requirements
            if isinstance(requirement, dict)
            and rail_name in _requirement_supply_rails(requirement)
        ]
        input_rails = sorted(
            source
            for source in external_input_rails
            if source != rail_name
        )
        if not consumers or len(input_rails) != 1 or input_rails[0] not in rails:
            continue
        input_rail = input_rails[0]
        # A compiler-added regulator may implement the declared rail only at its
        # exact reviewed voltage.  Nearest-voltage replacement would silently
        # rewrite the author's electrical requirement.
        reviewed = next(
            (row for row in _REVIEWED_RAIL_FAMILIES
             if row[1] == rail_voltage and row[2] <= rails[input_rail] < row[3]),
            None,
        )
        if reviewed is None:
            continue
        converter_family = reviewed[0]
        converter_id = _generated_regulator_requirement_id(rail_name, requirements)
        sheet_name = re.sub(r"[^A-Z0-9]+", " ", rail_name.upper()).strip() + " REGULATOR"
        completed = dict(candidate)
        completed["sheets"] = [dict(row) for row in completed.get("sheets") or []]
        completed["sheets"].append(
            {
                "name": sheet_name,
                "stem": sheet_name.replace(" ", "_"),
                "role": "regulator",
                "function": (
                    f"Step the {input_rail} input down to {rail_voltage:g} V for the {rail_name} "
                    "load rail."
                ),
            }
        )
        completed["requirements"] = [
            *(dict(row) if isinstance(row, dict) else row for row in requirements),
            {
                "id": converter_id,
                "sheet": sheet_name,
                "role": "regulator",
                "family": converter_family,
                "parameters": {"output_voltage": rail_voltage},
                "ports": {"input": input_rail, "output": rail_name, "gnd": "GND"},
                "functional_blocks": list(
                    consumers[0].get("functional_blocks") or []
                ),
            },
        ]
        refresh_touched_rail_nets(completed, {rail_name, input_rail})
        completed["assumptions"] = [
            *(completed.get("assumptions") or []),
            (
                f"{rail_name} ({rail_voltage:g} V) is generated by {converter_id} "
                f"({converter_family}, a reviewed buck) fed from the {input_rail} input, because "
                f"the design declared "
                f"the rail and bound {consumers[0].get('id')} to it without naming a source "
                "(defaulted)"
            ),
        ]
        # The added requirement is a recipe requirement like any other, and the recipe's owned
        # parts -- an adjustable regulator's feedback divider, its compensation network -- exist
        # only after the resolver's pass. Without it the parts unit drafts them from scratch,
        # which is how the 10 V motor rail ended up carrying the 3.3 V divider and failing §9.32
        # (live walkthrough, 2026-09-25). This is the same pass the slot already went through, so
        # every existing requirement keeps its own resolution.
        return completed

    return candidate


def refresh_touched_rail_nets(candidate: dict, touched: set[str]) -> None:
    """Rebuild the derived inter-sheet endpoints of the rails this completion touched.

    The derived nets are computed at decode, before this code runs, so a rebound port leaves the
    net list naming a rail its sheet no longer binds -- and the commit gate refuses exactly that
    ("inter-sheet net '+18V' endpoint on sheet 'DUAL H BRIDGE' has no requirement.ports value
    bound to that exact net name", live walkthrough 2026-09-25). Endpoints are recomputed for the
    touched rails only: every other net keeps the derivation's own rows.
    """
    if not touched:
        return
    requirements = [row for row in candidate.get("requirements") or [] if isinstance(row, dict)]
    bound: dict[str, set[str]] = {}
    generates: dict[tuple[str, str], bool] = {}
    for requirement in requirements:
        sheet = str(requirement.get("sheet") or "")
        for port, net in (requirement.get("ports") or {}).items():
            name = str(net)
            bound.setdefault(name, set()).add(sheet)
            port_key = str(port).casefold()
            if port_key.startswith(("output", "vout", "sw")) or port_key in {"out", "vout_sw"}:
                generates[(sheet, name)] = True
    nets: list[dict] = []
    seen: set[str] = set()
    for row in candidate.get("inter_sheet_nets") or []:
        if not isinstance(row, dict):
            nets.append(row)
            continue
        name = str(row.get("name") or "")
        if name not in touched:
            nets.append(row)
            continue
        seen.add(name)
        previous = {
            str(endpoint.get("sheet")): str(endpoint.get("direction") or "bidirectional")
            for endpoint in row.get("endpoints") or []
            if isinstance(endpoint, dict)
        }
        endpoints = []
        for sheet in sorted(bound.get(name, set())):
            direction = previous.get(sheet)
            if direction is None:
                direction = "output" if generates.get((sheet, name)) else "input"
            endpoints.append({"sheet": sheet, "direction": direction})
        if len(endpoints) >= 2:  # a net needs two ends; one sheet is not an inter-sheet net
            nets.append({"name": name, "endpoints": endpoints})
    for name in sorted(touched - seen):
        endpoints = [
            {
                "sheet": sheet,
                "direction": "output" if generates.get((sheet, name)) else "input",
            }
            for sheet in sorted(bound.get(name, set()))
        ]
        if len(endpoints) >= 2:
            nets.append({"name": name, "endpoints": endpoints})
    candidate["inter_sheet_nets"] = nets
    if isinstance(candidate.get("power_nets"), list):
        for name in sorted(touched):
            if bound.get(name) and name not in candidate["power_nets"]:
                candidate["power_nets"] = [*candidate["power_nets"], name]


def _rail_generators(candidate: dict) -> set[str]:
    """Every declared rail a part in this design generates, in either candidate shape."""
    generators: set[str] = set()
    for name, row in ((candidate.get("power") or {}).get("rails") or {}).items():
        if isinstance(row, dict) and row.get("from"):
            generators.add(str(name))
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        for port, net in (requirement.get("ports") or {}).items():
            key = str(port).casefold()
            if key.startswith(("output", "vout", "sw")) or key in {"out", "vout_sw"}:
                generators.add(str(net))
    return generators


def _rail_needs_no_generator(name: str, voltage: float) -> bool:
    """Only ground is inherently source-free; inputs need explicit ownership."""
    key = str(name).strip().casefold()
    return key in {"gnd", "ground", "0v"} or abs(float(voltage)) <= 0.05


def _load_current_disclosed_on(text: str, rail: str) -> bool:
    """Whether the text states how much current a load draws on this rail.

    ``external_load_budget_stated`` is anchored on the display/LED-string vocabulary ("5v",
    "hub75", "led string"); a motor or actuator rail is disclosed in its own words, so the
    drive-rail check reads both. Without this a deliberate shared rail could not be disclosed
    at all, and the check would park the stage -- the outcome it exists to avoid.
    """
    text = _text(text)
    if external_load_budget_stated(text):
        return True
    rail_words = re.escape(str(rail).casefold())
    load_words = r"(?:motor|actuator|solenoid|heater|driver|external load|load)"
    return bool(
        re.search(
            rf"(?:{rail_words}|{load_words})[^.;]{{0,80}}\d+(?:\.\d+)?\s*(?:a|ma)\b",
            text,
            re.I,
        )
        or re.search(
            rf"\d+(?:\.\d+)?\s*(?:a|ma)\b[^.;]{{0,80}}(?:{rail_words}|{load_words})",
            text,
            re.I,
        )
    )


def _architecture_drive_on_logic_rail(
    upstream: dict, candidate: dict
) -> list[StageDiagnostic]:
    """A drive part powered from the rail that powers the logic it is controlled by.

    The rating check has a cheap escape the writer found live (2026-09-25): after an 18 V rail
    was refused on a DRV8833 (``vm`` rated 10.8 V), the correction bound ``vm`` to the 3.3 V
    rail that powers the ESP32-C3 -- always inside the part's range, and always wrong: the
    actuators would run from the MCU's regulator, taking their current out of the logic budget.

    A deliberate shared rail is a real design, so the check is satisfied by *disclosing* the
    load's current on that rail (``external_load_budget_stated``); what it refuses is sharing
    the logic rail silently.
    """
    from kicraft.design.architecture_intent import _supply_port_name

    from kicraft.design.part_identity import reviewed_part, reviewed_supply_voltage_limits

    def needs_its_own_rail(requirement: dict) -> bool:
        """Whether the record says this part runs on a load rail rather than the logic rail.

        A level shifter or a display driver legitimately runs on the logic rail; a motor
        driver's reviewed record names a *motor supply* domain (the DRV8833's ``vm``). Only
        the latter is the escape this check exists to stop, so the check keys on the record,
        not on the role alone.
        """
        exact = str(requirement.get("exact_part") or "").strip()
        record = reviewed_part(exact) if exact else None
        if record is None:
            return False
        return any(
            re.search(r"\b(?:motor|load|coil|actuator)\b", str(label), re.I)
            for label, _low, high in reviewed_supply_voltage_limits(record)
            if high is not None
        )

    drives: dict[str, list[str]] = {}
    logic: dict[str, list[str]] = {}
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        role = str(requirement.get("role") or "")
        if role == "driver" and not needs_its_own_rail(requirement):
            continue
        requirement_id = str(requirement.get("id") or "")
        for port, net in (requirement.get("ports") or {}).items():
            port_key = str(port).casefold()
            if port_key in _GENERATOR_PORTS or not _supply_port_name(port_key):
                continue
            rail = str(net)
            if role == "driver":
                drives.setdefault(rail, []).append(requirement_id)
            elif role in _LOGIC_ROLES:
                logic.setdefault(rail, []).append(requirement_id)

    disclosure_text = _text([candidate, upstream.get("_stage_answers", [])])
    diagnostics: list[StageDiagnostic] = []
    for rail in sorted(set(drives) & set(logic)):
        if _load_current_disclosed_on(disclosure_text, rail):
            continue
        diagnostics.append(
            _diag(
                "architecture_drive_shares_logic_rail",
                "repair_required",
                "A drive part is powered from the same rail as the logic it is controlled by.",
                [
                    f"rail {rail!r} powers drive requirement(s) {drives[rail]} and logic "
                    f"requirement(s) {logic[rail]} — give the load its own regulated rail inside "
                    "the part's rated range (a converter whose output feeds the drive's supply "
                    "port, with the higher input voltage left as the board input), or state the "
                    "load's maximum current on that rail if sharing it is deliberate",
                ],
            )
        )
    return diagnostics


def _requirements_with_curated_ownership() -> frozenset[str]:
    """The family spellings a curated recipe or registered lowerer owns."""
    families: set[str] = set()
    try:
        from kicraft.design.lowering import registered_lowerers

        for lowerer in registered_lowerers():
            families.update(str(name).casefold() for name in lowerer.families)
    except Exception:  # noqa: BLE001 - a diagnostic never fails the stage
        pass
    try:
        from kicraft.design.recipes.registry import registered_recipes

        for recipe in registered_recipes():
            family = getattr(recipe.definition, "family", None)
            if family:
                families.add(str(family).casefold())
            families.update(
                str(alias).casefold()
                for alias in getattr(recipe.definition, "identity_aliases", ()) or ()
            )
    except Exception:  # noqa: BLE001
        pass
    return frozenset(families)


def _architecture_declared_contact_unnamed(candidate: dict) -> list[StageDiagnostic]:
    """A declared interface whose ports name no contact can never be verified.

    A declared interface is a claim about a real part, and the only thing that makes it
    checkable is the contact each port names. The BOM stage refuses an unnamed contact
    ("claim states no pin number; name the actual contact of <symbol>"), but the claim is
    committed with this stage and no BOM draft can change it, so the run can only exhaust its
    repair rounds. Measured over every committed architecture in the live store on 2026-09-30:
    seven carried an unnamed declared contact and all seven died at BOM with
    `unit_repair_exhausted` (projects 866, 971, 1019, 1034, 1057, 1074, 1105); no board that
    shipped has one. Naming the contact here costs one redraft instead of the whole run.
    """
    offenders: list[str] = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        # An explicit authored claim remains an auditable physical-contact
        # assertion even when a curated recipe/lowerer owns support circuitry.
        # Ownership cannot make a missing pin number observable.
        ports = requirement.get("declared_ports")
        if not isinstance(ports, list) or not ports:
            claim = requirement.get("declared_interface")
            ports = claim.get("ports") if isinstance(claim, dict) else None
        if not isinstance(ports, list) or not ports:
            continue
        unnamed = sorted(
            str(port.get("key") or port.get("function") or "?")
            for port in ports
            if isinstance(port, dict) and not str(port.get("pin") or "").strip()
        )
        if unnamed:
            offenders.append(f"{requirement.get('id')}: {', '.join(unnamed)}")
    if not offenders:
        return []
    return [
        _diag(
            "architecture_declared_contact_unnamed",
            "repair_required",
            "A declared interface must name the real contact behind each of its ports: name the "
            "reviewed part as `exact_part` and give every port that part's own contact number. "
            "When the ports belong to more than one physical part (a fuse in series with its "
            "output terminal, a shared ground return), split them into one requirement per part. "
            "An unnamed contact cannot be checked, and this claim is frozen once the stage "
            "commits.",
            sorted(offenders),
        )
    ]


_TYPED_VOLTAGE_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:v|volts?)(?!\w)",
    re.IGNORECASE,
)


def _typed_output_voltage(obligation: dict) -> float | None:
    """Return one explicit voltage from a typed conversion output, never a prose inference."""
    matches = _TYPED_VOLTAGE_RE.findall(str(obligation.get("output_kind") or ""))
    if len(matches) != 1:
        return None
    return float(matches[0])


def _conversion_source_obligations(upstream: dict) -> dict[str, dict]:
    """Read original conversion rows from committed upstream state by stable source id."""
    conversions: dict[str, dict] = {}
    for slot in ("intent", "functional_spec"):
        for obligation in (upstream.get(slot) or {}).get("obligations") or []:
            if not isinstance(obligation, dict) or obligation.get("kind") != "conversion":
                continue
            identifier = str(obligation.get("original_obligation_id") or "")
            if identifier:
                conversions.setdefault(identifier, obligation)
    return conversions


def _is_power_conversion(obligation: dict) -> bool:
    """Recognize typed power-to-power conversion, not signal or data conversion."""
    return all(
        re.search(r"\b(?:power|supply|rail)\b", str(obligation.get(key) or ""), re.I)
        for key in ("input_kind", "output_kind")
    )


def _requirement_obligation_ids(requirement: dict) -> set[str]:
    """Read a draft's source-id ownership before or after canonical restoration."""
    identifiers = {str(value) for value in requirement.get("obligation_ids") or []}
    identifiers.update(
        str(obligation.get("original_obligation_id"))
        for obligation in requirement.get("obligations") or []
        if isinstance(obligation, dict) and obligation.get("original_obligation_id")
    )
    return identifiers


def _requirement_output_evidence(requirement: dict, candidate: dict) -> list[tuple[float, str]]:
    """Pair numeric outputs with diagnostics; rail labels are never voltage evidence."""
    evidence: list[tuple[float, str]] = []
    parameters = requirement.get("parameters") or {}
    voltage = parameters.get("output_voltage")
    if isinstance(voltage, (int, float)) and not isinstance(voltage, bool):
        evidence.append((float(voltage), f"parameters.output_voltage={float(voltage):g} V"))
    rails = _architecture_rail_voltages(candidate)
    for port, net in (requirement.get("ports") or {}).items():
        key = str(port).casefold()
        if not (key.startswith(("output", "vout")) or key == "out"):
            continue
        rail_voltage = rails.get(str(net))
        if rail_voltage is not None:
            evidence.append((rail_voltage, f"ports.{port}={net!r} at {rail_voltage:g} V"))
    return evidence



def architecture_conversion_output_diagnostics(
    upstream: dict, candidate: dict
) -> list[StageDiagnostic]:
    """Reject a conversion whose owner's numeric output contradicts its original typed request."""
    conversions = _conversion_source_obligations(upstream)
    if not conversions:
        return []
    offenders: list[str] = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        for identifier in sorted(_requirement_obligation_ids(requirement) & conversions.keys()):
            obligation = conversions[identifier]
            requested = _typed_output_voltage(obligation)
            if requested is None:
                continue
            evidence = _requirement_output_evidence(requirement, candidate)
            conflicting = [
                description
                for voltage, description in evidence
                if abs(voltage - requested) > 0.05
            ]
            if conflicting:
                offenders.append(
                    f"requirement {requirement.get('id')!r} owns conversion {identifier!r}: "
                    f"original output_kind={obligation.get('output_kind')!r} requests "
                    f"{requested:g} V; " + "; ".join(conflicting)
                )
    if not offenders:
        return []
    return [
        _diag(
            "architecture_conversion_output_mismatch",
            "repair_required",
            "A typed conversion obligation's original requested output voltage conflicts with "
            "the implementing requirement. Preserve the original output and select a converter "
            "that implements it; do not relabel the rail or discard an exact identity.",
            sorted(offenders),
        )
    ]


_LED_OUTPUT_LABEL = re.compile(
    r"\b(?:ws2812[\w-]*|(?:addressable\s+)?led\s+(?:strip|string|load|output))\b",
    re.I,
)
_OTHER_CURRENT_OWNER = re.compile(
    r"\b(?:hub\s*75|panel|display|speaker|amplifier|amp|"
    r"(?:usb[- ]?c\s+)?pd\s+(?:input|source|supply)|"
    r"(?:input|source|supply|regulator|converter))\b",
    re.I,
)


def _owned_led_current_obligations(upstream: dict, requirement: dict) -> list[float]:
    """Recover original quantitative load demands belonging to this output."""
    owned = _requirement_obligation_ids(requirement)
    rows = list(requirement.get("obligations") or [])
    for slot in ("intent", "functional_spec"):
        rows.extend(
            row for row in (upstream.get(slot) or {}).get("obligations") or []
            if isinstance(row, dict) and str(row.get("original_obligation_id") or "") in owned
        )
    values = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") != "quantitative":
            continue
        quantity = str(row.get("quantity") or "")
        if not (re.search(r"\bcurrent\b", quantity, re.I)
                and re.search(r"\b(?:led|strip|string|load)\b", quantity, re.I)):
            continue
        scale = {"a": 1.0, "ma": 0.001}.get(str(row.get("unit") or "").strip().lower())
        relation = row.get("relation")
        value = row.get("maximum") if relation == "range" else row.get("value")
        if (scale is not None and relation in {"equal", "minimum", "maximum", "range"}
                and isinstance(value, (int, float)) and not isinstance(value, bool)):
            values.append(float(value) * scale)
    return values


def _led_clause_currents(
    fragment: str, label: re.Pattern | None, generic: bool,
    other_labels: tuple[re.Pattern, ...] = (),
) -> tuple[list[float], bool]:
    """Bind numeric claims to their nearest named load, not to an entire brief."""
    values = []
    ambiguous = False
    for sentence in re.split(r"(?<!\d)\.|\.(?!\d)|\n+", fragment):
        led_context = False
        for clause in re.split(r"[;,]", sentence):
            targets = list(label.finditer(clause)) if label else []
            if generic:
                targets.extend(_LED_OUTPUT_LABEL.finditer(clause))
            others = list(_OTHER_CURRENT_OWNER.finditer(clause))
            others.extend(match for other in other_labels for match in other.finditer(clause))
            # A continuation such as "LED load unspecified; output budgeted to
            # 0.25 A" retains its owner. An explicitly named different load does not.
            continued = led_context and not targets and not others
            led_context = (bool(targets) and not others) or continued
            if not targets and not continued:
                continue
            bound = []
            for current in _CURRENT_PATTERN.finditer(clause):
                def distance(owner):
                    return max(owner.start() - current.end(), current.start() - owner.end(), 0)

                target_distance = min(distance(owner) for owner in targets) if targets else 0
                if others and min(distance(owner) for owner in others) <= target_distance:
                    continue
                bound.append(float(current.group(1)) / (1000 if current.group(2).lower() == "ma" else 1))
            if len(set(bound)) > 1:
                ambiguous = True
            values.extend(bound)
    return values, ambiguous


def _recipe_port_envelope_failures(requirements, definitions, rail_voltages):
    """Check reviewed whole-circuit input envelopes on their actual bound ports."""
    failures = []
    for req in requirements:
        definition = definitions.get(str(req.get("id") or ""))
        if definition is None:
            continue
        for port in definition.ports:
            envelope = port.supply_voltage_range_v
            rail = (req.get("ports") or {}).get(port.name)
            voltage = rail_voltages.get(rail)
            if envelope is not None and voltage is not None and not envelope[0] <= voltage <= envelope[1]:
                failures.append(
                    f"requirement {req.get('id')!r} recipe {definition.recipe!r} port "
                    f"{port.name!r} bound to {rail!r} is {voltage:g} V; reviewed circuit "
                    f"operating range is {envelope[0]:g}–{envelope[1]:g} V"
                )
    return failures


def _architecture_pd_source_contract(
    upstream: dict, candidate: dict, brief: str, rails: dict, requirements: list[dict],
    definitions: dict, reserves: dict,
) -> list[StageDiagnostic]:
    """Check the reviewed trigger's operating source, not an adapter measurement."""
    diagnostics = []
    answers = [
        f"{answer.get('text', '')} {answer.get('answer', '')}"
        for answer in upstream.get("_stage_answers") or [] if isinstance(answer, dict)
    ]
    fragments = _text_fragments([brief, upstream.get("intent"), candidate.get("assumptions"),
                                 candidate.get("topologies"), answers])
    user_fragments = _text_fragments([brief, upstream.get("intent"), answers])
    sources = [
        row for row in requirements
        if (definition := definitions.get(str(row.get("id") or ""))) is not None
        and definition.family in {"usb-pd-selectable-trigger", "usb-pd-fixed-trigger"}
    ]
    def source_names(req):
        raw = (req.get("ports") or {}).get("vbus")
        protected = [
            (row.get("ports") or {}).get("output") for row in requirements
            if (definition := definitions.get(str(row.get("id") or ""))) is not None
            and getattr(definition, "pass_through", False)
            and (row.get("ports") or {}).get("input") == raw
        ]
        return (req.get("id"), raw, *protected)


    def owned_by(fragment, req):
        owners = []
        for source_req in sources:
            names = source_names(source_req)
            if any(
                name and str(name).casefold() not in {"pd", "source", "input", "supply"}
                and re.search(r"(?<!\w)" + re.escape(str(name)) + r"(?!\w)", fragment, re.I)
                for name in names
            ):
                owners.append(source_req)
        return (len(owners) == 1 and owners[0] is req) or (not owners and len(sources) == 1)

    for req in requirements:
        definition = definitions.get(str(req.get("id") or ""))
        if definition is None or definition.family not in {
            "usb-pd-selectable-trigger", "usb-pd-fixed-trigger"
        }:
            continue
        rail = (req.get("ports") or {}).get("vbus")
        voltage = rails.get(rail)
        if voltage is None:
            continue
        options = (definition.allowed_parameters or {}).get("voltage_options")
        if options:
            positions = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", str(options[0]))]
        else:
            positions = [float(value) for value in
                         (definition.allowed_parameters or {}).get("output_voltage", (voltage,))]
        if not positions:
            continue
        if voltage not in positions:
            diagnostics.append(_diag(
                "architecture_pd_source_voltage_unsupported", "repair_required",
                f"PD rail {rail} declares {voltage:g} V, but the selected reviewed trigger "
                f"supports {', '.join(f'{v:g}' for v in positions)} V. Preserve any explicit "
                "user input and choose a supported construction.",
                [str(rail)],
            ))
        # The reviewed selectable construction specifies 20 V plus transients;
        # its shunt supply is reviewed to 22 V. Check every direct load at that
        # envelope using the ordinary reviewed-part supply rating gate.
        if len(positions) > 1:
            peak = max(positions) * 1.1
            from kicraft.design.part_identity import reviewed_part, reviewed_supply_voltage_limits

            protected = {
                (row.get("ports") or {}).get("output")
                for row in requirements
                if (selected := definitions.get(str(row.get("id") or ""))) is not None
                and getattr(selected, "pass_through", False)
                and (row.get("ports") or {}).get("input") == rail
            }
            exposed = {rail, *protected}
            direct = []
            for row in requirements:
                if row is req or not exposed.intersection(_requirement_supply_rails(row)):
                    continue
                selected = definitions.get(str(row.get("id") or ""))
                direct.append({
                    **row,
                    "exact_part": row.get("exact_part") or
                                  (selected.exact_part if selected else None),
                })
            unqualified = []
            for row in direct:
                selected = definitions.get(str(row.get("id") or ""))
                bound_ports = [
                    port for port in (selected.ports if selected else ())
                    if (row.get("ports") or {}).get(port.name) in exposed
                    and port.direction == "power"
                ]
                has_envelope = bool(bound_ports) and all(
                    port.supply_voltage_range_v is not None for port in bound_ports
                )
                exact = row.get("exact_part")
                record = reviewed_part(str(exact)) if exact else None
                has_rating = record is not None and any(
                    high is not None for _, _, high in reviewed_supply_voltage_limits(record)
                )
                if not has_envelope and not has_rating:
                    unqualified.append(
                        f"requirement {row.get('id')!r} on {rail!r} has neither a reviewed "
                        "bound-port envelope nor a reviewed part supply rating"
                    )
            if unqualified:
                diagnostics.append(_diag(
                    "architecture_pd_selectable_supply_unqualified", "repair_required",
                    f"Selectable PD rail {rail} reaches a {peak:g} V transient envelope; "
                    "each direct load needs reviewed supply compatibility.",
                    unqualified,
                ))
            rated_candidate = {"rail_voltages": {name: peak for name in exposed}, "requirements": direct}
            for failure in _architecture_supply_over_rating(rated_candidate):
                diagnostics.append(_diag(
                    "architecture_pd_selectable_supply_exceeds_part_rating", "repair_required",
                    f"Selectable PD rail {rail} can reach {max(positions):g} V and its reviewed "
                    f"transient envelope reaches {peak:g} V; every directly connected load must "
                    "tolerate that envelope, not merely the nominal setting.",
                    failure.evidence,
                ))
            envelope_failures = []
            for operating_voltage in [*positions, peak]:
                envelope_failures.extend(_recipe_port_envelope_failures(
                    [row for row in requirements if row is not req], definitions,
                    {name: operating_voltage for name in exposed}
                ))
            if envelope_failures:
                diagnostics.append(_diag(
                    "architecture_pd_selectable_supply_exceeds_recipe_envelope", "repair_required",
                    f"Selectable PD rail {rail} includes positions through {max(positions):g} V "
                    f"and a reviewed {peak:g} V transient envelope; every bound load port "
                    "must tolerate all of them.",
                    envelope_failures,
                ))


        profiles = []
        for fragment in fragments:
            for clause in re.split(r"(?<!\d)\.|\.(?!\d)|[;\n]+", fragment):
                if not owned_by(clause, req):
                    continue
                source = re.search(
                    r"\b(?:usb[- ]?c\s+)?pd\s+(?:source|input|supply|contract)\b",
                    clause, re.I,
                )
                if not source:
                    for name in source_names(req):
                        if name and str(name).casefold() not in {"pd", "source", "input", "supply"}:
                            source = re.search(
                                r"(?<!\w)" + re.escape(str(name)) + r"(?!\w)", clause, re.I
                            )
                            if source:
                                break
                if not source:
                    continue
                volts = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*v\b", clause, re.I)
                currents = list(_CURRENT_PATTERN.finditer(clause))
                if len(volts) == 1 and len(currents) == 1:
                    match = currents[0]
                    others = list(_LED_OUTPUT_LABEL.finditer(clause))
                    others.extend(owner for owner in _OTHER_CURRENT_OWNER.finditer(clause)
                                  if not (owner.start() <= source.start() < owner.end()))

                    def distance(owner):
                        return max(owner.start() - match.end(), match.start() - owner.end(), 0)

                    if others and min(map(distance, others)) <= distance(source):
                        continue
                    profiles.append((float(volts[0]), float(match.group(1)) /
                                     (1000 if match.group(2).lower() == "ma" else 1),
                                     clause, fragment in user_fragments))
        conflicting = [profile for profile in profiles
                       if profile[3] and abs(profile[0] - voltage) >= 0.01]
        if conflicting:
            diagnostics.append(_diag(
                "architecture_pd_explicit_source_voltage_mismatch", "repair_required",
                f"PD rail {rail} is {voltage:g} V but an explicit source allocation names "
                "another voltage; retain the user input and choose a compatible reviewed source.",
                [profile[2] for profile in conflicting],
            ))
        matching = [profile for profile in profiles if abs(profile[0] - voltage) < 0.01]
        needed = reserves.get(rail, 0.0)
        if not matching:
            diagnostics.append(_diag(
                "architecture_pd_source_allocation_unspecified", "repair_required",
                f"State a numeric PD source voltage and current allocation for {rail} at "
                f"{voltage:g} V, with positive margin above the bound converter and load "
                f"planning reserve of {needed:g} A. This is an operating assumption, not a "
                "measurement; retain any explicit user input and loads.",
                [str(rail)],
            ))
        elif any(current > 3.0 or current <= needed + 1e-9
                 for _, current, _, _ in matching):
            diagnostics.append(_diag(
                "architecture_pd_source_allocation_invalid", "repair_required",
                f"PD source allocation at {voltage:g} V must exceed the {needed:g} A "
                "bound planning reserve and must not exceed 3 A: the reviewed receptacle "
                "construction does not guarantee E-marked operation above 3 A. Preserve "
                "explicit user requirements and select a supported construction if needed.",
                [profile[2] for profile in matching],
            ))
        controlled = any(
            (selected := definitions.get(str(row.get("id") or ""))) is not None
            and getattr(selected, "pass_through", False)
            and (row.get("ports") or {}).get("input") == rail
            for row in requirements
        )
        direct_5v_buck = any(
            (row.get("ports") or {}).get("input") == rail
            and rails.get((row.get("ports") or {}).get("output")) == 5.0
            and str(row.get("family") or "").startswith(("ap63205", "buck"))
            for row in requirements
        )
        if direct_5v_buck and not controlled:
            text = " ".join(fragment for fragment in fragments if owned_by(fragment, req)).lower()
            if not (re.search(r"\b5\s*v\b", text)
                    and re.search(r"startup|pre[- ]negotiat", text)
                    and re.search(r"fallback", text)
                    and re.search(r"(?:not|cannot|can't|no)\b.{0,80}\bregulat", text)
                    and re.search(r"\bnegotiated\b", text)
                    and re.search(r"external.{0,30}operation|operation.{0,30}external", text)
                    and re.search(r"available|availability|adequate", text)):
                diagnostics.append(_diag(
                    "architecture_pd_startup_envelope_unspecified", "repair_required",
                    "State the 5 V pre-negotiation source availability needed for startup; "
                    "do not assume an arbitrary 500 mA USB limit. A buck fed by 5 V fallback "
                    "does not establish regulated 5 V: full external-output operation requires "
                    "the negotiated input voltage and adequate source current. This is an "
                    "operating envelope, not a claim about any particular adapter.",
                    [str(rail)],
                ))
    return diagnostics


def _architecture_bound_load_budget(upstream: dict, candidate: dict, brief: str) -> list[StageDiagnostic]:
    """Budget external strip outputs and reviewed onboard allocations on their bound rails.

    Reserve full rated converter output unless every protected-rail converter
    output consumer has a bounded allocation. Linear regulators use a 5%
    planning allowance and switching converters 80% efficiency. Protected
    converter inputs use 95% of nominal voltage. None is a measurement or
    thermal proof.
    """
    from kicraft.design.architecture_intent import _supply_port_name
    from kicraft.design.recipes import get_recipe, registered_recipes

    rails = _architecture_rail_voltages(candidate)
    requirements = [r for r in candidate.get("requirements") or [] if isinstance(r, dict)]
    resolutions = {
        str(row.get("requirement_id")): row.get("recipe")
        for row in candidate.get("recipe_resolution") or [] if isinstance(row, dict)
    }
    definitions = {}
    for req in requirements:
        identifier = str(req.get("id") or "")
        selected = resolutions.get(identifier)
        if selected:
            try:
                definitions[identifier] = get_recipe(selected)
            except ValueError:
                continue
        else:
            matches = [
                item.definition for item in registered_recipes()
                if item.definition.family == req.get("family")
                and item.definition.default_for_family
            ]
            if len(matches) == 1:
                definitions[identifier] = matches[0]

    envelope_failures = _recipe_port_envelope_failures(requirements, definitions, rails)
    envelope_diagnostics = [
        _diag(
            "architecture_supply_outside_recipe_envelope", "repair_required",
            "A bound supply rail is outside the selected recipe port's reviewed circuit "
            "operating range. Choose a compatible construction or regulate the rail.",
            envelope_failures,
        )
    ] if envelope_failures else []

    reserves: dict[str, float] = {}
    strips = []
    protected_inputs = {
        (row.get("ports") or {}).get("output")
        for row in requirements
        if (definition := definitions.get(str(row.get("id") or ""))) is not None
        and getattr(definition, "pass_through", False)
    }
    for req in requirements:
        definition = definitions.get(str(req.get("id") or ""))
        if definition is None:
            continue
        ports = req.get("ports") or {}
        for port, current in definition.supply_current_reservations_a.items():
            rail = ports.get(port)
            if rail in rails:
                reserves[rail] = reserves.get(rail, 0.0) + current
        if definition.family in {"ws2812-external-strip", "ws2812-external-strip-header"}:
            rail = ports.get("vdd_5v")
            if rail in rails:
                strips.append((req, rail))
        output = ports.get("output") or ports.get("vout")
        input_rail = ports.get("input") or ports.get("vin")
        rating = definition.rated_output_current_a
        if (not getattr(definition, "pass_through", False)
                and rating is not None and output in rails and input_rail in rails
                and output != input_rail and rails[input_rail] > 0):
            if definition.family in {"me6211-3v3", "mcp1700-3v3", "ams1117-3v3"}:
                # Input current cannot be less than output current for an LDO.
                # The extra 5% is a planning allowance, not a datasheet Iq claim.
                input_current = rating * 1.05
            else:
                input_voltage = rails[input_rail] * (0.95 if input_rail in protected_inputs else 1.0)
                input_current = rating * rails[output] / (input_voltage * 0.8)
            reserves[input_rail] = reserves.get(input_rail, 0.0) + input_current

    # Reduce a converter's upstream planning allocation only when every bound
    # consumer of its output has an explicit reviewed reservation, a known
    # child-converter reservation, or an unambiguous external LED allocation.
    # Unknown consumers retain the full rated-output reservation above.
    load_fragments = _text_fragments([brief, upstream.get("intent"), candidate.get("assumptions")])
    for answer in upstream.get("_stage_answers") or []:
        if isinstance(answer, dict) and _external_load_budget_answered([answer]):
            load_fragments.append(f"{answer.get('text', '')} {answer.get('answer', '')}")
    labels = {
        identifier: re.compile(r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", re.I)
        for req, _rail in strips if (identifier := str(req.get("id") or ""))
    }
    strip_values = {}
    for req, rail in strips:
        identifier = str(req.get("id") or "")
        label = labels.get(identifier)
        other_labels = tuple(pattern for key, pattern in labels.items() if key != identifier)
        typed = _owned_led_current_obligations(upstream, req)
        values = list(typed)
        ambiguous = False
        for fragment in load_fragments:
            found, unclear = _led_clause_currents(fragment, label, len(strips) == 1, other_labels)
            values.extend(found)
            ambiguous |= unclear
        if values and all(value > 0 for value in values) and (typed or not ambiguous):
            strip_values[identifier] = max(values)

    for req in requirements:
        definition = definitions.get(str(req.get("id") or ""))
        ports = req.get("ports") or {}
        input_rail = ports.get("input") or ports.get("vin")
        output = ports.get("output") or ports.get("vout")
        if (definition is None or getattr(definition, "pass_through", False)
                or definition.rated_output_current_a is None
                or input_rail not in protected_inputs or output not in rails
                or input_rail not in rails or rails[input_rail] <= 0):
            continue
        covered = True
        external = 0.0
        consumer_count = 0
        for consumer in requirements:
            if consumer is req or output not in _requirement_supply_rails(consumer):
                continue
            consumer_count += 1
            consumer_def = definitions.get(str(consumer.get("id") or ""))
            cports = consumer.get("ports") or {}
            supply_ports = [
                port for port, net in cports.items()
                if net == output and (port in {"input", "input_positive"}
                                      or _supply_port_name(port))
            ]
            if (consumer_def is None or not supply_ports
                    or getattr(consumer_def, "pass_through", False)):
                covered = False
                break
            for port in supply_ports:
                identifier = str(consumer.get("id") or "")
                is_strip = consumer_def.family in {
                    "ws2812-external-strip", "ws2812-external-strip-header"
                } and port == "vdd_5v"
                if is_strip:
                    if identifier not in strip_values:
                        covered = False
                        break
                    external += strip_values[identifier]
                elif port not in consumer_def.supply_current_reservations_a and not (
                    port in {"input", "vin"} and consumer_def.rated_output_current_a is not None
                    and (cports.get("output") or cports.get("vout")) in rails
                ):
                    covered = False
                    break
            if not covered:
                break
        if not covered or not consumer_count:
            continue
        planned_output = reserves.get(output, 0.0) + external
        rated = definition.rated_output_current_a
        if planned_output >= rated:
            continue
        if definition.family in {"me6211-3v3", "mcp1700-3v3", "ams1117-3v3"}:
            old, new = rated * 1.05, planned_output * 1.05
        else:
            scale = rails[output] / (rails[input_rail] * 0.95 * 0.8)
            old, new = rated * scale, planned_output * scale
        reserves[input_rail] += new - old

    # Series paths consume only their own input reservation plus downstream
    # planning current, not their full rating or a converter efficiency penalty.
    paths = []
    for req in requirements:
        definition = definitions.get(str(req.get("id") or ""))
        ports = req.get("ports") or {}
        if (definition is not None and getattr(definition, "pass_through", False)
                and ports.get("input") in rails and ports.get("output") in rails):
            paths.append((req, definition, ports["input"], ports["output"]))
    path_diagnostics = []
    pending = list(paths)
    while pending:
        ready = [p for p in pending if not any(q[2] == p[3] for q in pending)]
        if not ready:
            break
        for req, definition, input_rail, output in ready:
            needed = reserves.get(output, 0.0)
            for strip, rail in strips:
                if rail == output:
                    identifier = str(strip.get("id") or "")
                    if identifier in strip_values:
                        needed += strip_values[identifier]
            for consumer in requirements:
                if consumer is req or output not in _requirement_supply_rails(consumer):
                    continue
                consumer_def = definitions.get(str(consumer.get("id") or ""))
                cports = consumer.get("ports") or {}
                supply_ports = [
                    port for port, net in cports.items()
                    if net == output and (port in {"input", "input_positive"}
                                          or _supply_port_name(port))
                ]
                unknown = not consumer_def or not supply_ports
                if not unknown:
                    for port in supply_ports:
                        if port in consumer_def.supply_current_reservations_a:
                            continue
                        if port in {"input", "vin"} and (
                            consumer_def.rated_output_current_a is not None
                            or getattr(consumer_def, "pass_through", False)
                        ) and (cports.get("output") or cports.get("vout")) in rails:
                            continue
                        if (consumer_def.family in {
                            "ws2812-external-strip", "ws2812-external-strip-header"
                        } and port == "vdd_5v" and
                                str(consumer.get("id") or "") in strip_values):
                            continue
                        unknown = True
                        break
                if unknown:
                    path_diagnostics.append(_diag(
                        "architecture_power_path_load_unspecified", "repair_required",
                        f"Series path {req.get('id')} has a downstream consumer without "
                        "a reviewed current reservation or bounded external load.",
                        [str(consumer.get("id") or ""), str(output)],
                    ))
            capacity = definition.rated_output_current_a
            if capacity is None or needed >= capacity - 1e-9:
                path_diagnostics.append(_diag(
                    "architecture_power_path_capacity_insufficient", "repair_required",
                    f"Series path {req.get('id')} must have reviewed capacity with positive "
                    f"margin above the {needed:g} A downstream planning reserve.",
                    [str(output)],
                ))
            reserves[input_rail] = reserves.get(input_rail, 0.0) + needed
            pending.remove((req, definition, input_rail, output))
    if pending:
        path_diagnostics.append(_diag(
            "architecture_power_path_cycle", "repair_required",
            "Series power paths must not form a cycle.", [str(p[0].get("id")) for p in pending],
        ))

    pd_diagnostics = envelope_diagnostics + path_diagnostics + _architecture_pd_source_contract(
        upstream, candidate, brief, rails, requirements, definitions, reserves
    )
    if not strips:
        return pd_diagnostics
    producers = _rail_producers(candidate, rails)
    # Architecture drafts can precede recipe_resolution; use the same registered
    # family default used above, never an authored converter headline.
    for producer in producers:
        if producer["rated_output_current_a"] is None:
            definition = definitions.get(producer["requirement_id"])
            if definition:
                producer["rated_output_current_a"] = definition.rated_output_current_a

    fragments = _text_fragments([brief, upstream.get("intent"), candidate.get("assumptions")])
    for answer in upstream.get("_stage_answers") or []:
        if isinstance(answer, dict) and _external_load_budget_answered([answer]):
            fragments.append(f"{answer.get('text', '')} {answer.get('answer', '')}")
    # An output can be identified by its requirement id; generic strip language is
    # usable only when there is one output. Multiple numeric claims are ambiguous.
    loads: dict[str, float] = {}
    diagnostics = pd_diagnostics
    for req, rail in strips:
        identifier = str(req.get("id") or "")
        label = labels.get(identifier)
        other_labels = tuple(pattern for key, pattern in labels.items() if key != identifier)
        typed_values = _owned_led_current_obligations(upstream, req)
        values = list(typed_values)
        ambiguous = False
        for fragment in fragments:
            found, unclear = _led_clause_currents(fragment, label, len(strips) == 1, other_labels)
            values.extend(found)
            ambiguous |= unclear
        if (ambiguous and not typed_values) or not values or any(value <= 0 for value in values):
            diagnostics.append(_diag(
                EXTERNAL_OUTPUT_ALLOCATION_CODE, "repair_required",
                f"Powered external LED output {identifier} on {rail} needs an unambiguous maximum "
                "current budget; when unspecified, default conservatively to 0.25 A and disclose it. "
                "Preserve any explicit user load instead of reducing it.",
                [identifier, rail],
            ))
            continue
        # Repeated disclosures describe the same load; the largest claim preserves
        # explicit higher requirements instead of silently replacing them by a default.
        loads[rail] = loads.get(rail, 0.0) + max(values)

    for rail, external in loads.items():
        sources = [p for p in producers if p["rail"] == rail]
        rated = [p for p in sources if p["rated_output_current_a"] is not None]
        if len(rated) != 1 or len(sources) != 1:
            diagnostics.append(_diag(
                "architecture_external_load_source_capacity_unspecified", "repair_required",
                f"External output on {rail} needs one bound source with a reviewed output-current rating.",
                [rail],
            ))
            continue
        source = rated[0]
        capacity = source["rated_output_current_a"]
        onboard = reserves.get(rail, 0.0)
        allowed = max(0.0, capacity - onboard)
        if external + onboard >= capacity - 1e-9:
            diagnostics.append(_diag(
                "architecture_5v_converter_has_no_headroom", "repair_required",
                f"Rail {rail} source {source['requirement_id']} is rated {capacity:g} A; "
                f"onboard and downstream reserves total {onboard:g} A, leaving an external "
                f"cap below {allowed:g} A with positive headroom. The bound external outputs "
                f"claim {external:g} A. Select a larger reviewed source for explicit loads; "
                "do not reduce user requirements. Downstream linear and switching conversion "
                "use planning allowances, not measured consumption.",
                [rail, source["requirement_id"], f"external {external:g} A", f"reserve {onboard:g} A"],
            ))
    return diagnostics


def _architecture(upstream: dict, candidate: dict, brief: str = "") -> list[StageDiagnostic]:
    diagnostics = architecture_power_requirement_diagnostics(upstream, candidate)
    diagnostics.extend(_architecture_bound_load_budget(upstream, candidate, brief))
    diagnostics.extend(_speaker_architecture_diagnostics(brief, upstream, candidate))
    diagnostics.extend(architecture_conversion_output_diagnostics(upstream, candidate))
    diagnostics.extend(_architecture_supply_over_rating(candidate))
    diagnostics.extend(_architecture_motor_clamp_standoff(candidate))
    diagnostics.extend(_architecture_drive_on_logic_rail(upstream, candidate))
    diagnostics.extend(_architecture_obligation_family_mismatch(candidate))
    diagnostics.extend(_architecture_derivation_diagnostics(upstream, candidate))
    diagnostics.extend(_architecture_declared_contact_unnamed(candidate))
    sheets = candidate.get("sheets") or []
    for sheet in sheets:
        if not isinstance(sheet, dict):
            continue
        name = str(sheet.get("name") or "")
        function = str(sheet.get("function") or "")
        # The escape hatch for "this rail-named sheet is actually a function" must accept every
        # inflected form of its own words. It read `regulat(?:or|ion)?`, which cannot match
        # "regulate", so a shipped Arduino-shield board whose POWER sheet "accept[ed] host power
        # rails and regulate[d] VIN to a 3.3 V rail" was refused as distribution-only (replay
        # 2026-09-26: 10 shipped boards). Same for convert/supply/input/protect/connect.
        physical_power_domain = re.search(
            r"\b(?:ldo|buck|boost|flyback|regulat\w*|convert\w*|suppl\w*|input\w*|sink\w*|"
            r"controll\w*|protect\w*|connect\w*|header\w*|terminal\w*|batter\w*|holder\w*|"
            r"implement\w*)\b",
            f"{name} {function}",
            re.I,
        )
        looks_like_power_only = is_power_or_ground_name(name) or bool(_POWER_RE.search(name))
        # A sheet named for a rail is only a distribution sheet when nothing else lives on it: a
        # "POWER INDICATOR" sheet holds a user-io requirement, and flagging it on the word "power"
        # alone asked the stage to delete a real circuit (live draft, 2026-09-25: two repair
        # rounds spent on a sheet the design needs).
        installed_roles = {
            str(row.get("role") or "")
            for row in candidate.get("requirements") or []
            if isinstance(row, dict) and str(row.get("sheet") or "") == name
        }
        distribution_only = not installed_roles or installed_roles <= {"power_input", "regulator"}
        if looks_like_power_only and not physical_power_domain and distribution_only:
            diagnostics.append(
                _diag(
                    "architecture_power_block_as_sheet",
                    "repair_required",
                    "A power net or distribution-only block was emitted as a physical sheet.",
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
    load_descriptions = {
        str(block.get("name") or "").lower(): _text(
            [block.get("name"), block.get("purpose")]
        ).lower().replace("_", " ").replace("-", " ")
        for block in (upstream.get("functional_spec") or {}).get("blocks") or []
        if isinstance(block, dict)
    }
    hub75_targets = {
        target for target in functional_power_targets
        if "hub75" in load_descriptions.get(target, target)
    }
    led_strip_targets = {
        target for target in functional_power_targets
        if re.search(
            r"\b(?:led|ws2812)\b.*\b(?:strip|string)\b",
            load_descriptions.get(target, target.replace("_", " ").replace("-", " ")),
        )
    }
    functional_powers_external = any(
        panel != strip for panel in hub75_targets for strip in led_strip_targets
    )
    board_powers_external = functional_powers_external or bool(
        re.search(
            r"board supplies power to both|power both from board",
            answer_text,
            re.I,
        )
    )
    stage_answers = upstream.get("_stage_answers", [])
    has_5v_load_budget = external_load_budget_stated(candidate) or _external_load_budget_answered(
        stage_answers
    )
    if board_powers_external and not has_5v_load_budget:
        diagnostics.append(
            _diag(
                EXTERNAL_LOAD_CURRENT_CODE,
                "repair_required",
                "Board-powered external loads have no maximum 5V current budget.",
                sorted(hub75_targets | led_strip_targets) or [answer_text],
            )
        )
    external_load_currents = [
        float(match.group(1)) / (1000 if match.group(2).casefold() == "ma" else 1)
        for answer in stage_answers
        if isinstance(answer, dict) and _external_load_budget_answered([answer])
        for match in re.finditer(
            r"\b(\d+(?:\.\d+)?)\s*(ma|a)\b",
            str(answer.get("answer") or ""),
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
        declared_sheets = {str(sheet.get("name")) for sheet in sheets if isinstance(sheet, dict)}
        producers = _rail_producers(
            candidate,
            {
                rail: voltage
                for rail, voltage in rail_voltages.items()
                if abs(float(voltage) - 3.3) <= 0.05
            },
        )
        sized = [
            producer
            for producer in producers
            if producer["rated_output_current_a"] is not None
            and producer["rated_output_current_a"] >= 0.5
            and producer["sheet"] in declared_sheets
            and not re.search(r"\b(?:mcu|esp32)\b", producer["sheet"], re.I)
        ]
        if not sized:
            diagnostics.append(
                _diag(
                    "architecture_mcu_regulator_incomplete",
                    "repair_required",
                    "ESP32-S3 needs its 3.3V rail generated by a regulator the recipe rates for >=0.5A "
                    "on a sheet of its own.",
                    [
                        "esp32-s3",
                        *(
                            f"{producer['rail']}: {producer['sheet']} "
                            f"{producer['requirement_id']} rated "
                            f"{producer['rated_output_current_a']}"
                            for producer in producers
                        ),
                        ">=0.5a",
                        "separate regulator sheet",
                    ],
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
    requirements_by_sheet: dict[str, list[dict]] = {}
    for requirement in architecture.get("requirements") or []:
        if isinstance(requirement, dict):
            requirements_by_sheet.setdefault(str(requirement.get("sheet") or ""), []).append(
                requirement
            )
    ic_role = re.compile(
        r"\b(?:controller|mcu|regulator|converter|buck|boost|amplifier|"
        r"level shifter|sensor|bridge|driver|hub)\b",
        re.I,
    )
    # `driver` is a role the pipeline assigns to a relay, an LED string or a transistor stage --
    # implemented by K/D/Q references, never a U. Reading it as an IC role refused shipped
    # relay-quad and LED-ring boards whose relays and WS2812 strings are exactly what the brief
    # asked for (replay 2026-09-26: 34 shipped boards). The sheet's own *title* keeps the word:
    # a sheet called MOTOR DRIVER with nothing on it is still worth refusing.
    ic_role_in_requirements = re.compile(
        r"\b(?:controller|mcu|regulator|converter|buck|boost|amplifier|"
        r"level shifter|sensor|bridge|hub)\b",
        re.I,
    )

    def _own_terms(sheet_name: str, requirements: list[dict], connector_owned: bool) -> set[str]:
        """The IC roles the sheet declares about itself: its own title and its typed requirements.

        A physical connector may be named for the external IC it connects to, so a
        connector-owned sheet contributes no title. Prose (`function`) is deliberately absent:
        it describes an *effect* and routinely names a part that lives on another sheet.
        """
        terms: set[str] = set()
        if not connector_owned:
            terms.update(match.lower() for match in ic_role.findall(sheet_name))
        for requirement in requirements:
            if requirement.get("role") == "connector":
                continue
            text = re.sub(
                r"[-_]", " ", f"{requirement.get('role', '')} {requirement.get('family', '')}"
            )
            terms.update(match.lower() for match in ic_role_in_requirements.findall(text))
        return terms

    def _is_connector_owned(requirements: list[dict], sheet_parts: list[dict]) -> bool:
        return (
            bool(requirements)
            and all(
                requirement.get("role") == "connector" and requirement.get("ports")
                for requirement in requirements
            )
            and any(str(part.get("ref") or "").startswith(("J", "P")) for part in sheet_parts)
        )

    # Terms some sheet actually implements with an IC (a U-reference on the declaring sheet).
    # A prose mention of one of these on another sheet is a reference to that part, not a claim
    # about the sheet it appears on: live replay 2026-09-26 refused the shipped seed-43 board
    # because "Expose the MCU UART … signals on a header" and "pulls the MCU reset input low"
    # name the MCU, which lives (with its U2) on the MCU sheet.
    implemented_terms: set[str] = set()
    for sheet in architecture.get("sheets") or []:
        if not isinstance(sheet, dict):
            continue
        sheet_name = str(sheet.get("name") or "")
        sheet_parts = parts_by_sheet.get(sheet_name, [])
        if not any(str(part.get("ref") or "").startswith("U") for part in sheet_parts):
            continue
        requirements = requirements_by_sheet.get(sheet_name, [])
        implemented_terms |= _own_terms(
            sheet_name, requirements, _is_connector_owned(requirements, sheet_parts)
        )

    unsupported_roles: list[str] = []
    for sheet in architecture.get("sheets") or []:
        if not isinstance(sheet, dict):
            continue
        sheet_name = str(sheet.get("name") or "")
        sheet_parts = parts_by_sheet.get(sheet_name, [])
        requirements = requirements_by_sheet.get(sheet_name, [])
        connector_owned = _is_connector_owned(requirements, sheet_parts)
        own_terms = _own_terms(sheet_name, requirements, connector_owned)
        # A sheet that carries a non-connector requirement has stated what it implements, and that
        # statement is enforced by the obligation and declared-interface gates on its own. Prose is
        # then only an *effect* it names, never a role it promises: live run KC-WWYS3W (project 1109,
        # brief 2) was refused at BOM because the POWER INDICATOR sheet's function reads "a visible
        # low-current indication that 5 V controller power is present" -- "controller" is the MCU's
        # word on another sheet -- while the sheet's own requirement (the reviewed LED build) was
        # realized. The title path is untouched, so a sheet *named* for an IC role with nothing on
        # it is still refused.
        realized_here = any(
            isinstance(requirement, dict) and requirement.get("role") != "connector"
            for requirement in requirements
        )
        # Prose only counts for a role the design implements nowhere: then the sheet promises
        # an active part nothing builds ("Connector and on-board amplifier" with no U anywhere).
        prose_terms = (
            set()
            if realized_here
            else {match.lower() for match in ic_role.findall(str(sheet.get("function") or ""))}
            - own_terms
        )
        if not own_terms and not (prose_terms - implemented_terms):
            continue
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
        check_requirement_physical_realization,
    )

    checks = (
        ("wiring_gate_9_11", check_net_coverage(bom)),
        ("wiring_gate_9_14", check_inter_sheet_nets_realized(architecture, bom)),
        ("wiring_gate_9_15", check_no_dangling_signal_nets(architecture, bom)),
        ("wiring_gate_9_29", check_mcu_programming_access(bom, architecture=architecture)),
        # §9.42's model-owned half: only the wiring graph can prove a declared
        # interface whose implementing component came from the model's BOM
        # groups. Recipe/lowerer-owned declared interfaces are proven at BOM
        # commit, where their expansions supply the connections.
        (
            "wiring_gate_9_42",
            check_requirement_physical_realization(
                architecture, bom, declared_interface_scope="model_owned"
            ),
        ),
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
        findings = _architecture(upstream_state, candidate, brief)
    elif stage == "bom":
        findings = _bom(upstream_state, candidate)
    elif stage == "wiring":
        findings = _wiring(upstream_state, candidate)
    else:
        findings = []
    # A row whose writer recorded no severity sorts with the advisory ones instead of raising.
    return sorted(
        findings,
        key=lambda finding: (finding.severity or "", finding.code, finding.evidence),
    )
