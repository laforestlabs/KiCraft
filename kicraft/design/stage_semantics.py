"""Pure, versioned semantic diagnostics for schema-valid stage candidates."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable

from kicraft.design.models import (
    Architecture,
    BOM,
    MAX_BOARD_COPPER_LAYERS,
    MIN_BOARD_COPPER_LAYERS,
    StageDiagnostic,
    board_copper_layers,
    is_power_or_ground_name,
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


def external_load_budget_stated(text) -> bool:
    """Does this text state how much current the external 5 V loads draw?

    One definition, read twice: the semantic check fires when the *slot* does not
    state it, and the driver asks the user only when the brief does not state it
    either. Asking for a number the user already gave would be a question nobody
    can answer better.
    """
    return bool(
        re.search(
            r"(?:5v|vbus|hub75|led string|external load)[^.;]{0,80}"
            r"\d+(?:\.\d+)?\s*(?:a|ma)\b",
            _text(text),
            re.I,
        )
    )


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
    from kicraft.design.part_identity import board_outline_fabrication_feature

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
    # invented here. A row the writer already wrote is left alone.
    layers = requested_board_copper_layers(brief)
    if layers is not None and board_copper_layers(completed.get("obligations")) is None:
        completed["obligations"] = [
            *(completed.get("obligations") or []),
            _board_stackup_obligation(layers),
        ]

    normalized_obligations = []
    for obligation in completed.get("obligations") or []:
        if not isinstance(obligation, dict) or obligation.get("kind") != "physical":
            normalized_obligations.append(obligation)
            continue
        feature = board_outline_fabrication_feature(str(obligation.get("component_class") or ""))
        normalized_obligations.append(
            {
                "kind": "fabrication",
                "original_obligation_id": obligation.get("original_obligation_id"),
                "feature": feature,
            }
            if feature is not None
            else obligation
        )
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
        if class_is_not_a_part(demanded) or off_board_source_class(demanded):
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
}

#: A count bound to the layer noun, so an unrelated number ("under 100 x 100 mm") never reads
#: as a stack-up. `two-layer`, `4 layer`, `four copper layers` all match.
_STACKUP_COUNT_RE = re.compile(
    r"\b(?P<count>\d{1,2}|"
    + "|".join(_STACKUP_COUNT_WORDS)
    + r")[\s-]+(?:copper[\s-]+)?layers?\b",
    re.IGNORECASE,
)


def requested_board_copper_layers(text: str) -> int | None:
    """The copper-layer count the brief states, or ``None`` when it states none.

    Only a count a printed stack-up can carry is returned, so a nonsensical request ("a
    hundred-layer board") stays unrecorded rather than failing the build with a number the
    writer never meant as a stack-up.
    """
    for match in _STACKUP_COUNT_RE.finditer(text or ""):
        raw = match.group("count").casefold()
        count = int(raw) if raw.isdigit() else _STACKUP_COUNT_WORDS.get(raw)
        if count is not None and MIN_BOARD_COPPER_LAYERS <= count <= MAX_BOARD_COPPER_LAYERS:
            return count
    return None


def _board_stackup_obligation(layers: int) -> dict:
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


def _unrealizable_obligation_classes(obligations) -> list[StageDiagnostic]:
    """Physical obligations whose class is a not-a-part fact or a puzzled variant name.

    Three cases, each safe to reject at the stage that writes them:

    * the class carries a token no physical class can have (an interface, bus, board
      format, package style, or printed-board feature), so it belongs in `constraints`,
      a `fabrication` row, or a `negative` row;
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
    for the same reason: they name something the board does place.
    """
    from kicraft.design.part_identity import (
        class_is_not_a_part,
        off_board_source_class,
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
        if not_a_part:
            diagnostics.append(
                _diag(
                    "intent_obligation_class_unrealizable",
                    "repair_required",
                    "A physical obligation names an interface, board format or board "
                    "fabrication feature rather than a part class.",
                    [f"{component_class} -> not a part class ({', '.join(not_a_part)})"],
                )
            )
        elif off_board_source_class(component_class):
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


def _functional_spec(brief: str, upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics: list[StageDiagnostic] = []
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
    diagnostics = []
    for requirement in requirements:
        family = _norm_token(requirement.get("family") or "")
        generic_power = family in {
            "powerinput",
            "powerconversion",
            "powerdistribution",
            "directbatteryrail",
        }
        distribution = family in {"powerdistribution", "directbatteryrail"}
        if not generic_power and requirement.get("role") not in {"power_input", "regulator"}:
            continue
        ports = requirement.get("ports") or {}
        port_aliases = {_norm_token(key): net for key, net in ports.items()}
        input_net = port_aliases.get("input") or port_aliases.get("vin")
        output_net = port_aliases.get("output") or port_aliases.get("vout")
        ground_net = port_aliases.get("gnd") or port_aliases.get("ground")
        parameters = requirement.get("parameters") or {}
        input_voltage = parameters.get("input_voltage", rails.get(input_net))
        output_voltage = parameters.get("output_voltage", rails.get(output_net))
        voltage_change = (
            type(input_voltage) in (int, float)
            and type(output_voltage) in (int, float)
            and abs(input_voltage - output_voltage) > 0.05
        )
        incomplete_conversion = voltage_change and (
            generic_power
            or not input_net
            or not output_net
            or not ground_net
            or len({input_net, output_net, ground_net}) != 3
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
                "Voltage conversion must have a typed regulator requirement with a physical "
                "converter family (and exact part where known), distinct input/output/GND ports "
                "before BOM. Keep the declared voltages and all functional ownership; split "
                "the external connector from its converter if they are separate hardware. "
                "A generic power-input requirement does not own a registered regulator."
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
        if _supply_port_name(str(port).casefold()) and str(net) not in rails:
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


#: Reviewed converter families by the rail each one produces, for a load that needs a rail of its
#: own. Every entry is a recipe the library carries with the divider its own data specifies, so the
#: pipeline picks a rail it can actually build and never invents a voltage: the highest one that
#: fits inside the part's rated range (a DRV8833's 2.7-10.8 V vm picks 10.0 V, the MP1584 instance).
_REVIEWED_RAIL_FAMILIES: tuple[tuple[str, float], ...] = (
    # Highest first: the selection takes the first entry that fits inside the part's rated range.
    # Only families whose part the offline catalog stocks at both JLCPCB assembly and the lcsc.com
    # retail storefront belong here -- a rail that cannot be ordered is not a rail (the 10 V
    # MP1584EN looked ideal until §9.26 reported 0 retail stock and the BOM could not commit).
    ("ap63205-5v", 5.0),
    ("tps54331-adjustable", 3.3),
)

#: The reviewed adjustable buck the pipeline uses when a load needs its own regulated rail
#: (3.5-28 V input, adjustable output). Named here, not chosen per design: it is the library's
#: general-purpose converter family, and the assumption records the choice.
_OVER_RATED_REGULATOR_FAMILY = "tps54331-adjustable"


def _lowerer_family_witnesses_class(family: str, component_class: str) -> bool:
    """Whether a registered lowerer of this family builds the demanded class itself.

    A family that constructs the class out of its own reviewed parts is a realization, not a
    substitute: `switch-input` emits the SW_Push symbol on the tactile-button footprint for a
    reset button. The witness table is the reviewed list of those relations
    (``kicraft.design.part_identity.lowerer_witnesses_physical_class``), so a family that merely
    looks switch-like -- a generic `pin-header` standing in for a demanded JST-XH connector --
    stays refused.
    """
    from kicraft.design.lowering import registered_lowerers
    from kicraft.design.part_identity import lowerer_witnesses_physical_class

    key = str(family or "").strip().casefold()
    if not key:
        return False
    return any(
        lowerer_witnesses_physical_class(registered.lowerer_id, component_class)
        for registered in registered_lowerers()
        if key in {name.casefold() for name in registered.families}
    )


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
    from kicraft.design.part_identity import reviewed_parts_for_feature

    diagnostics: list[StageDiagnostic] = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        family = str(requirement.get("family") or "")
        exact = str(requirement.get("exact_part") or "").strip().casefold()
        for obligation in requirement.get("obligations") or []:
            if not isinstance(obligation, dict) or obligation.get("kind") != "physical":
                continue
            component_class = str(obligation.get("component_class") or "")
            if not component_class:
                continue
            carriers = reviewed_parts_for_feature(component_class)
            if not carriers:
                continue
            if family in {part.family for part in carriers} or exact in {
                part.identity for part in carriers
            }:
                continue
            if _lowerer_family_witnesses_class(family, component_class):
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
    """Point a regulator whose family has no instance at its output voltage at the reviewed one.

    A requirement can name an adjustable family and a voltage that family is not registered for
    (live walkthrough 2026-09-25: `tps54331-adjustable` at 10.0 V, whose only instance is 3.3 V).
    The resolver then falls back to the instance's default, so the rail's feedback divider comes
    out sized for the wrong voltage and §9.32 refuses the commit. The reviewed family at that
    voltage is the same design intent, built from parts whose own data sizes it correctly.
    """
    retargeted: list[str] = []
    for requirement in candidate.get("requirements") or []:
        if not isinstance(requirement, dict) or str(requirement.get("role") or "") != "regulator":
            continue
        parameters = requirement.get("parameters") or {}
        target = parameters.get("output_voltage")
        if not isinstance(target, (int, float)):
            # The derived shape states the same fact as the rail this part's output port feeds.
            for port, net in (requirement.get("ports") or {}).items():
                port_key = str(port).casefold()
                if port_key.startswith(("output", "vout", "sw")) or port_key == "out":
                    target = (candidate.get("rail_voltages") or {}).get(str(net))
                    break
        if not isinstance(target, (int, float)):
            continue
        family = str(requirement.get("family") or "")
        if (family, float(target)) in _REVIEWED_RAIL_FAMILIES:
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
        reviewed = next(
            (row for row in _REVIEWED_RAIL_FAMILIES if abs(row[1] - float(target)) <= 0.05),
            None,
        )
        clamped = reviewed is None
        if reviewed is None:
            # No orderable converter produces this voltage: the rail moves to the nearest one the
            # library can build, and its consumers keep working (a motor driver's range usually
            # spans it), rather than the design naming a rail nothing can supply.
            reviewed = min(
                _REVIEWED_RAIL_FAMILIES, key=lambda row: abs(row[1] - float(target))
            )
        requirement["family"] = reviewed[0]
        # The exact part the requirement named belonged to the family it was on. Left in place it
        # reads as an explicitly named part on a family that does not carry it, and the resolver
        # refuses exactly that ("protected variant 'MP1584EN' has no verified recipe", live
        # walkthrough 2026-09-25) -- which blocks the whole resolution, not just this requirement.
        if str(requirement.get("exact_part") or "").strip():
            requirement["exact_part"] = None
        if clamped:
            requirement["parameters"] = {**parameters, "output_voltage": reviewed[1]}
            for port, net in (requirement.get("ports") or {}).items():
                port_key = str(port).casefold()
                if port_key.startswith(("output", "vout", "sw")) or port_key == "out":
                    rail_voltages = dict(candidate.get("rail_voltages") or {})
                    rail_voltages[str(net)] = reviewed[1]
                    candidate["rail_voltages"] = rail_voltages
        retargeted.append(str(requirement.get("id") or ""))
    return retargeted


def complete_over_rated_supply(candidate: dict) -> dict:
    """Give a load part the rail it can actually run on, then resolve what the pass wrote.

    The resolver pass runs on every candidate, not only on the ones this pass edited: a
    requirement can inherit a part its family no longer carries (live walkthrough, 2026-09-25:
    `exact_part: MP1584EN` beside `family: ap63205-5v`), which the resolver refuses as a
    protected variant with no recipe and which blocks the whole resolution.
    """
    return _resolve_added_requirements(_complete_load_supply_rails(candidate))


def _resolve_added_requirements(completed: dict) -> dict:
    """Run the recipe resolver, clearing a part the resolver says its family cannot carry."""
    try:
        from kicraft.design import models as _models
        from kicraft.design.recipes import resolve_architecture_recipes
        from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

        blocking = resolve_architecture_recipes(
            _models.Architecture.model_validate(completed), None
        ).blocking
        blocked_ids = {
            str(row.requirement_id)
            for row in blocking
            if getattr(row, "code", "") == "unsupported_protected_variant"
        }
        if blocked_ids:
            for row in completed.get("requirements") or []:
                if (
                    isinstance(row, dict)
                    and str(row.get("id") or "") in blocked_ids
                    and str(row.get("exact_part") or "").strip()
                ):
                    row["exact_part"] = None
        return apply_architecture_recipe_resolution(completed).model_dump(exclude_none=True)
    except Exception:  # a resolution problem is the stage's to report, not this pass's
        return completed


def _complete_load_supply_rails(candidate: dict) -> dict:
    """Give an over-rated load part its own regulated rail, before it is diagnosed.

    Live walkthrough (2026-09-25, seed 37): the brief states an 18 V DC input and a DRV8833
    (``vm`` rated 2.7-10.8 V) driving the actuators. The reviewed library has no dual-H-bridge
    rated for 18 V, so no part satisfies the stated input, and every correction round restated
    the impossibility: three drafts, no candidate. Catching the fault is not enough -- the rail
    the part can run on has to exist.

    So the pipeline adds it: a reviewed adjustable buck on the offending rail, its output at a
    voltage inside the part's rated range, the part's supply port rebound to that rail, and an
    assumption ending "(defaulted)" that names the choice. The higher input voltage stays the
    board input only. Nothing is invented about the *load*: its current and voltage stay as the
    brief and the writer stated them.
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
                + " moved to the reviewed converter family registered at that output voltage, "
                "because the named family has no instance there (defaulted)"
            ),
        ]
    rails = _architecture_rail_voltages(candidate)
    if not rails:
        return candidate
    requirements = candidate.get("requirements") or []
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            continue
        role = str(requirement.get("role") or "")
        if role not in {"driver", "analog_block"}:
            continue
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
        logic_rails = _logic_rails(candidate)
        supplied = [
            rail for rail in _requirement_supply_rails(requirement) if rails.get(rail) is not None
        ]
        over = [rail for rail in supplied if rails[rail] > allowed]
        # Riding the logic rail is the other half of the same fault: a load part on the rail that
        # powers the MCU takes its current out of the logic budget, and that is what the writer
        # reaches for once an over-rated rail is refused (seen live, 2026-09-25).
        on_logic = [rail for rail in supplied if rail in logic_rails]
        if not over and not on_logic:
            continue
        if over:
            source_rail = max(over, key=lambda rail: rails[rail])
        else:
            sources = [
                rail
                for rail, volts in rails.items()
                if rail not in logic_rails and rail not in on_logic and abs(float(volts)) > 0.0
            ]
            if not sources:
                continue
            source_rail = max(sources, key=lambda rail: rails[rail])
        if any(
            any(
                high is not None and high >= rails[source_rail]
                for _label, _low, high in reviewed_supply_voltage_limits(alternative)
            )
            for alternative in reviewed_parts_for_feature(str(requirement.get("family") or ""))
        ):
            continue
        label, low, _high = min(limits, key=lambda row: row[2])
        # A rail the library can build: the highest reviewed family that fits the part's range.
        reviewed = next(
            (
                (family, volts)
                for family, volts in _REVIEWED_RAIL_FAMILIES
                if float(low or 0.0) <= volts <= allowed
            ),
            None,
        )
        if reviewed is None:
            continue
        converter_family, voltage = reviewed
        new_rail = f"{str(requirement.get('id') or 'load').upper()}_RAIL"
        converter_id = f"{str(requirement.get('id') or 'load')}_regulator"
        sheet_name = f"{str(requirement.get('id') or 'load').upper()} REGULATOR"
        completed = dict(candidate)
        completed["requirements"] = [dict(row) if isinstance(row, dict) else row for row in requirements]
        offending = {*over, *on_logic}
        completed["requirements"][index]["ports"] = {
            key: (new_rail if str(value) in offending else value)
            for key, value in (requirement.get("ports") or {}).items()
        }
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
        # The writer's own statements are now false: they say this part runs on the source
        # rail, and (for the load) that the actuators are fed at the input voltage. A candidate
        # whose prose contradicts its bindings is not reviewable, so the claims are corrected
        # here before the disclosure below states the whole picture in one row.
        tokens = _load_part_tokens(requirement, exact)
        # The rail the part was wrongly on is the one its claims have to stop naming; for an
        # over-rated part that is the source rail, for a part riding the logic rail it is that
        # logic rail, and the converter's input is the source rail either way.
        wrong_rail = source_rail if over else (on_logic[0] if on_logic else source_rail)
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
        completed["assumptions"] = [
            *(completed.get("assumptions") or []),
            (
                f"{exact} runs from a regulated {voltage:g} V rail ({new_rail}) because its "
                f"reviewed {label} limit is {allowed:g} V and no reviewed part for "
                f"{requirement.get('family')!r} is rated for {source_rail} "
                f"({rails[source_rail]:g} V); {converter_id} steps {source_rail} down, and the "
                f"{source_rail} input stays the board input only (defaulted)"
            ),
        ]
        # The added requirement is a recipe requirement like any other, and the recipe's owned
        # parts -- an adjustable regulator's feedback divider, its compensation network -- exist
        # only after the resolver's pass. Without it the parts unit drafts them from scratch,
        # which is how the 10 V motor rail ended up carrying the 3.3 V divider and failing §9.32
        # (live walkthrough, 2026-09-25). This is the same pass the slot already went through, so
        # every existing requirement keeps its own resolution.
        return completed
    # A declared load rail that nothing generates is the same fault one step later: the writer
    # stated the voltage and bound the part, and left the converter out (live draft,
    # 2026-09-25: MOTOR_VIN at 10.8 V, `from: null`, nothing producing it, zero diagnostics).
    generators = _rail_generators(candidate)
    input_rail = _board_input_rail(rails)
    for rail_name, voltage in (candidate.get("rail_voltages") or {}).items():
        rail_name = str(rail_name)
        try:
            rail_voltage = float(voltage)
        except (TypeError, ValueError):
            continue
        if rail_name in generators or _rail_needs_no_generator(rail_name, rail_voltage, rails):
            continue
        consumers = [
            requirement
            for requirement in requirements
            if isinstance(requirement, dict)
            and rail_name in _requirement_supply_rails(requirement)
        ]
        if not consumers or input_rail is None or input_rail == rail_name:
            continue
        # The rail is built by a family the library carries: the nearest reviewed output voltage,
        # which is also the rail's voltage (a declared 10.8 V has no reviewed instance; 10.0 V
        # does, and it is inside any part that accepts 10.8 V).
        converter_family, rail_voltage = min(
            _REVIEWED_RAIL_FAMILIES, key=lambda row: abs(row[1] - rail_voltage)
        )
        converter_id = f"{rail_name.strip('+').replace(' ', '_').lower()}_regulator"
        sheet_name = f"{rail_name.strip('+').upper()} REGULATOR"
        completed = dict(candidate)
        rail_voltages = dict(completed.get("rail_voltages") or {})
        rail_voltages[rail_name] = rail_voltage
        completed["rail_voltages"] = rail_voltages
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


def _rail_needs_no_generator(name: str, voltage: float, rails: dict) -> bool:
    """Ground, a host-supplied socket rail, or the board's own input rail."""
    key = str(name).strip().casefold()
    if key in {"gnd", "ground", "0v"} or abs(float(voltage)) <= 0.05:
        return True
    if key in {"vbus", "+5v", "5v"} and abs(float(voltage) - 5.0) <= 0.5:
        return True
    others = [float(v) for rail, v in rails.items() if str(rail).strip().casefold() != key]
    return bool(others) and float(voltage) >= max(others)


def _board_input_rail(rails: dict) -> str | None:
    candidates = [
        (str(rail), float(volts))
        for rail, volts in rails.items()
        if abs(float(volts)) > 0.05
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[1])[0]


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


def _architecture(upstream: dict, candidate: dict) -> list[StageDiagnostic]:
    diagnostics = architecture_power_requirement_diagnostics(upstream, candidate)
    diagnostics.extend(_architecture_supply_over_rating(candidate))
    diagnostics.extend(_architecture_drive_on_logic_rail(upstream, candidate))
    diagnostics.extend(_architecture_obligation_family_mismatch(candidate))
    diagnostics.extend(_architecture_derivation_diagnostics(upstream, candidate))
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
    has_5v_load_budget = external_load_budget_stated(current_context)
    if board_powers_external and not has_5v_load_budget:
        diagnostics.append(
            _diag(
                EXTERNAL_LOAD_CURRENT_CODE,
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
            and producer["rated_output_current_a"] >= 1.0
            and producer["sheet"] in declared_sheets
            and not re.search(r"\b(?:mcu|esp32)\b", producer["sheet"], re.I)
        ]
        if not sized:
            diagnostics.append(
                _diag(
                    "architecture_mcu_regulator_incomplete",
                    "repair_required",
                    "ESP32-S3 needs its 3.3V rail generated by a regulator the recipe rates for >=1A "
                    "on a sheet of its own.",
                    [
                        "esp32-s3",
                        *(
                            f"{producer['rail']}: {producer['sheet']} "
                            f"{producer['requirement_id']} rated "
                            f"{producer['rated_output_current_a']}"
                            for producer in producers
                        ),
                        ">=1a",
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
        # Prose only counts for a role the design implements nowhere: then the sheet promises
        # an active part nothing builds ("Connector and on-board amplifier" with no U anywhere).
        prose_terms = {
            match.lower() for match in ic_role.findall(str(sheet.get("function") or ""))
        } - own_terms
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
        ("wiring_gate_9_29", check_mcu_programming_access(bom)),
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
        findings = _architecture(upstream_state, candidate)
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
