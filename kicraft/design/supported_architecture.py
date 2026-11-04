"""Conservative native construction for reviewed ESP32 control products.

The source intent, not a previous architecture or evaluation result, is the authority.
An unrecognised fact returns control to the normal authored stage.
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from kicraft.design.architecture_intent import ArchitectureIntent
from kicraft.design.models import FunctionalSpec, IntentSlot, is_board_level_quantitative_obligation


_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4}


def _numbers(text: str, noun: str) -> set[int]:
    tokens = re.findall(r"\b(one|two|three|four|[1-9]\d*)\s+(?:user\s+)?" + noun, text, re.I)
    return {_NUMBERS[token.lower()] if token.lower() in _NUMBERS else int(token) for token in tokens}


def _plan(intent: dict) -> dict | None:
    if not isinstance(intent, dict) or set(intent) - set(IntentSlot.model_fields):
        return None
    try:
        source = IntentSlot.model_validate(intent)
    except (ValidationError, TypeError, ValueError):
        return None
    goal = re.sub(r"^\s*design\b", "build", source.goal.lower())
    text = " ".join([goal, *source.constraints]).lower()
    family_text = " ".join([text, *source.named_parts]).lower()
    c3 = bool(re.search(r"\besp32-c3\b", family_text))
    s3 = bool(re.search(r"\besp32-s3\b", family_text))
    if c3 == s3:
        return None
    if c3 and not re.search(r"\brelay", text):
        return None
    if s3 and not re.search(r"\btemperatur", text):
        return None
    # The native implementation has no native USB, RF antenna, analogue sensor,
    # specified memory, load qualification, or arbitrary peripheral contract.
    if re.search(r"(?<![\w.])(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:[munµμ]?a|amps?|amperes?)\b", text):
        return None
    if re.search(r"\b(?:usb|wi[ -]?fi|bluetooth|ble|antenna|wireless|radio|ethernet|analog(?:ue)?|adc|humidity|display|screen|mains|isolat\w*|surge|hot[ -]?plug|battery|charging|current|ampere|watts?|waterproof|ip\d+|flash|psram|memory|thermocouple|thermistor|humidity|canbus|rs[ -]?485|spi|uart|sensor output|sensor interface)\b", text):
        return None
    if source.form_factor and (
        source.form_factor.standard or source.form_factor.shape != "rect"
        or source.form_factor.size_mm is not None
        or source.form_factor.corner_radius_mm is not None
        or source.form_factor.chamfer_mm is not None
        or source.form_factor.note
    ):
        return None
    exact = "esp32-c3-mini-1-h4x" if c3 else "esp32-s3-mini-1u-n4r2"
    generic = "esp32-c3" if c3 else "esp32-s3"
    if any(part.strip().lower() not in {generic, exact} for part in source.named_parts):
        return None
    if c3:
        text = re.sub(r"\bindependent\s+(?=relays?\b)", "", text)
    else:
        text = re.sub(r"\bindependent\s+(?=(?:user\s+)?(?:push\s*)?buttons?\b)", "", text)
        text = re.sub(r"\buser\s+(?=(?:push\s*)?buttons?\b)", "", text)
    text = re.sub(r"\bdimensions\s+kept\s+under\b", "dimensions under", text)
    text = re.sub(r"\bmust\s+be\s+under\b", "under", text)
    text = re.sub(
        r"\bkept\s+under(?=\s+\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?\s*mm\b)",
        "dimensions under", text,
    )
    text = re.sub(r"\bboard\s+size\s+(?=under\b)", "board dimensions ", text)
    text = re.sub(r"\bpowered\s+from\b", "power input", text)
    text = re.sub(r"\boperated\s+with\b", "with", text)
    text = re.sub(r"\b(?:include|add)\s+(?=(?:a\s+)?(?:power\s+led|reset\s+button)\b)", "use ", text)
    # A count of generic outputs is a separate requirement, not proof that it
    # refers to the counted relay channels.
    if _numbers(text, r"outputs?\b"):
        return None
    # A constrained vocabulary is intentional: unfamiliar requirements are not
    # evidence that the reviewed circuit satisfies them.
    allowed_words = set("build a an the and with using use based on board module relay relays output outputs screw terminal terminals input supply power led reset button buttons push pushbutton pushbuttons dc v volt volts voltage temperature acquisition sensor three two one four dimensions under below less than by mm x layer layers stack up pcb rectangular".split())
    vocabulary_text = text.replace("×", " x ").replace(exact, "module").replace(generic, "module")
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", vocabulary_text)
    if any(word not in allowed_words and not re.fullmatch(r"\d+(?:\.\d+)?", word) for word in words):
        return None
    if c3 and (re.search(r"\btemperatur", text) or re.search(r"\b(?:user|push)\s*buttons?\b", text)):
        return None
    if c3 and re.search(r"\bbuttons?\b", text) and not re.search(r"\breset\s+button\b", text):
        return None
    if s3 and re.search(r"\b(?:relay|relays|led|reset)\b", text):
        return None
    if not source.obligations:
        return None
    owners: dict[str, list[str]] = {name: [] for name in ("core", "input", "relay", "led", "button", "board")}
    voltage = None
    count = None
    for row in source.obligations:
        item = row.model_dump()
        kind = item["kind"]
        identifier = item["original_obligation_id"]
        owner = None
        if kind == "physical":
            cls = item["component_class"]
            if cls in {"microcontroller", "microcontroller-module", generic + "-module"}:
                owner = "core"
            elif cls in {"screw-terminal", "screw-terminal-connector"}:
                owner = "input"
                if c3:
                    # Both distinct terminal facts are witnessed by the input
                    # connector and the relay contact connectors.  Their IDs
                    # carry no machine-readable location semantics.
                    owners["relay"].append(identifier)
            elif cls == "relay" and c3:
                owner = "relay"
            elif cls in {"led", "light-emitting-diode"} and c3:
                owner = "led"
            elif cls in {"pushbutton", "push-button", "momentary-button"}:
                owner = "core" if c3 and re.search(r"\breset\s+button\b", text) else "button" if s3 else None
            elif cls == "tactile-switch" and c3 and re.search(r"\breset\s+button\b", text):
                owner = "core"
            elif cls in {"temperature-sensor", "temperature-sensor-ic"} and s3:
                owner = "sensor"
        elif kind == "quantity":
            subject = item["subject"]
            if (c3 and subject in {"relay", "relays"}) or (s3 and subject in {"pushbutton", "push-button", "momentary-button"}):
                if count is not None and count != item["minimum"]:
                    return None
                count = item["minimum"]
                owner = "relay" if c3 else "button"
            elif c3 and subject in {"screw-terminal", "screw-terminal-connector"}:
                if item["minimum"] > 4:
                    return None
                owner = "relay"
        elif kind == "quantitative":
            quantity = item["quantity"].lower()
            unit = item["unit"].lower()
            if quantity in {"supply voltage", "input voltage", "dc input voltage"} and item["relation"] == "equal" and unit in {"v", "v dc", "vdc"}:
                if voltage is not None and voltage != item["value"]:
                    return None
                voltage = item["value"]
                owner = "input"
            elif (quantity in {"board width", "board height", "board dimensions"}
                  and is_board_level_quantitative_obligation(item)
                  and item["relation"] == "maximum" and unit == "mm" and item["value"] >= 100):
                owner = "board"
            elif quantity in {"pcb stack-up layers", "stack-up layers", "layers", "copper layers", "pcb copper layers"} and item["relation"] == "equal" and unit in {"layer", "layers"} and item["value"] in {2, 4}:
                owner = "board"
        if owner is None:
            return None
        owners.setdefault(owner, []).append(identifier)
    if voltage is None or not 6 <= voltage < 30 or (c3 and voltage != 24):
        return None
    mentioned = _numbers(text, r"relays?\b") if c3 else _numbers(text, r"(?:push\s*)?buttons?\b")
    if count is None and len(mentioned) == 1:
        count = next(iter(mentioned))
    if count is None or any(value != count for value in mentioned):
        return None
    if count not in ({2, 3, 4} if c3 else {1, 2, 3, 4}):
        return None
    terminal_counts = [
        row.minimum for row in source.obligations
        if row.kind == "quantity" and row.subject in {"screw-terminal", "screw-terminal-connector"}
    ]
    if terminal_counts and (not c3 or any(value > count for value in terminal_counts)):
        return None
    terminal_mentions = _numbers(text, r"screw[ -]terminals?\b")
    if terminal_mentions and (
        not terminal_counts or any(value not in terminal_counts for value in terminal_mentions)
    ):
        return None
    # Explicit voltage text must not contradict the typed rail.
    for token in re.findall(r"\b(\d+(?:\.\d+)?)\s*v(?:olts?)?\b", text):
        if float(token) != voltage:
            return None
    board_rows = [row.model_dump() for row in source.obligations if row.kind == "quantitative"]
    layer_rows = {
        row["value"] for row in board_rows
        if row["quantity"].lower() in {"pcb stack-up layers", "stack-up layers", "layers", "copper layers", "pcb copper layers"}
    }
    if len(layer_rows) > 1:
        return None
    for token in re.findall(r"\b(one|two|three|four|\d+)[ -]+layers?\b", text):
        layers = _NUMBERS[token] if token in _NUMBERS else int(token)
        if not any(row["quantity"].lower() in {"pcb stack-up layers", "stack-up layers", "layers", "copper layers", "pcb copper layers"} and row["relation"] == "equal" and row["value"] == layers for row in board_rows):
            return None
    if re.search(r"\b(?:dimensions|mm)\b", text):
        dimensions: dict[str, float] = {}
        for row in board_rows:
            quantity = row["quantity"].lower()
            if quantity not in {"board width", "board height", "board dimensions"}:
                continue
            if quantity in dimensions and dimensions[quantity] != row["value"]:
                return None
            dimensions[quantity] = row["value"]
        stated_dimensions = re.findall(
            r"\bdimensions\s+under\s+(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*mm\b",
            text,
        )
        if not stated_dimensions:
            return None
        for width, height in stated_dimensions:
            for quantity, value in (("board width", width), ("board height", height)):
                limits = [dimensions[key] for key in (quantity, "board dimensions") if key in dimensions]
                if not limits or any(limit != float(value) for limit in limits):
                    return None
    return {"c3": c3, "count": count, "voltage": voltage, "owners": owners, "obligations": [row.model_dump() for row in source.obligations]}


def _spec(plan: dict) -> dict:
    owners = plan["owners"]
    blocks = [
        {"name": "POWER INPUT", "category": "power", "purpose": "DC input and 3.3 V regulation", "obligation_ids": owners["input"]},
        {"name": "CONTROL", "category": "process", "purpose": "ESP32 control and UART recovery", "obligation_ids": owners["core"]},
    ]
    connections = [{"from_block": "POWER INPUT", "to_block": "CONTROL", "signal_type": "power", "description": "3.3 V logic supply"}]
    if plan["c3"]:
        blocks += [
            {"name": "RELAY OUTPUT", "category": "drive", "purpose": "Independent switched relay contacts", "obligation_ids": owners["relay"]},
            {"name": "POWER INDICATOR", "category": "interface", "purpose": "Indicate logic power", "obligation_ids": owners["led"]},
        ]
        connections += [
            {"from_block": "CONTROL", "to_block": "RELAY OUTPUT", "signal_type": "digital", "description": "Independent relay controls"},
            {"from_block": "POWER INPUT", "to_block": "RELAY OUTPUT", "signal_type": "power", "description": "24 V coil supply"},
            {"from_block": "POWER INPUT", "to_block": "POWER INDICATOR", "signal_type": "power", "description": "3.3 V indicator supply"},
        ]
    else:
        blocks += [
            {"name": "TEMPERATURE", "category": "sense", "purpose": "I2C temperature acquisition", "obligation_ids": owners.get("sensor", [])},
            {"name": "USER BUTTONS", "category": "interface", "purpose": "Independent user inputs", "obligation_ids": owners["button"]},
        ]
        connections += [
            {"from_block": "TEMPERATURE", "to_block": "CONTROL", "signal_type": "bus", "description": "I2C temperature data"},
            {"from_block": "USER BUTTONS", "to_block": "CONTROL", "signal_type": "digital", "description": "Independent user button inputs"},
            {"from_block": "POWER INPUT", "to_block": "TEMPERATURE", "signal_type": "power", "description": "3.3 V sensor supply"},
            {"from_block": "POWER INPUT", "to_block": "USER BUTTONS", "signal_type": "power", "description": "3.3 V input pullups"},
        ]
    if owners["board"]:
        blocks.append({"name": "BOARD", "category": "mechanical", "purpose": "Board dimensions and stack-up", "obligation_ids": owners["board"]})
    assumptions = ["Engineering choice: reviewed 3.3 V DC/DC module and ROM UART programming access."]
    if not plan["c3"]:
        assumptions.append("Engineering choice: TMP102 I2C temperature sensor and active-low user buttons.")
    else:
        assumptions.append("Engineering choice: 24 V relay coils; contact load and installation remain subject to product review.")
    return FunctionalSpec.model_validate({"blocks": blocks, "connections": connections, "obligations": plan["obligations"], "assumptions": assumptions}).model_dump()


def supported_functional_spec(intent: dict) -> dict | None:
    plan = _plan(intent)
    return _spec(plan) if plan is not None else None


def supported_architecture_intent(intent: dict, functional_spec: dict) -> dict | None:
    plan = _plan(intent)
    if plan is None:
        return None
    try:
        if FunctionalSpec.model_validate(functional_spec).model_dump() != _spec(plan):
            return None
    except (ValidationError, TypeError, ValueError):
        return None
    owners = plan["owners"]
    requirements = [
        {"id": "input", "sheet": "MAIN", "role": "power_input", "family": "screw-terminal", "exact_part": "WJ126V-5.0-02P-14-00A", "ties": {"negative": "GND"}, "functional_blocks": ["POWER INPUT"], "obligation_ids": owners["input"]},
        {"id": "regulator", "sheet": "MAIN", "role": "regulator", "family": "ylptec-k7803m-1000r3", "supply_bindings": {"input": "VIN"}, "parameters": {"output_voltage": 3.3}, "functional_blocks": ["POWER INPUT"]},
        {"id": "core", "sheet": "MAIN", "role": "mcu_core", "family": "esp32-c3-mini-1-module" if plan["c3"] else "esp32-s3-module", "exact_part": "ESP32-C3-MINI-1-H4X" if plan["c3"] else "ESP32-S3-MINI-1U-N4R2", "supply": "+3V3", "programming": "uart", "functional_blocks": ["CONTROL"], "obligation_ids": owners["core"]},
    ]
    signals = []
    if plan["c3"]:
        requirements += [
            {"id": "relay", "sheet": "MAIN", "role": "driver", "family": "relay-output", "parameters": {"channel_count": plan["count"], "motor_voltage_v": 24.0}, "supply_bindings": {"vmotor": "VIN"}, "functional_blocks": ["RELAY OUTPUT"], "obligation_ids": owners["relay"]},
            {"id": "led", "sheet": "MAIN", "role": "user_io", "family": "led-current-resistor", "parameters": {"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 2}, "ties": {"drive": "+3V3", "gnd": "GND"}, "functional_blocks": ["POWER INDICATOR"], "obligation_ids": owners["led"]},
        ]
        for i in range(1, plan["count"] + 1):
            signals.append({"name": f"RELAY_CONTROL_{i}", "from": f"core.output_relay_{i}", "to": f"relay.control_{i}"})
            for contact in ("common", "normally_open", "normally_closed"):
                signals.append({"name": f"RELAY_{i}_{contact.upper()}", "from": f"relay.relay_{i}_{contact}", "to": []})
    else:
        requirements += [
            {"id": "sensor", "sheet": "MAIN", "role": "sensor", "family": "tmp102-temperature", "supply": "+3V3", "functional_blocks": ["TEMPERATURE"], "obligation_ids": owners.get("sensor", [])},
        ]
        for line in ("sda", "scl"):
            signals.append({"name": line.upper(), "from": f"core.{line}", "to": f"sensor.{line}"})
        for i in range(1, plan["count"] + 1):
            requirements.append({"id": f"button_{i}", "sheet": "MAIN", "role": "user_io", "family": "switch-input", "supply": "+3V3", "parameters": {"pull_policy": "external", "active_level": "low"}, "functional_blocks": ["USER BUTTONS"], "obligation_ids": owners["button"]})
            signals.append({"name": f"BUTTON_{i}", "from": f"button_{i}.signal", "to": f"core.input_button_{i}"})
    payload = {"sheets": [{"name": "MAIN", "stem": "MAIN", "role": "main", "function": "Power, control and application interfaces"}], "requirements": requirements, "signals": signals, "power": {"rails": {"VIN": {"voltage": plan["voltage"], "from": "input.positive"}, "+3V3": {"voltage": 3.3, "from": "regulator.output"}}}, "mcu_present": True, "comms_protocols": [] if plan["c3"] else ["i2c"], "obligations": plan["obligations"], "assumptions": _spec(plan)["assumptions"]}
    return ArchitectureIntent.model_validate(payload).model_dump(by_alias=True)
