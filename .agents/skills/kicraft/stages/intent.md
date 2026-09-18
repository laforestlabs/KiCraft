Stage 1: Intent. Draft the complete `intent` slot and commit it through the workflow in the parent `SKILL.md`; this file defines the slot contract.

Slot shape (`IntentSlot`):

- `goal`: one or two sentences describing what the user is building.
- `constraints`: list of hard requirements they have stated explicitly (voltage, size, BOM cost, target fab, regulatory, etc.). Don't invent constraints.
- `named_parts`: list of any specific MPNs, ICs, connectors, or batteries the user has named. Empty list if none.
- `inferred_expertise`: one of `"beginner"` / `"intermediate"` / `"expert"`, inferred from vocabulary and constraint specificity.
- `assumptions`: defaults you applied because the user didn't say. Each entry MUST end with `(defaulted)` so the user can spot and override (e.g. `"target fab: JLCPCB (defaulted)"`).
- `form_factor` (optional): set ONLY when the user asks for a non-rectangular board outline. `{"shape": "...", "size_mm": <headline dimension if stated>}`. Parametric shapes: `"circle"`, `"rounded_rect"` (+`"corner_radius_mm"`), `"chamfered_rect"` (+`"chamfer_mm"`). Named shapes are also allowed: `"hexagon"`, `"octagon"`, `"triangle"`, `"pentagon"`, `"star"`, `"heart"`, `"gear"`, `"snowman"`. Omit the field (or use `"rect"`) for a conventional rectangular board — don't infer a shape the user didn't ask for. A deterministic extractor also fills this from the brief, so it's a safety net, but set it when the request is clear (especially for paraphrased shapes the keyword matcher might miss). Also keep the user's shape wording in `constraints`.

Classify every explicit package, quantity, voltage/frequency/unit, interface,
inclusion/exclusion, and mechanical requirement in `constraints`; classify every
named IC/family/connector in `named_parts`. Do not merely copy the brief into
`goal`. Bad: RP2040/QFN-56/USB-C/12 MHz/castellated brief with both lists empty.
Valid: `named_parts: ["RP2040"]` and constraints containing the package,
interface, clock, and castellation requirements.

`obligations`: the typed form of those explicit requirements — one row per physical class,
quantity, adjustability, conversion behavior, numerical limit, board fabrication feature, or
absent class. A board *fabrication* feature (a printed copper area acting as a heatsink, a
thermal-via field, an edge treatment) or the *absence* of a class ("no microcontroller") is
never a `physical` obligation: use `kind: "fabrication"` with its `feature` and any stated
`minimum`/`unit`, or `kind: "negative"` with the `absent_class`.

A **prototyping area** (a pad field the user solders into — the defining feature of a
prototyping shield, board, perfboard or pad field) is one of those board features: record
`kind: "fabrication"` with `feature: "prototyping-area"`, and keep the brief's wording in
`constraints`. State no `minimum`/`unit` unless the brief states a size. Nothing else can
carry it — it names no part, owns no pin and draws no net — so a brief that asks for one
and a slot that omits it is an incomplete capture.

A `physical` obligation's `component_class` names a *part class*, not a fact about the
board or its wiring. Write the class, not the user's phrase or a qualifier: the reviewed
vocabulary spells the Arduino shield interface `stacking-header` (not
`stacking-through-hole-header`) and a 3.3 V regulator `voltage-regulator` (not
`smt-voltage-regulator`). An interface or bus (`i2c-interface`), a board format
(`arduino-uno-format-board`) and a printed-copper feature (`thermal-via-copper-pour`) are
not part classes: a board format goes in `constraints`, a printed-board feature uses
`kind: "fabrication"`, and an absent class uses `kind: "negative"`.

A part class the reviewed library does not cover yet is legitimate — name it plainly
(`gps-module`, `air-quality-sensor`) and never substitute an unrelated reviewed class just
to look familiar. If the shell tools reject the class at intent time, follow the evidence
it returns.

`project_stem` rule (top-level state field, NOT inside the slot — pass via `--project-stem`):

Pick the 2-3 most significant words from the goal, uppercase-and-underscore them, cap at 32 chars. Examples: goal "USB-powered Li-ion charger" → `"USB_LIION_CHARGER"`; goal "ESP32 weather station" → `"ESP32_WEATHER_STATION"`.

Open-question discipline:

- `blocking: true` — reserve for things that materially change the project (battery vs USB, single-board vs multi-board).
- `material: true` (default; not blocking) — worth surfacing at the next stage boundary. Affects topology or part choice.
- Cosmetic clarifications — DON'T emit a question; record the default in `assumptions` instead.

Keep it tight. This stage captures what the user said, not what they should build.
