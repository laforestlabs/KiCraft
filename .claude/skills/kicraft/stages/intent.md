Stage 1: Intent. You are running inside the KiCraft stage sub-agent. Your job is to draft the `intent` slot of the conversation state and commit it. Follow SKILL.md's "Workflow (follow exactly, in order)" section to do that — this file specifies what the slot must look like.

Slot shape (`IntentSlot`):

- `goal`: one or two sentences describing what the user is building.
- `constraints`: list of hard requirements they have stated explicitly (voltage, size, BOM cost, target fab, regulatory, etc.). Don't invent constraints.
- `named_parts`: list of any specific MPNs, ICs, connectors, or batteries the user has named. Empty list if none.
- `inferred_expertise`: one of `"beginner"` / `"intermediate"` / `"expert"`, inferred from vocabulary and constraint specificity.
- `assumptions`: defaults you applied because the user didn't say. Each entry MUST end with `(defaulted)` so the user can spot and override (e.g. `"target fab: JLCPCB (defaulted)"`).

`project_stem` rule (top-level state field, NOT inside the slot — pass via `--project-stem`):

Pick the 2-3 most significant words from the goal, uppercase-and-underscore them, cap at 32 chars. Examples: goal "USB-powered Li-ion charger" → `"USB_LIION_CHARGER"`; goal "ESP32 weather station" → `"ESP32_WEATHER_STATION"`.

Open-question discipline:

- `blocking: true` — reserve for things that materially change the project (battery vs USB, single-board vs multi-board).
- `material: true` (default; not blocking) — worth surfacing at the next stage boundary. Affects topology or part choice.
- Cosmetic clarifications — DON'T emit a question; record the default in `assumptions` instead.

Keep it tight. This stage captures what the user said, not what they should build.
