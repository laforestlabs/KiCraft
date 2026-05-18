Stage 1: Intent. Read the conversation so far and write the `intent` slot of `.kicraft/state.json`.

Slot shape (`IntentSlot`):

- `goal`: one or two sentences describing what the user is building.
- `constraints`: list of hard requirements they have stated explicitly (voltage, size, BOM cost, target fab, regulatory, etc.). Don't invent constraints.
- `named_parts`: list of any specific MPNs, ICs, connectors, or batteries the user has named. Empty list if none.
- `inferred_expertise`: one of `"beginner"` / `"intermediate"` / `"expert"`, inferred from vocabulary and constraint specificity.
- `assumptions`: defaults you applied because the user didn't say. Each entry MUST end with `(defaulted)` so the user can spot and override (e.g. `"target fab: JLCPCB (defaulted)"`).

Open questions discipline:

- `blocking: true` — the stage cannot produce useful output without an answer. Reserve for things that materially change the project (battery vs USB, single-board vs multi-board).
- `material: true` (default; not blocking) — worth surfacing at the next stage boundary. Affects topology or part choice (e.g. output current).
- Cosmetic clarifications — DON'T emit a question; record the default in `assumptions` instead.

Set `project_stem` on the top-level state at the same time: pick the 2-3 most significant words from the goal, uppercase-and-underscore them, cap at 32 chars (e.g. goal "USB-powered Li-ion charger" → `"USB_LIION_CHARGER"`).

Keep it tight. This stage captures what the user said, not what they should build.
