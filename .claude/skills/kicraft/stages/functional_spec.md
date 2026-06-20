Stage 2: Functional Spec. You are running inside the KiCraft stage sub-agent. Your job is to draft the `functional_spec` slot of the conversation state and commit it. Follow SKILL.md's "Workflow (follow exactly, in order)" section — this file specifies what the slot must look like.

Given the captured `intent` (available in the `state` field of stage-prep's output), decompose the project into abstract functional blocks and their inter-block signal flow. DO NOT commit to topologies, part numbers, or component-level detail yet — that's Stage 3 and Stage 4.

Slot shape (`FunctionalSpec`):

- `blocks`: list of `FunctionalBlock`, each with:
  - `name` — uppercase identifier (e.g. `USB_INPUT`, `CHARGER`, `LDO_3V3`).
  - `category` — one of `sense` / `process` / `drive` / `power` / `interface`.
  - `purpose` — one sentence.
  - `count` — number of identical instances (default 1). Set `count: N` on ONE block when the design asks for N copies of the SAME function (e.g. "3 axes of stepper outputs" → one `STEPPER_AXIS` block with `count: 3`, NOT three near-duplicate blocks). The architecture stage expands a `count`-N block into N grouped sheets that the layout solves once and replicates. Keep genuinely distinct functions as separate blocks.
- `connections`: list of `BlockConnection`, each with `from_block`, `to_block`, `signal_type` (`power` / `ground` / `digital` / `analog` / `clock` / `bus` / `rf` / `other`), and a short description.
- `assumptions`: list of defaults applied, each ending with `(defaulted)`.

Constraints (enforced by Pydantic):

- Block names must be unique.
- Every `connection.from_block` and `to_block` must reference a block in this list.

Block-boundary heuristics:

- Each block should map cleanly to one schematic sheet later.
- A block is "a coherent function someone would describe as a unit" (e.g. "the charger", "the boost converter"), not "an op-amp" or "three caps".
- Power input, ground, and rail outputs each get their own block if they connect to multiple downstream blocks.
- Aim for 3-8 blocks total for a typical hobbyist project. More is fine for complex designs.

Open-question discipline matches Stage 1: `blocking`, `material`, or silent default in `assumptions`.
