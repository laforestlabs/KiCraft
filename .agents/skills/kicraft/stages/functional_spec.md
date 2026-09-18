Stage 2: Functional Spec. Draft the complete `functional_spec` slot and commit it through the workflow in the parent `SKILL.md`; this file defines the slot contract.

Given the captured `intent` (available in the `state` field of stage-prep's output), decompose the project into abstract functional blocks and their inter-block signal flow. DO NOT commit to topologies, part numbers, or component-level detail yet — that's Stage 3 and Stage 4.

Slot shape (`FunctionalSpec`):

- `blocks`: list of `FunctionalBlock`, each with:
  - `name` — uppercase functional identifier (e.g. `USB_INPUT`, `POWER_CONVERSION`).
  - `category` — one of `sense` / `process` / `drive` / `power` / `interface`.
  - `purpose` — one sentence.
  - `count` — number of identical instances (default 1). Set `count: N` on ONE block when the design asks for N copies of the SAME function (e.g. "3 axes of stepper outputs" → one `STEPPER_AXIS` block with `count: 3`, NOT three near-duplicate blocks). The architecture stage expands a `count`-N block into N grouped sheets that the layout solves once and replicates. Keep genuinely distinct functions as separate blocks.
- `connections`: list of `BlockConnection`, each with `from_block`, `to_block`, `signal_type` (`power` / `ground` / `digital` / `analog` / `clock` / `bus` / `rf` / `other`), and a short description.
- `assumptions`: list of defaults applied, each ending with `(defaulted)`.

Constraints (enforced by Pydantic):

- Block names must be unique.
- Every `connection.from_block` and `to_block` must reference a block in this list.
- Each (`from_block`, `to_block`, `signal_type`) combination must be unique;
  changing the description does not create another connection.

Connection discipline:

- Emit only actual causal power or signal flows required for the stated functions,
  once per directed block pair and signal type. Combine descriptions of the same flow.
- Several independent signals of the same type between the same blocks share
  ONE edge: list them together in `description`. For example, two digital
  channels from `INPUT` to `PROCESS` need one `digital` connection describing
  both channels, not two duplicate connections. This does not remove either
  channel; architecture defines their individual nets.
- Do not enumerate every block pair or every `signal_type` enum value. Those are
  allowed labels, not a checklist of connections the board must have.
- Fan-out to different consumers, a real reverse-direction flow, and distinct
  signal types between the same blocks are valid when the functions require them.

Block-boundary heuristics:

- A block is a coherent user-visible function, not a rail, ground, mechanical
  hole, crystal, passive support network, component, or chosen topology.
- Do not introduce LDO/buck/boost/ESD/controller choices unless the brief or
  intent explicitly requires them. Every default introduced here must appear in
  `assumptions` ending `(defaulted)`.
- Aim for 3-8 blocks total for a typical hobbyist project. More is fine for
  complex designs.
- A board feature that names no component function is NOT a block: do not emit a
  block for a prototyping pad field (a prototyping shield, perfboard or pad
  field), and draw no connection to or from it. A block is a user-visible
  function and the pad field carries no signal of its own; its sheet and its
  bare 2.54 mm pad grid are derived downstream from the intent's `fabrication`
  obligation.

Open-question discipline matches Stage 1: `blocking`, `material`, or silent default in `assumptions`.

External-load power boundary:

- If the board drives a display, LED string, motor, heater, or other potentially
  high-current external load and the accepted intent does not explicitly say
  whether the board supplies that load's power, return one `blocking: true`
  question before drafting. Do not silently add a board-to-load power connection.
