# replay_workspace fixture

A committed, already-synthesized KiCraft workspace (`USB_PD_TRIGGER`, a USB-C PD
trigger board: 4 leaf sheets — USB_C_INPUT, PD_CONTROLLER, VOLTAGE_SELECT,
OUTPUT) used by `tests/test_replay_command.py` to exercise `kicraft replay` end
to end.

It carries exactly the artifacts the placement engine consumes:
- `USB_PD_TRIGGER.kicad_sch` (root) + the four leaf `.kicad_sch` files,
- `USB_PD_TRIGGER.kicad_pcb` (the seed/board),
- `USB_PD_TRIGGER.kicad_pro`,
- `USB_PD_TRIGGER_autoplacer.json` (UI seed file; not read by the placer).

There is intentionally **no** `.kicraft/state.json` next to it, so the test
drives the `replay --project DIR` discovery path (no state needed).

The deterministic-placement test compares the per-leaf
`leaf_pre_freerouting.kicad_pcb` boards across two runs — that is the placement
output, which `replay` guarantees is reproducible (pinned seed + hash seed).
The composed parent board is NOT asserted byte-stable: it consumes the *routed*
leaf boards and so inherits FreeRouting's best-effort nondeterminism.
