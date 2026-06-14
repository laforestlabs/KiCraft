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

## `.experiments/subcircuits/` — frozen leaf artifacts (parent corpus)

`scripts/replay_corpus.py --mode parent` validates parent-frame placement
(Part 2 Levers 2.1/2.3). A full replay's *parent* is NOT reproducible — leaf
stamping (pour/vias → `size_reduction` → block bbox) is nondeterministic — so
the parent gate freezes the leaf inputs instead: the committed
`.experiments/subcircuits/<leaf>/{metadata,debug,solved_layout}.json` +
`leaf_routed.kicad_pcb` are a deterministic compose snapshot. Given frozen
leaves + thread pinning, compose's parent placement is byte-identical run to run
(verified).

Absolute paths inside these JSON files are tokenized as
`__KICRAFT_PROJECT_DIR__`; the corpus substitutes the real copy dir at run time,
so the fixture relocates cleanly. Regenerate (e.g. after a deliberate
parent-placement change) with `scripts/replay_corpus.py --mode parent --update`.
