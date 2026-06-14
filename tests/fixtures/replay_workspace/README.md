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

## `PARENT_LOCAL_CONN` — parent-local-connector fixture (Lever 2.1)

`USB_PD_TRIGGER` + one extra connector `J3` placed at the **root** level (in no
leaf, `edge:bottom`). It is the only corpus board that exercises
`_snap_parent_local`'s connector branch: in `USB_PD_TRIGGER` every edge
connector lives inside a leaf, so the parent-local allowlist is empty and that
branch never runs. Lever 2.1 (Phase 3 of
`docs/plans/place-route-root-cause-v2.md`) deletes that branch and re-routes
loose parent-level connectors through the leaf path; this fixture is its gate.

Built by `scripts/build_parent_local_conn_fixture.py` (hand-derived, no LLM
synthesis). Notes:
- **Baseline behaviour:** the leaf connectors J1/J2/SW1 land flush, but the
  parent-local J3 **strands ~4 mm inboard** — `_snap_parent_local` snaps it to
  the pre-repair outline while a leaf defines the board extremity, and J3 is
  never pinned as an extremity. `tests/test_connector_edge_gap.py` encodes this
  as a strict `xfail` (`test_parent_local_connector_not_stranded`) that flips to
  pass once Lever 2.1 lands.
- **Parent-mode only:** it has a `.parent.golden.json` but no `.leaf.golden.json`
  (the seed PCB carries J3, which leaf-regeneration via `replay` doesn't model).
  `replay_corpus.py` SKIPs a mode with no golden, so `--mode both` is clean.
- **`parent_compose_spacing_mm: 3.5`** in its autoplacer config: a 4th edge
  connector packs the dense board tight enough to short at the 2.0 default, so
  the gate composes it at the clearance it was frozen with (read by both
  `replay_corpus.py` and the edge-gap test).
