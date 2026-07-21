# Self-eval 2026-07-20 — results of the review-fix wave + next steps

Batch `logs/self_eval/20260720T113207Z` (34 briefs, deepseek-v4-flash design /
minimax-m3 judge, $1.18, 7.1h wall). First batch on the 2026-07-19 review fixes
(PR-A..PR-F, `2d78ff5`..`88a856d`) and the first graded by the repaired rubric
(PR-G: real friction axis — levels NOT comparable to pre-7-20 batches, which
scored a constant 3 there).

## Headline vs baseline `20260719T014949Z`

| | baseline 7-19 | this batch | Δ |
|---|---|---|---|
| fab-ready | 16/34 | **24/34** | **+8** |
| mean / median final | 71.0 / — | 75.3 / 76.5 | +4.3 |
| grades | 15B 14C 5D | **23B** 9C **2D** | D's 5→2 |
| gates | 2 unprogrammable_mcu + 1 erc | **1 erc** | strap gates gone |
| errored briefs | — | 0 | |
| spend | $1.13 | $1.18 | flat |

**10 briefs flipped to fab-ready** (rc-lowpass-bnc B/88.5, r2r-dac B/79.5,
esp32-s3-sensor B/77.5, proto-shield C/62 ← the PR-B scaffold fix,
esp32-dual-motor B/80, gpio-expander B/76.5, servo-driver-16 B/81 ← the #1-breadth
GND-strand class, round-led-ring B/76 ← the circle genre that failed 5 live
boards, rounded-c3-devboard C/71.5, chamfered-badge C/72). **2 flipped down**
(usb-pd-trigger: `connector_misoriented:SW1` — an honest facings-gate rejection
of a bad placement roll; lora-node: strand+unconnected) — both single-run,
within the ~12-pt noise floor; treat as indicative, not proven regressions.

Retry composition: intent retries **5 → 0** (the §5.1 flat-output fix
eliminated its class); bom 28→32 / wiring 9→12 (un-normalized — more designs
survived deeper, so later stages simply ran more).

## Remaining failure classes (10 boards), ranked by breadth

1. **C1 unconnected / dense routing — 7 boards** (06 unc=5, 09 unc=1,
   10 unc=24, 13 unc=6, 14 unc=1, 24 unc=6, 27 unc=10). Now unambiguously the
   dominant blocker; everything else the review fixed. Owner: **C1 v2 richer
   pathfinding/rip-up** (deferred from the C1 memory —
   `no_clear_path` family, plus track-endpoint anchors). run_10's unc=24
   (RP2040 QFN-56 + QSPI) is the stress fixture.
2. **`illegal_routed_geometry` now FIRES on 5 of those boards** (06, 13, 14,
   24, 27) — the previously-dead outline-containment check (2.6) is catching
   freerouting escaping Edge.Cuts on dense boards, honestly. New lever this
   opens: **post-route escape remediation** — rip up copper outside the
   outline and hand the resulting opens to the existing
   `unconnected_repair`/GND-spine machinery instead of shipping the escape.
   Pairs naturally with C1 v2 (same boards).
3. **Zero-pin mechanical symbols kill stage-prep wiring — run_20**
   (encoder-oled-panel, D/49.5, no build, BOTH batches). The model legitimately
   picks `Mechanical:MountingHole` (zero pins); `stage-prep wiring` treats
   "resolved but exposes no pins" as fatal (`cli_app` rc=4) and the run dies
   before the wiring model ever runs — no retry, no park. Fix: exempt zero-pin
   symbols from the offender list (nothing to wire — skip them in
   `symbol_pinouts` and in §9.11 coverage). Small, surgical, un-bricks the
   whole front-panel genre.
4. **Multi-unit emitter incompleteness — run_28** (audio-jack-buffer, D/45,
   ERC, both batches). PR-D made the gates/wiring see a dual op-amp's unit-B
   pins, but the EMITTER still instantiates only unit A, so `U1.5/6/7` never
   reach the netlist ("pin missing from netlist"). Owner: emitter multi-unit
   instantiation — the completion `validation.py:1745`'s §9.30 docstring
   already anticipates ("the emitter learns to instantiate all units").
5. **Connector-mouth follow-ups** — run_05 `connector_misoriented:SW1`
   (slide-switch facing, the KC-YJ7Q69 family's open unzoned-flush extension)
   and the strand insets on 14/24 (the known perp-mouth 1.0–1.5mm family).

## Ranked next steps

- **N1 (owns 7 boards): C1 v2 pathfinding/rip-up.** Design from the C1 memory:
  richer rip-up/reroute for `no_clear_path` edges + `no_pad_anchor`
  track-endpoint anchors. Fixture: run_10 (unc=24), plus 06/13/24/27.
  This is its own multi-day PR; everything else below is small.
- **N2 (pairs with N1, same boards): outline-escape remediation.** On
  `malformed_board_geometry`, rip escaped segments and re-run the unconnected
  repair + GND spine before the verdict; keep the honest rejection when the
  remediation cannot close the opens.
- **N3 (un-bricks a genre, ~1h): zero-pin symbol exemption** in stage-prep
  wiring + §9.11 (skip, don't offend). Pin a test on
  `Mechanical:MountingHole`.
- **N4 (owns run_28): emitter multi-unit instantiation** (draw every unit,
  place unit B beside unit A, netlist all pins). §9.30 then validates it.
- **N5: facings/flush follow-ups** — extend the unzoned-flush gate to switch
  mouths (run_05) and finish the perp-mouth flush inset (14/24).
- **N6 (carry-over from the review backlog):** §6.2 semantic net-name↔pin
  lint, §6.3 architecture-GPIO reconcile, drv8833 sourcing decision (human),
  live-DB junk-row DELETE (human), §9/§10 hygiene+perf items, deferred 3.3.
- **Measurement discipline:** next batch after N1/N2 should target ≥28/34;
  run N-of-3 medians before claiming any single-brief regression (noise
  floor). Friction-axis history restarts at this batch.

## Verdict

The 2026-07-19 review wave did what it claimed: +8 fab-ready (courtyard,
scaffold, GND-strand, shaped-outline and retry classes all cleared), zero
infra errors, strap gates gone, intent retries eliminated, at flat spend.
The board is now bottlenecked on exactly one big thing (dense-board routing
completion) and three small seams (zero-pin symbols, multi-unit emitter,
switch facings).

## Status 2026-07-21 (implementation wave)

- **N3 SHIPPED**: stage-prep wiring now carries the same zero-pin
  `Mechanical:*` exemption as BOM commit (§9.11 already passed these
  vacuously); regression test on `Mechanical:MountingHole`.
- **N4 SHIPPED**: full multi-unit instantiation — placement expands each
  functional unit into its own placeable entity (`PlacedPart.unit`), the
  router resolves each pin against its owning unit's placement, the emitter
  draws one `(symbol)` block per unit, §9.30 deleted. TL072 dual-buffer
  golden: ERC clean + §9.13 sees all unit-B pins.
- **N5 half-SHIPPED**: (a) the adapter now runs mouth detection for ANY
  edge-zoned ref, prefix-blind like the facings gate — the run_05
  `connector_misoriented:SW1` class places deliberately now. (b) The
  14/24 strand insets are NOT the leaf-level perp-mouth family: run_14's
  edge is set by a FOREIGN leaf's cap poking 0.95 mm past J2's mouth
  line, run_24's by leaf blocks composing 2.2 mm short of the line a
  sibling leaf set. Needs a compose-level shared mouth-line constraint —
  open follow-up.
- **N2 SHIPPED, premise corrected**: the 5 named boards have ZERO copper
  outside the outline — their `illegal_routed_geometry` is real
  `clearance` / `copper_edge_clearance` violations. The shipped pass
  (`geometry_repair.rip_illegal_copper` + accept-or-revert wrapper) rips
  DRC-named track/via items AND outline escapes, re-runs the repair
  machinery, honestly reverts when the re-close loses ground (currently
  the case on 13/14/27 — the re-close is what N1 phase 3 owns).
- **N1 phases 1–2 SHIPPED** (track-endpoint anchors, board-diagonal gap
  cap, free-anchor strict margins, geometry-worse accept gates); phase 3
  (grid A* + bounded rip-up) designed in
  `docs/plans/c1-v2-pathfinding-design.md`. run_10 after phases 1–2:
  every edge genuinely attempts; all 21 remaining are honest
  `no_clear_path` — the pathfinder is the sole remaining owner.
