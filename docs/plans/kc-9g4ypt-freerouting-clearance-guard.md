> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# KC-9G4YPT investigation → fix plan: FreeRouting clearance guard (+ rc6 replayability)

*Written 2026-07-12 by the kicraft-investigate skill run on KC-9G4YPT. This file is the
implementation contract for a follow-up agent. Everything below is replay-/artifact-verified;
file:line references are against branch `placement-streamline` @ `86e2a17`.*

> **STATUS 2026-07-13: GAP 1 + GAP 2 IMPLEMENTED and $0-verified.** GAP 1: re-builds of 1/601
> and 1/600 from committed state both flipped the ring leaf from rejected-every-round
> (`illegal_routed_geometry`, sub-µm clearance) to accepted-every-round (601: 2/2 leaves,
> pinned score 99.07; 600: 3/3). Both still exit rc6, but for the DOWNSTREAM known issue this
> plan scoped out: parent compose can't nest the MCU leaf inside the ring (`shape fit rejected:
> circumscribed 82.9x82.9 mm exceeds requested size_mm 60.0x60.0`) + candidate shorts — the
> KC-HN59RJ shaped-outline **nesting** follow-up now owns the rest of this genre. GAP 2:
> replay of the freshly-failed 1/601 workspace restored the seed snapshot and ran the full
> pipeline (pre-fix: instant "no matching components"). GAP 3 not implemented (deferred —
> coordinate with the RoundScheduler refactor, `docs/plans/autoexperiment-round-scheduler.md`).

## TL;DR

KC-9G4YPT (project `1/601`, "round 60 mm LED ring, 12× WS2812B, ATtiny412") failed rc6 — the
parent was never composed — because the **LED RING leaf was rejected in all 9 rounds by the
`no_illegal_geometry` acceptance gate over two clearance violations of 0.8 µm**
(`required 0.1530 mm, actual 0.1522 mm`, GND track vs WS2812B pad 4 of D3/D9). The router
(FreeRouting) was given a DSN clearance rule **exactly equal** to the board's KiCad DRC rule
(both 0.153 mm), and FreeRouting's internal geometry approximation of **rotated** pads differs
from KiCad's exact shapes by ~1 µm — so any wire FreeRouting places exactly at minimum
clearance can measure just *under* the rule in KiCad DRC. The fix is a small **clearance guard**:
route at `rule + guard_um`, verify at `rule`.

Not a one-off: ≥4 distinct designs carry the identical sub-micron signature
(`actual 0.1520–0.1522 vs 0.1530`), two of them today. Ring arrays (`86e2a17`) didn't cause it —
they made it **deterministic** (fixed ring placement + non-cardinal rotations ⇒ every round
produces the same violating route, so no round can luck past it, and the whole genre of round
LED-ring briefs is now 0% buildable).

A second, independent finding: **rc6 builds overwrite the project's seed `.kicad_pcb` with the
promoted partial board**, which breaks `kicraft replay` on exactly the runs you most need to
replay (95 rc6 runs in the current corpus). Fix is to snapshot the pre-promote board and teach
the replay resolver to prefer it.

---

## Evidence (verified, with paths)

Run: `/home/kicraft/.kicraft/projects/1/601` (KC-9G4YPT, created 2026-07-12T12:41Z, rc6).
Synthesis + ERC clean (0 errors). Leaves: MCU accepted, LED RING rejected 9/9 rounds.

- **The rejection** (`.experiments/subcircuits/9459187a-…__f67b9e7bc1/debug.json`, every round):
  `rejection_reasons=['illegal_routed_geometry']`, DRC `shorts=0 unconnected=0 clearance=2`,
  report: `Clearance violation (clearance 0.1530 mm; actual 0.1522 mm)` — GND track
  (0.5 mm wide, F.Cu) vs `Pad 4 [DATA_D8] of D9` at (133.65, 85.11) mm and
  `Pad 4 [DATA_D2] of D3` at (163.35, 125.16) mm (real mm, PCB coords).
- **Attribution to FreeRouting** (not our stamped copper): the violating segment
  (`start 132.8969 83.9956 → end 134.5813 83.9956`, width 0.5, net GND) is present in **all
  three `route_cache/*.kicad_pcb`** outputs (local frame `13.5773→15.2617, 8.0391`) and **absent
  from `leaf_pre_freerouting.kicad_pcb`**; it carries no `(locked …)` flag (breakout-stub/
  strand-repair copper is stamped locked). `pre_route_validation` had `clearance=0`.
- **Rotated pads**: D9 rot = **161°**, D3 = **−19°** (ring tangent orientations). Pad 4 is a
  1.5×1.0 mm rect pad. `pcbnew` exact-shape check: the segment **collides at 0.153 mm clearance
  and clears at 0.152 mm** — true gap ≈ 0.1522, matching DRC.
- **Zero-margin configuration** (the source): board DRC rule `min_clearance = 0.153` and
  netclasses Default/Power `clearance = 0.153` (`kicraft/design/synthesis/kicad_pro.py:26,47`);
  FreeRouting floor `freerouting_min_clearance_mm = 0.153` (`kicraft/autoplacer/config.py:354`).
  The DSN inherits the 0.153 netclass rule (no fine-pitch override fired on this board —
  no "fine-pitch routing rule" line in `.kicraft/build.log`), so FreeRouting routes wires at
  *exactly* 153 µm from **its own polygonal approximation** of the pads; KiCad DRC then measures
  the true shapes and finds 152.0–152.2 µm.
- **Cross-run breadth** (scan of 365 runs with layout artifacts, 2026-07-12):
  - `SUBMICRON_CLEARANCE_MISS(0.1530→0.1522)`: `1/600` (KC-CV4NE3, **same LED-ring brief**,
    today 02:27Z) and `1/601` (KC-9G4YPT, today 12:41Z) — both deterministic, all rounds.
  - `(0.1530→0.1521)` and `(→0.1520)`: `1/581` (KC-HE2Q5T, nRF52840 BLE beacon, 2026-07-08)
    and self-eval `run_13_nrf52-beacon` (2026-07-08) — **non-ring designs**, i.e. the family
    predates ring arrays and hits ordinary boards intermittently (a re-placement round can
    jitter past it; the ring's deterministic placement cannot).
  - Leaf gate `no_illegal_geometry` failed on **19 designs**, latest today.
- **Why rc6 followed**: with 1/2 leaves accepted the parent phase skips compose entirely
  (`hierarchical_autoexperiment/round_000N/` contain only `round_config.json`), the build
  promotes the best partial board and exits: `error: the layout engine produced no routed
  parent board`.
- **Replay on current code** (`placement-streamline` working tree): reproduces rc6
  (`Round N/3 … leafs=1/2 parent_route=fail` every round). The replay also exposed GAP 2: the
  LED RING leaf re-solve fails with `Leaf subcircuit 'LED RING' has no matching components in
  the full board state` because the rc6 promote overwrote `ROUND_LED_RING.kicad_pcb` (the full
  seed board) with the MCU-only partial board (`cli_app.py:4473 shutil.copy(routed, pcb)`).

Prior-art dedup: **NEW.** No memory/plan covers sub-micron router-vs-DRC clearance skew. Related
but distinct prior art: the µm *value*-rounding trap that motivated choosing 0.153 over 0.1524
(`config.py:17-24`, `kicad_pro.py:19-24`) — that protects the rule *value* from DSN integer
rounding, not the pad *shape* approximation; and `kicraft-array-leaf-purity-and-backside-header`
(FreeRouting ignores DSN keepout/boundary for wires) — a different FR-1.9.0 geometry gotcha.
Not a regression of `86e2a17` (evidence predates it); `86e2a17` raised its severity for the
ring genre from intermittent to deterministic.

---

## GAP 1: FreeRouting routes at exactly the DRC minimum — zero tolerance for its ~1 µm geometry skew  `[code]`

```
GAP 1: router clearance == DRC clearance leaves no room for FR shape-approximation error   [code]
  evidence:  ≥4/365 designs (1/581, run_13_nrf52-beacon, 1/600, 1/601), latest 2026-07-12 (2 today);
             deterministic 9/9 rounds on ring-array leaves; replay-verified rc6 on current code
  detect:    leaf acceptance gate no_illegal_geometry DID catch it (that is why rc6, not a shipped
             dirty board) — the hole is upstream prevention, not detection: the router is asked to
             satisfy a rule with 0 µm of margin it physically cannot hold
  source:    kicraft/autoplacer/freerouting_runner.py:export_dsn (DSN rule chain _patch_dsn_clearance
             → _inject_netclass_clearances) — the single point that sets what FreeRouting routes to
  fix:       route at (rule + guard_um), keep DRC verifying at (rule); guard: unit test on the DSN
             rewrite + LED-ring build-replay regression (below)
  verify:    $0 re-build of 1/601 state.json → LED RING leaf accepted (was rejected 9/9), build
             reaches parent compose (rc6 → rc7-or-better; target rc0)
  prior-art: NEW (see dedup note above)
```

### Implementation

1. **Config knob** — `kicraft/autoplacer/config.py`, next to `freerouting_min_clearance_mm`
   (~line 354):

   ```python
   # FreeRouting's internal geometry (polygonal pad approximations, integer DSN
   # units) differs from KiCad's exact shapes by up to ~1 µm, so a wire FR places
   # exactly at the clearance rule can measure just UNDER it in KiCad DRC
   # (observed 0.1520-0.1522 vs 0.1530 on rotated pads; KC-9G4YPT/KC-CV4NE3/KC-HE2Q5T).
   # Route with this guard ABOVE the DRC rule; the board keeps verifying at the
   # real rule, so the guard can never mask a genuine violation.
   "freerouting_clearance_guard_um": 5,
   ```

   5 µm is deliberate: ≥5× the observed worst skew (1.0 µm), yet 3.3% of the rule — no
   measurable routability cost.

2. **Guard pass** — `kicraft/autoplacer/freerouting_runner.py`: new module function
   `_apply_dsn_clearance_guard(dsn_path: str, guard_um: int) -> None` that rewrites **every**
   clearance token in the DSN, bare and typed:

   - `(clearance N)` → `(clearance N+guard)`
   - `(clearance N (type T))` → `(clearance N+guard (type T))`

   Reuse the regexes already in `_patch_dsn_clearance` (:694). It must run **last**, after
   `_inject_netclass_clearances`, so injected per-netclass rules get the guard too (they are
   raw netclass values and would otherwise re-open the zero-margin hole for e.g. a Power-class
   wire — exactly this run's 0.5 mm GND track). No-op when `guard_um <= 0`.

3. **Wire it through** — `export_dsn` (:565) gains a keyword `clearance_guard_um: int = 0`;
   call the guard pass at the end (after line 620). `route_board` (:1246, the only production
   caller) passes `int(config.get("freerouting_clearance_guard_um", 5))` at its `export_dsn`
   call (:1305).

4. **Do NOT touch** `_set_board_clearance_um` (:1211) or the fine-pitch lowering
   (`_resolve_fine_pitch_rule` :1165). The guard lives **only inside the DSN**: the routed
   board's netclass/min_clearance rules — what DRC checks — stay at the un-guarded values.
   Fine-pitch interaction is then automatically correct: FR routes at `target + guard`, the
   board is capped at `target`, DRC margin = guard. Do not bump `(width …)` tokens.

5. **Leave the acceptance gate alone.** No epsilon/waiver in `validate_routed_board` — a
   tolerance there would mask genuine violations (fix-at-source rule; see
   `kicraft-fix-at-source-no-hacks`).

### Tests

- **Unit** (place near existing DSN-patch tests — `grep -rn "_patch_dsn_clearance\|export_dsn" tests/`):
  feed a small DSN string with a bare rule, a typed `smd_smd` rule, and two class rules through
  `_patch_dsn_clearance` + `_inject_netclass_clearances` + `_apply_dsn_clearance_guard(…, 5)`;
  assert every clearance token gained exactly +5 and widths are untouched. Include the
  fine-pitch-lowered path (target < global).
- **Regression guard**: unit-level is sufficient as the committed test; the end-to-end proof is
  the verification recipe below (document its result in the commit message).

### Verification recipe ($0, no LLM)

`kicraft replay --project` does **not** work on 1/601 (GAP 2 clobbered its seed board), so
re-build from the committed state instead — synthesis emission is deterministic and LLM-free:

```bash
WORK=$(mktemp -d)
cp -a /home/kicraft/.kicraft/projects/1/601/. "$WORK/"          # KC-9G4YPT
rm -rf "$WORK/generated"
cd "$WORK" && /home/kicraft/KiCraft/.venv/bin/python -m kicraft.design.cli_app \
    build .kicraft/state.json generated
```

Expected delta, pre-fix → post-fix:
- LED RING leaf: `accepted=False reject=['no_illegal_geometry']` (9/9 rounds) →
  `accepted=True` (the same GND route now sits ≥0.155 µm-nominal from pads, DRC-measured
  ≥0.153 ⇒ `clearance=0`);
- build: `leafs=1/2 parent_route=fail`, rc6 → leaves 2/2, parent compose runs (rc7 or rc0 —
  any parent-stage issue past that point is a *separate* finding, do not fold it in here).
- Repeat for `1/600` (KC-CV4NE3, same brief) — same expected delta. For `1/581`
  (KC-HE2Q5T, intermittent flavor) run 2–3 replays before claiming a delta
  (route noise; see `kicraft-self-eval-2026-06-24-findings`).

---

## GAP 2: rc6 promote overwrites the seed `.kicad_pcb` — failed runs are unreplayable  `[infra]`

```
GAP 2: partial-board promote destroys the replay input for rc6 runs                    [infra]
  evidence:  1/1 replay-verified today (1/601: replay leaf solve dies with "Leaf subcircuit
             'LED RING' has no matching components in the full board state"); mechanism applies
             to every rc6 run (95 in the corpus) — and to the CMA-ES tuner, which uses replay
  detect:    _resolve_replay_workspace already reads the workspace; promote provenance
             (source_kind=partial) is already written at promote time (cli_app.py:4446) but the
             resolver never consults it — it silently replays a wrong (partial) board
  source:    kicraft/design/cli_app.py:_promote_verify_fab (:4394) — the `shutil.copy(routed, pcb)`
             at :4473 replaces the only full-component board in the workspace
  fix:       snapshot the pre-promote full board; replay resolver prefers the snapshot when
             provenance says partial; guard: unit test on the resolver
  verify:    `cli_app replay --project <copy of 1/601 generated/ROUND_LED_RING>` runs the LED RING
             leaf solve instead of erroring "no matching components"
  prior-art: NEW (interacts with kicraft-no-fallback-previews: the promote itself is intended —
             keep it; only preserve the input it replaces)
```

### Implementation

1. In `_promote_verify_fab` (`cli_app.py:4394`), immediately before the promote copy (:4473),
   snapshot the board being replaced when the promote source is partial (and it's cheap enough
   to do unconditionally): copy the current `pcb` to a stable sibling under the experiments
   tree, e.g. `artifact_paths`-owned `…/.experiments/pre_promote_seed.kicad_pcb` (add an
   accessor in `kicraft/cli/artifact_paths.py` rather than hard-coding the path — that module
   is the artifact-layout contract, see `docs/ARTIFACTS.md`).
2. In `_resolve_replay_workspace` (`cli_app.py:5021`): when promote provenance exists with
   `source_kind == "partial"` (or `"leaf"`), and the snapshot exists, use the snapshot as the
   replay board instead of the promoted `.kicad_pcb`; when provenance says partial and **no**
   snapshot exists (all pre-fix runs), fail with an explicit message ("this rc6 run predates
   seed-snapshotting; re-build from state.json") instead of the current misleading
   "no matching components" from deep inside leaf extraction.
3. Do **not** change what gets promoted/rendered — the partial-board promote is deliberate
   (`kicraft-no-fallback-previews`).

### Tests

- Unit: a fake workspace with promote provenance `source_kind=partial` + snapshot present →
  resolver returns the snapshot; provenance partial + no snapshot → the explicit error; no
  provenance (rc7/rc0 runs) → unchanged behavior.

---

## GAP 3 (lever, smaller): deterministic-leaf search futility  `[efficiency]`

On this build the 3×3 leaf mutation search evaluated the LED RING leaf **9 times with an
identical outcome** (ring placement is deterministic; the route cache key —
`leaf_routing.py:_deterministic_route_signature` (:142) — correctly includes the freerouting
knobs, and the 3 distinct param sets produced 3 cache entries with byte-identical violating
copper), then the parent phase burned 3 more rounds re-checking `1/2 leaves`. ~5 min of
guaranteed-identical retries per build, and mutation search gives false hope on any
deterministic-placement leaf. Sketch: in the leaf loop (`cli/autoexperiment.py` /
`cli/solve_subcircuits.py`), after K=2 rounds where a leaf's `(route_signature,
rejection_reasons)` pair repeats exactly, stop re-solving that leaf and surface the rejection
loudly. **Note:** both files carry uncommitted WIP on `placement-streamline` (assignment-search
tuning) — coordinate with that work; implement only if it doesn't collide.

---

## Per-run verdict + audit appendix (KC-9G4YPT)

**Verdict:** synthesis clean (6 stages ok, ERC 0), died at rc6. Root cause = GAP 1 (leaf LED
RING rejected 9/9 for two 0.8 µm clearance misses on D3/D9 pad 4 vs a FreeRouting GND wire);
parent compose never attempted with 1/2 leaves. MCU leaf accepted (its 2 failed rounds were
`leaf_pre_stamp_legality_repair` retries on UPDI/MCU_DATA_OUT — recovered round 3).

Audits (all orthogonal to the failure):
- **§6 provenance**: all parts curated-default or stock-KiCad except **U1 attiny412 =
  home-fetched** — curated-library coverage gap; if attiny412 recurs across runs, vendor it
  (`add-part --from-lcsc C1337190 --into vendored` + `refresh_sample_previews.py`).
- **§7 BOM realness**: 6/6 priced parts REAL and in stock (catalog dump 0 d old); explicit-C#
  MPNs all MATCH (ATTINY412-SSNR, WS2812B-B/T, S2B-PH-SM4-TB). TP1–3 test points unpriced
  (no LCSC path — expected for testpoints). Clean BOM.
- **§8 wheel-spin**: all stages converged attempt 1–2; BOM hit its 6-round tool cap
  (18 tool calls) — the known cost pattern (`kicraft-pipeline-cost-bom-retries`), converged.
- **§8.5 intent adherence**: `intent.form_factor = {shape: circle, size_mm: 60.0}` correctly
  captured from the brief; delivery blocked by rc6 (adherence untestable until GAP 1 lands —
  re-check the promoted outline is a 60 mm circle in the GAP 1 verification build).
- Cross-run datum (not this run's blocker): `leaf_pre_stamp_legality_repair` is the
  **broadest** leaf round-failure family (70/365 designs, latest today) — owned by the
  self-eval N-plan (`docs/plans/self-eval-2026-07-11-fix-plan.md`); do not band-aid here.
- Gate-report cosmetics: the `drc_clearance` acceptance gate reported `passed=True
  (mode=unconstrained)` for the same 2 violations that failed `no_illegal_geometry` — confusing
  but by design (the strict verdict lives in the illegal-geometry gate); optional cleanup:
  annotate the `drc_clearance` gate result with `deferred_to: no_illegal_geometry`.
- Scanner note for future investigators: the cross-run `SUBMICRON_CLEARANCE_MISS(0.1530→0.4000
  / →0.5650)` rows produced by this investigation's ad-hoc scan are regex artifacts (first-match
  against multi-violation report_text), not real near-misses; the real family is 0.1520–0.1522.

## Suggested implementation order

1. GAP 1 (config knob + guard pass + unit tests) — the fab-blocking fix; verify via the
   recipe on 1/601 and 1/600.
2. GAP 2 (snapshot + resolver) — restores the replay primitive for rc6 runs, including
   verifying #1 on future failures.
3. GAP 3 — optional, coordinate with the placement-streamline WIP in the same files.

Deploy note: pipeline change ⇒ restart **both** `kicraft-web` and the build worker
(`deploy/restart-web.sh`, `deploy/restart-build-worker.sh`).
