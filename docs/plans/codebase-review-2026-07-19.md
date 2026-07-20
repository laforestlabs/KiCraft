# Full codebase review & fix plan — 2026-07-19

**How this was produced.** Seven parallel review passes over the whole repo (autoplacer, cli,
server, design/synthesis, supporting packages, a line-by-line audit of every canned LLM prompt,
and a reconstruction of the full LLM thinking traces from boards 631–639), cross-checked against
the live DB and build logs for the last 14 days. Every finding below was verified by reading the
actual code (several by executing it); none are style nits. Findings that duplicate known/open
issues in the memory index were dropped or are explicitly marked as *new evidence on a known issue*.

**Verification protocol.** Nothing here is replay-proven yet. Per house rules, each place/route
fix must be `$0`-verified with `kicraft replay` (full build for round-loop changes — `--quality=fast`
does not exercise the round loop) before shipping, and prompt changes should be A/B-checked against
a handful of recent briefs. Fix at the source; no masking gates.

---

## 1. Where we are (live baseline — read with cohort caveats)

41 failed / 31 ok (43% ok) over 14 days — **but that window spans five pipeline deploys**
(7-16 12:42 + 20:17, 7-17 17:51 + 22:24, 7-18 ~12:00 `eaba0a0`+`7c82a8d`, 7-19 11:55 UTC
`36e732f` V1-V8), so older failures may already be fixed. Every *code* finding in this plan was
made against HEAD = `36e732f` (post-V1-V8) and is current by construction; only the live-data
prioritization needs cohort discipline:

- **Current-code cohort** (built after the 7-18 midday deploy): failures 636, 637, 638, 639.
  Fingerprints: `unconnected=9` + sprawl (636, stepper); synthesis ERC output-conflict +
  multi-unit TL072 (637 — built *before* the 7-19 V3 deploy, so partially re-testable);
  circle-outline non-conformance (638 — known open, leaf grid-assignment tuning); and
  **639, built on today's V1-V8 code**: `connector_stranded:J2@-1.37mm(right)`, unconnected=4 —
  the known-open perp single-header mouth-inset family. Critically, 639's log shows the round
  loop **promoted** that strand-rejected parent ("3/5 promoted routed parent") with only the
  terminal verify refusing fab — live current-code confirmation of §2.2. 639 also repeated the
  intent-nesting retry (§5.1), confirming it on current code.
- **Older cohort (7-13 → 7-16, pre-T1-T3) — RE-VERIFIED CURRENT via Step-0 triage
  (2026-07-19)**: cold full-build replays of 623 and 628 on HEAD (`36e732f`, quality=good,
  seed 0) reproduced the exact failure classes: 623 → `courtyards_overlap` (courtyard=1, rc7);
  628 → `illegal_routed_geometry` + `courtyards_overlap` (courtyard=1, rc7). T1-T3 did **not**
  fix this class. Current-cohort replays likewise reproduced: 639 →
  `connector_stranded:J2@-1.37mm(right)` (same mm figure, unconnected 9 vs 4 live — normal
  routing variance); 636 → unconnected=7 + sprawl aspect 3.92 (live: 9 / 5.94). All four
  promoted a routed parent at step 3/5 and were rejected only by the terminal verify — the §2
  honesty gap demonstrated on HEAD in every replay. (Single replay per board; class-level
  reasons matched exactly, so no repeat runs were needed for the class verdict.)
- Infra deaths **625** (stream error, no retry) and **627** (`PadBindingError`) are cohort-proof:
  the defective code paths were re-verified at HEAD (§4.1, §4.2).
- Trace analysis covered 631–639: 631–638 ran pre-V1-V8 synthesis prompts; 639 ran today's.
  Trace findings called out below were re-checked against HEAD prompt text (all quoted strings
  verified current), and the drv8833 retail-stock check was run against the live catalog today.

The plan below is ordered by (expected fab-ready-rate impact × confidence) / effort. With Step-0
complete, the courtyard items in §2/§3 carry **full priority** — all four triaged failure classes
(courtyard overlap, illegal geometry, connector strand, unconnected sprawl) are confirmed live on
HEAD.

---

## 2. P0 — the round-loop honesty chain (owns the courtyard/unconnected fab-gate cluster)

The live pattern "Round 3/3 score=20 tier=not_routed … [discard], final verify finds
courtyard/unconnected/stranded" is not one bug — it is a chain of five, all in the
acceptance/scoring path. The search *cannot see* the defects the fab gate later rejects, so it can
neither prefer clean candidates nor learn between rounds. These are honesty bugs verified at HEAD
regardless of how often each defect class currently occurs — and §2.2 is *proven current* by board
639 (built on V1-V8 code today: strand-rejected parent promoted, terminal gate refused).
Fix as one workstream; verify with a full-build replay of 639/636 (current cohort) plus
628/623 (courtyard family — which doubles as the Step-0 triage of whether courtyard still
reproduces on HEAD, see §12).

- **2.1 Courtyard overlap is invisible to round-loop acceptance** (bug-high).
  `validate_routed_board` (`autoplacer/freerouting_runner.py:1683-1856`) never reads
  `drc["courtyard"]`; `_route_parent_board` (`cli/_compose_route.py:345-390`) adds a post-hoc
  rejection for `unconnected` only (:374-383) — no courtyard equivalent. A parent whose only defect
  is `courtyards_overlap` returns `accepted=True`, exits 0, and `_score_round` tiers it
  **functional**. The only courtyard check is `cli_app._verify_routed_board`, after the round budget
  is spent. *Fix:* reject on `drc.courtyard > 0` in `_route_parent_board`, mirroring unconnected.

- **2.2 A strand-rejected parent returns rc=0 and still scores "functional"** (bug-high).
  `compose_subcircuits.py:3943-3989` deliberately returns 0 for the promotable-strand-only case with
  `validation["accepted"]=False` (:3894), but `_score_round`'s `parent_routed=True` branch
  (`cli/autoexperiment.py:~777-791`) never consults `parent_routed_validation["accepted"]` — unlike
  the sibling `not parent_routed` branch. The rejected board gets kept, pinned by `_pin_best_parent`,
  and biases later rounds *toward* the defect. *Fix:* check `accepted` in that branch; route False
  through the `routed_dirty`-style penalized scoring.

- **2.3 A DRC timeout / missing kicad-cli / tool crash reads as "0 violations"** (bug-high, found
  independently by two reviewers). `_run_kicad_cli_drc` zero-inits counts
  (`freerouting_runner.py:1540-1560`) and on timeout/missing binary only flips flags (:1669-1672);
  a nonzero kicad-cli exit with an empty report also parses to `total=0` (returncode stored at :1575,
  never checked). Neither compose call site (`compose_subcircuits.py:2912-2918`, `:3711-3725`) nor
  `validate_routed_board` checks `timed_out`/`missing_cli`/returncode before trusting `shorts==0` —
  and `_winner_key` prefers exactly those "clean" numbers, so a timed-out candidate can *win*.
  *Fix:* treat timed-out / missing / rc≠0-with-empty-report as disqualifying at all three sites.

- **2.4 The composer computes courtyard/keepout conflicts per candidate, then throws them away**
  (bug-high). `same_side_overlap_conflicts` and `tht_keepout_violations` are computed and stored
  (`compose_subcircuits.py:2045-2046, 2102, 2114, 2119-2120`) but never read by the accept gate
  (`accepted = shorts == 0`, :3016-3041) or the score (:2939-2944). A candidate with a *detected*
  collision can win the K-candidate search. *Fix:* fold both counts into `accepted` (or a heavy
  score penalty) so the search prefers geometrically-legal candidates.

- **2.5 No round-to-round learning for courtyard, and `_best_config` freezes on all-fail runs**
  (bug-medium + architectural). The scheduler's only adaptive levers are refit-backoff and the
  unconnected congestion valve (`autoexperiment.py:1343-1378`, `_round_scheduler.py:292-326`);
  `courtyard_padding_mm`/`placement_clearance_mm` are in `CONFIG_SEARCH_SPACE` (`config.py:599-600`)
  with no forcing function. And since `_best_config` only updates on kept rounds
  (`autoexperiment.py:3197-3200`), an all-3-rounds-fail run mutates the *same* starting config three
  times independently. *Fix:* add a `_parent_rejected_courtyard` signal biasing those two params
  (pattern-match the congestion valve), and consider a parent-side analogue of
  `_auto_pin_best_leaves` (full-history reselection; today `_pin_best_parent` trusts live "kept"
  tracking, which 2.1–2.3 can corrupt).

Related, same theme, slightly lower priority:

- **2.6 `illegal_routed_geometry`'s upstream flag is dead code**: `malformed_board_geometry` is
  initialized False (`freerouting_runner.py:1698`), consumed as a hard-fail reason (:1850-1851) and
  in two other modules — but *never set True anywhere*. There is no copper-inside-outline
  containment check even though freerouting 1.9.0 is known to ignore DSN boundaries for wires.
  Live boards 631/628 failed with `illegal_routed_geometry` only at the terminal gate. *Fix:*
  implement the outline-containment check and set the flag, so the round loop sees it (bug-high).

- **2.7 Congestion valve can push `parent_seed_area_overhead` to 7.0 (documented ceiling 3.5)** —
  `autoexperiment.py:2567-2582` multiplies an already-mutated value by up to 2.0 with no re-clamp.
  *Fix:* clamp to the search-space max after scaling (bug-medium).

---

## 3. P0 — leaf/compose geometry: how illegal boards get frozen in the first place

These are the producers of the defects §2 fails to see. Same replay verification set.

- **3.1 `resnap_to_grid` unconditionally undoes the final courtyard legalization pass on the
  DEFAULT leaf path** (bug-high). `placement_solver.py:712-730` (Step 16,
  `_resolve_courtyard_overlaps`) moves gridded passives to clear overlaps; :736-739 then
  unconditionally snaps every gridded occupant back to its pre-computed slot
  (`leaf_grid_assignment.py:532-545`, no legality check). Gridded passives are not exempt from
  Step 16 (only `array_member` pairs, :3665-3667), so any fix Step 16 makes is silently reverted.
  The leaf gate only rejects *gross* overlap, so a minor one ships frozen into the parent —
  reproducing `courtyards_overlap` at the fab gate. *Fix:* skip the resnap for occupants Step 16
  moved (or exempt gridded occupants from Step 16 and re-check slot legality before resnap).

- **3.2 Opposite-layer stacking never checks body/courtyard rects** (bug-high).
  `can_overlap_sparse`'s stacking branch (`subcircuit_composer.py:2535-2646`) compares only
  pads/drills/traces; `component_rects` (the real physical envelope) is used only in the coarse
  prefilter. THT-heavy leaves (battery holders, connector-only leaves) route through this branch
  almost always (`compose_subcircuits.py:2069-2079`), so a big plastic body can overlap another
  leaf's geometry with zero rejection. *Fix:* add a `component_rects` pairwise check to the branch.

- **3.3 [DEFERRED from PR-B 2026-07-20]** — collides with the V2 battery-holder exemption
  deliberately shipped 2026-07-19 and verified clean on run_31; revisit with that context,
  not as a drive-by. Original finding: **Trivial leaves (no internal nets) skip DRC entirely** (bug-medium, **check intent first**:
  the recent V2 battery-holder gate exemption was deliberate — this finding generalizes it to *all*
  trivial leaves and to KiCad's own courtyard/clearance DRC, which `repair_leaf_placement_legality`
  does not cover). `leaf_routing.py:1327-1526` copies the placed board to `leaf_routed.kicad_pcb`
  with hardcoded `accepted: True`. *Fix (if intent allows):* run `validate_routed_board` on the
  trivial-leaf stamp too, keeping the V2 exemption scoped to the specific gate it addressed.

- **3.4 Pour failures are swallowed while poured nets stay exempt from the unconnected gate by
  name** (bug-high; *new evidence on the known GND-strand issue*). `leaf_routing.py:971-994`
  (GND) and :999-1031 (power) catch all exceptions and continue; `leaf_acceptance._is_poured_net`
  (~:202-210) exempts by net-name pattern unconditionally. A mid-pour pcbnew exception leaves pads
  genuinely stranded yet gate-exempt — matching the live small-N unconnected GND/power class.
  *Fix:* record pour success in `validation`; exempt a net only when its pour ran clean.

- **3.5 GND clearance floor `break`s after the first footprint** (bug-high). `gnd_pour.py:702-707`
  computes the max GND-pad clearance across the board — except the `break` exits after the first
  GND-bearing footprint, ignoring stricter netclass overrides elsewhere. This re-opens the exact
  KC-UXASHQ via-too-close class the code cites. *Fix:* remove the `break`.

- **3.6 SA displacement move clamps to the raw board edge without pad half-extents**
  (bug-medium). `placement_solver.py:2536-2537` is the only clamp site in the file not using
  `_pad_half_extents`; SA can promote off-board candidates into `best_comps`. *Fix:* use the same
  clamp convention as everywhere else.

- **3.7 Fine-pitch clearance relief triggers off a hardcoded 0.2mm instead of the board's parsed
  default clearance** (bug-medium). `freerouting_runner.py:1221` vs the real value used at
  :724-727. Boards with a wider default (0.3mm power classes) never get relief for pad gaps in
  0.2–0.3mm → FreeRouting is handed an unsatisfiable rule → near-miss unrouted nets. *Fix:*
  derive the threshold from the parsed default clearance.

- **3.8 Stale obstacle cache degrades unconnected-repair** (bug-medium; *new evidence on C1*).
  `unconnected_repair.py:318` computes `obstacles` once; successful ties inside the loop
  (:367-379) never refresh it, so the pre-filter wastes `max_attempts` on candidates that collide
  with just-stamped ties. *Fix:* append newly stamped geometry after each success.

- **3.9 Form-factor scaffold headers have no `body_center` → collision math on a phantom box up
  to 11.4mm off** (bug-high on the shield path; plausible contributor to board 628's
  courtyard+form-factor failure). `form_factors/compose_scaffold.py:104-115` sets `pos` = pin 1
  and leaves `body_center=None`, so `Component.bbox()` falls back to `pos` (Arduino digital_high
  header: center x=30.2 vs pos 18.8). Every other construction site sets it. *Fix:* pass the
  already-computed `(min+max)/2` center into the `Component(...)` call.

---

## 4. P0 — crashes & silent run-killers with live incidents

- **4.1 Mid-stream LLM connection errors permanently fail the run (board 625)** (bug-high).
  `server/client.py:111-230`: retries wrap only the initial POST; once 2xx, `resp.iter_lines()`
  has no retry (the docstring admits it), so `InvalidChunkLength`/`ChunkedEncodingError`
  propagates to `_run_design`'s outer except (`web.py:2130`) and the project is marked failed.
  Nothing has been committed at that point — a full re-POST is safe. *Fix:* bounded retry around
  the stream-accumulation loop, discarding partial buffers.

- **4.2 §9.27 skips symbol/footprint mismatch for zero-numbered-pad footprints → uncaught
  `PadBindingError` crash (board 627)** (bug-high). `cli_app.py:297-347`: `_footprint_pad_numbers`
  returns `None` (unresolvable) or `set()` (resolved, only NPTH `""` pads — e.g. plain
  `MountingHole:*`); `if not pads: continue` conflates them, so `Mechanical:MountingHole_Pad` +
  a padless footprint passes the gate, then `write_empty_pcb` raises `PadBindingError` inside
  `synthesize.run()` (`synthesize.py:259`) *before* validation collection, unwrapped by any
  handler (`cli_app.py:5218-5236`, `main()` :6365) → hard crash instead of a bounce-able failure.
  *Fix:* `if pads is None: continue`, plus catch `PadBindingError` in `synthesize.run()` and
  surface it as a `SynthesisValidationError` so wiring can retry.

- **4.3 S-expression block matcher is not string-aware — 481 stock symbols silently parse to
  zero pins** (bug-high). `symbol_library.py:34-47` counts parens inside quoted strings; the whole
  `Conn_*_Row_Letter_Last` family (180 symbols) and ~300 STM32H5/H7 variants truncate, and
  `lookup_pins` returns an empty pin list indistinguishable from a legit zero-pin symbol —
  the same empty-vs-unresolved conflation as 4.2, one layer deeper, reachable from the BOM stage's
  own `search_symbols` tool. *Fix:* skip quoted content (honoring `\"`) while scanning.

- **4.4 Pin-type normalization applies to the embedded schematic symbol but not to
  `lookup_pins`** (bug-high, proven by execution on the stale vendored tps54331 on this host).
  `extract_symbol_block` retypes switch-node/regulator-output pins (`symbol_library.py:241-270`);
  `symbol_pinout.lookup_pins` — feeding §9.11/§9.16/§9.20/§9.29 *and* the emitter's PWR_FLAG
  logic (`emitter.py:1114-1166`) — reads raw types. Divergence produces the exact spurious
  "Power output and Power output" ERC short the normalization was built to fix. *Fix:* factor the
  normalization into `_resolve_extends_chain` so both readers see one corrected view.

- **4.5 router/placement silently drop nets on symbol-resolution failure** (bug-high).
  `router.py:212-341` and `placement.py:155-203` catch lookup errors and substitute empties with
  no log and no `bom.assumptions` entry; a fully-unresolvable net vanishes from the schematic and
  surfaces only as a baffling ERC "pin not connected". *Fix:* record every drop loudly.

- **4.6 Array-decap downsizer force-shrinks any package outside its rank table** (bug-high).
  `array_decap_footprints.py:45,104-107`: unknown-but-larger packages (2010/2220/2512) default to
  rank 99 → always downsized, inverting the "never enlarge, only shrink" contract; a deliberate
  bulk-cap choice gets silently rewritten to 0603. *Fix:* unknown package ⇒ leave alone.

- **4.7 LCSC-pin denylist eats real part C1812** (bug-high). `fab_export.py:33-58` treats "1812"
  as a package token; C1812 is a real in-stock part → blank `lcsc` column in the fab BOM and a
  defeated §9.26 read. *Fix:* require package context (or catalog cross-check) before excluding.

---

## 5. P0 — synthesis retry eliminations (cheapest wins in the whole report)

BOM/wiring retries dominate LLM spend (~90%). Trace analysis of boards 631–639 found the retries
are highly systematic, not noise:

- **5.1 Intent "goal" nesting bug — 4 of 9 recent boards burned a retry on it** (prompt fix,
  one sentence). The model wraps the slot under an `"intent"` key (matching the
  `CURRENT DESIGN STATE` JSON it is shown, `stage_driver.py:819-820`) because
  `_stage_extra("intent")` (:166-169) is the only stage that introduces "a top-level field beside
  the slot" (`project_stem`). *Fix:* add to the intent extra: *"Output the slot's fields directly
  at the JSON top level — do NOT wrap them under an `intent` key; `project_stem` is a sibling key
  in the same flat object: `{"project_stem": "X", "goal": "...", ...}`."*

- **5.2 Vendored `drv8833` "production" bundle is a guaranteed §9.26 bounce** (data fix at the
  source). `C50506` has 3,299 assembly stock but **0 lcsc.com retail** — and the prompt explicitly
  tells the model to skip re-verifying bundle rows, so every DRV8833 brief eats a mandatory retry
  (reproduced on board 631). Root cause: `_format_core_defaults_block`'s dry filter
  (`stage_driver.py:296-313`) and `validate-part` check assembly stock only, never retail. *Fix:*
  re-point drv8833 sourcing to `C191171` (DRV8833CPWPR, 2,267/1,811 — the substitute board 631's
  own retry found), and add the retail check to bundle curation/validation so no other vendored
  bundle can regress the same way.

- **5.3 `bom.md` still recommends the below-floor `ch340n` bundle** — `bom.md:8` vs
  `architecture.md:64`/`wiring.md:27` (both say ch340c); `core_blocks.json:417-424` already
  documents the bug. *Fix:* delete the ch340n clause.

- **5.4 `bom.md` routes expertise on a value that doesn't exist** — `advanced` vs the schema's
  `expert` (`models.py:174`). *Fix:* one-word change.

- **5.5 Bounce messages truncate offenders to 20 with no total** — 14 sites in `cli_app.py`
  (`"offenders": bad[:20]`), re-serialized verbatim by `_retry_feedback`
  (`stage_driver.py:692-702`). On big arrays the model fixes the visible 20 and gets bounced with
  a *different* 20. *Fix:* append `offenders_total` + "fix ALL instances of this class, not just
  those listed."

- **5.6 `search_symbols`/`search_footprints` AND-match silently drops the stock-KiCad section for
  natural phrasing** — `symbol_library.py:339-356` requires every term as a literal substring, so
  "pin header 1x03 male" matches nothing (real name `Conn_01x03`) with no signal; board 637 then
  guessed the nonexistent `Connector:Conn_01x03_Male` and bounced — defeating the prompt's own
  "NEVER guess" rule. *Fix:* token aliasing (pin/header/male/female → conn) and/or an explicit
  "no stock symbol matched all of {terms} — retry with fewer/simpler keywords" line on zero hits.

- **5.7 Repeated memoized tool calls only get a soft notice** — board 635 called the identical
  `lookup_symbol` 4×, the 4th immediately after its own "now write the final JSON"; each repeat is
  a paid round trip. `lookup_lcsc_id` already has a hard cap. *Fix:* after 2 identical repeats,
  return a steer-only payload (no data), mirroring the hard cutoff.

- **5.8 Deterministic BOM-reconcile treats partial fulfillment as full success** (pipeline bug,
  produced a functionally dead board). Board 639's wiring parked asking for a crystal + 3 caps +
  u.FL + 0-ohm; `apply_deterministic_bom_adds` (`session.py:403-501`, `_PASSIVE_ASK_RE` :307-312)
  added the caps it could parse, silently dropped the crystal/u.FL/R7, and `maybe_bom_reconcile`
  (:518-566) saw non-empty `added` → re-drove wiring only, with "do NOT park on the same deficit
  again". Result committed: load caps wired `XTAL_N/XTAL_P`→GND **with no crystal**, no antenna
  network. *Fix:* return unfulfilled asks and fall through to the LLM `[bom, wiring]` pass for the
  remainder.

---

## 6. P1 — gate gaps: model errors that currently ship

From the trace analysis — each names the gate that should own it.

- **6.1 Multi-unit symbols: stage-prep surfaces only unit A's pins to §9.11** (board 637: TL072
  unit B never wired, accepted; the model itself flagged it after the fact). Owning fix: multi-unit
  enumeration in the `symbol_pinout`/stage-prep extraction; §9.11 can then see the full pin list.
  (Same root as 4.3/4.4: the pin-data seam is the weakest layer in synthesis.)

- **6.2 No lint checks that a semantically-named net lands on the matching pin function**
  (board 639: `USB_DP`/`USB_DN` committed to `U0TXD`/`U0RXD` while the correctly-identified real
  USB pins were marked no-connect; §9.11/§9.17/§9.19 are topology-only and electrical_review
  passed it — the board would ship unable to enumerate). *Fix:* a new wiring-commit lint
  cross-checking net-name substrings (`USB_D`, `SWD`, `I2C_SDA/SCL`, `UART_TX/RX`) against the
  resolved pin's own name when both are present; warn (or reject on exact-function conflicts).

- **6.3 Architecture invents generic `GPIO0..N` inter-sheet nets that don't exist on the resolved
  symbol** — cost board 639 a 93k-char single-turn wheel-spin and caused 6.2's miswire. *Fix:*
  deterministic reconcile of architecture net names against the resolved symbol pinout at wiring
  stage-prep (surface "these architecture nets have no matching pin" instead of letting the model
  freelance).

- **6.4 `BOM.component_zones` is unvalidated** while `PlacementSection.component_zones` is strict
  (`models.py:590-601` vs :718-767) — a typo'd zone key silently no-ops through
  `synthesis/autoplacer.py:115-139` and degrades placement with no error. *Fix:* apply the same
  validator at BOM commit.

- **6.5 Form-factor reconcile injects BOM parts at wiring-commit, bypassing §9.26/§9.27/§9.28**
  (`cli_app.py:3140-3169`; gates are `stage == "bom"`-only). Latent (today's only user is the
  curated scaffold), but any future reconcile feature inherits the hole. *Fix:* re-run the cheap
  parts-only identity gates on any out-of-band BOM mutation.

- **6.6 Form-factor conformance gate silently no-ops on any parse exception**
  (`cli_app.py:4015-4049` bare except → `None` = "no standard requested"). *Fix:* when
  `template.validated and enforce_enabled()`, a parse failure is a gate failure.

- **6.7 §9.8 library-interface gate matches any project file with the right label set**
  (`validation.py:381-488`) — two sheets pulling the same library slug can hide one corrupt file
  behind the other's match. *Fix:* claim files per entry / resolve sheet stem → file.

---

## 7. P1 — canned-prompt overhaul (beyond the P0 items in §5)

The full audit is in the review notes; the highest-value structural changes:

- **7.1 Add one worked JSON example per stage.** Today the model gets prose + a raw
  `model_json_schema()` dump (BOM: 7.4k chars of `$defs`/`anyOf`) and no concrete instance —
  the single best lever for a mid-tier model (deepseek-v4-flash) on nested-optional shapes.
- **7.2 De-duplicate `bom.md` vs `_stage_extra("bom")`** (~90 near-identical lines paid on every
  attempt of the costliest stage; also removes reconciliation ambiguity like "in stock" vs the
  quantified 100-unit floor). Keep the quantified tool-aware version in `_stage_extra`.
- **7.3 Add proactive guidance for gates the model only discovers via bounces:** §9.29 strap pins
  (RP2040 BOOTSEL / ESP32 BOOT+EN — shipped today with no prompt text), §9.31 repeated-block
  coverage ("wire EVERY identical connector"), §9.25 polarized caps (`Device:CP` never mentioned),
  §9.32 regulator feedback-divider math. Each is one or two sentences in the owning stage spec.
- **7.4 Disclose the hard 6-round tool budget** (`_BOM_MAX_ROUNDS`) in the EFFICIENCY bullet so the
  model front-loads batched lookups.
- **7.5 Give wiring the COMPACT OUTPUT instruction BOM already has** — wiring's payload scales
  with the same arrays but has a *lower* token floor (8192 vs 16384) and the same
  truncation-death mode.
- **7.6 Fix the expert-fallback question in `bom.md:104-113`** that tells a web user to run
  `kicraft add-part …` CLI commands they don't have.
- **7.7 Anchor the review/judge prompts numerically**: add "do not compute or assert a numeric
  value not given verbatim in the digest" to both `electrical_review.py` and `eval/judge.py`
  `_SYSTEM` prompts — the Vref hallucination was patched for regulators only; the checklist still
  invites unanchored thermal/current/divider claims.
- **7.8 Sheet-name punctuation**: one line in `architecture.md` ("no hyphens — use a space")
  saves the observed retry.

---

## 8. P1 — eval/tuning integrity (these distort what we choose to fix)

- **8.1 `interaction_friction` (weight 6) is a hard-coded constant on the web/self-eval path** —
  `eval/metrics_web.py:164-165` pins `expected_question_band=None` and `perm.excess=0`, so
  `score_friction` (`eval/scoring.py:139-169`) always returns 3; the claimed Class-J compensation
  doesn't exist. 6% of every grade is dead weight. *Fix:* real band heuristic or a Class-J
  question-appropriateness item + stop scoring the axis on this path.
- **8.2 Tuning eval cache omits `quality` from its key** (`tuning/store.py:41`, `runner.py:58`) —
  cross-quality cache poisoning of the CMA-ES objective via the shared default `tuning.db`.
  *Fix:* add `quality` to the PK and `lookup()`.
- **8.3 `REF_DRC` calibration is stale** (`tuning/reward.py:34-40` tuned against
  `MISSING_BOARD_PENALTY=999`, now 100). *Fix:* recompute.
- **8.4 Judge JSON parsing has no direct unit tests** (`eval/judge.py`) — table-driven
  malformed-reply fixtures.

---

## 9. P2 — server hygiene, security, UX

- **9.1 Stale red "failed" pill — root cause found** (known quirk): `open_project` paints correct
  statuses from DB truth, then the event replay pushes the stale persisted `build_done{ok:false}`
  through `StageTabs.push` → `_finish` repaints failed (`web.py:5062-5081`, :5452-5466,
  `stagetabs.py:844-857`). *Fix:* filter terminal events from replay (or re-apply
  `set_statuses` after it).
- **9.2 Design-quota TOCTOU** — `start()` check-then-insert unlocked (`web.py:5190-5196`);
  `enqueue_build` already shows the `BEGIN IMMEDIATE` pattern. *Fix:* atomic `try_create_project`.
- **9.3 `_flip` visibility toggle re-checks tier but not ownership** (`web.py:3017-3031`;
  `set_visibility` has no user scoping). Defense-in-depth: re-check `p.user_id == u.id`.
- **9.4 Capability tokens never expire** (`render_serving.py:74-96`) — add an `exp` to the signed
  payload; pages already re-mint per load.
- **9.5 `events.jsonl` unbounded growth + full rewrite per save** — coalesce per-run deltas
  (mirror `stagetabs._Run.buf`).
- **9.6 Orphaned project dirs on crash mid-delete** — DB row first, `rmtree` second, nothing ever
  sweeps (`accounts.py:1588-1625`). Piggyback an hourly sweep on `_orphan_reaper`.
- **9.7 `rules_panel._apply` doesn't re-check the tier gate at commit** (sibling panel does).
- **9.8 Manual-layout redo restores stale state** — `applyWithHistory` never clears `future`
  (`layout_canvas.js:354-360`); one-line fix.
- **9.9 DRC-overlay diagnostic is dead on the live host** — `render_drc_overlay.py` composites
  via `magick` (absent in prod) with no PIL/cairosvg fallback; reachable from real build
  diagnostics. Give it the `pcb_renderer.py` fallback.
- **9.10 Loud-drop consistency**: `library.py` loaders, `silk_plan._switch_positions`,
  `electrical_review._pin_names`, `_parent_stamp_subprocess._resolve_net` (floating copper with
  netcode 0), `solve_subcircuits` stale-outline fallback (:1297-1316), and the leaf
  `routing_exception` catch-all (no traceback; can abort the whole search as "structurally
  unroutable", `solve_subcircuits.py:589-613` + `autoexperiment.py:186-194`) — all swallow real
  errors silently. One sweep: narrow the excepts, log with context.
- **9.11 Subprocess robustness**: `export_fab` gerber/drill/CPL calls have no timeout
  (`fab_export.py:111-127` — the KC-Y3V9XU shape); `solve_hierarchy.py` timeouts don't kill the
  process group (FreeRouting jar orphans); `_run_command`'s `timeout_s` is declared but never
  passed (wire it or delete it).

---

## 10. P2 — performance (siblings of the fixed V1 quadratic hang)

- **10.1 `extract_leaf_blocker_set` recomputed uncached** at 3 call sites × K candidates
  (pcbnew load + up to 400k-cell grid scan; depends only on the artifact). Cache per artifact path.
- **10.2 GND repair round-trips the whole board file per single tie**
  (`add_breakout_stubs` LoadBoard/Save per call × ≤10 ties × ≤5 iterations) — batch the specs
  (the API already accepts a list) or thread a live `BOARD`.
- **10.3 `_collect_net_clusters` O(n²) with no bbox prefilter** (`gnd_pour.py:292-415`) — on the
  #1 rc7-breadth path; mirror the compose-scorer envelope prefilter.
- **10.4 Winner candidate's stamped board deleted, then re-stamped from scratch**
  (`compose_subcircuits.py:3247-3298` rmtrees the file `_compose_stamp.py` then recreates).
  Move it to the canonical path.

---

## 11. Also found (small but real)

`(extends)` merge drops derived-only properties (439 stock cases); `find_part` falls through a
hash-broken high-priority tier to a stale lower tier (dormant — all 96 vendored bundles currently
clean); `ensure_courtyard_clears_pads` doesn't run at the stamp seam for hand-authored project-tier
overrides; `diff_rounds.py` reads a schema `_write_round_detail` no longer writes (the tool always
prints "no changes"); `symbol_pinout` lru_cache has no mtime key (stale pins after re-vendor in the
long-running web process); `lookup_footprint.pad_count` counts NPTH pads (hides the 4.2 trap from
the BOM model); `_align_large_pairs` axis choice contradicts its own comment (confirm intent);
canonical `solved_layout.json` is clobbered every outer leaf round with monotonicity restored only
at phase end (watchdog kill mid-phase ships the *last* round, not the best); `PlacementBoard`
width/height can be half-specified and silently dropped; leaf_library's promoted-leaf reuse is
never consulted on the build path (confirms dead-code doc Wave 3.3); `check_netlist_faithfulness`
ignores kicad-cli's return code.

**Test-coverage gaps worth closing with the fixes:** tuning store cache-key surface (how 8.2
shipped), `security/scans.py` parsers (zero tests), judge JSON parsing, the §9.27
empty-pad-set path, round-loop acceptance on courtyard/timeout DRC results.

---

## 12. Suggested sequencing

0. **Step-0 triage — DONE 2026-07-19**: cold full-build replays of 623/628/639/636 on HEAD
   (scratch copies, quality=good, seed 0). **All four failure classes reproduced** —
   623 `courtyards_overlap`; 628 `illegal_routed_geometry`+`courtyards_overlap`;
   639 `connector_stranded:J2@-1.37mm`; 636 unconnected=7+sprawl. Courtyard is NOT fixed by
   T1-T3; §2/§3 keep full priority. These four scratch replays double as the before/after
   fixtures for PR-A/PR-B: re-replay the same four after each PR and compare class-level
   verdicts (not raw DRC counts — those vary run to run).
1. **PR-A (round-loop honesty) — IMPLEMENTED + REPLAY-VERIFIED 2026-07-19** (2.1–2.4 + 2.7;
   2.6 outline-containment split out to PR-B, it needs its own arc/circle geometry + tests).
   Post-fix replays of the Step-0 fixture set: **623 flipped rc7 → rc0 FAB-READY**
   (courtyard=0; rounds honestly `tier=functional parent_route=ok`); 628 same rc7 defects but
   rounds now honestly `tier=routed_dirty [discard]` (producer fix = PR-B 3.9 scaffold
   body_center); 639 same rc7 strand (owning fix = known perp-mouth follow-up); 636 same rc7
   unconnected class (C1 family), single-run geometry noticeably less sprawled (aspect
   3.92 → 1.44 — single replay, treat as anecdote not evidence). No new test failures
   (2779 passed; the 8 reds are pre-existing, stash-verified on clean HEAD). Implementation
   notes vs the plan: 2.1 mirrors the verify gate's minor/gross courtyard severity split
   exactly (minor clips stay warnings; unmeasured is conservatively blocking) and
   `_promotable_strand_only` now promotes courtyard-only rejects for inspection; 2.2 factors
   `_routed_dirty_score` shared by both routed_dirty entry points; 2.3 = `drc_failed`
   rejection in `validate_routed_board` (rc≠0 with no violations) + candidate-search
   `stamp_drc_unreliable` gate (missing_cli stays tolerated at candidate level — every
   candidate equally blind, final gate still rejects); 2.4 via the objective stamped-DRC
   courtyard count (score penalty + `_winner_key` hard preference) rather than the envelope
   conflict lists, which `_stamp_parent_board` overwrites in the candidate loop.
2. **PR-B (leaf/compose geometry)**: 3.1, 3.2, 3.5, 3.6, 3.9 (+3.3 after intent check); replay the
   same set + one shield brief.
3. **PR-C (crash/robustness)**: 4.1 (stream retry), 4.2 (§9.27 + PadBindingError), 4.6, 4.7 —
   small, independent, ship fast.
4. **PR-D (pin-data seam)**: 4.3 + 4.4 + 6.1 together — one consistent, string-aware, normalized,
   multi-unit-complete pin view for every consumer; then 4.5 loud drops.
5. **PR-E (retry eliminations)**: everything in §5 — mostly prompt/data one-liners plus the
   reconcile fix 5.8; measure by retry counts in the next self-eval batch.
6. **PR-F (prompt overhaul)**: §7; A/B a few briefs before/after.
7. **PR-G (eval integrity)**: §8 before the next self-eval batch so the batch grades honestly.
8. §9/§10 opportunistically behind the above.

Expected effect: §2+§3 target the current dominant failure fingerprint directly (courtyard/
unconnected/illegal-geometry at the terminal gate after 3 blind rounds); §5 removes the most
common retry burns observed in the traces (44% of recent runs on 5.1 alone); §8 makes the next
self-eval batch trustworthy as the measuring stick for all of it.
