> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Fix: FreeRouting re-route hang on its own copper + useless rc6 failure card

Date: 2026-08-11. Filed from the KC-Z879KB (run 1/695) investigation.
Two independent fixes. Fix 1 is the fab-blocking one; Fix 2 is why the user
could not tell what had failed.

## Plain-language background

KiCraft does not route boards itself. It places components, draws the copper
inside each circuit block ("leaf"), then hands the whole board to an external
autorouter program, FreeRouting 1.9, to connect the blocks. Hand-over format
is a text file called a DSN; results come back as a SES file.

Two things about that hand-over matter here:

1. **Two-pass routing ("power-first").** FreeRouting has no notion of "route
   the fat power wires first", so KiCraft runs it twice: pass 1 routes only
   the power nets, pass 2 routes everything else with pass 1's wires marked
   as fixed ("locked") so they are not moved.
2. **FreeRouting 1.9 has a parser bug: it freezes forever if a net's locked
   wires form a closed geometric loop** (it prints
   `The normalization of net 'X' failed.` and then makes no progress until
   KiCraft's 120-second timer kills it, which shows up as `rc=-1` and no SES
   output). A closed loop is legitimate copper on a real board — a net may
   branch and rejoin — so before each run KiCraft edits the DSN copy only:
   it finds each loop and snips one wire a hair short, opening the loop. The
   real `.kicad_pcb` keeps its continuous copper; only FreeRouting's view
   gets the gap. That sanitizer is `_break_locked_wire_cycles` in
   `kicraft/autoplacer/freerouting_runner.py`, added 2026-07-27 for the
   LED-ring board.

### The failure this plan fixes

The loop detector builds its graph from wire **endpoints only** (rounded to
0.1 DSN units). Pass 1's own output defeats that model two ways at once:

- **T-junctions.** A wire can start in the *middle* of another wire's segment
  (a branch point), not at its end. The detector never looks at segment
  interiors, so it never registers the branch as a graph node.
- **Float drift.** On the round trip pass 1 → board file → pass 2, coordinates
  shift by sub-micron amounts (0.2 µm observed), so two points that are the
  same junction on the board compare as different endpoints.

Result: a hair-thin loop — branch point → short stub → almost-the-same point
→ back along the original segment — is invisible to the detector, FreeRouting
freezes on it, every retry and every search round burns its full 120 s
timeout, and the build dies with "no routed parent board" (rc6).

In KC-Z879KB the loop was on net VOUT_2 at DSN coordinates
(151.82, −97.03) mm, with a mirror-image copy on VOUT_1 (the second USB-A
port is a replica of the first). Both came from pass 1's own routing of the
power nets.

**Proof this is the whole story** (already run, on current HEAD `7152634e`):
exporting the failed run's composed-parent board to DSN and running
FreeRouting reproduces the freeze (`normalization of net 'VOUT_2' failed.`,
killed at timeout). Hand-opening the two sliver loops with the detector's own
snip trick makes FreeRouting route the entire board in **3 seconds**.

**Breadth:** 6 builds since 2026-07-28 show the same fingerprint (power-first
pass succeeded, main pass killed at timeout, no SES): project runs 1/667,
1/675, 1/678, 1/680, 1/693, 1/695. This is a floor, not a count — the
evidence line is only preserved for some runs (see Fix 2).

---

## Fix 1 — teach the loop detector about T-junctions and float drift

**File:** `kicraft/autoplacer/freerouting_runner.py`, function
`_break_locked_wire_cycles` (~line 1184). Fix at this point only — the
`.kicad_pcb` must keep its real copper, and no downstream workaround
(re-rolls, longer timeouts, seed changes) can ever help: the loop is a
geometry constant of the input, so every attempt freezes identically.

### What to change

Today the function reads each `(wire (path ...))` DSN entry, unions its first
and last point in a union-find, and "opens" any wire whose endpoints are
already in the same set (that wire closes a loop) by retreating its final
point along its last segment. Replace the endpoint-only graph with a
segment-aware one:

1. **Snap near-coincident points.** When registering a point, merge it with
   any already-registered point on the same net within a small tolerance
   (suggestion: 1 µm in DSN units; the observed drift was 0.2 µm — keep it
   far below the 10%-of-width snip gap so the snip still opens loops).
2. **Split segments at T-junctions.** For every wire endpoint, test whether
   it lies on the interior of any other same-net wire segment (point-to-
   segment distance below the same tolerance, and strictly between the
   segment's ends). When it does, treat that segment as two graph edges that
   meet at the junction point — either by literally splitting the polyline in
   the in-memory graph, or by inserting the junction as a shared node. Only
   the graph needs the split; the DSN text does not.
3. With snapping + splitting in place, the existing union-find sees the
   sliver loop as a cycle and the existing retreat-snip opens it. Keep the
   current snip mechanics (gap = min(max(width/10, 1% of segment), half the
   segment)) and the zero-length-wire drop unchanged.

Mind the existing details: net names may be quoted; multi-point polylines
union first-to-last today but interior vertices already act as path nodes, so
build the graph per *segment*, not per wire; keep the whole function
best-effort (any parse trouble leaves the DSN unchanged) — it must never
break routing.

### Tests

- New fixture next to `tests/data/fr_hang_5v_loop.dsn`: a DSN whose locked
  wiring contains the KC-Z879KB pattern — wire A is a polyline J→N→K, wire B
  is N→K′ where K′ is K + 0.2 µm — asserting
  `_break_locked_wire_cycles` reports ≥1 opened wire and that the resulting
  graph is acyclic. (Constructing it from the real run: the two wire entries
  are quoted in the investigation; VOUT_2 entry
  `(wire (path F.Cu 500  151819 -97032.7  152479 -97692.3)(net VOUT_2)(type fix))`
  with the branch segment `(151159 -96372.9) → (152479 -97692.5)` it lands
  on.)
- Extend `tests/test_dsn_wire_cycle_guard.py` rather than adding a parallel
  file; keep the existing exact-endpoint cases passing (regression cover for
  the LED-ring loop).
- Integration proof (manual, do not automate against the jar in CI): on the
  fixture, FreeRouting 1.9 reaches `Auto-routing was completed` instead of
  printing `normalization of net ... failed`.

### End-to-end verification

Replay the failed run (frozen seed, no LLM, ~10 min):

```bash
PY=/home/kicraft/KiCraft/.venv/bin/python
SRC=/home/kicraft/.kicraft/projects/1/695/generated/USB_C_HOST_POWER_SPLITTER
WORK=$(mktemp -d); cp -a "$SRC" "$WORK/replay"
# keep .experiments/pre_promote_seed.kicad_pcb — rc6 replay seed; the promoted
# board in the run dir is an unrouted preview, not evidence of anything
"$PY" -m kicraft.design.cli_app replay --project "$WORK/replay" --quality good --seed 0
"$PY" -m kicraft.design.cli_app artifacts --project "$WORK/replay"
```

Expected: the parent route produces a SES within seconds per attempt (no
120 s timeouts), a routed parent board exists, and the run leaves the rc6
family (rc0, or an honest rc7 with specific DRC violations). Also re-check
the other five runs' composed parents the same way if cheap. Success metric
for the whole fix: parent attempts stop dying at exactly the timeout with
rc=-1 on these inputs.

---

## Fix 2 — make the rc6 failure card name the real cause (and stop inventing fake ones)

**Why the user saw nothing useful.** When a build fails, KiCraft persists a
structured diagnostic (`pcb_errors` in `.kicraft/state.json`, built by
`build_pcb_errors` in `kicraft/design/cli_app.py`, ~line 4560; the web UI
renders it as the failure card). For the "route produced no board at all"
case it takes a generic fallback that:

- fills `details` with 120-character slices of raw log tails — in KC-Z879KB
  these were window-manager debug spam (`Adding duplicate image handler ...`)
  and a timing-JSON fragment, cut mid-word;
- attaches whatever DRC violations were last recorded against *any* board —
  here two warn-only silkscreen clips blamed on LED `D1` at coordinates
  (145, 118) mm, physically off this 24×59 mm board (they belonged to a
  different board frame), with an `x_mm`/`y_mm` that look like "the location
  of the problem";
- sets `overlay_path: null`, because the board-image error overlay is drawn
  only from located violations of the failing board — so there is no visual
  indication at all, and the canvas shows the unrouted preview board with
  nothing marking it as unrouted;
- suggests "spread the blocks, and retry place/route" — placement was never
  the problem.

Meanwhile the one sentence that explains everything — FreeRouting's
`The normalization of net 'VOUT_2' failed.` — was captured and thrown away:
`parse_freerouting_output` (same file as Fix 1, ~line 1275) keeps only the
**first** 2000 characters of FreeRouting's output, and the timeout path
(`run_freerouting`, ~line 1418) returns it into a stats dict that the
exception path never reads. The raised error says only
`FreeRouting produced no SES output after 2 attempts (rc=-1)`.

### What to change (three small edits, two files)

1. **`freerouting_runner.py` — keep the end of the output, name the hang.**
   In `parse_freerouting_output`, keep the **last** 2000 characters of
   stdout/stderr, not the first (on a freeze the banner is worthless; the
   final lines are the diagnosis). In the timeout branch of
   `run_freerouting` and the final `RuntimeError` in
   `route_with_freerouting`, include FreeRouting's last non-empty output line
   in the message, e.g.
   `FreeRouting hung and was killed at the 120 s timeout; its last output was: "The normalization of net 'VOUT_2' failed."`
2. **`cli_app.py` `build_pcb_errors` fallback — extract signatures, don't
   slice logs.** Before falling back to raw log fragments, scan the round
   logs for the known FreeRouting failure signatures (the table already
   exists in `kicraft/cli/triage.py` — `FR round signatures`; factor it into
   a shared helper rather than duplicating the regexes) and write a specific
   `explanation`, e.g. "The autorouter froze while reading the locked power
   wiring on net VOUT_2 and was killed at the timeout. This is a pipeline
   bug, not something wrong with the design." Only if no signature matches,
   keep the log slices — but strip noise lines (the
   `Adding duplicate image handler` class) and never cut mid-word.
3. **`cli_app.py` — never attach foreign or warn-only violations to a failure
   card.** In the fallback path, drop violation types that cannot fail a
   build (`silk_*`, minor courtyard clips) and drop any violation whose
   coordinates fall outside the failing board's outline — both are proof the
   violation came from a different board frame. If that empties the list, the
   card correctly shows "no board location exists for this failure" instead
   of a fake one.

Out of scope (note for later, do not build now): watermarking the unrouted
preview in the web canvas. The preview intentionally ships so the user can
inspect placement; the card from step 2 is the designated place to say it is
unrouted.

### Tests

- `build_pcb_errors` fed a summary whose evidence tails contain the
  freeze-then-killed sequence produces: explanation naming the net and the
  timeout, empty `violations`, empty `footprint_refs`, `overlay_path=None`,
  and a `next_action` that does not blame placement.
- A summary containing only warn-only silk violations produces a card with
  zero violations attached.
- `parse_freerouting_output` on a >2000-character stdout keeps the tail.
- Match existing conventions in `tests/test_build_queue.py` /
  the cli_app diagnostic tests; deterministic, no FreeRouting invocation.

### End-to-end verification

Re-run the parent phase of run 1/695 with Fix 1 *not yet applied* (or the
detector kill switch if one exists) so the freeze still happens: the
persisted `pcb_errors` must name the net and the watchdog kill, carry no silk
violations, and the build log's final error line must not point at
"rejected candidates" when the candidate search accepted everything (8/8
here — consider deleting or rewording that sentence in
`kicraft/design/cli_app.py`'s rc6 message: it sent the user to an empty
directory).

---

## Sequencing and guardrails

- Land Fix 1 and Fix 2 as separate commits; Fix 2 first is fine (it makes the
  next occurrence of any routing-freeze class self-explaining, which pays off
  even beyond this bug).
- Do not "fix" this by raising the timeout, retrying more, or adding a
  post-route repair pass — the loop is deterministic input geometry; only
  opening it in the DSN works. Masking gates and downstream band-aids have
  been rejected on principle in this codebase; the DSN sanitizer is the
  designated boundary for FreeRouting-1.9 quirks.
- Do not touch the `.kicad_pcb` copper: the snip is DSN-only by design
  (`import_ses` merges results onto the original board).
- FreeRouting 2.x is not a substitute fix: the runner pins 1.9 because 2.1
  regresses `max_passes` handling (see the note at the top of
  `freerouting_runner.py`).

## Reference material

- Failed run: `/home/kicraft/.kicraft/projects/1/695` (board KC-Z879KB),
  composed parent entering the frozen pass:
  `.experiments/subcircuits/subcircuit__8a5edab282/parent_power_routed.kicad_pcb`
- Prior art: `tests/data/fr_hang_5v_loop.dsn`,
  `tests/test_dsn_wire_cycle_guard.py`, memory note
  `kicraft-freerouting-non-ansi-dsn-hang` (the sibling sanitizer for the
  other FreeRouting freeze).
- Triage tooling: `python -m kicraft.cli.triage run KC-Z879KB` reproduces the
  verdict this plan was written from.
