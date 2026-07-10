# Handoff — turn on the standard-form-factor feature + rerun the Arduino shield brief end-to-end

**Goal for this session:** flip the standard-form-factor feature ON, run the Arduino-Uno
prototyping-shield brief from scratch (synthesis → build), and read the honest result. The
electrical "replace & rewire" half is built and gated; this session validates it on a **real build**
(the ERC-correctness that unit tests can't prove) and then finishes the one remaining piece
(mechanical placement of the standard headers at their fixed positions).

Everything is on branch `placement-streamline`, committed `ab89e0f … 3e82281`. Nothing is pushed.

---

## 0. TL;DR — the one command

```bash
cd ~/KiCraft
OUT="logs/form_factor_dogfood/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
KICRAFT_FORM_FACTOR_ENFORCE=1 .venv/bin/python -m kicraft.eval.self_eval \
    --only proto-shield --out "$OUT" --parallel 1 --no-judge > "$OUT/run.log" 2>&1
```

`--only proto-shield` targets exactly the Arduino shield brief (`benchmark.py` slug `proto-shield`:
*"An Arduino-Uno-format prototyping shield with stacking through-hole headers and an onboard SMT
3.3 V regulator."*). This runs the **full pipeline with the real LLM ($)** — synthesis (all 5
stages, auto-answering questions) then place/route/verify. `--no-judge` skips the LLM grader (we
only need the build verdict). The env prefix reaches **both** halves: the synthesis stage-commit is
a subprocess spawned by `stage_driver._run`, which passes `env={**os.environ, …}`, and the build
subprocess inherits `os.environ` via `self_eval._build_env` — so one env prefix on the launching
shell covers the whole run.

The run dir is `$OUT/<slug>__.../` (a normal project tree: `.kicraft/state.json`,
`generated/<stem>/`, `.kicraft/build.log`).

---

## 1. What is already built (so you know what you're turning on)

| Piece | State | Where |
|---|---|---|
| Brief → `intent.form_factor.standard = "arduino_uno_shield"` | done | `design/synthesis/form_factor.py`, registry `kicraft/form_factors/__init__.py` |
| Validated Arduino R3 datum (real coords, 0.16″ offset) | done | `form_factors/__init__.py` (`validated=True`) |
| **Electrical reconcile** — replace LLM stacking headers with the standard's as real BOM parts, bind power, signal pins no-connect | done, **env-gated** | `form_factors/reconcile.py`, wired in `design/cli_app.py` at the wiring stage-commit |
| Mechanical-conformance check (geometry, not net names) | done | `form_factors/conformance.py` + investigate §8.5 |
| Compose fixed-outline + locked-connector fork | done, **cfg-gated, dormant** | `form_factors/compose_scaffold.py` + `cli/compose_subcircuits.py` |
| **Mechanical placement of the real headers at fixed positions** | **NOT done** | this session |

**Master switch:** `KICRAFT_FORM_FACTOR_ENFORCE` (env; default off) turns on the electrical
reconcile. The compose fork is a *separate* cfg gate (`cfg["form_factor_enforce"]`) that nothing
sets yet — so with only the env on, **compose does NOT inject or pin anything** (no duplication).

**What the env-on run produces:** a shield with the correct **schematic/BOM** (4 standard Arduino
headers replacing the LLM's free ones, power bound, signals no-connect) but a **free-placed layout**
(headers wherever the solver puts them). That is the expected Phase-1 intermediate — it isolates
the ERC question from the placement question.

---

## 2. Phase 1 — run it & validate the electrical half (do this first)

Run the TL;DR command. Then inspect (`RUN=$OUT/<the slug dir>`):

**(a) Did the reconcile fire?** `grep form_factor "$RUN/events.jsonl"` — expect a
`form_factor replaced N LLM stacking connector(s) … added 4 arduino_uno_shield header(s)` note.

**(b) Is the BOM the standard interface?**
```bash
.venv/bin/python - "$RUN" <<'PY'
import json,sys; from pathlib import Path
st=json.loads((Path(sys.argv[1])/".kicraft"/"state.json").read_text())
for p in st["bom"]["parts"]:
    print(p["ref"], p["value"], "|", (p.get("sourcing_note") or "")[:40])
PY
```
Expect: the regulator + caps + exactly **4** parts marked `standard form factor: arduino_uno_shield`
(roles digital_high/digital_low/power/analog); **no** generic `Conn_01x08` LLM headers left.

**(c) THE KEY CHECK — did ERC stay clean?** The reconcile changes the netlist right before emit +
ERC, so this is the real validation:
```bash
cat "$RUN"/generated/*/*_erc.rpt | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); \
print('ERC errors:', sum(1 for s in d['sheets'] for v in s['violations'] if v['severity']=='error'))"
# or just: /kicraft-investigate <the run dir>
```
- **ERC = 0 errors → the electrical reconcile is validated. Proceed to Phase 2.**
- **ERC errors → this is the finding to fix.** Likely suspects and fixes:
  - `power_pin_not_driven` on +5V/VIN: the Arduino *supplies* 5V, but on this board nothing drives
    it → the header's +5V pin needs a PWR_FLAG. Extend the reconcile to add a PWR_FLAG (or mark the
    rail externally-driven) for standard power pins. See `emitter.py:_power_nets_with_driver` and the
    PWR_FLAG assigner.
  - `pin_not_connected` on signal pins: the no-connect endpoints should suppress this; if they don't,
    check the emitter honors `bom.no_connect_pins` for connector pins (it should emit `(no_connect)`).
  - A `§9.x` synthesis-check gate (net coverage / cohesion) tripping on the rewired netlist: read
    `.kicraft/synthesis_check.json`; decide whether the gate or the reconcile is wrong (fix at source).

**(d) Build verdict:** `tail -20 "$RUN/.kicraft/build.log"`. Expect it to reach place/route. It will
**not** be mechanically conformant yet (free-placed) — confirm with the conformance check:
```bash
RB=$(find "$RUN" -name parent_routed.kicad_pcb | sort | tail -1)
[ -n "$RB" ] && .venv/bin/python - "$RB" <<'PY'
import sys; from kicraft.form_factors import get_template
from kicraft.form_factors.conformance import board_local_pads, check_conformance
pads,wh=board_local_pads(sys.argv[1])
print(check_conformance(get_template("arduino_uno_shield"), pads, wh).summary())
PY
```
Expect `NON-CONFORMANT` in Phase 1 (headers not at fixed positions). That's the cue for Phase 2.

**Baseline compare (optional):** rerun the same command **without** the env prefix → the old
behavior (10 free headers, free size). Confirms the feature is the only difference.

---

## 3. Phase 2 — finish the mechanical placement (make it conformant)

The blocker: the reconcile puts the 4 headers on a normal sheet, so they become a **leaf** (placed
as a rigid unit). The compose Half-1 fork instead *injects synthetic* connectors — enabling both
would duplicate refs. Compose must **place the real header parts** at the template positions.

Two candidate approaches (investigate the compose flow first — prior mapping is in the plan and the
two Explore findings referenced there):

- **(A) Fixed-layout interface leaf, pinned at the parent origin (recommended).** Make the header
  sheet's leaf lay its 4 headers out at the exact template coordinates (a leaf whose internal
  placement is forced/locked), then pin that leaf at parent `(0,0)` with `enable_board_size_search`
  off and the parent outline set to the template rect. Reuses the leaf machinery; no parent-local
  surgery. Seam: `solve_subcircuits` (force a leaf's component positions) + `compose_subcircuits`
  (pin one leaf + set outline — the Half-1 outline-pinning code at `_ff_scaffold.outline` already
  exists, just drive it from the real headers).
- **(B) Headers as parent-local.** Keep them out of any leaf (via `component_zones` extraction so
  `extract_parent_local_components` picks them up), then reuse the committed Half-1 lock
  (`compose_scaffold.build_scaffold`) but locking the **real** refs instead of injecting synthetic
  ones. Needs confidence that a `component_zones` part on a sheet is excluded from its leaf.

Whichever: **coordinate the refs** (compose must lock the SAME refs the reconcile emitted — read them
from `state.bom` parts whose `sourcing_note` carries `standard form factor:`), and drive compose from
the env master switch (set `cfg["form_factor_enforce"]` in `write_autoplacer_json` when
`reconcile.enforce_enabled()`), so both halves turn on together.

**Validate Phase 2** by rerunning the TL;DR command and re-checking:
- conformance check (§2d) → **CONFORMANT** (32/32 pins, outline 68.58×53.34),
- `inspect_parent` DRC (`python -m kicraft.cli.inspect_parent "$RB" --output-dir /tmp/x`) → 0 shorts,
  0 unconnected (the headers' no-connect signal pins shouldn't count as unconnected),
- eyeball the board: 4 headers on the edges at Arduino spacing, regulator/caps inside.

Then flip **PR3**: run `check_conformance` at promote and fail/downgrade a non-conformant shield
(report-only until here). Owner: `conformance.py` is ready; wire it into the promote/verify gate.

---

## 4. Safety / rollback

- The whole feature is **off unless `KICRAFT_FORM_FACTOR_ENFORCE` is set** — no normal build is
  affected. `unset KICRAFT_FORM_FACTOR_ENFORCE` (or leave it unset) fully disables it.
- For a **web** run instead of the CLI harness: set the env in the web app's environment **and** the
  build worker's (`.env` / systemd unit), then restart **both** (`deploy/restart-web.sh` +
  `deploy/restart-build-worker.sh`), submit the brief in the UI. The reconcile is in synthesis (web
  app) and compose is in the build worker, so both need the var.
- Branch has **6 pre-existing `test_parent_outline_repair.py` failures** (`_compose_validate.py:339`,
  present on clean HEAD) — unrelated to this feature; don't be alarmed.

---

## 5. Reference

- Plan + full status: `docs/plans/standard-form-factor-templates.md`.
- Registry/datum: `kicraft/form_factors/__init__.py`; reconcile: `reconcile.py` + `synthesis.py`;
  compose fork: `compose_scaffold.py` + `cli/compose_subcircuits.py` (search `_ff_scaffold`);
  conformance: `conformance.py`; wiring point: `design/cli_app.py` (search `form_factor`).
- Tests (all green, gate off): `tests/test_form_factor*.py` (95 tests).
- Commits this line of work: `ab89e0f`(GAP1) `3f12c73`(registry) `7fa4ee6`(datum) `e725de8`(emit)
  `ad33434`(conformance) `590c6fa`(§8.5) `b7393b0`(compose Half 1) `dddb49a`+`e6ca872`(Half 2 electrical).
- Investigate any failure with `/kicraft-investigate <run dir>` — its §8.5 audit already reports
  form-factor conformance.
