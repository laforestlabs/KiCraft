# Handoff: ESP32 antenna keepout, silkscreen refdes, edge-mount connectors

Self-contained implementation brief for another agent. Source of truth for the
original design is the approved plan at
`~/.claude/plans/velvet-snuggling-sutherland.md`; this file folds in what was learned
during a partial Fix 0 attempt and records the exact current repo state.

## Current repo state (verified clean)

- Branch `fix/footprint-keepouts-and-placement` is correctly based on `main` (PR #9
  route-cache merge `974acd2` is in history). A partial Fix 0a edit was made and then
  **reverted**, so the working tree is clean — `git status` shows only this untracked
  handoff doc. Nothing is committed yet; no work is half-applied.
- You can implement directly on this branch (or cut a fresh one). Nothing to salvage
  or untangle.

Repo facts:
- Use `.venv/bin/python` for everything (has `pcbnew` 9.0.9 + `easyeda2kicad`).
- pcbnew quirk: `Footprint.Zones()` returns a `ZONES` object that is **iterable** but
  has **no `.GetCount()`** — use `list(fp.Zones())` / `len(list(fp.Zones()))`.
- Commit style: no Claude attribution. PR via Codeberg/Forgejo `curl --netrc`
  (origin `git@codeberg.org:LaForestLabs/KiCraft.git`), merge commit + delete branch.

---

## Why (context)

A parent-board render of the esp32-led-matrix project surfaced three pipeline-level
defects (the user does not want this specific board patched — fix the pipeline so they
can't recur):

1. **ESP32 antenna keep-clear ignored.** The placer's only obstacle model is the
   footprint **courtyard bbox** (`adapter.py` `load()`); it has no keep-out concept
   except mounting holes. Compounded by the board using a vendored easyeda2kicad
   footprint (`WIRELM-SMD_ESP32-S3-WROOM-1`) that has an 18×25.5 courtyard and **no
   keepout zone** (the stock `RF_Module:ESP32-S3-WROOM-1` carries a ~48×21 keep-clear;
   the import dropped it). Measured in the MCU leaf: switches sit ~3.8 mm outside U1's
   courtyard, i.e. in the antenna near-field the keep-clear would protect. Secondary:
   courtyard overlap itself is best-effort (`_resolve_overlaps_bounded`,
   `placement_solver.py:560`, 200-iter clamped push), so cramped leaves can leave
   residual overlaps.
2. **Silkscreen refdes overlap.** No stage manages per-footprint reference
   designators; on a 1.5 mm LED array the default ~1 mm "D147" label overlaps
   neighbors. No hide/shrink/relocate pass exists.
3. **USB-C not at board edge.** Parent edge-pinning is fully built and the project's
   `autoplacer.json` already carries `{"J1": {"edge": "bottom"}}` — so this is NOT a
   missing hint. The existing edge constraint was not applied (the parent compose for
   that run was discarded). Root-cause lives in the parent compose path.

User-confirmed decisions:
- Silkscreen: hide refdes on array members / parts whose refdes overflows its
  courtyard (move them to F.Fab, keep for assembly); shrink-to-fit + nudge the rest.
- Edge-mount: deterministically auto-derive an edge `component_zone` for edge-mount
  connectors when the BOM didn't set one, AND tighten `bom.md`.
- Antenna keepout extent: footprint carries a **modest on-module strip** (parity with
  the sister WROOM-32E, avoids bloating placement/pour); the larger RF near-field
  clearance is enforced at placement time via a tunable config family spec (Fix 1).

---

## Fix 0 — Repair ESP32 library footprints (mostly done; redo cleanly)

Measured state of the two antenna-bearing footprints in `kicraft/parts_library/`:

| Footprint | Format | Courtyard covers antenna | Keepout |
|---|---|---|---|
| `esp32-s3-wroom-1/.../WIRELM-SMD_ESP32-S3-WROOM-1.kicad_mod` | **legacy `(module …)`** | yes (18×25.5, +7.5 mm past pads at −y end) | **none → ADD** |
| `esp32-wroom-32e-n4/.../WIFI-SMD_ESP32-WROOM-32E.kicad_mod` | modern `(footprint …)` v20241229 | yes (+7.53 mm at −x end) | exists but `footprints allowed` → **tighten** |

**Courtyards already cover the antenna on both — no courtyard change needed.** (The
user's request to "fix the courtyard" is, per measurement, already satisfied; only the
keep-out is the real gap. The library-invariant test below codifies it.)

**0a. WIRELM (ESP32-S3-WROOM-1): add antenna keep-out.** The file is legacy KiCad-5
`(module …)` format, which can't express modern keepout flags (`pads`/`footprints`).
The proven approach is to add the zone via **pcbnew** and `FootprintSave`, which
rewrites the footprint in modern format (clean ~113/−91 line diff; pads (49) and 3D
model are preserved). Working script (verified to add the zone before an unrelated
verify-line crash on `.GetCount()`):

```python
import os, pcbnew
LIB = "<repo>/kicraft/parts_library"
mm = pcbnew.FromMM
pretty = os.path.join(LIB, "esp32-s3-wroom-1", "esp32-s3-wroom-1.pretty")
fp = pcbnew.FootprintLoad(pretty, "WIRELM-SMD_ESP32-S3-WROOM-1")
for z in list(fp.Zones()):       # idempotent
    fp.Remove(z)
zone = pcbnew.ZONE(fp)
zone.SetIsRuleArea(True)
zone.SetDoNotAllowTracks(True); zone.SetDoNotAllowVias(True)
zone.SetDoNotAllowPads(True);   zone.SetDoNotAllowCopperPour(True)
zone.SetDoNotAllowFootprints(True)
ls = pcbnew.LSET(); ls.AddLayer(pcbnew.F_Cu); ls.AddLayer(pcbnew.B_Cu)
zone.SetLayerSet(ls)
try: zone.SetZoneName("antenna_keepout")
except Exception: pass
chain = pcbnew.SHAPE_LINE_CHAIN()
# on-module antenna strip: full module width, top edge to just above pads
for x, y in [(-9.0,-16.39),(9.0,-16.39),(9.0,-10.0),(-9.0,-10.0)]:
    chain.Append(mm(x), mm(y))
chain.SetClosed(True)
poly = pcbnew.SHAPE_POLY_SET(); poly.AddOutline(chain)
zone.SetOutline(poly); fp.Add(zone)
io = pcbnew.PCB_IO_MGR.PluginFind(pcbnew.PCB_IO_MGR.KICAD_SEXP)
io.FootprintSave(pretty, fp)
# verify with list(...), NOT .GetCount()
chk = pcbnew.FootprintLoad(pretty, "WIRELM-SMD_ESP32-S3-WROOM-1")
z = list(chk.Zones())[0]
assert z.GetIsRuleArea() and z.GetDoNotAllowFootprints() and z.GetDoNotAllowCopperPour()
```

(Alternative if a minimal text diff is preferred over pcbnew reformatting: the modern
`(zone …)` s-expr can be hand-written, but the file must then be migrated to
`(footprint …)` format — pcbnew is simpler and guaranteed valid.)

**0b. WROOM-32E: tighten flag.** Single text edit in the modern file — change the one
line inside its `(keepout …)` block from `(footprints allowed)` to
`(footprints not_allowed)`. (Optionally also widen its keepout, but parity is fine.)

**0c. Test — `tests/test_library_antenna_keepouts.py`** (invariant; guards future
re-vendored footprints). Discover every footprint in `kicraft/parts_library`
whose name matches `WROOM|WIFI|BLE|NRF|ESP32|ESP8266`; assert each has ≥1 rule-area
zone with `GetDoNotAllowCopperPour()` AND `GetDoNotAllowFootprints()` AND
`GetDoNotAllowTracks()`, and a non-empty `F.CrtYd` courtyard. Include a guard test
asserting discovery finds both known modules (so the parametrization can't vacuously
pass). `pytest.importorskip("pcbnew")`. This test passed in the partial run.

**0d. Sweep:** only the two ESP32 modules match today; the invariant test makes it
durable.

Note: already-generated projects pick up fixed footprints only on re-synth (existing
"re-synth for footprint changes" workflow). Do **not** re-run the esp32-led-matrix
board.

---

## Fix 1 — Antenna keep-clear as a first-class keepout (placer + router)

Reuse the existing mounting-hole keepout rail: `compose_subcircuits` builds
`parent_local_keep_in_rects` → stamped as F.Cu/B.Cu rule-areas
(`_parent_stamp_subprocess.py:198-211`) and fed to the placer via `parent_keep_in_rects`
cfg → `placement_solver._resolve_keep_in_rects` (`placement_solver.py:2252`).

**1a. `kicraft/autoplacer/hardware/keepout_extract.py` (new).** Given a loaded board,
return owner-tagged board-coord keepout rects from two sources:
- **Preserve:** footprint-internal rule-areas (`fp.Zones()` where `GetIsRuleArea()`
  and footprints/pads not allowed). Covers stock + library footprints after Fix 0.
- **Inject:** config-driven per-footprint-family keep-clear spec (`antenna_keepouts`
  in `config.py`, keyed by footprint-name pattern, e.g. `*ESP32-S3-WROOM-1*` →
  local-frame rect; default the generous ~48×21 stock geometry). When a placed
  footprint matches and has no internal keepout, synthesize the rect in local frame
  and transform to board coords by the footprint's placed position/rotation. This is
  the larger RF near-field clearance the placer enforces (kept out of the footprint so
  it doesn't bloat the .kicad_mod).

**1b. Feed the placer.** Add `BoardState.keepout_rects` (list of `{rect, owner_ref}`;
new small type) in `kicraft/autoplacer/brain/types.py`. Populate it in
`adapter.load()` (`hardware/adapter.py`, ~`390+`). In `compose_subcircuits`, derive
parent-frame rects from placed child footprints and append to
`parent_local_keep_in_rects`. Add `placement_solver._resolve_keepout_rects` (sibling of
`_resolve_keep_in_rects`) that pushes any **unlocked, non-owner** component courtyard
out of each rect (owner footprint exempt); call it in `solve()` beside the keep-in
pass (`placement_solver.py:1003-1010`). Count keepout overlaps in
`legality_diagnostics`.

**1c. Routing/pour survival.** Footprint-internal rule-areas travel with the
footprint; injected rects go into the stamped parent `keepouts` JSON (already stamps
rule-areas). Fix `freerouting_runner.clear_zones` (`freerouting_runner.py:240`) to
**preserve rule areas** like `adapter.py:689` does
(`[z for z in board.Zones() if not z.GetIsRuleArea()]`) so the keep-clear reaches the
DSN export. Confirm `gnd_pour` already skips rule-areas (comment at `gnd_pour.py:129`).

**1d. (Optional)** Surface unresolved residual courtyard overlaps from
`_resolve_overlaps_bounded` through `legality_diagnostics`/round JSON.

**Tests:** `test_keepout_extract.py` (preserve case + inject case + rotated transform);
`test_placement_keepout.py` (`solve()` pushes an unlocked part out of a keepout rect,
owner exempt, legality flags unresolved overlap); extend freerouting-runner tests
(`clear_zones` keeps rule-area zones, removes copper pours).

---

## Fix 2 — Silkscreen refdes legalization (geometric)

**`kicraft/autoplacer/hardware/silk_refdes.py` (new)**, operating on a loaded pcbnew
board; call at the end of parent stamping (`_parent_stamp_subprocess.py`) and leaf
stamping (`_stamp_subcircuit_subprocess.py`). For each footprint's reference text:
- Compute refdes text bbox vs the footprint courtyard bbox.
- **Hide on silk → move to F.Fab/B.Fab** (`ref.SetLayer(F_Fab/B_Fab)`, keeps it for
  assembly) when the refdes bbox exceeds its courtyard (with margin) or overlaps a
  neighbor's courtyard. This geometric rule captures dense array members without
  plumbing the `array_member` flag. (`SetVisible(False)` is the fallback.)
- **Else keep + fit:** shrink (`SetTextSize`/`SetTextThickness`) to fit the courtyard
  and nudge to the clearest adjacent side clear of this part's pads / neighbor
  courtyards (`SetPosition`).

**Test:** `test_silk_refdes.py` — tiny part w/ oversized refdes → moved to Fab; normal
part → kept/shrunk/repositioned clear.

---

## Fix 3 — Edge constraint: root-cause (primary) + deterministic fallback

**3a. Root-cause (primary).** `{J1: edge:bottom}` exists but wasn't honored. Trace
`component_zones` → `derive_attachment_constraints` (`subcircuit_composer.py:235+`) →
parent placement in `compose_subcircuits`. Determine whether (i) the parent compose
simply failed/was discarded (no constraint ran) or (ii) the constraint is derived but
the edge-attachment math / leaf-block placement doesn't move the connector's leaf to
the parent outline. Fix the broken link. Add `test_edge_constraint_applied.py`: a
connector with an `edge` zone composed into a parent lands its pad edge on the parent
outline within the configured inset/overhang — the regression that would have caught
the J1 float.

**3b. Deterministic fallback (defense-in-depth).** In `synthesis/autoplacer.py`
`write_autoplacer_json` (where `component_zones` is assembled, ~lines 53-81): for each
connector part whose footprint matches a curated edge-mount family
(`USB_C_Receptacle_*`, `USB_A_*`, `USB_B_*`, edge BarrelJack, known vendored
equivalents like `USB-C_SMD-TYPE-C-31-M-12`), if the ref isn't already in
`component_zones`, inject `{ "edge": <default_side> }` (configurable
`default_edge_connector_zone`, e.g. `"bottom"`). Pure name match; orientation handled
downstream by `detect_opening_direction` + `_best_rotation_for_edge`. Test:
`test_edge_zone_derivation.py`.

**3c.** Tighten `.claude/skills/circuitchat/stages/bom.md` `component_zones` bullet
from example to rule: any connector that mates off-board MUST get an `edge`/`corner`
zone; LLM may pick the edge, else synthesis defaults it.

---

## Files (summary)

- `kicraft/parts_library/esp32-s3-wroom-1/.../WIRELM-SMD_ESP32-S3-WROOM-1.kicad_mod`
  (add keepout, Fix 0a)
- `kicraft/parts_library/esp32-wroom-32e-n4/.../WIFI-SMD_ESP32-WROOM-32E.kicad_mod`
  (flag flip, Fix 0b)
- `kicraft/autoplacer/hardware/keepout_extract.py` (new), `…/silk_refdes.py` (new)
- `kicraft/autoplacer/brain/types.py` (`keepout_rects`),
  `…/hardware/adapter.py` (populate), `…/brain/placement_solver.py`
  (`_resolve_keepout_rects` + legality)
- `kicraft/cli/compose_subcircuits.py` (parent rects + Fix 3 root-cause),
  `kicraft/autoplacer/brain/subcircuit_composer.py` (edge path)
- `kicraft/autoplacer/freerouting_runner.py` (`clear_zones` preserve rule areas)
- `kicraft/cli/_parent_stamp_subprocess.py`,
  `kicraft/autoplacer/hardware/_stamp_subcircuit_subprocess.py` (call silk pass)
- `kicraft/circuitchat/synthesis/autoplacer.py` (edge auto-derive),
  `.claude/skills/circuitchat/stages/bom.md` (rule)
- `kicraft/autoplacer/config.py` (`antenna_keepouts` family spec,
  `default_edge_connector_zone`, margins)

## Verification

1. `.venv/bin/python -m pytest tests/test_library_antenna_keepouts.py
   tests/test_keepout_extract.py tests/test_placement_keepout.py
   tests/test_silk_refdes.py tests/test_edge_zone_derivation.py
   tests/test_edge_constraint_applied.py -q` + full suite stays green.
2. Targeted: load the esp32 leaf via `adapter.load()` → WROOM antenna keepout appears
   in `BoardState.keepout_rects`; `solve()` leaves no non-WROOM courtyard in the rect;
   silk pass on stamped parent → no two refdes silk bboxes overlap.
3. Re-emit `autoplacer.json` → USB-C ref carries an `edge` zone.
4. Commit on a fresh `fix/` branch, open + merge a Codeberg PR (curl --netrc, merge
   commit, delete branch, no Claude attribution).
