# Standard form-factor templates — honoring mechanical intent as a hard constraint

**Status:** proposed (design). Motivated by KC-99A9M8 ("An Arduino-Uno-format prototyping
shield with stacking through-hole headers and an onboard SMT 3.3 V regulator"), which the
pipeline free-placed as generic headers — producing a board that could not physically mate
with an Arduino even if it built, and which in fact failed the build outright (a header column
overflowed the content canvas; see the GAP 1 reframe fix in `leaf_routing.py`).

## The gap

A brief that names a **standard mechanical form factor** ("Arduino Uno shield", "Raspberry Pi
HAT", "Feather", "Pi Zero", "mikroBUS") is stating a *hard* constraint: a fixed board outline,
fixed connector positions (an Uno shield's stacking headers sit at standardized coordinates —
including the deliberate 100-mil-off gap between D7 and D8), and often a fixed mounting-hole
pattern. Today the pipeline has **no way to represent or honor any of this**:

- `IntentSlot.form_factor` (`models.py:173`) is the only brief-level mechanical field, and it
  carries **shape only** (rect / rounded_rect / circle / named aesthetic shapes) + an *advisory*
  `size_mm`. `FunctionalSpec` and `Architecture` (`models.py:204,325`) have **zero** mechanical
  fields.
- The parent board outline is **always derived from the placed-component bbox** and only ever
  *grows* to enclose placed geometry (`compose_subcircuits._compute_final_outline:737`,
  `subcircuit_composer._derive_board_outline:2529`). Even the shape hook merely *circumscribes*
  the named shape around whatever the solver placed (`_compose_validate._fit_requested_shape:199`,
  `shapes/__init__.py:170` — *"the placed circuit stays put; the shape grows around it"*).
- There is **no template registry** and **no data** for any named standard — no outline
  coordinates, no connector positions, no mounting-hole patterns. The strings `arduino`/`shield`
  appear only in a benchmark brief and passing comments (`tuning/benchmark.py:91`,
  `cli_app.py:3206`).

Net: "Arduino-Uno-format shield" cannot be honored on the automatic path. The board is sized by
whatever the solver happens to place, so the shield's outline is wrong and its headers land
wherever placement likes — mechanically non-conformant by construction.

## What already exists to build on

- **Exact-lock placement primitive** (`Component.locked`, `types.py:146`): a locked part is
  excluded from force/SA, its rotation is frozen (`placement_solver.py:1811`), and its position
  is restored after every phase (`_restore_pinned_positions`, `placement_solver.py:649`). Setting
  `pos` + `rotation` + `locked=True` freezes an exact (x, y, θ). Already auto-set for array
  members and mounting holes/edge connectors.
- **Manual-layout mode** (`compose_subcircuits.py:1335`): already **skips the solver** and honors
  user-supplied exact placements + an exact `manual_layout.board_outline` (`OutlineSpec`,
  `layout_editor/outline.py`). This is the strongest existing hook — a template is essentially an
  *auto-generated* manual-mode fixed outline + a set of pre-locked connector/hole components.

So the feature is mostly **data + plumbing**, not a new solver primitive.

## Design

### 1. Data: a form-factor registry
New module `kicraft/form_factors/` (sibling of `kicraft/shapes/`), one entry per standard:

```python
FormFactorTemplate(
    key="arduino_uno_shield",
    aliases=["arduino shield", "uno shield", "arduino-uno-format", ...],
    outline=OutlineSpec(...),                 # exact Edge.Cuts incl. the USB/notch cut
    fixed_connectors=[                        # ref-role, footprint, exact (x, y, rot)
        FixedConnector(role="digital_hi", footprint="PinHeader_1x10_...", x=..., y=..., rot=0),
        FixedConnector(role="digital_lo", footprint="PinHeader_1x08_...", x=..., y=..., rot=0),
        FixedConnector(role="analog",     footprint="PinHeader_1x06_...", x=..., y=..., rot=0),
        FixedConnector(role="power",      footprint="PinHeader_1x08_...", x=..., y=..., rot=0),
    ],
    mounting_holes=[(x, y, dia), ...],
    pin_semantics={...},                      # role/pin -> canonical net (D0..D13, A0..A5, 5V, 3V3, GND, VIN, RESET, AREF)
)
```
Start with **Arduino Uno R3 shield** only; the schema generalizes to HAT/Feather/mikroBUS later.
Coordinates come from the published mechanical drawings (author once, guard with a golden test).

### 2. Intent → constraint flow
- Extend `extract_form_factor` (`design/synthesis/form_factor.py`) to **detect a named standard**
  in the brief (alias match) and set a new `IntentSlot.form_factor.standard` key (or a dedicated
  `IntentSlot.mechanical` slot). This is deterministic — no LLM needed for the match.
- `write_autoplacer_json` (`design/synthesis/autoplacer.py:200-228`) emits, when a standard is
  present: the **fixed outline** (not a shape-to-circumscribe), the **pre-locked connector
  components** at their template coordinates, and the **mounting holes**.
- Synthesis (BOM/wiring) must bind the shield's function pins to the template's `pin_semantics`
  so the design's nets connect to the right header pins (this is the real electrical work — the
  headers aren't decoration, they're the board's I/O contract).

### 3. Compose honors the fixed outline
- `_compute_final_outline` / `_fit_requested_shape` gain a **fixed-outline** branch: when a
  template outline is supplied, use it verbatim (clamp/validate that placed parts fit inside it —
  and *fail loudly* if they don't, rather than growing it). This is the inverse of today's
  grow-to-content behavior.
- The templated connectors + holes enter placement **pre-locked**, so the solver places only the
  remaining parts *within* the fixed outline and *around* the locked headers.

### 4. Gate: mechanical-conformance check (new promote gate)
A deterministic gate that fails the build (or downgrades fab-ready) when a standard was requested
but the delivered board's outline / connector positions don't match the template within tolerance.
This is what makes the constraint *real* rather than advisory — see the investigate-skill
intent-adherence audit (below), which is the diagnostic counterpart.

## Scope / sequencing

**Decisions taken** (2026-07-10): connector model = **replace & rewire** (the template's headers
ARE the board's I/O; inject them pre-locked, drop the LLM's free headers, wire the design's
function to the standard nets). Datum source = **transcribed from an open-source KiCad library**
(Alarm-Siren/arduino-kicad-library) — the Arduino Uno R3 shield datum is now `validated=True`.

- **PR1 — data + detection [DONE, committed].** `kicraft/form_factors/` registry
  (`FormFactorTemplate`/`FixedConnector`/`MountingHole`), Arduino Uno R3 shield template,
  `match_standard` alias matching, `FormFactor.standard` field, `extract_form_factor` detection at
  intent-commit. Golden tests. Surfaces the standard in state.json; no downstream behavior change.
- **datum validation [DONE, committed].** Authoritative coordinates from the KiCad library
  (KiCad-native top-left frame), `validated=True`. The famous 0.16″ D7–D8 offset is exact.
- **PR2a — emission [DONE, committed].** `to_autoplacer_dict()` + a `form_factor_standard` block
  in `<stem>_autoplacer.json` (outline + fixed connectors + holes). Informational; consumers gate
  on `validated`. Survives the project-config loader.
- **PR3 foundation — conformance check [DONE, committed].** `conformance.py`:
  `check_conformance` (geometry, not net names) + `board_local_pads` (pcbnew-free board reader) +
  wired into the investigate §8.5 audit. On KC-99A9M8's board: `NON-CONFORMANT 4/32, outline
  121.9×45.2 != 68.58×53.34`. Read-only; reports, does not place.

- **PR2b Half 1 — compose enforcement [DONE, committed].** `compose_scaffold.py` (`build_scaffold`
  locked connector Components + fixed outline; `resolve_scaffold` gate) + a thin gated fork in
  `_compose_artifacts`: pins the parent placement canvas + final outline to the template rect and
  injects the locked connectors, so the solver auto-places everything around them and
  `_validate_parent_geometry` rejects a design that doesn't fit. Gated on `cfg.form_factor_enforce`
  + a validated `form_factor_standard`; dormant otherwise (existing compose tests unchanged). The
  synthesis data layer (`scaffold.py`: `standard_header_parts` / `standard_placements` /
  `canonical_power_bindings`) is also done + tested.

- **PR2b Half 2 electrical — replace & rewire reconcile [DONE, committed `e6ca872`].**
  `reconcile.py::reconcile_standard_form_factor`, wired at the wiring stage-commit next to the
  wiring normalizers, **env-gated by `KICRAFT_FORM_FACTOR_ENFORCE` (default OFF)**: replaces the
  LLM's 2.54 mm stacking connectors with the standard's headers as real `BomPart`s (recycling the
  freed J-refs), prunes every reference to the dropped parts (connections / no-connects /
  component_zones / thermal_refs / signal_flow_order / ic_groups / arrays), binds the header
  power/ground pins by net name, and marks signal pins (D0..D13/A0..A5/SCL/SDA) no-connect (they
  mate with the host below). `standard_form_factor_bom_delta` returns the signal no-connects too.
  Validated on KC-99A9M8's real committed state: 13 parts → 7 (regulator + 2 caps + 4 standard
  headers), state re-validates. 20 unit tests; 240 pass with the gate off.

- **PR2b Half 2 mechanical — place the real headers at fixed positions [DONE, validated on a real
  build].** Resolved the leaf-vs-scaffold coupling: compose now (a) resolves the scaffold up front,
  (b) drops the leaf made entirely of the standard-header refs, (c) pops those refs out of
  `parent_local` before the loose-connector wrap, and (d) injects ONE locked copy per header at the
  template pos+rotation — the stamp then moves the real seed footprint (matched by ref) there, so no
  duplicate and no synthetic footprint. Three coordinated pieces made it work end-to-end:
  - **scaffold rotation** (`compose_scaffold._stamp_rotation_deg`): a KiCad vertical single-row
    header advances +Y at rot 0, but the Arduino edge headers advance +X (`axis="x"`), so the stamp
    rotation is **90°**. Without it pin-1 lands right but the row runs off the wrong axis.
  - **real refs** (`autoplacer.py` emits `form_factor_standard.header_refs` role→BOM ref; the
    scaffold uses them): the reconcile recycles/renumbers J-refs, so the scaffold must lock the SAME
    refs the schematic carries, not a guessed `J1..`.
  - **exact-lock survives the solver** (`placement_solver._pin_edge_components`): a component that
    arrives `locked=True` is exact-placed — record its pinned target but never edge-repin/re-orient
    it (the nearest-edge fallback was clobbering the scaffold rotation).
  - **reconcile also prunes emptied sheets** (`reconcile._prune_emptied_sheets`): consolidating the
    headers onto one host sheet leaves the old connector sheets empty → a degenerate leaf that aborts
    the build; drop them (and their inter-sheet nets) from the architecture.
  Validated by a no-LLM replay of KC-99A9M8's committed state (reconcile → `build`): **CONFORMANT
  32/32 pins, outline 68.58×53.34, DRC 0 shorts / 0 unconnected, fab package exported.** The original
  brief *failed* the build (structurally unroutable free headers); the feature makes it buildable.
- **PR2b Half 2 electrical — ERC hardening [DONE, validated].** The real build surfaced ERC/synthesis
  findings the unit tests could not: the binding was rewritten to be **design-aware**
  (`synthesis.standard_form_factor_bom_delta` + `reconcile`): a header pin binds to a rail the design
  ALREADY carries (alias-normalized: `+5V`≡`5V`, `+3V3`≡`3V3`), else it is no-connect — including the
  reserved/NC pin and every unused digital/analog/AREF/RESET/IOREF/VIN pin. This fixed all four
  findings at the source (net-merge + double-power-output on `+3V3`/`3V3`, dangling AREF/RESET/IOREF
  labels, uncovered NC pin): the emitted schematic is now **ERC clean (0 errors), §9.13 netlist
  faithful**.
- **PR3 gate — enforcement [DONE].** `_check_form_factor_conformance` runs `check_conformance` on the
  promoted board at the verify gate: when enforcement placed the board a non-conformant result **fails
  fab** (the board can't mate); with enforcement off it is an advisory only (the board is free-placed
  by design). Also fixed a bug in the conformance reader itself: `board_local_pads` did not apply the
  footprint rotation to pad offsets, so a correctly-rotated (conformant) header read as
  non-conformant.

## Open questions (for PR2b)
- **Header ↔ role/pin binding.** How does the design's I/O map onto D0..D13/A0..A5/power — does the
  LLM target canonical nets at the wiring stage (keyed off the template), or a post-hoc mapper?
  This is the crux of the rewire half.
- **Footprint reuse.** Vendor the exact shield footprints/outline as a `leaf_library/` part so the
  injection reuses the manual-mode `OutlineSpec` path directly?
- **ICSP + board notch.** The 2×3 ICSP header and the top-right corner notch aren't modeled yet
  (noted in the template) — add as keepout/outline refinements.
- **Multiple stacked form factors** (a shield that is also a specific size) — precedence rules.
