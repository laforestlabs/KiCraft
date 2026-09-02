# KiCraft

KiCad PCB automation toolkit — automated placement, routing, scoring, and
experiment management for KiCad projects via the pcbnew Python API.

## Layers

KiCraft is a multi-layer pipeline. Top-down:

1. **KiCraft** (portable Agent Skills + Python helpers) -- turns a natural-
   language project description into the hierarchical KiCad 9 file set
   (root + leaf `.kicad_sch`, `.kicad_pro`, `_autoplacer.json`). The five
   LLM-authored stages live in standards-compliant skills under
   `.agents/skills/`; deterministic validation, synthesis, placement, routing,
   and fab export are Python CLIs. See [KiCraft](#kicraft-chat--kicad-files).
2. **placement + routing + scoring** (Python, this repo) -- everything below.

## Installation

```bash
# Development install (editable). Synthesis imports KiCad 9's bundled
# `pcbnew`; system Python already has it. For an isolated venv, create it
# with `--system-site-packages` so `pcbnew` is visible (see pipx note below).
pip install -e .

# With KiCraft design helpers (Pydantic, kicad-skip, and easyeda2kicad).
# The ordinary skill uses the active agent's configured LLM; KiCraft itself
# needs provider credentials only for the provider-backed debug/server path.
pip install -e ".[design]"

# With all optional dependencies
pip install -e ".[scoring,experiment,design,tuning,server,loadtest,dev]"
```

### End-user install (pipx)

`kicraft build` imports KiCad's bundled `pcbnew`, and
`add-part --from-lcsc` imports `easyeda2kicad`. An isolated pipx venv sees
neither by default, so install with `--system-site-packages` and the
`[design]` extra:

```bash
pipx install --system-site-packages "kicraft[design]"
```

Already installed without them? Symptoms are `add-part --from-lcsc` printing
`easyeda2kicad not installed`, or `synthesize` raising
`ModuleNotFoundError: pcbnew`. Remediate with:

```bash
pipx reinstall --system-site-packages kicraft
pipx inject kicraft easyeda2kicad   # only if the [design] extra was omitted
```

## KiCraft (chat -> KiCad files)

A multi-turn agent workflow that takes a project description in plain English
and emits the hierarchical KiCad 9 file set consumed by placement and routing.
No prior schematic is required.

### Architecture

- The five design stages (`intent`, `functional_spec`, `architecture`, `bom`,
  and `wiring`) use the `ConversationState` Pydantic schema in
  `.kicraft/state.json`.
- Portable Agent Skills live under `.agents/skills/`. The ordinary `kicraft`
  skill uses whichever LLM the active agent runtime provides.
- `.omp/config.yml` points OMP at that generic directory as a custom skill
  source, so the repository skills override stale user-level copies with the
  same names.
- The optional `kicraft-debug` skill calls the production provider-backed stage
  runtime and pauses before each durable commit for explicit review.
- The deterministic synthesis/build step is the `kicraft` CLI.

### Run it

Open this repository in any Agent Skills-compatible coding agent and ask it to
use the `kicraft` skill:

```text
Use the kicraft skill to design a USB-C powered 3.3 V status LED board.
```

For production-provider stage inspection:

```text
Use the kicraft-debug skill to walk through each KiCraft LLM stage.
```

Describe your project ("USB-C powered 3.3V regulator with status LED,
JLCPCB target, under $5 BOM"). The ordinary flow steps through all five
stages and commits each slot. The debug flow uses the production provider
and retains its prompt, raw response, tool trace, candidate, and diagnostics
under `.kicraft/debug/`, while keeping `.kicraft/state.json` unchanged until
explicit acceptance. After wiring is committed, ask the ordinary skill to
run the deterministic build.

```bash
kicraft build .kicraft/state.json ./generated --quality good
```

### Using the skills in another project

Copy the standards-compliant skill directories into the target project's
`.agents/skills/` discovery root:

```bash
mkdir -p your-project/.agents/skills
cp -r /path/to/KiCraft/.agents/skills/kicraft your-project/.agents/skills/
cp -r /path/to/KiCraft/.agents/skills/kicraft-debug your-project/.agents/skills/
```

OMP users should also copy the small discovery bridge:

```bash
mkdir -p your-project/.omp
cp /path/to/KiCraft/.omp/config.yml your-project/.omp/
```

Clients that use a different Agent Skills discovery root can copy the same
skill directories there unchanged; each skill is self-contained around its
`SKILL.md`.

Install the ordinary design CLI with
`pip install -e "/path/to/KiCraft[design]"`. It uses the active agent's LLM
configuration and does not require a KiCraft provider key. The provider-backed
`kicraft-debug` skill additionally requires
`pip install -e "/path/to/KiCraft[server,design]"` and the OpenRouter
configuration used by the production web stage driver.

### Agent capabilities

The skills use generic capabilities rather than vendor-specific permission
syntax. Grant the active agent only the capabilities listed in each
`SKILL.md`: read project/skill files, write `/tmp/kicraft_*`, and run the named
`kicraft` commands. No slash commands, vendor-specific delegation API, or
vendor-specific settings file is required.

### CLI reference

```bash
kicraft stage-prep <stage> [STATE]
# Single-shot collector. Returns JSON: {state, extras} where extras carries
# the leaf-library output for architecture and BATCHED symbol pinouts for
# wiring (replacing N per-part lookup-symbol calls with one).

kicraft stage-commit <stage> --slot-file F.json \
  [--questions-file Q.json] [--history-message M] [--project-stem S] \
  [--invalidate-downstream] [STATE] [--no-archive] [--archive-root DIR]
# Atomic: validate the proposed slot, merge into state.json, append history,
# archive. Returns {"ok": true, ...} or {"ok": false, "errors": [...]} so the
# active agent can correct a rejected candidate.

kicraft-stage-debug debug-draft --workspace PATH --stage STAGE \
  --brief-file BRIEF [--instruction-file GUIDANCE] [--answers-file ANSWERS]
# Runs the production provider and atomically saves a pending review artifact
# without committing or stamping state.json.

kicraft-stage-debug debug-commit --workspace PATH --stage STAGE \
  --history-message-file MESSAGE
# Stale-hash checks state.json, commits the exact accepted slot through
# stage-commit, invalidates downstream stages, and finalizes the debug trace.

kicraft validate .kicraft/state.json
# prints {ok, project_stem, slots_filled, open_questions, blocking_questions}
# exit codes: 0 ok, 2 schema error, 3 library validation error

kicraft list-leaves
# prints the "Available leaves" markdown block. The architecture stage
# receives this via stage-prep's extras; this command is for ad-hoc inspection.

kicraft lookup-symbol Library:Name
# prints one symbol's pin inventory. The wiring stage receives all needed
# pinouts via stage-prep's batched extras; this command is for ad-hoc lookup.

kicraft synthesize .kicraft/state.json ./generated [--smoke]
# wraps kicraft.design.synthesize.run; --smoke adds the (slow)
# solve-subcircuits check
```

### What gets written

For project stem `MYPROJ`, synthesis writes:

```
MYPROJ/
  MYPROJ.kicad_sch                    # hierarchical root
  <SHEET>.kicad_sch ...               # one per leaf sheet
  MYPROJ.kicad_pro                    # design rules + Default/Power netclasses
  MYPROJ_autoplacer.json              # ic_groups, power_nets, signal_flow_order, ...
  MYPROJ.kicad_pcb                    # empty pcbnew stub
```

Everything past that is the existing layout/routing pipeline:

```bash
cd MYPROJ
autoexperiment MYPROJ.kicad_pcb --schematic MYPROJ.kicad_sch --rounds 20
```

### Validation

Every synthesis run executes SS9.1-SS9.6 from
`docs/kicraft_schematic_prompt.md` against the written files
(schematic version, footprints non-empty, pin directions valid,
Sheetfile refs resolve, autoplacer JSON valid, every named ref in
the schematic). Synthesis raises `SynthesisValidationError` and prints
the failing check rather than shipping a broken file set.

### Notes

- Custom footprints (`<PROJECT>.pretty/`) are out of scope for v1 --
  if the BOM references a footprint not in the stock KiCad libraries,
  the skill surfaces a blocking question instead.
- State persists as `.kicraft/state.json` in the project directory —
  gitignore it (or commit it; it's plain JSON and reviewable).

## Leaf Library (reuse vetted designs across projects)

A *leaf* is a single hierarchical sheet — a pre-routed PCB fragment plus
its schematic, BOM, and autoplacer settings. Once you've solved one
("USB-C 1S LiPo charger") and pinned a round you trust, you can promote
it into the global Leaf Library. The KiCraft pipeline then reuses
it verbatim every time the LLM judges it a match for a new project,
collapsing leaf-level design surface to a vetted, pinned solution.

### Where leaves live

```
$KICRAFT_LEAF_LIB                # default: ~/.kicraft/leaves/
  usb-c-lipo-charger/
    manifest.json
    leaf_routed.kicad_pcb        # canonical name (matches .experiments/...)
    metadata.json
    solved_layout.json
    schematic.kicad_sch
    autoplacer_fragment.json
    bom.csv
    renders/
      front_all.png
      back_copper.png
      copper_both.png
      thumbnail.png
```

The pinned-PCB triad (`leaf_routed.kicad_pcb` + `metadata.json` +
`solved_layout.json`) is the same triad the existing pin manager (under
`.experiments/subcircuits/<leaf_key>/`) consumes, so an imported leaf
drops in as a pre-solved round and the parent composer treats it as
already-solved on the next `autoexperiment --parents-only`.

### Promote a leaf

> **Note:** the Experiment Manager GUI that hosted the "Leaf Library" promote flow was
> removed 2026-06-22 (recover from git history if needed). Leaf promotion is pending a
> re-home in the web app — see `docs/plans/refactor-roadmap.md`. The pinned-round import
> format described above still applies.

The wizard runs the renders, writes a manifest with the
content-addressed hash, and atomically writes the new directory into
`$KICRAFT_LEAF_LIB`. To replace an existing leaf, bump the version
(e.g. `0.1.0 -> 0.1.1` for a patch, `0.2.0` for additive interface
changes, `1.0.0` for breaking ones).

### Automatic reuse during KiCraft

When the skill enters the architecture stage it runs
`kicraft list-leaves` and reads the result so the model can
see the curated catalog. The active agent picks matches by setting
`Sheet.from_library = "<name>@<version>"` and `Sheet.library_instance = N`.
`kicraft validate` then verifies the leaf's hierarchical-
label interface matches the sheet's endpoints exactly. Synthesis then:

1. Skips the LLM-generated leaf schematic for that sheet.
2. Copies the leaf's `schematic.kicad_sch` to the project (with refdes
   renumbered to fit the host project's existing refs).
3. Writes the renumbered triad into
   `<project>/.experiments/subcircuits/<leaf_key>/round_lib0001_*`.
4. Pins the import via `pins.json` so `ensure_applied()` keeps it locked
   across solver runs.
5. Adds a `library_leaves` audit record to `<project>_autoplacer.json`.

### CLI

```bash
kicraft-leaf list             # rows: name@version, hash, tags, description
kicraft-leaf show <name>      # full manifest as JSON
kicraft-leaf path             # resolved library directory
```

Promotion and removal are GUI-only — there's no `kicraft-leaf promote`.
The intent is that curating the library is a deliberate, human-in-loop
step, not a scriptable one.

## Quick Start

KiCraft is project-agnostic. Create a `<project>_autoplacer.json` (or
`autoplacer.json`) in your KiCad project root to configure IC groups,
component zones, signal flow, and other project-specific settings.

See `examples/llups_autoplacer.json` for a full example.

### Solve subcircuits

```bash
cd /path/to/your/kicad-project
solve-subcircuits project.kicad_sch --pcb project.kicad_pcb --rounds 3 --route
```

### Run experiment loop

```bash
autoexperiment project.kicad_pcb project.kicad_sch --rounds 20 --workers 2
```

### Two-phase guided experiment (leaf pinning)

Explore leaf candidates first, lock the ones you like, then iterate on the
parent only:

```bash
# 1. Solve only leaves -- snapshots every round to .experiments/subcircuits/<leaf>/round_NNNN_*
autoexperiment project.kicad_pcb --schematic project.kicad_sch --rounds 30 --leaves-only

# 2. Pin chosen rounds via the GUI Analysis page (Hierarchical Progression -> Accepted
#    Leaf Gallery -> "Pin from prior experiment-round snapshots"), or write
#    .experiments/pins.json by hand.

# 3. Run only the parent compose phase against the pinned leaves
autoexperiment project.kicad_pcb --schematic project.kicad_sch --rounds 10 --parents-only
```

The composer calls `pins.ensure_applied()` before loading artifacts, so
pinned leaves stay locked even if a stray leaf-solve overwrites the
canonical files. The best parent of any run is also copied to
`.experiments/best/parent_routed.kicad_pcb` and `<projectname>_best.kicad_pcb`
at the project root for fab handoff.

### Score a layout

```bash
score-layout project.kicad_pcb
```

### Configuring Placement

Placement constraints live in `<project>_autoplacer.json` at the project
root. The autoplacer auto-discovers it; the file is project-agnostic --
it references components by ref designator (e.g. `J1`) or subcircuit
sheet name (e.g. `BATT`), nothing is hardcoded to a particular project.
See `examples/llups_autoplacer.json` for a full working file.

#### Pinning components to edges, corners, or zones

`component_zones` constrains where individual refs land on the board.
Three constraint types, all applied at solve time:

```json
{
  "component_zones": {
    "J1":  { "edge":   "left" },
    "J2":  { "edge":   "right" },
    "H4":  { "corner": "top-left" },
    "H86": { "corner": "bottom-right" },
    "BT1": { "zone":   "bottom" },
    "BT2": { "zone":   "bottom" }
  }
}
```

* `edge: left|right|top|bottom` — connector body flush with the named
  edge. Pads face inward, housing overhangs outward by
  `connector_edge_inset_mm`.
* `corner: top-left|top-right|bottom-left|bottom-right` — typically
  mounting holes. Locks at the corner with `mounting_holes.keepout.size_mm`
  reserved (see [Mounting holes](#mounting-holes) below).
* `zone: top|bottom|left|right|...` — soft confinement to a region.
  The component is locked inside the zone; the solver may place it
  anywhere within it.

When a ref is part of a child subcircuit (e.g. `BT1` inside the `BATT`
subcircuit), the constraint is propagated to the *block* during parent
composition, so the BATT block ends up zone-bottom-pinned at parent
solve time.

#### Backside THT components: enabling stacking

By default, the parent composer prevents two leaves from overlapping
when both have any same-layer copper -- a strict same-layer-outline
gate that catches CHARGER-style continuous-F.Cu shorts. PTH pads
register as copper on **both** F.Cu and B.Cu, so a leaf whose only
front-side copper is the shadow of through-hole pads (battery
holders, screw terminals, large barrel jacks) is treated as having
real F.Cu occupancy and rejects SMT-on-front stacking inside its
bbox -- even when the body centre is empty and stacking would be
physically fine.

To re-enable opposite-side stacking on top of such a leaf, open the
GUI's **Setup → Per-Component Placement Rules** tab, expand the leaf,
and toggle the sheet-level switch:

> ☐ → ☑ **Backside THT anchor (allow SMT-front leaves to stack on top)**

Click **Save to project JSON…**, review the diff modal, and confirm.
The toggle persists as a sheet-name entry under
`parent_placement.backside_through_hole_leaves`. For headless or CI
workflows the same key can be set by hand:

```json
{
  "parent_placement": {
    "backside_through_hole_leaves": ["BATT", "TERMINAL_BLOCK"]
  }
}
```

What the override does:

* Marks the leaf as having no F.Cu intent for the same-layer-outline
  gate, so SMT-on-front leaves may overlap inside its bbox.
* Tells the post-SA stack pass (`_stack_compatible_blocks`) to use
  it as a stacking anchor, actively row-packing front-only candidates
  inside its body.
* Leaves the per-pad sparse-rect overlap checks in place -- if a
  candidate's pads happen to land on the THT pads themselves
  (corner positions), the predicate still rejects that specific
  geometry.

The flag is keyed by `sheet_name` so any project can list its own
THT-back leaves; nothing is hardcoded to a single board. CHARGER-style
leaves with continuous F.Cu routing **must not** be enabled -- the
override there would let the original same-layer shorts return.

#### Candidate search

`parent_placement.candidate_search` controls the K-candidate search
loop. The picker stamps each candidate, runs DRC, scores by a
composite of opposite-side packing, courtyard overlap, ratsnest
length, and bbox density, and picks the highest-scoring candidate
whose stamped DRC reports `shorts == 0`:

```json
{
  "parent_placement": {
    "candidate_search": {
      "k": 4,
      "time_budget_s": 240.0
    }
  }
}
```

`shorts == 0` is the only hard gate. Geometry-validation results
(components/pads outside the auto-grown outline) are recorded on each
candidate for diagnosis but no longer reject -- letting routing run
on a violating layout produces a routed PNG showing exactly where the
problem is, which is more actionable than aborting the round.

#### Mounting holes

The `mounting_holes` config block parameterizes the corner-anchored
mechanical holes (H4, H86, etc.):

```json
{
  "mounting_holes": {
    "count": 2,
    "screw": "M3",
    "hole_diameter_mm": 3.2,
    "pad":     { "shape": "hexagon", "size_mm": 3.0 },
    "keepout": { "shape": "hexagon", "size_mm": 4.0 }
  }
}
```

| Field | Meaning |
|---|---|
| `count` | Target number of holes (informational; actual count is whatever the source PCB project ships). |
| `screw` | Free-form screw reference (`"M2.5"`, `"M3"`, `"#4-40"`). Documents intent; not parsed. |
| `hole_diameter_mm` | Drill clearance diameter for the screw. M3 default = 3.2 mm (clearance fit + fab tolerance). |
| `pad.shape` | `hexagon` (default), `circle`, or `square`. Reserved for the future footprint generator. |
| `pad.size_mm` | Outer extent of the exposed copper / annular ring around the hole, measured from hole center to the closest edge of the shape (radius for `circle`, half-width-across-flats for `hexagon` / `square`). Reserved for the future footprint generator. |
| `keepout.shape` | Same vocabulary as `pad.shape`. Reserved -- placer currently treats keepout as rectangular. |
| `keepout.size_mm` | Component-free zone radius (face-to-flat distance for `hexagon` / `square`). The placer reads this directly: a corner-anchored hole sits this far inboard from each board edge. Default 4 mm = M3 head 3 mm + 1 mm fab slack; bump to ~5 mm if you ship M3 + washer. |

Only `keepout.size_mm` currently affects geometry. The shape fields and
`pad.*` are plumbing for an upcoming change that lets KiCraft generate
or validate mounting-hole footprints directly from this block.

### Mutation Search Bounds

The GUI Setup tab exposes 39 searchable parameters that the evolutionary
optimizer mutates during `autoexperiment` runs. Users can narrow or widen
the search range (min/max) for each parameter to focus exploration.

Bounds persist across sessions automatically and flow to the optimizer via
`gui_param_ranges.json`. A shared `normalize_bounds()` helper validates all
bound inputs (rejects NaN/Infinity, clamps to spec domain, swaps inverted
ranges). The only enforced cross-parameter constraint is physical:
`via_drill_mm < via_size_mm` (annular ring requirement).

**Parameter groups:**

| Group | Params | Controls |
|-------|--------|----------|
| Placement Physics | 8 | Force strengths, cooling, iterations, convergence |
| Board Geometry | 6 | Dimensions, margins, grid snap, clearances |
| Edge & Connectors | 6 | Courtyard, connector gaps, insets, pad margins |
| SA Refinement | 6 | Temperature, cooling, move radius, swap/rotation |
| Routing | 7 | Trace widths, via dimensions, zone margin, KiCad Routing Tools |
| Thermal | 1 | Keep-away radius around hot components |
| Component Behavior | 1 | THT backside area threshold |
| Zone Pour | 4 | Clearance, fill thickness, thermal relief |

Run `autoexperiment` with `--param-ranges <file.json>` to override bounds
from the command line.

## CLI Commands

### KiCraft
- `kicraft validate STATE.json` — Validate a `.kicraft/state.json`
  against the `ConversationState` schema + library-pick rules.
- `kicraft list-leaves` — Print the "Available leaves" markdown
  block the architecture stage shows to the model.
- `kicraft synthesize STATE.json OUT_DIR [--smoke]` — Emit the
  KiCad file set from a complete state. The portable `kicraft` Agent Skill
  under `.agents/skills/kicraft/` owns the LLM-authored design stages.

### Core Pipeline
- `solve-subcircuits` — Hierarchical subcircuit placement and routing
- `compose-subcircuits` — Assemble solved subcircuits into parent boards
- `solve-hierarchy` — Full hierarchical solve (leaves → parents)

### Experiment Management
- `autoexperiment` — Automated experiment loop with parameter search
- `clean-experiments` — Clean up experiment artifacts (before/after/nuke)
- `watch-status` — Live terminal monitor for running experiments

### Scoring & Rendering
- `score-layout` — Score PCB layout quality
- `render-pcb` — Render PCB layers to PNG
- `render-drc-overlay` — DRC violation overlay on PCB render
- `render-failure-heatmap` — Routing failure heatmap

### Analysis
- `plot-results` — Plot experiment or scoring dashboards (auto-detects format)
- `diff-rounds` — Diff between experiment rounds
- `generate-report` — Generate scoring report
- `token-report`: Token usage + estimated cost from supported agent transcript JSONL files

### Board Manipulation
- `move-component` — Move a component to absolute position
- `align-components` — Align components along an axis
- `arrange-grid` — Arrange components in a grid
- `add-gnd-zone` — Add GND copper zone
- `cleanup-routing` — Clean up routing artifacts
- `split-schematic` — Split flat schematic into hierarchical sheets

### Inspection
- `list-footprints` — List all footprints with positions
- `check-trace-widths` — Check trace widths against minimum
- `run-drc` — Run design rule check
- `net-report` — Network connectivity report
- `inspect-subcircuits` — Inspect subcircuit hierarchy
- `inspect-solved-subcircuits` — Inspect solved subcircuit artifacts

## Package Structure

```
.omp/config.yml          # OMP bridge to the generic skill directory
.agents/skills/           # Portable Agent Skills (SKILL.md standard)
├── kicraft/              # Five-stage design interview + build handoff
├── kicraft-debug/        # Provider-backed stage review before commit
├── kicraft-investigate/  # Artifact-driven failure investigation
├── self-eval/            # Curated end-to-end regression batch
└── verify/               # Deterministic place/route replay verification
kicraft/
├── design/               # State schema, deterministic stage CLI, synthesis
│   ├── models.py         # Pydantic state slots
│   ├── library.py        # Leaf-library helpers
│   ├── synthesize.py     # Deterministic state -> file set step
│   ├── synthesis/        # KiCad emitters and deterministic checks
│   └── cli_app.py        # `kicraft` stage/build commands
├── autoplacer/          # Placement and routing engine
│   ├── config.py        # Default config + project config loader
│   ├── kicad_routing_tools.py  # pinned KRT adapter and copper custody
│   ├── routing_board.py        # pcbnew/DRC/copper helpers
│   ├── brain/           # Pure algorithms (no pcbnew dependency)
│   └── hardware/        # KiCad pcbnew API adapter
├── scoring/             # Layout quality scoring checks
├── cli/                 # CLI entry-point scripts
└── logging_config.py    # Structured logging setup
```

## Requirements

- Python 3.10+
- KiCad 9 with pcbnew Python bindings (on system PATH)
- KiCad Routing Tools source `0.20.2` at commit
  `3ceb773722bea67aa3685e7ee430c0c0d17ef38d` with native router `0.20.1`

## License

MIT
