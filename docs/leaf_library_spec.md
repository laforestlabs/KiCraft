# Leaf Library — Implementation Spec (v1)

A curated, hand-promoted library of trusted "golden leaves" that the
CircuitChat pipeline reuses verbatim when they match a user's project,
collapsing leaf-level LLM design surface to a vetted, pinned solution.

This document is the implementation brief for the next session. The
design decisions below were settled in a prior conversation and should
be treated as fixed unless explicitly revisited.

---

## 1. Goals and non-goals

### Goals

- Allow a user to promote a pinned, tested leaf from a working project
  into a global library on disk.
- Allow CircuitChat to automatically reuse a library leaf — schematic,
  pinned PCB layout, and BOM, verbatim — whenever the LLM judges it a
  match for the current project's functional spec.
- Support multiple independent instances of the same leaf in one
  project (e.g. two identical 3.3 V regulators on different rails).
- Re-evaluate reuse decisions on every architecture-stage run so a
  leaf that no longer fits is silently dropped.
- Provide GUI affordances for promote and remove, gated by the user.

### Non-goals for v1

- Custom KiCad symbol or footprint libraries bundled inside a leaf
  (stock KiCad libraries only).
- BOM substitution / alternate parts.
- Migrating an existing project from leaf v1 to v2 (projects record
  the version they imported and are self-contained from then on).
- Cloud / git registry of leaves.
- Project-local library overrides (one global library only).
- Tag-based or electrical-spec-structured matching (LLM judges from
  descriptions and interfaces).
- A CLI promote or remove path (GUI is the only curation surface).
- An "edit a leaf in place" affordance (re-promote a new version).
- Auto-stamping a leaf based on autoexperiment score.
- A "force from-scratch" per-project override.
- Capturing originating CircuitChat slots, scores, or DRC numbers in
  the manifest (renders are kept; everything else is dropped).

---

## 2. Library on disk

### Location

Default `~/.kicraft/leaves/`, override via `$KICRAFT_LEAF_LIB`.

Created on first use. The loader treats a missing directory as an
empty library, never as an error.

### One entry per leaf name

A leaf name is unique across the library; promoting a new version
replaces the directory after a GUI confirmation. The version + content
hash live in the manifest so projects that already imported the prior
version remain self-contained.

### Per-leaf directory layout

```
$KICRAFT_LEAF_LIB/
  <leaf-name>/
    manifest.json
    schematic.kicad_sch          # the leaf sheet (single .kicad_sch file)
    pinned_layout.kicad_pcb      # solved + routed PCB fragment for the leaf
    autoplacer_fragment.json     # leaf-scoped slice of the source autoplacer.json
    bom.csv                      # leaf-scoped BOM rows
    renders/
      front_all.png
      back_copper.png
      copper_both.png
      thumbnail.png              # 256x256 PNG, used by the GUI list view
```

`<leaf-name>` is a kebab-case slug validated against
`^[a-z][a-z0-9-]*[a-z0-9]$`.

### Manifest schema

`manifest.json` is JSON (consistency with `autoplacer.json` and
`pins.json`; no new parser dependency).

```json
{
  "schema_version": "1",
  "name": "usb-c-lipo-charger",
  "version": "1.2.0",
  "content_hash": "sha256:0123...",
  "description": "USB-C 5V VBUS input, 1A LiPo charger using AP2112K with charge status LED. ISET=1.5kΩ -> 500mA charge current.",
  "tags": ["power", "charger", "lipo", "usb-c"],
  "watch_out_for": "Single-cell LiPo only. Not suitable for 2S+ packs.",
  "interface": {
    "hierarchical_labels": [
      {"name": "VBUS_IN",  "direction": "input"},
      {"name": "VBAT",     "direction": "bidirectional"},
      {"name": "GND",      "direction": "passive"},
      {"name": "CHG_STAT", "direction": "output"}
    ]
  },
  "refs": ["U1", "U2", "C1", "C2", "C3", "R1", "R2", "R3", "R4", "F1", "J1", "D1"],
  "dependencies": {
    "kicad_symbol_libs":   ["Regulator_Linear", "Connector"],
    "kicad_footprint_libs": ["Package_TO_SOT_SMD", "Connector_USB"],
    "kicad_version_min":   "9.0.0"
  },
  "provenance": {
    "source_project_stem":      "llups",
    "source_sheet_name":        "CHARGER",
    "source_experiment_round":  47,
    "promoted_at":              "2026-05-17T14:23:00Z",
    "kicad_version":            "9.0.0"
  }
}
```

### Field semantics

| Field | Notes |
|---|---|
| `schema_version` | Fixed string `"1"` for this spec. Future migrations bump this. |
| `name` | Kebab-case slug, unique in the library. |
| `version` | Semver. Standard semantics — major = breaking interface change, minor = additive interface, patch = internal change. |
| `content_hash` | `sha256:` over a canonical tarball of every file in the leaf directory *except* `manifest.json` itself. Verified on every load. |
| `description` | One paragraph, plain text, written by the user during promotion. The single input the LLM uses to decide fitness. |
| `tags` | Optional. Informational for human browsing only; the LLM does **not** match on tags in v1. |
| `watch_out_for` | Optional. Free-text caveats shown in the GUI and in the LLM's retrieval context. |
| `interface.hierarchical_labels` | Exact set of hierarchical labels in the leaf schematic, with KiCad pin directions. The parent project's `inter_sheet_nets` must connect to these names verbatim. |
| `refs` | Every reference designator that appears in the leaf (schematic and PCB). Validated against `^[A-Z]+[0-9]+$` — no suffix forms in v1. |
| `dependencies.kicad_symbol_libs` / `kicad_footprint_libs` | Stock KiCad library names the leaf depends on. Used to fail fast if a target system is missing them. |
| `dependencies.kicad_version_min` | Minimum KiCad version. Compared lexicographically by dotted segments. |
| `provenance` | Metadata from the promotion event. Read-only after writing. |

### Validation rules for the manifest

A loader rejects a manifest if any of the following fail:

1. `schema_version == "1"`.
2. `name` matches `^[a-z][a-z0-9-]*[a-z0-9]$`.
3. `version` parses as a valid semver triple.
4. `content_hash` recomputed matches the stored value.
5. Every entry in `refs` matches `^[A-Z]+[0-9]+$`.
6. Every `interface.hierarchical_labels[*].name` matches
   `^[A-Z][A-Z0-9_]*$`.
7. Every `interface.hierarchical_labels[*].direction` is one of
   `input | output | bidirectional | passive` (matches existing
   `PinDirection` literal in `kicraft/circuitchat/models.py`).
8. The `schematic.kicad_sch`'s hierarchical labels exactly match the
   manifest's declared interface (set equality on names + directions).
9. Every ref in `refs` appears at least once in `schematic.kicad_sch`
   and at least once in `pinned_layout.kicad_pcb`.

A loader that rejects a leaf logs the reason and excludes that leaf
from the available-leaves list passed to the LLM. The user sees the
broken leaf in the GUI list with an error chip.

---

## 3. Refdes strategy — renumber on import

### Algorithm

Given:
- `leaf.refs`: the list of refs in the leaf manifest.
- `project_refs`: every ref already present in the project's
  schematic + PCB + autoplacer.json (the union, before this import).
- `instance`: 1 for the first import of this leaf in this project, 2
  for the second, etc.

Steps:

```python
def renumber_leaf(leaf_refs: list[str],
                  project_refs: list[str]) -> dict[str, str]:
    """Return {leaf_ref -> project_ref}."""
    # Group existing project refs by letter-class.
    by_class: dict[str, list[int]] = defaultdict(list)
    for r in project_refs:
        cls, num = parse_ref(r)        # ("U", 7) for "U7"
        by_class[cls].append(num)

    next_in_class: dict[str, int] = {
        cls: max(nums) + 1 for cls, nums in by_class.items()
    }

    ref_map: dict[str, str] = {}
    # Sort leaf refs by (class, number) so the allocation order is
    # deterministic.
    for leaf_ref in sorted(leaf_refs, key=parse_ref):
        cls, _ = parse_ref(leaf_ref)
        n = next_in_class.get(cls, 1)
        ref_map[leaf_ref] = f"{cls}{n}"
        next_in_class[cls] = n + 1
    return ref_map
```

For multi-instance reuse, call `renumber_leaf` once per instance,
with each call seeing the prior instance's allocated refs included
in `project_refs`. Instances are processed in architecture-stage
order (`library_instance = 1` first, then 2, etc.).

### What gets renumbered

For each library-backed sheet:

1. **`schematic.kicad_sch`**: rewrite the `Reference` property of
   every symbol instance. Edit the string value only — preserve
   position, rotation, hidden flag, font size, layer. Use
   `kicad-skip` for the parse/edit; do **not** do textual substitution.

2. **`pinned_layout.kicad_pcb`**: rewrite the `(fp_text reference
   "<old>" ...)` element of every footprint. Same constraint —
   string value only, all other attributes preserved. Also rewrite
   any `(path ... "<old>")` ref tracking inside footprint blocks.

3. **`autoplacer_fragment.json`**, applying the map to every
   ref-shaped occurrence:
   - `ic_groups`: rewrite keys *and* member-list values.
   - `group_labels`: rewrite keys.
   - `thermal_refs`: rewrite values.
   - `signal_flow_order`: rewrite values.
   - `component_zones`: rewrite keys.
   - `parent_placement.backside_through_hole_leaves`: sheet names,
     **not** refs — leave alone.

4. **`bom.csv`**: rewrite the `ref` column. Other columns
   (`value`, `symbol`, `footprint`, `mpn`, etc.) are unchanged.

5. **Defensive scan**: parse the pinned PCB for any `(fp_text user
   "...")` whose value matches `^[A-Z]+[0-9]+$` *and* is a key in
   the renumber map. Log a warning per match and rewrite the
   string. This catches hand-placed silkscreen labels that
   accidentally encode a refdes.

### Silkscreen handling

KiCad renders the `(fp_text reference ...)` element directly onto
the silkscreen layer; the silkscreen text *is* the reference field.
Renumbering the reference automatically updates the rendered silk —
no separate silkscreen mutation. The position / rotation / layer
of each ref text on the silk is preserved because we edit only the
string value.

### Persisting the renumber map

After all library leaves have been imported, write a new top-level
`library_leaves` key in the project's `<project>_autoplacer.json`:

```json
"library_leaves": {
  "CHARGER": {
    "source":         "usb-c-lipo-charger@1.2.0",
    "source_hash":    "sha256:0123...",
    "instance":       1,
    "ref_map":        {"U1": "U7", "U2": "U8", "C1": "C12", ...}
  },
  "CHARGER_2": {
    "source":         "usb-c-lipo-charger@1.2.0",
    "source_hash":    "sha256:0123...",
    "instance":       2,
    "ref_map":        {"U1": "U9", "U2": "U10", "C1": "C13", ...}
  }
}
```

Keyed by the sheet name in the project's architecture (which the
LLM choose; may differ from the leaf's `provenance.source_sheet_name`).

### Pinning the imported layout

For each library-backed sheet, after renumbering the PCB fragment,
the synthesis stage:

1. Writes the renumbered fragment to
   `<project>/.experiments/subcircuits/<sheet_name>/round_imported_leaf_routed.kicad_pcb`
   and a sibling `metadata.json` describing the source.
2. Adds an entry to `<project>/.experiments/pins.json` (schema
   `pins.v1`, same shape the existing pin manager produces)
   referencing the imported round path.

The existing pin manager (`kicraft/autoplacer/brain/pins.py`)
already handles "leaves with pinned solutions" — the importer just
seeds its inputs.

---

## 4. CircuitChat integration

### Sheet model changes

Edit `kicraft/circuitchat/models.py`. Add two optional fields to
`Sheet`:

```python
class Sheet(BaseModel):
    name: str
    stem: str
    function: str
    from_library: str | None = None     # "<name>@<version>" or None
    library_instance: int | None = None  # 1, 2, ... or None
```

Add a model validator: if `from_library` is set, `library_instance`
must be >= 1; if `library_instance` is set, `from_library` must be
set. For non-library sheets both are None.

### Architecture stage retrieval

Edit `kicraft/circuitchat/stages/architecture.py` and
`kicraft/circuitchat/prompts/architecture.md`.

Before each architecture-stage LLM call:

1. Load every valid manifest from `$KICRAFT_LEAF_LIB`.
2. Inject a "Available leaves" section into the system prompt, one
   block per leaf:

```
### usb-c-lipo-charger@1.2.0

**What it does**: USB-C 5V VBUS input, 1A LiPo charger using AP2112K
with charge status LED. ISET=1.5kΩ -> 500mA charge current.

**Interface (hierarchical labels)**:
- VBUS_IN (input)
- VBAT (bidirectional)
- GND (passive)
- CHG_STAT (output)

**Watch out for**: Single-cell LiPo only. Not suitable for 2S+ packs.
```

3. Add a behavioral directive to the prompt:

> "If one of the available leaves is a good match for a sheet you
> would otherwise design from scratch, set
> `Sheet.from_library = '<name>@<version>'` and
> `Sheet.library_instance = N` (1 for the first instance, 2 for
> the second, etc., if the same leaf is reused multiple times).
> The sheet's hierarchical-label interface MUST exactly match the
> leaf's declared interface. Use the leaf's label names verbatim in
> `inter_sheet_nets`. Do not redefine the leaf's interface. If no
> leaf is a good match, design the sheet from scratch as usual."

4. After the LLM responds, validate every Sheet with `from_library`
   set:
   - The reference `<name>@<version>` matches a leaf in the library.
   - The hierarchical labels referenced by the architecture's
     `inter_sheet_nets` for this sheet are exactly the leaf's
     declared interface (set equality on names + directions).
   - `library_instance` is sequential per leaf (1 first, 2 second,
     no gaps).

   On validation failure, treat it like any other architecture
   validation failure — re-prompt the LLM with the specific
   diagnostic.

### Reevaluation on every run

The retrieval list is rebuilt and the prompt re-injected on every
architecture-stage invocation. The LLM is free to:
- Pick a leaf it didn't pick last turn.
- Drop a leaf it previously picked.
- Switch from one leaf to another.

The state mutation is just the new `from_library` /
`library_instance` values on `Sheet` entries — downstream stages
read the current architecture, so dropped leaves automatically
become from-scratch sheets and vice versa.

### BOM stage handling

Edit `kicraft/circuitchat/stages/bom.py`. Before the LLM call:

1. Partition sheets into `library_sheets` (have `from_library`)
   and `generated_sheets`.
2. For each library sheet, load its manifest and parse its
   `bom.csv`. Collect those `BomPart` rows into
   `library_bom_parts` — note: these rows still have the leaf's
   own refs (e.g. `U1`); renumbering happens at synthesis time.
3. Hide library sheets from the LLM. The BOM stage prompt only
   mentions `generated_sheets`; the LLM emits BOM rows only for
   those.
4. After the LLM responds, merge `library_bom_parts` (refs still
   in leaf form) with the LLM's `BomPart` rows, then run the
   existing `BOM` model validators.

The state holds the merged BOM. Library-sourced BomPart rows are
flagged with an optional `source_leaf: str | None = None` field
on `BomPart` so the synthesis stage knows to apply the renumber
map.

### Synthesis stage

Edit `kicraft/circuitchat/synthesis/emitter.py` and
`kicraft/circuitchat/synthesis/autoplacer.py`.

For each `Sheet`:

- **If `from_library` is None**: emit the sheet as today (generate
  `.kicad_sch` from `BomPart` rows for this sheet, write hierarchical
  labels, etc.).

- **If `from_library` is set**:
  1. Load the leaf manifest. Verify `content_hash` matches.
  2. Verify dependencies: every `kicad_symbol_libs` /
     `kicad_footprint_libs` entry exists on the target system. Fail
     fast with `SynthesisValidationError` if any are missing.
  3. Compute the renumber map for this instance, with
     `project_refs` = the union of (a) refs already allocated to
     prior sheets in this synthesis run and (b) refs from any
     prior library-instance of any leaf in this synthesis run.
  4. Copy the leaf's `schematic.kicad_sch` to
     `<project>/<sheet.stem>.kicad_sch`, applying the renumber map
     to every `Reference` property.
  5. Copy the leaf's `pinned_layout.kicad_pcb` to
     `<project>/.experiments/subcircuits/<sheet.name>/round_imported_leaf_routed.kicad_pcb`,
     applying the renumber map to every `fp_text reference` and
     footprint `path` ref.
  6. Write `<project>/.experiments/subcircuits/<sheet.name>/round_imported_metadata.json`
     describing the source.
  7. Merge the leaf's `autoplacer_fragment.json` into the
     project's autoplacer JSON model (in-memory), applying the
     renumber map to every ref-shaped key/value.
  8. Add a `pins.json` entry for the imported round (matching the
     existing `pins.v1` schema).
  9. Record the renumber map in the in-memory autoplacer JSON's
     `library_leaves` key (see section 3).

After all sheets are processed:
- Write the assembled `<project>_autoplacer.json` (including
  `library_leaves`).
- Write the parent `.kicad_pro` and `.kicad_pcb` stub as today.
- Run the validation pass (section 5).

The existing parent composer (`subcircuit_composer.py`) consumes
the pin manifest at solve time, so library-backed sheets are
treated as pre-solved by every downstream stage.

---

## 5. Synthesis validation additions

Extend `kicraft/circuitchat/synthesis/validation.py`.

Two new checks:

- **SS9.7 — Ref uniqueness.** Every ref designator across the
  project (schematic + PCB + autoplacer.json) is globally unique.
  Failure mode: a renumber map collision. Fail with
  `SynthesisValidationError` listing the duplicates.

- **SS9.8 — Library interface match.** For every sheet with
  `from_library`, the hierarchical labels in the emitted
  `.kicad_sch` exactly match the manifest's declared interface
  (set equality on names + directions). Failure mode: the leaf
  on disk was edited between architecture and synthesis. Fail
  with `SynthesisValidationError` naming the mismatch.

Both run inside the existing mechanical-stage validation pass,
before the file set is declared written.

---

## 6. CLI surface (`kicraft-leaf`)

A new console_script in `pyproject.toml`:

```toml
[project.scripts]
kicraft-leaf = "kicraft.cli.leaf:main"
```

v1 exposes **read-only** commands only — curation is GUI-driven.

```
kicraft-leaf list
    Print every leaf in the library: name, version, hash, tags,
    description first line. One row per leaf. Reads from
    $KICRAFT_LEAF_LIB or default.

kicraft-leaf show <name>
    Pretty-print one leaf's full manifest. Exit 1 if not found.

kicraft-leaf path
    Print the resolved library path. Useful for scripting.
```

No `extract`, `promote`, `remove`, `pack`, or `unpack` in v1.

---

## 7. GUI — Leaf Library tab

New top-level tab. Register in `kicraft/gui/app.py` alongside the
existing tabs (CircuitChat, Setup, Analysis, etc.). Implementation
file: `kicraft/gui/pages/leaf_library.py`.

### Tab layout — two sections

**(A) Promote a new leaf — top section.**

A vertical stepper / wizard:

1. **Source project**: defaults to the currently loaded project in
   the GUI session if one is loaded. Otherwise a directory picker.

2. **Pick a leaf**: scans the source project for sheets that have
   a pinned snapshot — read from `<project>/.experiments/pins.json`
   and `<project>/.experiments/subcircuits/<leaf>/round_NNNN_leaf_routed.kicad_pcb`.
   Lists the candidate sheets with their hierarchical-label
   interface and a thumbnail per available pinned round (using
   the existing `render_pcb.py` if no cached thumbnail exists).
   User picks one sheet and one round.

3. **Fill metadata**: form fields:
   - **Name** (kebab-case slug, validated against the manifest
     rule). If a leaf with this name already exists in the
     library, the form shows the existing version and offers an
     auto-bumped patch version as the default.
   - **Version** (semver, defaulted as above; `0.1.0` for first
     promotion).
   - **Description** (multi-line text area, required).
   - **Tags** (chips with autocomplete from existing tags across
     the library).
   - **Watch out for** (optional multi-line note).

   Read-only pre-populated views of:
   - Interface (hierarchical labels parsed from the leaf
     `.kicad_sch`).
   - BOM rows (sliced from the project's `BOM` state, filtered to
     parts with `sheet == <selected sheet name>`).
   - KiCad symbol / footprint library dependencies (scanned from
     the BOM's `symbol` / `footprint` columns — the part before
     the colon is the library name).
   - Refs (sorted, deduplicated, parsed from the schematic).

4. **Preview & confirm**: shows the three renders (`front_all`,
   `back_copper`, `copper_both`) plus the populated manifest. A
   single **Confirm and write** button.

   On confirm:
   - If `<library_dir>/<name>/` exists and contains a leaf with
     the same `version`, refuse and show "version already exists
     — bump and retry."
   - Otherwise, atomically write the new leaf directory (write to
     a sibling `.tmp` dir first, `os.replace` into place to avoid
     partial writes).
   - Compute and store the `content_hash`.
   - Render and cache the three views into `renders/` and produce
     a 256×256 `thumbnail.png` (downscaled from `copper_both`).
   - Show a success toast linking to the leaf directory.

**(B) Installed leaves — bottom section.**

A card grid, one card per leaf:

- Thumbnail (256×256 from `renders/thumbnail.png`).
- Name + version.
- Description (truncated to 3 lines, expandable on click).
- Tags as chips.
- Source-project provenance (`from llups, round 47, 2026-05-17`).
- A **Remove** button.

On Remove click: confirmation modal with the warning *"This will
not affect projects that have already imported this leaf. Existing
projects keep their imported copy in their own files."* Confirming
deletes the leaf directory.

No "edit" affordance. To change a leaf, promote a new version.

### Banner in the CircuitChat tab

Edit `kicraft/gui/pages/circuitchat.py`. When the state's
`architecture` contains one or more sheets with `from_library`
set, render a small status line above the chat input:

> *"Reusing 2 leaves from the Leaf Library:
> usb-c-lipo-charger@1.2.0, ldo-3v3-500ma@1.0.0. Skipping LLM
> generation for those sheets."*

Update on every architecture-stage commit (the same hook that
refreshes the state-slot panel).

---

## 8. New / modified files

### New modules

| Path | Purpose |
|---|---|
| `kicraft/leaf_library/__init__.py` | Package marker + public re-exports. |
| `kicraft/leaf_library/manifest.py` | `Manifest` pydantic model + JSON load/dump + validation rules (section 2). |
| `kicraft/leaf_library/loader.py` | Resolve `$KICRAFT_LEAF_LIB` / default; iterate library; load + verify hashes; surface broken-leaf reasons. |
| `kicraft/leaf_library/renumber.py` | `renumber_leaf(...)` + `parse_ref(...)` (section 3). |
| `kicraft/leaf_library/extractor.py` | Build a leaf directory from a source project + sheet name + round (used by the GUI promote wizard). |
| `kicraft/leaf_library/installer.py` | Apply a leaf to an in-progress synthesis: copy + renumber the sch, the pinned pcb, the autoplacer fragment, the BOM; write the pins.json entry; record the ref_map. |
| `kicraft/cli/leaf.py` | `kicraft-leaf list / show / path`. |
| `kicraft/gui/pages/leaf_library.py` | The Leaf Library tab (section 7). |

### Modified modules

| Path | Change |
|---|---|
| `kicraft/circuitchat/models.py` | Add `from_library`, `library_instance` to `Sheet`. Add optional `source_leaf` to `BomPart`. Add cross-field validators. |
| `kicraft/circuitchat/prompts/architecture.md` | Add the "Available leaves" injection point + behavioral directive. |
| `kicraft/circuitchat/stages/architecture.py` | Load library, inject leaf summaries into the system prompt, validate `from_library` and `library_instance` against the loaded library + against `inter_sheet_nets`. |
| `kicraft/circuitchat/stages/bom.py` | Partition sheets; merge library-sourced BOM rows; flag with `source_leaf`. |
| `kicraft/circuitchat/synthesis/emitter.py` | Route library-backed sheets through the installer; route others through the existing emitter path. |
| `kicraft/circuitchat/synthesis/autoplacer.py` | Merge library autoplacer fragments (renumber-mapped) into the project autoplacer JSON. Write the `library_leaves` map. |
| `kicraft/circuitchat/synthesis/validation.py` | SS9.7 (ref uniqueness), SS9.8 (library interface match). |
| `kicraft/gui/app.py` | Register the Leaf Library tab. |
| `kicraft/gui/pages/circuitchat.py` | Banner for library-backed reuse. |
| `pyproject.toml` | Add `kicraft-leaf` console_script. |
| `README.md` | One section describing the Leaf Library — location, how to promote via GUI, how reuse is automatic during CircuitChat. |

---

## 9. Test plan

### Unit tests

- `tests/leaf_library/test_manifest.py`
  - Round-trip JSON load/dump.
  - Each validation rule (section 2) has a positive and negative case.
  - `content_hash` recomputation matches stored hash.
- `tests/leaf_library/test_renumber.py`
  - Single-instance: leaf refs land in the next available slot per
    letter-class, no collisions with `project_refs`.
  - Multi-instance: second call sees first call's allocations.
  - Refs with suffix forms (`U1A`) are rejected at manifest load,
    never reach the renumberer.
  - Determinism: same inputs produce the same map across runs.
- `tests/leaf_library/test_loader.py`
  - Missing directory -> empty library, no error.
  - Malformed manifest -> excluded, reason logged.
  - Hash mismatch -> excluded, reason logged.
- `tests/leaf_library/test_extractor.py`
  - Given a small fixture project with a pinned leaf, the
    extractor produces a directory that the loader accepts.
- `tests/leaf_library/test_installer.py`
  - Round-trip: extract a leaf from project A, install it into a
    fresh project B, verify the emitted sch/pcb/autoplacer all
    use the renumbered refs and no orphan original refs remain.
  - Multi-instance install: two instances of one leaf produce two
    non-overlapping ref ranges and two pins.json entries.

### Integration tests

- `tests/circuitchat/test_library_reuse.py`
  - Populate `$KICRAFT_LEAF_LIB` with a fixture leaf.
  - Run the architecture stage with a spec that should match;
    assert `Sheet.from_library` is set on the matched sheet.
  - Run the BOM stage; assert library BOM rows are present with
    leaf-local refs and `source_leaf` flagged.
  - Run synthesis; assert the project on disk contains the
    renumbered sch, the pinned pcb in `.experiments/subcircuits/`,
    a pins.json entry, and the `library_leaves` map in
    autoplacer.json.
  - SS9.7 and SS9.8 pass on the emitted project.
- `tests/circuitchat/test_library_reevaluation.py`
  - Run architecture once with a matching spec -> leaf picked.
  - Mutate the spec so the leaf no longer fits.
  - Run architecture again -> `Sheet.from_library` is None.

### Manual smoke test (in the spec for the implementer)

1. Promote the existing `llups` project's `CHARGER` leaf via the
   GUI Leaf Library tab.
2. Start a fresh CircuitChat conversation with a similar project
   description ("USB-C 5V to 1S LiPo with status LED + 3.3V LDO").
3. Verify the CircuitChat banner shows the reused leaf.
4. Run `--synthesize` and inspect the emitted project: the
   CHARGER sheet's refs are renumbered to fit the new project,
   the pinned pcb shows up under `.experiments/subcircuits/CHARGER/`,
   and the `library_leaves.CHARGER` entry in autoplacer.json
   records the renumber map.
5. Run `autoexperiment --parents-only` against the emitted
   project — the parent composer should consume the pinned leaf
   directly without re-solving it.

---

## 10. Implementation order

A suggested order that keeps the system runnable at each step:

1. **Manifest + loader.** Pydantic model, validation, loader,
   `kicraft-leaf list / show / path`. Nothing depends on this yet
   but it can be tested standalone.
2. **Extractor + GUI promote wizard.** Lets the user populate the
   library from a real project. CircuitChat reuse not yet wired.
3. **GUI Leaf Library tab (installed leaves list + remove).**
   Read-side of the GUI is now complete.
4. **Renumber + installer.** Pure functions, tested in isolation.
5. **Architecture stage retrieval.** Wire the loader into the
   architecture system prompt + add the Sheet validators. At this
   point the LLM can pick leaves but synthesis ignores the picks.
6. **BOM stage partition.** Library sheets are now correctly
   excluded from BOM generation.
7. **Synthesis stage emitter + autoplacer merge.** Library
   reuse is now end-to-end. Banner in CircuitChat tab.
8. **Validation SS9.7 / SS9.8.**
9. **README section + smoke test.**

Each step ends with a green test suite and a manually runnable
slice of the feature.
