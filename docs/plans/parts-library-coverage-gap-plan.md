# Parts-Library Coverage Gap: vendor missing high-use parts + standing worklist

## 1. Problem

The curated library (`kicraft/parts_library/`) has no canonical bundle for the
3.5 mm audio jack or the quad op-amp, so every audio/analog design falls back
to the BOM stage's auto-fetch (home tier, `~/.kicraft/parts/<slug>/`). The
provenance audit (`triage audits`, block `[A]`) flags these as `home-fetched`
("auto-fetched — curated library lacks it — coverage gap").

Evidence (three sibling runs of the *same* brief, plus three R-2R DAC runs):

| run | date | home-fetched parts |
|---|---|---|
| `1/733` | 2026-08-24 | `pj-320d` (J1–J4), `mcp6004` (U1–U2) |
| `1/681` | 2026-08-07 | `pj-320a-4p`, `mcp6004`, `dc005-barrel-jack` |
| `1/680` | 2026-08-07 | `pj-320a`, `tl074` |

The three runs picked **different slugs** for the *same* conceptual parts
(`pj-320d` / `pj-320a` / `pj-320a-4p` for the jack; `mcp6004` / `tl074` for
the op-amp). The slug is deterministic given the LCSC id — the variance is the
model choosing different LCSC parts per run because no canonical block steers
it. This is exactly what the core-blocks catalog exists to fix.

Non-blocking, but it costs money and reproducibility: every auto-fetch is extra
tool rounds on the hosted model, and the variant drift makes boards
non-reproducible run-to-run.

## 2. Machinery to reuse (do NOT re-invent)

- **Tier resolver** — `kicraft/parts_library/loader.py`. Search order:
  project → vendored (`kicraft/parts_library/`) → home (`~/.kicraft/parts/`)
  → extra. Vendoring writes to `vendored_parts_dir()` = `kicraft/parts_library/`,
  which shadows any stale home-tier cache.
- **Vendoring** — `kicraft add-part --from-lcsc C<NNNNN> --into vendored`
  (handler `_cmd_add_part`, `kicraft/design/cli_app.py:2220`). Fetches
  symbol + footprint + 3D model from EasyEDA, writes
  `<libname>/{manifest.json,<libname>.kicad_sym,<libname>.pretty/}`,
  finalizes `content_hash`. Default maturity = `prototype`.
- **Maturity** — `kicraft promote-part --tier vendored --name <slug> --to <…>`
  (`_cmd_promote_part`, `cli_app.py:2527`). `production` requires a real
  `3d/*.step|stp|wrl`. `reviewed` = human-checked, no 3D requirement.
- **Canonical block registry** — `kicraft/parts_library/core_blocks.json`
  (+ `core_blocks.py`). One default part per functional block. A `bundle` row
  names a vendored bundle; MPN/LCSC are *derived from the manifest at sync time*
  (never duplicated). The account store re-syncs `core_components` from this
  catalog on every store init (`accounts.py:_sync_core_components_from_catalog`),
  and `stage_driver._format_core_defaults_block` injects the table into the
  architecture/BOM prompt as the preferred-part guidance.
- **Enumeration** — `python -m kicraft.cli.part_query_report` (per-machine
  telemetry `~/.kicraft/part_queries.jsonl`): `fetches` counter = "LCSC fetched
  into the library" (miss candidates); `lib_hits` = popularity. Also
  `triage audits` block `[A]` (`collect_library_provenance`, `triage.py:747`)
  resolves the *final* tier of every BOM part from `state.json`.

## 3. Part A — vendor the specific missing parts

Run from the repo root with the venv python. `easyeda2kicad` must be installed
(already is — the BOM stage uses it).

Confirmed LCSC ids (from the three runs' `.kicraft/bom_prices.json`):

```bash
REPO=/home/kicraft/KiCraft; PY="$REPO/.venv/bin/python"

# 3.5 mm SMD audio jack — PJ-320D (LCSC C431535, stock ~50k)
"$PY" -m kicraft.design.cli_app add-part --from-lcsc C431535 --into vendored --name pj-320d

# Quad op-amp — MCP6004-I/SL (LCSC C1346056) — the unity-gain-buffer workhorse
"$PY" -m kicraft.design.cli_app add-part --from-lcsc C1346056 --into vendored --name mcp6004
```

Then promote each to `reviewed` (human check) and, once the 3D model is present
(`add-part` fetches it by default), to `production`:

```bash
# NOTE: promote-part takes the slug positionally; validate-part takes a path.
"$PY" -m kicraft.design.cli_app promote-part pj-320d --to reviewed --tier vendored
"$PY" -m kicraft.design.cli_app promote-part mcp6004 --to reviewed --tier vendored
# after eyeballing the 3D model + validate-part:
"$PY" -m kicraft.design.cli_app validate-part kicraft/parts_library/pj-320d
"$PY" -m kicraft.design.cli_app validate-part kicraft/parts_library/mcp6004
```
**Variant sweep (cover the drift).** The same three runs also emitted
`pj-320a` (C77093 — confirm), `pj-320a-4p` (C431533), `tl074`, and
`dc005-barrel-jack`. Resolve the remaining ids and vendor them too, so whichever
slug a *legacy* BOM references resolves curated:

- `1/680` ids to disambiguate: `C6964, C77093, C2691448, C2884926, C8401`.
- `1/681` ids to disambiguate: `C15850, C18185602, C492425, C431533`.

Disambiguate with `lookup-lcsc-id` / the offline catalog, then vendor each with
an explicit `--name` matching the slug its run already emitted. This is
belt-and-suspenders: after Part B steers new designs to `pj-320d`/`mcp6004`,
the variant bundles only serve to clean up historical runs and guard against
model fallback.

## 4. Part B — register canonical blocks in core_blocks.json

Vendoring alone does not steer the BOM stage. Register a `bundle` block per
missing functional role so the architecture/BOM prompt tells the model which
default part to use. Edit `kicraft/parts_library/core_blocks.json`
(`blocks` array; schema in `core_blocks.py:CoreBlock` — exactly one of
`bundle`/`stock`, category in `power|sensors|drivers|interface|passives`).

Proposed rows (pick `sort_order` consistent with neighbours; the catalog is
git-edited, DB re-syncs automatically on next store init):

```json
{
  "function_key": "audio-jack-3-5mm",
  "display_name": "3.5mm TRS audio jack",
  "category": "interface",
  "qualifier": "SMD, tip/ring/sleeve + detect",
  "package": "PJ-320D SMD",
  "selection_notes": "C431535 PJ-320D. Legacy runs also picked PJ-320A (C77093) and PJ-320A-4P (C431533); the vendored pj-320a / pj-320a-4p bundles shadow those if the model falls back.",
  "sort_order": <next in interface>,
  "bundle": "pj-320d"
},
{
  "function_key": "opamp-quad-general",
  "display_name": "Quad op-amp (general/unity-gain buffer)",
  "category": "drivers",
  "qualifier": "rail-to-rail, 1MHz GBW, single supply",
  "package": "SOIC-14",
  "selection_notes": "C1346056 MCP6004-I/SL. For higher slew/low-noise use TL074 (vendored tl074 bundle) — pick per the design's signal requirement.",
  "sort_order": <next in drivers>,
  "bundle": "mcp6004"
}
```

Category choice is a judgement call (`interface` vs `drivers` for a jack;
`drivers` for a signal op-amp). Verify against how `_format_core_defaults_block`
groups/renders categories before finalizing.

**Verification of the steering path (do not skip):** confirm the architecture
BOM stage actually consults the injected core-defaults table for these roles.
Trace `stage_driver._format_core_defaults_block` → the prompt template → the
architecture/BOM tool schema. If a block only *suggests* and the model can
still ignore it, that is a follow-up gap, not a blocker for this plan — but the
plan must record what actually steers the pick (see §7 open questions).

## 5. Part C — standing "missing parts" worklist (the generalizable piece)

The recurring loop must not depend on a human triage investigation. Add a
cross-run aggregation of home-fetched parts to `triage scan`, which already
walks every run dir (`projects/`, `~/.kicraft/self_eval/`, `logs/self_eval/`).

Change in `kicraft/cli/triage.py`:

1. In `collect_scan`, for each run with a resolvable `state.json` BOM, call the
   existing `collect_library_provenance(run)` and tally the `home-fetched` (and
   `missing-lib`) `sym_lib`/`fp_lib` slugs into a `Counter` keyed by slug,
   recording the run ids per slug.
2. Emit a new scan bucket, e.g.:

   ```
   home-fetched libraries (#designs; >1 = VENDOR IT):
     pj-320d: 1  e.g. ['1/733']
     mcp6004: 2  e.g. ['1/733','1/681']
     ...
   ```

   Rank by #designs, then by `latest` run mtime — mirroring the existing scan
   buckets' `latest=`/`sha=` discipline so a stale hit (pre-dating a vendor) is
   distinguishable from a regression.
3. Pin it in `tests/test_triage_cli.py` the same way the other scan buckets are
   pinned against artifact drift.

This turns "which parts keep missing the library" into a continuously
re-emitted worklist instead of a per-investigation discovery. Keep
`part-query-report` as the *live* (per-machine, recent) complement; the scan
bucket is the *corpus-complete* (historical, state.json-derived) view.

## 6. Acceptance criteria

- `triage audits 1/733` block `[A]` now reports `curated-default` for J1–J4
  (`pj-320d`) and U1–U2 (`mcp6004`) — no `home-fetched` flags on this run.
- `triage audits 1/681` and `1/680` show the same after their variant slugs are
  vendored (`pj-320a-4p`, `dc005-barrel-jack` / `pj-320a`, `tl074`).
- `core_blocks.json` validates (`CoreBlockCatalog.model_validate`) and the new
  blocks appear in the synced `core_components` table and in
  `_format_core_defaults_block` output.
- `triage scan` prints the new `home-fetched libraries` bucket, and the pinned
  `test_triage_cli.py` test passes.
- A fresh audio-design synthesis (offline `synthesize` on a frozen seed, or a
  new run) picks the vendored `pj-320d`/`mcp6004` slug and does **not** emit a
  new `add_part_from_lcsc` fetch for those roles — observable in
  `~/.kicraft/part_queries.jsonl` (`add_part_from_lcsc`/`fetched` count stays
  flat for C431535/C1346056).

## 7. Open questions / risks

- **Steering strength.** Confirm whether a `core_components` row *deterministically*
  selects the part or is advisory prompt text. If advisory, the auto-fetch can
  still fire when the model deviates — note it and, if it recurs, consider a
  deterministic architecture-stage part-family pre-resolution
  (`cli_app.py:_unresolved_architecture_parts`) for these roles.
- **Symbol/footprint name matching.** The vendored bundle must expose the same
  `symbol_name`/`footprint_name` the BOM emits (`PJ-320D`,
  `AUDIO-SMD_PJ-320D-1`). Verify with `list-parts` after vendoring; if
  EasyEDA's exported names differ, pass `--symbol-name` / adjust the footprint
  name so existing BOM refs (`pj-320d:PJ-320D`,
  `pj-320d:AUDIO-SMD_PJ-320D-1`) resolve.
- **`add-part` also runs hygiene normalizers** (courtyard rebuild, PTH
  normalization, emech pin-type retyping — `_ensure_vendored_courtyard_clearance`,
  `_normalize_emech_pin_types`). If a fetch fails one, fix at the bundle, not by
  suppressing the check.
- **`refresh_sample_previews.py` is NOT part of vendoring.** It regenerates
  landing-page sample-board 3D previews; it does not touch the parts library.
  Only run it if a vendored part is later adopted into a showcase sample.

## 8. Non-goals

- No post-hoc edits to the failed board `1/733` (its failure is per-run wiring
  model output, not this library gap).
- No changes to the BOM-stage auto-fetch fallback itself — it is correct to
  keep as a safety net; this plan only removes the *need* for it on the covered
  parts.
