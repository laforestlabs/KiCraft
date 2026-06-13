# Implementation plan: single source of truth for KiCraft parts

Status: APPROVED PLAN, NOT YET IMPLEMENTED (written 2026-06-12).
Audience: an implementing agent with no prior session context. Everything
needed is in this file plus the referenced code. Verify line numbers before
editing; they were checked on 2026-06-12 against main + the 3D-models PR
stack (#83-#88), which this work assumes is merged.

## Working notes for the implementer

- Use the repo venv for everything: `.venv/bin/python` (has pcbnew +
  easyeda2kicad; network works).
- Origin is Codeberg/Forgejo: no `gh`. Create PRs with
  `curl --netrc https://codeberg.org/api/v1/repos/LaForestLabs/KiCraft/pulls`.
  Do NOT self-merge to main; the user merges. No Claude attribution in
  commits.
- EasyEDA API rate limit: HTTP 403 after ~17 rapid fetches, ~10 min
  cooldown; space fetches ~20 s apart (each part = 2+ requests: CAD data +
  3D binary).
- Editing any file inside a bundle invalidates its manifest
  `content_hash`; finish every bundle touch with
  `kicraft validate-part <dir> --update-hash`.
  (`kicraft` here = `.venv/bin/python -m kicraft.design.cli_app`.)
- Known pre-existing test failures on clean main (NOT regressions):
  `tests/test_solve_subcircuits_layout_persistence.py::test_best_round_to_layout_prefers_routed_board_geometry`
  and `tests/test_web_index_autoopen.py::test_cloned_project_opens_with_bom`.

## Context (why)

KiCraft has three overlapping part lists that drifted apart:

1. **Vendored bundles** (`kicraft/parts_library/<name>/`, 23 today): the
   files the pipeline actually resolves (symbol/footprint/3D models,
   manifest with mpn/sourcing/maturity). All carry 3D models and strict
   model-path validation as of PRs #83-#88.
2. **Core-components registry** (`kicraft/server/core_components_seed.json`,
   43 rows: function_key -> default LCSC pick + display metadata + price/
   stock snapshots). Seeded ONCE into the accounts DB
   (`AccountStore._maybe_seed_core_components`, accounts.py ~1966, called
   from `__init__` ~327, guarded by the `core_components_seed_version`
   app-setting); after that the DB is authoritative and the JSON is dead
   weight. Admin edits (full CRUD, /admin/core-components, web.py
   ~3712-4014) live only in the DB. Rendered per run into the
   architecture/BOM prompts by `stage_driver._format_core_defaults_block`
   (~182-216); the adoption rule tells the model to fetch the C-number via
   `add_part_from_lcsc`.
3. **Home-tier bundles** (`~/.kicraft/parts`): prototype-maturity caches
   auto-written by that fetch flow (`stage_driver.py` ~334-338,
   `add-part --from-lcsc ... --into home`).

Problems: only 2 of 43 registry defaults are vendored (ldo-3v3-1a =
ams1117-3v3/C6186, boost-5v-1a = mt3608/C84817), so standard picks get
re-fetched per box as unreviewed prototypes; TP4056 and WS2812B exist as
DIFFERENT variants on each side (vendored tp4056=C16581 vs registry
C725790; vendored ws2812b=C114586 vs registry WS2812B-B/T C2761795);
registry rows duplicate manifest metadata; seed JSON and DB drift by
design.

## User decisions (fixed; do not relitigate)

1. **Repo is canonical.** The DB becomes a cache re-synced from the repo on
   every `AccountStore` init. The admin page keeps ONLY enable/disable and
   jlcparts price/stock refresh; part/block edits happen via git.
2. **Vendor ALL bundle-backed registry defaults now**: 32 new bundles + 2
   conflict re-vendors = 34 EasyEDA fetches; library grows 23 -> 55.
3. **Conflicts adopt the REGISTRY picks**: re-vendor tp4056 from C725790
   and ws2812b from C2761795 (same bundle dir names, content replaced).
4. **Loader precedence is a plain tier swap** (no maturity conditionals):
   project > vendored > home > extra. One rule: curated repo content beats
   auto-fetched caches; a deliberate override lives in the project tier.

## Target architecture

### New canonical catalog

`kicraft/parts_library/core_blocks.json` + pydantic model
`kicraft/parts_library/core_blocks.py` (follow the `manifest.py` pattern).
Move `CORE_COMPONENT_CATEGORIES` and the function-key regex here from
accounts.py; accounts.py imports them.

Row fields: `function_key, display_name, category, qualifier,
package (authored prose), selection_notes, sort_order`, plus EXACTLY ONE of:

- `bundle: "<vendored-bundle-name>"`: mpn/lcsc are DERIVED from that
  bundle's manifest (`mpn`, `sourcing["lcsc"]`) at sync time; never stored
  in the catalog.
- `stock: {"series": "..."}`: the 6 passives rows (res-0402/0603/0805,
  cap-mlcc-0402/0603/0805); stock KiCad `Device:R`/`Device:C` symbols.
- transitional `default_mpn` + `default_lcsc`: allowed only while
  vendoring is in flight; forbidden by the validator/guard in the final PR.

No `price_usd`/`stock`/`snapshot_date`/`enabled` in the catalog: those are
DB-owned runtime state. Two rows may share one bundle (gyroscope and
imu-6axis both -> lsm6ds3tr-c). Provide `load_core_catalog()` and
`resolve_block()` (flattens a row to the DB shape; reads the bundle
manifest with `load_manifest` ONLY: no content-hash verification, because
sync runs on every AccountStore init and must not hash ~50 MB of STEP).

Example rows:

```json
{
  "function_key": "ldo-3v3-1a",
  "display_name": "LDO 3.3V (<=1A)",
  "category": "power",
  "qualifier": "<=1A @ 3.3V out, Vin to 15V",
  "package": "SOT-223",
  "selection_notes": "JLC Basic jellybean; ...",
  "sort_order": 20,
  "bundle": "ams1117-3v3"
},
{
  "function_key": "res-0402",
  "display_name": "Resistor series 0402",
  "category": "passives",
  "qualifier": "1% thick film, E24 values",
  "package": "0402",
  "sort_order": 310,
  "stock": {"series": "UNI-ROYAL 0402WGF series"}
},
{
  "function_key": "tof-distance",
  "display_name": "ToF distance sensor",
  "category": "sensors",
  "qualifier": "I2C, ranges to 2m",
  "package": "SMD-12 4.4x2.4mm",
  "sort_order": 140,
  "default_mpn": "VL53L0CXV0DH/1",
  "default_lcsc": "C91199"
}
```

Seed the initial catalog content from `core_components_seed.json` (43 rows:
2 bundle-backed, 6 stock, 35 transitional).

### DB becomes a re-synced cache (accounts.py)

- Replace `_maybe_seed_core_components` with
  `_sync_core_components_from_catalog()` called from `__init__` on every
  open. One transaction: upsert by `function_key` (overwrite canonical
  fields: display_name, category, qualifier, default_mpn, default_lcsc,
  package, selection_notes, sort_order, bundle; PRESERVE enabled,
  price_usd, stock, snapshot_date, created_at; touch updated_at only on
  change), delete rows whose key left the catalog, delete the
  `core_components_seed_version` app-setting. Log a one-line
  inserted/updated/deleted diff summary.
- Add a nullable `bundle TEXT` column via additive `ALTER TABLE` guarded by
  `PRAGMA table_info` (same pattern as the `build_jobs.kind` migration,
  accounts.py ~480-483). Add `"bundle"` to `_CORE_COMPONENT_FIELDS` and
  `_normalize_core_component`.
- Resilience: catalog or manifest read errors must warn and keep existing
  DB rows; never raise out of `__init__` (matches the web.py ~1691-1696
  "registry trouble never blocks a run" stance).
- Delete the seed JSON file and its constants. Remove public
  `create_core_component`/`delete_core_component`; keep
  `update_core_component` (enable/disable) and
  `record_core_component_snapshot` (refresh).

### Admin page shrink (web.py ~3712-4014)

Remove the New/Edit/Delete dialogs and buttons. Keep enable/disable
(`do_set_enabled`), jlcparts `do_refresh`, and the read-only table with a
new `bundle` column next to LCSC. Header prose: "Synced from the repo
catalog (kicraft/parts_library/core_blocks.json) on every restart; part
and block edits happen via git; this page owns only enable/disable and
price/stock snapshots." Fix the stale "pipeline does not consume this
registry yet" docstring (it does, since PR #82).
`Settings.enable_core_defaults` / `KICRAFT_CORE_DEFAULTS`
(server/config.py ~138, 202, 240) stays untouched.

### Prompt rendering + stage specs

`stage_driver.py _format_core_defaults_block` (~182-216):

- Table gains a `bundle` column:
  `| function_key | block | qualifier | default part | LCSC | package | bundle |`.
- Adoption rule split by row kind:
  - bundle-backed: "rows with a `bundle` are ALREADY in the parts library:
    take the exact symbol/footprint ids from the Available parts table
    (extras.parts_block) or list_parts; do NOT call add_part_from_lcsc or
    lookup_lcsc_id for them."
  - passives/stock: unchanged.
  - transitional (LCSC, no bundle): keep today's fetch sentence; delete it
    at the guard flip.
- Same rewrite in the `BOM_TOOLS` blurb for `add_part_from_lcsc` and the
  CORE DEFAULTS paragraph in `_stage_extra("bom")` (~152-158).
- Stage specs live at `.claude/skills/kicraft/stages/`:
  - `bom.md` ~line 47: the "adopt before researching" paragraph currently
    says "fetch its bundle with the given C-number in ONE call"; rewrite
    per the split above.
  - `architecture.md` lines 7 and 57; line 57 names "the vendored ch340n
    bundle" as a CH340C alternative: drop that clause once `ch340c` is
    vendored (batch V3).
- `kicraft/design/library.py _format_available_parts_block` (~158-205)
  stays the structural source of exact ids; the core table carries only
  bundle NAMES, never symbol/footprint ids, so there is no duplication.

### Prompt budget fixes (LOAD-BEARING; land in PR1)

Measured on 2026-06-12 with the real formatters: core block = 5.1 KB;
parts block = 14.2 KB for 23 bundles (8.9 KB of that is `watch_out_for`
notes); `list_parts` tool output is ALREADY truncated today by the
`[:8000]` cap in `_bom_executor` (stage_driver.py ~315) and the BOM extras
blob by `json.dumps(extras)[:24000]` (~511, wiring gets 40000). At 55
bundles the parts table alone is ~34 KB, so without fixes the new "read
ids from list_parts" rule points at silently truncated data.

Fix all three: cap rendered `watch_out_for` to ~140 chars/part in
`_format_available_parts_block`; raise the list_parts cap to ~40000 (and
the `[:5000]` list-parts echo in the add_part branch, ~344); raise the BOM
extras budget 24000 -> 48000. Re-measure in the final PR with all 55
bundles.

### Loader tier swap (loader.py)

`resolve_tier_dirs` (~95-109) currently orders project > HOME > VENDORED >
extra. Swap to **project > VENDORED > HOME > extra**. Update the module
docstring (lines 1-18) and `kicraft/parts_library/__init__.py` prose.
This makes stale home-tier caches harmless after the re-vendors, on every
machine, with no cleanup scripts and no maturity conditionals.

## Vendoring workflow

Tooling to land first (PR1):

- Extend `add-part --into` with `vendored` (`_resolve_dest_dir`,
  cli_app.py ~694-697, + argparse choices ~2740): writes straight into
  `kicraft/parts_library/<slug>/` reusing the existing slug/sanitize/3D
  logic (3D is fetched by default since PR #85).
- New `scripts/render_check_bundles.py`: the probe-render harness (build a
  one-footprint board per bundle, `kicad-cli pcb render` top + oblique
  PNGs into a gitignored out dir for human review). This has been written
  ad hoc twice already (see commit 3360700's review notes); make it a repo
  script. Accept bundle paths or `--all-vendored`.

Per part: `kicraft add-part --from-lcsc C### --into vendored --name <slug>`
-> author `description`/`tags`/`watch_out_for`/`datasheet_url` in the
manifest -> `validate-part <dir> --update-hash` ->
`scripts/render_check_bundles.py` + human eyeball ->
`promote-part <slug> --to production --tier vendored` -> flip the catalog
row from `default_lcsc` to `bundle: <slug>`.

Slug table (32 new + 2 re-vendors; batches sized for the EasyEDA limit):

| batch | function_key | LCSC | slug |
|---|---|---|---|
| V1 power | ldo-3v3-500ma | C82942 | me6211c33 |
| V1 | ldo-3v3-3a | C151391 | az1084c-3v3 |
| V1 | boost-5v-2a | C87357 | tps61088 |
| V1 | buck-3v3-1a | C141836 | tlv62569 |
| V1 | buck-3v3-2a | C780769 | ap63203 |
| V1 | buck-5v-2a | C2071056 | ap63205 |
| V1 | buck-adj-3a | C90761 | tps54331 |
| V1 | fuel-gauge-1s | C2682616 | max17048 |
| V1 | current-sensor | C49851 | ina226 |
| V2 sensors | photodiode | C146236 | pd15-22c |
| V2 | adc-external | C37593 | ads1115 |
| V2 | hall-effect | C314698 | ah49e |
| V2 | tof-distance | C91199 | vl53l0x |
| V2 | accelerometer | C110926 | lis2dh12 |
| V2 | gyroscope + imu-6axis | C967633 | lsm6ds3tr-c (one bundle, two rows) |
| V2 | pir | C90465 | as312 |
| V2 | temp-humidity | C2757850 | aht20 |
| V2 | pressure-sensor | C779278 | bmp388 |
| V2 | ambient-light | C504893 | veml7700 |
| V2 | magnetometer | C404328 | mmc5603nj |
| V2 | temp-sensor-1wire | C376006 | ds18b20 |
| V2 | cap-touch | C42422128 | ttp223 |
| V2 | i2s-microphone | C27636198 | msm261d |
| V3 drv+iface | stepper-driver | C38437 | a4988 |
| V3 | dc-motor-driver | C50506 | drv8833 |
| V3 | servo-pwm-driver | C2678753 | pca9685 |
| V3 | audio-amp-i2s | C910588 | ns4168 |
| V3 | audio-amp-analog | C112137 | pam8302a |
| V3 | rtc | C269877 | bm8563 |
| V3 | usb-uart-bridge | C84681 | ch340c |
| V3 | io-expander | C42420608 | tca9555 |
| V3 | i2c-mux | C130026 | tca9548a |
| V4 conflicts | lipo-charger-1s | C725790 | tp4056 (--overwrite) |
| V4 | addressable-led | C2761795 | ws2812b (--overwrite) |

Rate-limit pacing: V1 = 9 parts in one sitting (20 s spacing); V2 = 14
parts as 7 + ~10 min pause + 7; V3 = 9; V4 = 2. THT parts (as312 TO-5,
ds18b20 TO-92) may lack EasyEDA 3D models: `add-part` succeeds without a
model but `promote-part --to production` refuses; leave such stragglers at
`reviewed` (ip2368 precedent) and note it in the PR.

## Re-vendor reference fixups (PR-V4)

Verified-complete list of references to the OLD tp4056/ws2812b content
outside `kicraft/parts_library/` that must change when the bundles are
overwritten (symbol/footprint names and LCSC ids change):

1. `tests/test_web_bom_vendor.py:31-39`: expects manifest LCSC `C16581`;
   update to `C725790` plus any verbatim symbol/footprint strings.
2. `tests/test_web_bom_pricing.py:22-23`: `_price_key` ->
   `("id", "C16581")`; update to `C725790`.
3. `kicraft/server/web.py:6214-6215` `_DEMO_STATE` BOM ids
   (`"tp4056:TP4056"`, `"tp4056:SOP-8"`) and `:6234` `_DEMO_PRICES` key
   `"id:C16581"` (the demo cost column breaks otherwise).
4. `kicraft/server/stagetabs.py:858-882`: canned demo event stream with
   `lookup_lcsc_id`/`add_part_from_lcsc` on C16581; update ids AND model
   the new bundle-backed flow (the demo currently showcases exactly the
   behavior the new adoption rule forbids for bundle-backed rows).
5. The new manifests need re-authored description/tags/watch_out_for/
   datasheet_url, then promote + catalog flip.

Checked and NOT requiring changes: `docs/kicraft_wiring_spec.md`,
`tests/skill-eval/scenarios/S04-ambiguous-brief.md`, dw01a manifest prose,
`tests/test_silk_refdes.py` (uses ws2812b-2020), `kicraft/server/
sample_projects/*` (no tp4056/ws2812b), `kicraft/leaf_library/`,
`kicraft/server/examples.py` (family prose only).

## PR chunks, in order

1. **PR1: catalog + sync + prompt (transitional; code-only).**
   `core_blocks.{py,json}`; accounts.py sync (+`bundle` column migration,
   delete seed file/constants, remove create/delete store methods); admin
   shrink; stage_driver renderer/blurbs; bom.md + architecture.md wording;
   the three prompt-budget fixes; `add-part --into vendored`;
   `scripts/render_check_bundles.py`; pyproject `[tool.setuptools.
   package-data]`: swap `core_components_seed.json` for `core_blocks.json`
   (note: parts bundles are not in package-data at all; box runs from a
   checkout, unchanged behavior). Tests per the section below.
2. **PR2: loader tier swap** (project > vendored > home > extra).
   Independent, but MUST merge before PR-V4.
3. **PR-V1, PR-V2, PR-V3: vendoring batches** per the slug table. Each PR:
   bundles + authored manifests + render review + production maturity +
   catalog rows flipped from `default_lcsc` to `bundle:`. The CI guards
   (`test_vendored_bundles_load.py`, `test_3d_models.py`, the new catalog
   guard) keep every batch honest.
4. **PR-V4: conflict re-vendor** (tp4056, ws2812b) + the reference fixups
   above + catalog flips.
5. **PR5: guard flip + cleanup.** Validator (or guard test) forbids
   `default_lcsc` rows; delete the transitional schema fields and the
   transitional renderer sentence; re-measure prompt sizes with all 55
   bundles and adjust budgets; confirm the architecture.md ch340n clause
   is gone.

## Tests

- `tests/test_core_components.py`: replace seeding tests with sync
  semantics: fresh DB == catalog; restart preserves enabled/price/stock/
  snapshot_date but overwrites canonical fields; key removed from catalog
  -> row deleted; hand-deleted row resurrects (inverted from today);
  bundle rows' mpn/lcsc equal the manifest; two rows sharing one bundle
  both sync. Drop create/delete CRUD tests. Replace `test_seed_json_is_
  valid` with a catalog guard: pydantic-valid, unique keys, every `bundle`
  resolves to a vendored manifest, transitional-row count == the expected
  number for the current batch (reaches 0 at PR5).
- `tests/test_stage_driver_core_defaults.py`: `_seed_rows()` currently
  reads the seed JSON path; rebuild from synthetic rows or the sync
  output; assert the bundle column renders and the bundle-backed adoption
  wording is present.
- `tests/test_web_core_components.py`: New/Edit/Delete UI gone;
  enable/disable + refresh present.
- `tests/parts_library/test_loader.py`: tier-order cases (vendored shadows
  home; home still resolves when not vendored; project shadows vendored;
  `load_all_with_overrides` shadow reporting).
- PR-V4: the two C16581 test fixups listed above.

## Risks / gotchas

- **Sync clobbers admin edits by design**: canonical fields revert on
  restart; admin-created rows are deleted. Back up the box `accounts.db`
  before the first PR1 deploy; the sync diff log line is the audit trail.
- **content_hash**: sync does NOT verify hashes (perf); the CI guards are
  the only protection, keep them green per batch.
- **Repo size**: `kicraft/parts_library` is 49 MB today (STEP-dominated);
  +32 bundles adds roughly 70-100 MB. Decide plain git vs git-lfs for
  `parts_library/**/3d/*` explicitly in PR-V1 (Codeberg has soft limits;
  re-vendors also leave old blobs in history).
- **Prompt growth**: every BOM run lists 55 bundles even for trivial
  boards. If cost telemetry regresses, a follow-up can filter the parts
  block by architecture categories (out of scope here).
- **Clone-vendor listings** (CH340C/TTP223/TCA9555 are UMW/clone rows):
  EasyEDA symbol quality varies; the per-part render review is the catch.
- **gyroscope/imu-6axis share one bundle**: sync and guards must not
  assume bundle uniqueness across rows.
- **Box home tier**: the tier swap makes stale home caches harmless;
  optional hygiene `rm -rf ~/.kicraft/parts/{tp4056,ws2812b}` in deploy
  notes. If anyone relied on a home bundle overriding a vendored one, the
  project tier is the supported override location.

## Verification

1. Per PR: the test files above, plus
   `for d in kicraft/parts_library/*/; do kicraft validate-part "$d"; done`.
2. Sync e2e: instantiate `AccountStore(tmp_db, tmp_projects)` twice;
   assert rows == catalog, derived mpn/lcsc == manifests, second init is a
   no-op; flip `enabled` + stamp a snapshot, re-init, assert preserved.
3. Prompt e2e: render `_format_core_defaults_block` +
   `_format_available_parts_block` for the full library; run a real
   `stage-prep bom` on a scratch workspace and assert the extras blob fits
   the raised budget with zero truncation; eyeball the adoption wording.
4. Render review per batch via `scripts/render_check_bundles.py` before
   any `promote-part`.
5. Optional: drive one curated example brief (e.g. the WS2812 example in
   `kicraft/server/examples.py`) through the BOM stage and confirm
   bundle-backed defaults are adopted via list_parts with zero
   `add_part_from_lcsc` calls for vendored rows; or run the `/self-eval`
   loop as the broad regression.
6. Box (user-driven): deploy PR1, restart `kicraft-web`, check the sync
   diff log and the admin page (bundle column, no edit/delete); after
   PR-V4 confirm `find_part("tp4056")` resolves the vendored tier despite
   home-tier leftovers.
