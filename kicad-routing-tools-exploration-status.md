# KiCadRoutingTools exploration handoff

Last updated: 2026-08-12

## Execution status

- Branch: `experiment/kicad-routing-tools`.
- Required `skill://verify` instructions were read. The final verification must use the real `kicraft.design.cli_app replay` build tail and app artifact resolver, followed by fresh `kicad-cli pcb drc` output.
- The working tree already contained unrelated modifications. Do not touch `.claude/commands/kicraft-investigate.md`, `kicraft/server/*`, `tests/test_stagetabs_helpers.py`, `tests/test_web_support_reports.py`, or `scripts/self_eval_model_compare.py`.
- Todo state: all five approved implementation/evidence steps complete.

### Work completed in this session

1. Read the approved plan from `local://kicad-routing-tools-exploration-plan.md` and the `verify` skill.
2. Confirmed the upstream live check directly:
   ```bash
   cd /tmp/KiCadRoutingTools
   /tmp/krt-venv/bin/python -c 'from py_router.startup_checks import run_all_checks; print(run_all_checks())'
   ```
   Observed stdout `0.20.1`, rc 0, wall time 0.43 s.
3. Edited `kicraft/autoplacer/routing_backends.py` to strengthen `preflight_routing_backend`:
   - added expected native version `0.20.1`;
   - added a success-only cache keyed by resolved checkout and interpreter;
   - requires a readable `VERSION` with `0.20.2`;
   - requires Git metadata and observed `HEAD=3ceb773722bea67aa3685e7ee430c0c0d17ef38d`;
   - executes the exact upstream `startup_checks.run_all_checks()` command with the configured interpreter and checkout cwd;
   - reports observed source version, commit, interpreter, and native version;
   - leaves failures uncached.
4. Verified the installed strict preflight. It returned observed source version `0.20.2`, commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, interpreter `/tmp/krt-venv/bin/python`, and native version `0.20.1`.
5. Completed the deterministic stamped-parent ablation. Durable evidence is in `/tmp/kicraft-krt-parent-ablation/results.json` (169,066 bytes), with per-variant boards and logs in sibling directories:

   | Variant | rc | Wall s | Accepted | Shorts | Unconnected | Clearance | Total DRC | Traces | Vias | Length mm | Missing traces/vias |
   |---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
   | `current` | 0 | 1.506 | no | 0 | 0 | 10 | 87 | 131 | 15 | 295.472925 | 6 / 0 |
   | `keep` | 0 | 1.385 | no | 0 | 0 | 10 | 90 | 134 | 15 | 299.329827 | 0 / 0 |
   | `keep_no_rip` | 0 | 1.385 | no | 0 | 0 | 10 | 90 | 134 | 15 | 299.329827 | 0 / 0 |
   | `kicraft_planes` | 0 | 1.398 | no | 0 | 0 | 10 | 90 | 134 | 15 | 299.329827 | 0 / 0 |
   | `project_rules` | 0 | 1.400 | no | 0 | 0 | 10 | 90 | 153 | 15 | 301.146984 | 0 / 0 |

   Every variant produced a fresh output and one parsed upstream summary. `project_rules` preserved all 52 stamped traces and 13 vias and had zero shorts, but authoritative validation rejected it for genuine clearance violations. The Step 1 decision is therefore **not viable yet**.
6. Implemented the Step 2 ownership boundary in `routing_backends.py` and generalized `freerouting_runner.propagate_sibling_project_rules`:
   - unconditional `--keep-input-copper`; no forced track/via geometry;
   - strict preflight before sidecars/launch;
   - mandatory project rules, optional custom rules, and output propagation;
   - distinct paths, stale-output deletion, exact no-rip/no-plane environment;
   - all summaries retained, first summary backs legacy counters;
   - observed trace/via preservation with route rejection on loss.
   A real adapter smoke routed the stamped parent in 2.313 s, propagated `.kicad_pro`, returned the pinned runtime identities, retained one summary, and reported zero missing traces/vias. Shared validation honestly rejected the output with 10 clearances, zero shorts, and zero unconnected.
7. Implemented Step 3: backend selection is part of deterministic route cache identity; timeout remains cache-neutral; skipped/error router labels use normalized backend names; backend-unavailable KRT errors re-raise. LSP diagnostics are clean for the affected files, and the existing signature test passes.
8. Added the focused Step 4 tests. The exact approved command passed:
   ```text
   22 passed, 1 warning in 0.96s
   ```
   It covers strict/cached preflight, command invariants, project/custom-rule custody, strict process ordering/environment, stale output, timeout/nonzero/no-output behavior, summary precedence, copper-loss rejection, backend-aware signatures, KRT error escalation/labels, and parent FreeRouting-workaround bypass.
9. Completed the sequential project-696 replay A/B. Durable evidence is `/tmp/kicraft-krt-replay/results.json`; logs, resolver JSON, DRC JSON, boards, and per-backend artifacts remain under `/tmp/kicraft-krt-replay/`.
   - FreeRouting: rc 0, 327.200 s; leaf phase 217 s; parent phase 106 s; 3/3 selected leaves accepted; final verify 0 shorts/0 unconnected; fresh severity-error DRC 0/0.
   - KRT: rc 0, 204.063 s; leaf phase 136 s; parent phase 65 s; 3/3 selected leaves accepted; final verify 0 shorts/0 unconnected; fresh severity-error DRC 0/0; final parent adapter fingerprint matched 113/113 traces and 15/15 vias.
   - End-to-end KRT child verification still reported loss of 2/40 traces and 1/5 vias on replicated USB A PORT 2.
   - Provenance was fresh and run IDs matched for both backends, but promoted bytes differed from resolver-routed/winning-round bytes. KRT MD5s: promoted `6a02849df6d71a6144e42b219d1e22d0`; routed and winning `3e0ee02e4fd7eec587417e0913f73a93`.
10. Updated `docs/kicad-routing-tools-experiment.md` with corrected source/tag/native identities, ownership semantics, rule staging, limitations, the old comparison, ablation, full replay, checksums, and the final **Not viable yet** decision. FreeRouting remains the default.

### Completion status

The approved plan is complete. No implementation work remains. The classification is **Not viable yet** because the project-rules ablation failed authoritative clearance validation, the end-to-end replay child verifier found copper loss, and promoted/routed/winning board bytes did not match.

### Files changed by this session so far

- `kicraft/autoplacer/routing_backends.py` — strict runtime identity, KRT ownership policy, process defenses, summary capture, and observed preservation.
- `kicraft/autoplacer/freerouting_runner.py` — project/custom-rule propagation shared by both backends.
- `kicraft/autoplacer/brain/leaf_routing.py` — backend-aware deterministic cache identity and corrected backend-label scope.
- `kicraft/cli/solve_subcircuits.py` — truthful backend labels and unavailable-backend escalation.
- `tests/test_routing_backends.py` — command, strict preflight, process-boundary, rule-custody, summary, failure, and preservation contracts.
- `tests/test_array_placement.py` — backend-aware deterministic signature contract.
- `tests/test_leaf_place_quality_gate.py` — KRT unavailable-error escalation and truthful generic-failure labels.
- `docs/kicad-routing-tools-experiment.md` — complete evidence and decision record.
- `kicad-routing-tools-exploration-status.md` — this handoff.

### Remaining work

None in the approved plan. Future work is explicitly outside this task: resolve the genuine-clearance ablation rejection, explain/fix downstream replicated-leaf copper loss, and make post-promote bytes/provenance agree before expanding the corpus.

---

# Authoritative approved plan

## Context

Continue the existing `experiment/kicad-routing-tools` branch evaluation of KiCadRoutingTools as an optional KiCraft routing backend, without touching the unrelated modified server/support files in the same working tree. The current adapter is pinned to upstream source commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, whose `VERSION` file is `0.20.2`; this commit is post-tag and must not be described as the `v0.20.2` tag commit (`042bf137f21344d1290eb1f30fecd915301b75a9`). The matching checkout, native `grid_router.so`, and Python environment already exist at `/tmp/KiCadRoutingTools` and `/tmp/krt-venv`.

The first rule-aware A/B in `/tmp/kicraft-krt-comparison-rules/results.json` was promising on one cleared leaf (KiCadRoutingTools: 0 unconnected, 0 shorts, 2.252 s; FreeRouting: 1 unconnected, 0 shorts, 6.09 s) but unsafe on the cleared parent (KiCadRoutingTools: 2 unconnected, 109 shorts; FreeRouting: 3 unconnected, 0 shorts). The KRT parent log also says it removed 9 input segments and changed one segment layer while the adapter falsely reported `preserved_existing_copper=true`. Upstream source inspection explains the mismatch: KiCraft does not pass `--keep-input-copper`, inherits enabled pre-existing-copper rip and plane-finalization paths, and explicitly forces KiCraft's `signal_width_mm` over the sibling project's per-net netclass widths. The intended end state is an architecture-aligned adapter that treats stamped leaf copper, KiCraft's plane stages, and project rules as authoritative, followed by a controlled parent ablation and a real replay A/B; FreeRouting remains the production default regardless of the result.

## Approach

### 1. Preserve and extend the existing evidence before changing behavior

- Verify the installed runtime with `preflight_routing_backend` using `routing_backend="kicad-routing-tools"`, `kicad_routing_tools_path="/tmp/KiCadRoutingTools"`, and `kicad_routing_tools_python="/tmp/krt-venv/bin/python"`. Strengthen preflight first so it requires `VERSION=0.20.2`, a Git checkout at `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, and executes `from py_router.startup_checks import run_all_checks; print(run_all_checks())` under the configured interpreter with the checkout as `cwd`. This live upstream check is known to return native version `0.20.1` in the installed environment; a missing/unverifiable source revision, dependency, or native module stops the experiment rather than silently labeling another runtime with constants.
- Build one deterministic composed-parent input in `/tmp/kicraft-krt-parent-ablation/project` from the frozen leaves in `tests/fixtures/replay_workspace/PARENT_LOCAL_CONN`: copy the fixture, replace `__KICRAFT_PROJECT_DIR__` tokens in its copied `.experiments/**/*.json`, remove only stale `parent_pre_freerouting.kicad_pcb` artifacts, and run the exact compose command from `scripts/replay_corpus.py:_run_parent` with `--parent PARENT_LOCAL_CONN --spacing-mm 3.5 --stamp --seed 0` and `PINNED_ENV`. Resolve the placed/stamped board through `kicraft.cli.artifact_paths.resolve_parent_board(..., kind="placed")`; do not clear its tracks, vias, or zones, because the experiment must exercise preservation of stamped leaf copper.
- From that one stamped board and its same-stem `.kicad_pro` (plus `.kicad_dru` if present), run five KiCadRoutingTools variants into separate output directories, always starting from a fresh copy:
  1. `current`: the current `_krt_command` and inherited upstream defaults.
  2. `keep`: `current` plus `--keep-input-copper`.
  3. `keep_no_rip`: `keep` plus `KICAD_RIP_PREEXISTING=0`.
  4. `kicraft_planes`: `keep_no_rip` plus `KICAD_PLANE_FINALIZE=0`.
  5. `project_rules`: `kicraft_planes` with the current explicit `--track-width`, `--via-size`, and `--via-drill` option/value pairs removed so upstream reads the sibling project.
- Save `/tmp/kicraft-krt-parent-ablation/results.json`. For every variant record wall time, subprocess return code, every parsed `JSON_SUMMARY`, KiCraft `validate_routed_board` acceptance/short/unconnected/clearance/total counts, trace/via/length totals, and a trace/via preservation report. Build that report with `Counter(fingerprint_trace(...))` and `Counter(fingerprint_via(...))` from `kicraft.autoplacer.brain.copper_accounting` over `import_routed_copper` results; record expected, matched, and missing multisets separately. The required stamped-copper result is zero missing traces and zero missing vias.
- This ablation is diagnostic, not a knob-selection contest. The adapter policy in Step 2 remains `project_rules` even if a weaker variant scores better: KiCraft-owned copper/planes and unchanged project design rules are hard constraints, not quality dials. If `project_rules` loses copper, fails, has any shorts, or is rejected by `validate_routed_board`, preserve the artifacts and report KiCadRoutingTools as not viable yet; never fall back to upstream ripping/finalization or narrower explicit geometry.

### 2. Make the adapter match the KiCraft ownership boundary

- In `kicraft/autoplacer/routing_backends.py:_krt_command`, make the ownership policy unconditional: always append `--keep-input-copper`; remove the unconditional `--track-width`, `--via-size`, and `--via-drill` arguments. Omission is upstream commit `3ceb...`'s contract for reading Default/per-net netclass geometry from the sibling project. Keep `--no-fix-drc-settings`, the optional explicit `kicad_routing_tools_clearance_mm`, layer selection, ordering, iteration, and intra-route rip-up budgets. Do not add config switches for input-copper custody or upstream plane ownership; those are invariants, not user-tunable quality knobs.
- In `preflight_routing_backend`, require and report the observed source version and Git commit rather than trusting the adapter constants. Execute upstream `startup_checks.run_all_checks()` with the configured interpreter/checkout, require native version `0.20.1`, and include that observed native version in the result. Put the KRT-specific check behind a success-only cache keyed by resolved checkout and interpreter so repeated leaf rounds do not pay the 0.45 s native startup cost; failures remain uncached. In `route_with_kicad_routing_tools`, invoke this strict preflight before creating sidecars or launching a route, and use its observed source/native identities in returned stats.
- Require a sibling `.kicad_pro`: reuse the existing `pcb_path`/same-directory candidate search and temporary same-stem staging, but raise `RoutingBackendUnavailableError` with prefix `KiCadRoutingTools requires a sibling .kicad_pro` before launch if none exists. Rename/generalize `freerouting_runner._propagate_sibling_pro` to carry both `.kicad_pro` and optional `.kicad_dru`; use the same helper to stage input rules and overwrite/carry the authoritative rule files beside the KRT output before temporary input sidecars are removed. After propagation, require the output `.kicad_pro` to exist.
- Require distinct resolved input/output board paths, then unlink any pre-existing output before launch so rc 0 cannot satisfy the output check with stale data or destroy its own input. Launch with a copied environment containing exact `KICAD_RIP_PREEXISTING="0"` and `KICAD_PLANE_FINALIZE="0"`. Do not set or expose `KICAD_FINALIZE_RIP`: the upstream plane finalizer is disabled because KiCraft owns plane creation and repair.
- Parse every `JSON_SUMMARY` line into a new `"json_summaries"` list; retain the first summary as the source for legacy scalar counters because later upstream records are reconciliation subsets, not complete replacements. Preserve the raw streams and command for diagnosis.
- Make preservation telemetry observed, not configured. Fingerprint input and output traces/vias with the existing copper-accounting functions; return an `"input_copper_preservation"` structure and set `"preserved_existing_copper"` only when no input trace/via is missing. Because `--keep-input-copper` is an adapter invariant, any missing item raises a route error after retaining the output for diagnosis instead of allowing corrupted copper into shared post-processing.
- Keep the shared post-route path in `kicraft/cli/_compose_route.py` unchanged: KiCraft still pours planes, runs GND/power repair, imports copper, validates DRC/connectivity, and rejects shorts/unconnected after either backend. Only the FreeRouting branch may run DSN/SES conversion, GND-plane probe/fallback, power-first, zone-clearing, or pass-scaling workarounds.

### 3. Isolate deterministic leaf caches by backend

- In `kicraft/autoplacer/brain/leaf_routing.py:_deterministic_route_signature`, include the normalized `routing_backend(cfg)` value in the serialized cache key. The current filter includes backend-specific tuning keys but omits the selector itself, so switching an otherwise identical deterministic array leaf can reuse copper produced by the other router.
- Keep the function signature unchanged. Extend `tests/test_array_placement.py::test_deterministic_route_signature` to assert FreeRouting and KiCadRoutingTools produce different signatures while timeout-only changes for either backend remain cache-neutral.
- In `kicraft/cli/solve_subcircuits.py`, derive skipped/error `"router"` labels from `routing_backend(cfg)` and make the place-quality message backend-neutral. Re-raise `RoutingBackendUnavailableError` alongside `FreeroutingUnavailableError` so a strict KRT preflight failure becomes one clear host/configuration failure rather than one mislabeled `routing_exception` per leaf.

### 4. Lock the corrected contract with focused tests

- Update `tests/test_routing_backends.py::test_krt_command_preserves_rules_and_existing_copper` to require `--keep-input-copper`, `--no-fix-drc-settings`, no force/rip flags, and no default `--track-width`/`--via-size`/`--via-drill`/`--clearance` override.
- Add process-boundary tests around `route_with_kicad_routing_tools` using the existing monkeypatch/tmp-path style: strict preflight precedes launch; same input/output is rejected; stale output is deleted; `Popen(env=...)` receives both disabled upstream controls; `.kicad_pro` and optional `.kicad_dru` are staged and propagated to output; absence of project rules raises before launch; all summaries are retained with legacy counters sourced from the first; timeout/nonzero/no-output behavior remains unchanged; and a missing input trace/via makes preservation false and rejects the route.
- In `tests/test_leaf_place_quality_gate.py`, add focused `_solve_one_round` cases proving a KRT `RoutingBackendUnavailableError` re-raises and a generic KRT failure is labeled `kicad-routing-tools`; reuse the file's lightweight fake solver/placement style. Extend the existing parent dispatch test to assert KRT stats identify the backend and FreeRouting-only helpers remain bypassed. Do not rename public result fields or remove `freerouting_stats` in this experiment.

### 5. Run the real pipeline A/B and record the decision

- After the focused tests and parent ablation pass, use `/home/kicraft/.kicraft/projects/1/696/generated/USB_C_USB_A_SPLITTER/` as the representative full replay input. Its original `quality=good` FreeRouting build on this branch completed with 18/18 components, 0 shorts, and 0 unconnected, so it is a stronger baseline than the synthetic committed fixture. Never replay project 696 in place.
- Create fresh sequential scratch roots `/tmp/kicraft-krt-replay/freerouting` and `/tmp/kicraft-krt-replay/krt`: copy the generated project and ancestor `.kicraft/state.json`, restore the copied top-level board from the copied `.experiments/pre_promote_seed.kicad_pcb`, then delete copied `.experiments` and provenance. In each copied autoplacer JSON, change only backend selection/path/interpreter: explicit FreeRouting in one, and the strict pinned KRT checkout/environment in the other.
- Replay each through state/output mode with identical `--quality good --seed 0 --no-fab`, one after the other so wall time is not distorted by CPU contention. This drives the real autoexperiment leaf routes, composed-parent route, promotion, and verify tail without synthesis, LLM use, or fab export.
- For each replay, retain its log and return code, then resolve results only with `kicraft.design.cli_app artifacts --kind all --json`. Compare the promoted board MD5 to the resolver-selected routed artifact and its winning-round source within that same replay, check provenance freshness/run IDs, and run fresh `kicad-cli pcb drc --format json --severity-error` on the promoted board. Record phase wall times, leaf acceptance/failures, final backend stats, all upstream summaries, input-copper verification, shorts, unconnected, total/clearance DRC, traces, vias, total length, and artifact identity.
- Update the existing `docs/kicad-routing-tools-experiment.md`, not a new document. Correct the tag/commit identity, input-copper and default plane-finalization claims, project-rule staging (including `.kicad_dru`), and single-ended limitations. Add the existing rule-aware table, the stamped-parent ablation, and the full replay A/B with exact runtime/project/quality/seed and one conclusion:
  - **Not viable yet** if KRT deletes stamped copper, produces any short, fails authoritative DRC/clearance validation, cannot complete leaf plus parent, or does not produce a fresh routed promotion with matching provenance.
  - **Promising; expand the corpus** only if stamped-copper loss and shorts are zero, authoritative validation/replay verification pass, KRT unconnected and genuine DRC counts are no worse than FreeRouting on leaf and parent, and replay artifact/provenance checks pass.
  A single representative project still cannot justify changing the production default, so `"routing_backend": "freerouting"` remains unchanged in either case.

## Critical files & anchors

- `kicraft/autoplacer/routing_backends.py` — strict runtime preflight, command/environment policy, project-rule custody, summary parsing, stale-output defense, and observed copper preservation.
- `kicraft/autoplacer/freerouting_runner.py` — generalize the existing sibling-project propagation helper to carry `.kicad_pro` and `.kicad_dru` for both route outputs.
- `kicraft/autoplacer/brain/leaf_routing.py` and `kicraft/cli/solve_subcircuits.py` — backend-aware cache identity plus truthful leaf failure attribution/escalation.
- `tests/fixtures/replay_workspace/PARENT_LOCAL_CONN/` — frozen-leaf composed-parent preservation fixture with `parent_compose_spacing_mm=3.5`.
- `/home/kicraft/.kicraft/projects/1/696/generated/USB_C_USB_A_SPLITTER/` — successful real-project baseline for scratch-only full replay.

## Verification

Run from `/home/kicraft/KiCraft` unless a command says otherwise.

1. Focused behavior:
   ```bash
   .venv/bin/pytest -q tests/test_routing_backends.py tests/test_array_placement.py::test_deterministic_route_signature tests/test_leaf_place_quality_gate.py
   ```
   Expected: all selected tests pass; command/process tests prove project rules and input copper are retained, upstream plane/rip paths are disabled, stale output is impossible, preflight errors are attributed correctly, and backend switches cannot hit old route caches.

2. Installed runtime:
   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from kicraft.autoplacer.routing_backends import preflight_routing_backend; print(preflight_routing_backend({"routing_backend":"kicad-routing-tools","kicad_routing_tools_path":"/tmp/KiCadRoutingTools","kicad_routing_tools_python":"/tmp/krt-venv/bin/python"}))'
   ```
   Expected: observed `backend='kicad-routing-tools'`, source `version='0.20.2'`, commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, and native `0.20.1` from the live upstream startup check. The documentation separately identifies `v0.20.2`'s tag commit as `042bf...`.

3. Controlled new behavior: execute the Step 1 scratch harness and inspect `/tmp/kicraft-krt-parent-ablation/results.json`. Required invariant: composed stamped input -> `project_rules` routed output with zero missing input traces and vias. Any nonzero loss is a failed integration regardless of DRC score. Also require a new output file and retain every short/unconnected/summary result rather than suppressing it.

4. Full pipeline, sequentially:
   ```bash
   .venv/bin/python -m kicraft.design.cli_app replay /tmp/kicraft-krt-replay/freerouting/.kicraft/state.json /tmp/kicraft-krt-replay/freerouting/generated --quality good --seed 0 --no-fab
   .venv/bin/python -m kicraft.design.cli_app replay /tmp/kicraft-krt-replay/krt/.kicraft/state.json /tmp/kicraft-krt-replay/krt/generated --quality good --seed 0 --no-fab
   .venv/bin/python -m kicraft.design.cli_app artifacts --project /tmp/kicraft-krt-replay/freerouting/generated/USB_C_USB_A_SPLITTER --kind all --json
   .venv/bin/python -m kicraft.design.cli_app artifacts --project /tmp/kicraft-krt-replay/krt/generated/USB_C_USB_A_SPLITTER --kind all --json
   ```
   For each promoted board:
   ```bash
   kicad-cli pcb drc --format json --severity-error --output /tmp/kicraft-krt-replay/<backend>-drc.json /tmp/kicraft-krt-replay/<backend>/generated/USB_C_USB_A_SPLITTER/USB_C_USB_A_SPLITTER.kicad_pcb
   ```
   Expected: each replay yields an honest rc, artifact/provenance report, and within-replay identity check. KRT is classified by the Step 5 rubric; a missing routed artifact, short, stale promotion, MD5 mismatch, or copper loss is not success.

## Assumptions & contingencies

- Treat all current working-tree modifications as pre-existing work. Modify only the routing runner, FreeRouting sidecar helper, leaf cache/error callsites, focused tests, and existing experiment document named above; leave `.claude/commands/kicraft-investigate.md`, `kicraft/server/*`, `tests/test_stagetabs_helpers.py`, `tests/test_web_support_reports.py`, and `scripts/self_eval_model_compare.py` untouched.
- `/tmp/KiCadRoutingTools` and `/tmp/krt-venv` are disposable but currently complete and pinned. If either disappears during execution, recreate the source checkout at commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, install its pinned requirements, and run upstream `python build_router.py --tag v0.20.2` for the separately versioned native component; never claim `3ceb...` is the tag commit or substitute a newer source revision mid-comparison.
- If the KRT full replay stops before parent composition, retain that failure as the leaf result and use the `PARENT_LOCAL_CONN` ablation for parent coverage. Do not weaken rules, increase rip authority, or enable upstream plane finalization to force a green result.
- If the fresh FreeRouting replay also fails, use its observed same-project result as the baseline and limit the conclusion to relative evidence; the historical project-696 success is context, not a substitute for the new run. KRT still cannot be called promising unless it satisfies every hard preservation/short/artifact criterion above.
