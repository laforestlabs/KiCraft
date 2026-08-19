# KiCadRoutingTools routing experiment

This document records the KiCadRoutingTools routing evaluation. As of 2026-08-19,
KiCadRoutingTools is the default router on kicraft.io; FreeRouting remains
available as an alternate backend.

## Pinned upstream

- Repository: https://github.com/drandyhaas/KiCadRoutingTools
- Source checkout `VERSION`: `0.20.2`
- Evaluated source commit: `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`
- `v0.20.2` tag commit: `042bf137f21344d1290eb1f30fecd915301b75a9`
- Observed native `grid_router` version: `0.20.1`
- Python dependencies: NumPy, SciPy, Shapely

The evaluated source commit is post-tag; it is **not** the `v0.20.2` tag commit. The native component is separately versioned and was built with upstream `python build_router.py --tag v0.20.2`.

Configure a checkout at the evaluated source commit:

```json
{
  "routing_backend": "kicad-routing-tools",
  "kicad_routing_tools_path": "/opt/KiCadRoutingTools",
  "kicad_routing_tools_python": "/opt/krt-venv/bin/python"
}
```

Preflight observes rather than assumes the runtime identity. It requires `VERSION=0.20.2`, Git `HEAD=3ceb...`, and a successful live `startup_checks.run_all_checks()` reporting native `0.20.1`. Successful checks are cached by resolved checkout and interpreter; failures are not cached.

Use `"routing_backend": "freerouting"` to opt into the alternate FreeRouting route;
omitting the key selects the KRT production default.

## Integration behavior

The adapter invokes `py_router/route.py` directly from input `.kicad_pcb` to a distinct output `.kicad_pcb` and enforces KiCraft's ownership boundary:

- `--keep-input-copper` and `--no-fix-drc-settings` are unconditional.
- The adapter does not pass `--force-reroute`, `--rip-existing-nets`, `--track-width`, `--via-size`, or `--via-drill`. Omitting the geometry overrides lets upstream read Default and per-net netclass geometry from the sibling project.
- A sibling `.kicad_pro` is mandatory. The authoritative `.kicad_pro` and optional `.kicad_dru` are staged beside temporary inputs and propagated beside routed outputs.
- `KICAD_RIP_PREEXISTING=0` disables upstream's pre-existing-copper rip path.
- `KICAD_PLANE_FINALIZE=0` disables upstream's otherwise enabled plane finalizer because KiCraft owns plane creation and repair. `KICAD_FINALIZE_RIP` is not set or exposed.
- Pre-existing outputs are deleted before launch. Input and output cannot resolve to the same path.
- Every `JSON_SUMMARY` record is retained. The first record backs legacy scalar counters because later reconciliation summaries are subsets.
- Input and output traces/vias are fingerprinted. Missing input copper sets `preserved_existing_copper=false`, retains the routed output for diagnosis, and raises instead of admitting corrupted copper to shared post-processing.

Leaf placement, breakout/escape preparation, post-route plane pouring and repairs, copper import, KiCad DRC, connectivity validation, and acceptance remain shared. Only the FreeRouting branch uses DSN/SES conversion, the GND-plane probe/fallback, power-first routing, zone clearing, or pass scaling.

## Configuration

- `kicad_routing_tools_timeout_s` (default 120)
- `kicad_routing_tools_max_iterations` (default 200000 per route)
- `kicad_routing_tools_max_ripup` (default 3; intra-route rip-up budget only)
- `kicad_routing_tools_ordering` (`mps`, `inside_out`, or `original`)
- `kicad_routing_tools_clearance_mm` (default null: use project/netclass values)
- `kicad_routing_tools_layers` (default null: use all board copper layers)

Input-copper custody and plane ownership are invariants, not configuration switches.

## Known limitations

- Upstream supports KiCad 9 and 10 only. Older KiCad board formats are not claimed as compatible.
- The external checkout and its Python/native dependencies are not installed by KiCraft; strict preflight reports missing or mismatched pieces.
- The integration uses the single-ended `route.py` flow. It does not invoke upstream's separate differential-pair, BGA/QFN fanout, plane-routing, or length-matching stages.
- Upstream emits human diagnostics plus `JSON_SUMMARY` records. KiCraft records all summaries but treats its own KiCad DRC/connectivity validation as authoritative.
- For compatibility with existing result consumers, `freerouting_stats` remains populated as a legacy field. New code should read `routing_stats` and `backend`/`router`.

## Evidence

### Initial rule-aware cleared-board comparison

Source: `/tmp/kicraft-krt-comparison-rules/results.json`. These boards had their input routing cleared, so this comparison did not test stamped-copper custody.

| Board | Backend | Wall s | Accepted | Shorts | Unconnected | Clearance | Total DRC | Traces | Vias | Length mm |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Leaf | FreeRouting | 6.090 | no | 0 | 1 | 2 | 23 | 14 | 0 | 32.75 |
| Leaf | KiCadRoutingTools | 2.252 | no | 0 | 0 | 4 | 10 | 38 | 1 | 39.71 |
| Parent | FreeRouting | 13.184 | no | 0 | 3 | 11 | 189 | 115 | 6 | 362.38 |
| Parent | KiCadRoutingTools | 17.627 | no | 109 | 2 | 140 | 202 | 146 | 24 | 290.01 |

The old KRT adapter reported `preserved_existing_copper=true` without measuring it. Its log showed nine input segments removed and one segment moved to another layer. This comparison motivated the custody ablation; it is not evidence for the corrected adapter.

### Stamped-parent ownership ablation

Source: `/tmp/kicraft-krt-parent-ablation/results.json`. The input was composed deterministically from `tests/fixtures/replay_workspace/PARENT_LOCAL_CONN` with spacing 3.5 mm, stamp enabled, and seed 0. It contained 52 traces and 13 vias. Each variant started from a fresh copy.

| Variant | Wall s | rc | Accepted | Shorts | Unconnected | Clearance | Total DRC | Traces | Vias | Length mm | Missing traces/vias |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `current` | 1.506 | 0 | no | 0 | 0 | 10 | 87 | 131 | 15 | 295.473 | 6 / 0 |
| `keep` | 1.385 | 0 | no | 0 | 0 | 10 | 90 | 134 | 15 | 299.330 | 0 / 0 |
| `keep_no_rip` | 1.385 | 0 | no | 0 | 0 | 10 | 90 | 134 | 15 | 299.330 | 0 / 0 |
| `kicraft_planes` | 1.398 | 0 | no | 0 | 0 | 10 | 90 | 134 | 15 | 299.330 | 0 / 0 |
| `project_rules` | 1.400 | 0 | no | 0 | 0 | 10 | 90 | 153 | 15 | 301.147 | 0 / 0 |

`project_rules` is the adapter policy regardless of the scores of weaker variants. It produced a new board, retained all stamped traces/vias, and had zero shorts and zero unconnected items. Authoritative `validate_routed_board` still rejected it for ten genuine clearance violations. That failure alone prevents a viable classification.

### Full sequential replay A/B

Source project: `/home/kicraft/.kicraft/projects/1/696/generated/USB_C_USB_A_SPLITTER/`. Each backend ran in a fresh scratch copy under `/tmp/kicraft-krt-replay`, restored from `pre_promote_seed.kicad_pcb`, with `quality=good`, seed 0, `--no-fab`, and no synthesis/LLM work. The runs were sequential. Complete logs, resolver output, DRC JSON, every KRT summary, preservation telemetry, and checksums are recorded in `/tmp/kicraft-krt-replay/results.json`.

| Metric | FreeRouting | KiCadRoutingTools |
|---|---:|---:|
| Replay rc / wall | 0 / 327.200 s | 0 / 204.063 s |
| Leaf phase wall | 217 s | 136 s |
| Parent phase wall | 106 s | 65 s |
| Selected leaves accepted | 3/3 | 3/3 |
| Best round score | 70.793 | 70.776 |
| Parent validation accepted | yes | yes |
| Parent shorts / unconnected / clearance / total DRC | 0 / 0 / 0 / 20 | 0 / 0 / 0 / 18 |
| Fresh promoted severity-error DRC / unconnected | 0 / 0 | 0 / 0 |
| Promoted traces / vias / length | 121 / 17 / 103.095 mm | 141 / 21 / 182.399 mm |
| Retained upstream JSON summaries | 0 | 3 |

The two directly routed KRT leaf winners and the final KRT parent each retained one upstream summary. The final parent adapter check matched all 113 input traces and all 15 input vias, with zero missing. Both directly routed leaves were accepted with zero shorts, zero unconnected items, and zero genuine clearances; the second USB-A leaf was replicated from the first.

The later end-to-end child-copper verifier was not clean: it reported the replicated USB A PORT 2 leaf missing 2/40 traces and 1/5 vias in the final parent. The adapter-boundary fingerprint check and the downstream child-manifest check therefore disagree, and the stricter end-to-end result is a failure.

Both resolver reports had fresh provenance and matching run IDs, but independent byte checks failed:

| Backend | Promoted MD5 | Resolver-routed MD5 | Winning-round MD5 | All equal |
|---|---|---|---|---|
| FreeRouting | `25e8d1d3aa49433b06045bf146f97c41` | `a18301b01577b5c7ce41502a1fda22fc` | `a18301b01577b5c7ce41502a1fda22fc` | no |
| KiCadRoutingTools | `6a02849df6d71a6144e42b219d1e22d0` | `3e0ee02e4fd7eec587417e0913f73a93` | `3e0ee02e4fd7eec587417e0913f73a93` | no |

The promoted boards were saved after provenance was written, so fresh run IDs did not imply byte identity. The new FreeRouting replay has the same drift, which limits router-relative interpretation, but it does not satisfy the KRT promotion criterion.

## Decision

**Not viable yet.**

The corrected adapter is materially safer and the representative replay was faster, completed leaf plus parent routing, passed shared validation, preserved adapter input copper, and produced zero severity-error DRC or unconnected items. It still fails the approved hard gates:

1. The `project_rules` stamped-parent ablation was rejected for genuine clearances.
2. The full replay's child-copper verifier reported two missing traces and one missing via on the replicated leaf.
3. The promoted KRT PCB did not byte-match the resolver-selected routed artifact or winning-round source.

Do not weaken input-copper custody, re-enable upstream plane finalization, or force narrower geometry to obtain a green result. The experiment's original recommendation was to keep `"routing_backend": "freerouting"` as the production default; kicraft.io now intentionally selects KRT by product decision. A broader corpus is not justified until the clearance, end-to-end preservation, and promotion-identity failures are resolved.

## Recovered 34-prompt self-eval router A/B — completed 2026-08-17

The original 34-prompt source run was invalidated by a post-route diagnostics bug introduced immediately before the run. `build_leaf_contact_sheet()` used an undefined `out` variable after routing and KiCad validation had completed. The broad exception handler mislabeled that `NameError` as a routing failure. The only surviving prompt had no internal leaf nets and skipped the broken path.

The fix restores `out = Path(output_path)`, creates its parent directory, and has a focused regression test. A replay of the frozen `hex-env-sensor` workspace reproduced `name 'out' is not defined` before the fix. The fixed production replay no longer raised it.

### Recovered source

The original source batch remains unchanged at `logs/self_eval/krt-router-source-20260814`. A separate recovered source is retained at `logs/self_eval/krt-router-source-20260814-contact-sheet-fixed`.

For each of the 32 generated projects stopped by the diagnostics bug, the main PCB had never been promoted over and therefore remained the canonical full-component input board. That board was copied to `.experiments/pre_promote_seed.kicad_pcb` in the recovered source. `recovery.json` records the method and all recovered prompts. No synthesis, LLM, or judge was rerun.

`dual-rail-supply` remains excluded: synthesis produced no project after repeatedly emitting a regulator feedback divider for about 0.853 V while naming the output as a 12 V rail.

- Source interval: `2026-08-14T20:09:12+00:00`–`2026-08-15T01:57:26+00:00`
- Design model: `deepseek/deepseek-v4-flash`
- Judge model: `minimax/minimax-m3`; rubric `2`
- Recorded design plus judge cost: `$1.0167`
- Recovered source summary SHA-256: `d5d7d63d1159eb050b6e01b3e45db7121ec53774423e10ee5d52422c5d61598e`
- Router batch: `logs/krt_self_eval_router_ab/full-corpus-contact-sheet-fixed-20260815`
- Router manifest SHA-256: `7a1226ae7fd227451495a673b236dc34992bb918a9ac6a0bb8be7484371ee911`
- Runtime-tree SHA-256: `2053a0a6a41237a8cfc5c95d5d272b3b96273cb98d416f2e5bedf63379271433`
- Driver SHA-256: `b70f22c31510a767057775c89895d604a74521da5c1af16db5c0998838aad101`
- FreeRouting: `1.9.0`
- KiCadRoutingTools: source `0.20.2`, commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`, native `0.20.1`
- KiCad CLI / pcbnew: `9.0.9` / `9.0.9-9.0.9~ubuntu24.04.1`

### Method and coverage

The fixed experiment replayed **33/34 prompts**, three placement seeds, and both backends: **198 complete cells**. Every cell used the same recovered input board, non-backend configuration, placement seed, `quality=good`, and real build tail. Router order alternated within each prompt and seed. Replay did no synthesis, LLM, judging, fab export, tuning overlay, or placement-objective run.

The SSH connection was interrupted twice. The checksum-bound resume path reused 185 complete cells and rebuilt only the interrupted cell plus the remaining 12. Final evidence contains 198 unique cells, no missing or invalid evidence, and no `harness_error` rows.

### Routing and build result

The build command reported manufacturing-ready status in **64/99 FreeRouting cells** and **85/99 KRT cells**. By majority across three seeds:

| Metric | FreeRouting | KiCadRoutingTools |
|---|---:|---:|
| Majority manufacturing-ready prompts | 22/33 | 29/33 |
| Median full replay wall time | 462.326 s | 346.554 s |
| 90th percentile full replay wall time | 1293.339 s | 855.308 s |
| Median retained router time | 6.460 s | 3.960 s |
| 90th percentile retained router time | 26.390 s | 9.714 s |
| Faster paired cells | 7/99 | 92/99 |

KRT had seven prompt-level manufacturing-ready wins where FreeRouting did not:

- `usb-pd-trigger`
- `usb-c-full-breakout`
- `nrf52-beacon`
- `lora-node`
- `can-node`
- `daq-8ch`
- `stepper-a4988`

FreeRouting had no prompt-level manufacturing-ready win over KRT. The other prompts were ties under the registered majority rule. This is strong evidence that KRT routes this corpus faster and reaches the normal build-complete gate more often.

### Safety result

The manufacturing-ready count is not sufficient to approve KRT. Independent evidence checks found widespread disagreement between the selected routed board and the final saved board, plus missing composed child-board copper evidence.

| Hard safety regression | FreeRouting cells | KRT cells |
|---|---:|---:|
| Final saved board did not match selected routed artifact | 83/99 | 86/99 |
| Child-board copper missing or not provable | 61/99 | 78/99 |
| Router adapter preservation missing or failed | 0/99 | 39/99 |
| Shape check regression | 3/99 | 3/99 |

The first two failures affect the FreeRouting control too, so they identify shared promotion/composition defects rather than a KRT-only routing defect. They still invalidate a manufacturing claim: a clean router result is not useful when the pipeline cannot prove that the same copper reached the final file.

KRT adds a separate problem. In 39 cells, its adapter-level evidence did not prove that copper already present at the router boundary was retained. Both backends degraded the round LED ring to a rectangular outline in all three seeds.

### Historical evaluation decision (2026-08-17)

**Not viable as a production replacement based on this evaluation. Keep FreeRouting available as the alternate backend.**

KRT is the better routing performer in this corpus by build completion and time. It cannot be promoted while adapter copper preservation, child-copper composition, final-artifact identity, and shaped-outline handling fail their hard gates. The control-arm failures also require repair before either backend's build-complete label can be treated as a trustworthy manufacturing verdict.

Authoritative retained evidence:

- `logs/krt_self_eval_router_ab/full-corpus-contact-sheet-fixed-20260815/manifest.json`
- `logs/krt_self_eval_router_ab/full-corpus-contact-sheet-fixed-20260815/results.jsonl`
- `logs/krt_self_eval_router_ab/full-corpus-contact-sheet-fixed-20260815/summary.json`
- `logs/krt_self_eval_router_ab/full-corpus-contact-sheet-fixed-20260815/report.md`
- 198 checksum-bound `cells/` directories
