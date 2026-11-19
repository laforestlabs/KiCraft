# KiCad Routing Tools integration

KiCraft has one autorouter: KiCad Routing Tools (KRT). There is no backend selector or compatibility dispatch.

## Pinned router

- Repository: `https://github.com/drandyhaas/KiCadRoutingTools.git`
- Source version: `0.20.2`
- Commit: `f6fa93122db463ff3f3a1617605c69f7959d2cd9`
- Native router reported by startup checks: `0.20.1`
- KiCraft's bundled connectivity repair and streaming route observer atop upstream `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`; no native binary changes.

Provision from the KiCraft repository root; the repaired commit is carried in the
bundle, not assumed to exist in the upstream GitHub repository:

```sh
git clone https://github.com/drandyhaas/KiCadRoutingTools.git /opt/KiCadRoutingTools
git -C /opt/KiCadRoutingTools checkout 3ceb773722bea67aa3685e7ee430c0c0d17ef38d
git -C /opt/KiCadRoutingTools fetch "$PWD/deploy/tuning-i7/kicad-routing-tools-connectivity.bundle" main
git -C /opt/KiCadRoutingTools checkout f6fa93122db463ff3f3a1617605c69f7959d2cd9
python3 -m venv --system-site-packages /opt/krt-venv
/opt/krt-venv/bin/pip install -r /opt/KiCadRoutingTools/requirements.txt
cd /opt/KiCadRoutingTools
/opt/krt-venv/bin/python build_router.py --tag v0.20.2
```

Configure:

```sh
export KICRAFT_KICAD_ROUTING_TOOLS_PATH=/opt/KiCadRoutingTools
export KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/opt/krt-venv/bin/python
```

`preflight_kicad_routing_tools` rejects a missing route CLI, wrong source version, wrong Git revision, failed startup checks, or a native version other than `0.20.1`. Successful checks are cached by resolved checkout and interpreter; failures are never cached.

Multipoint victim recovery uses the original pad topology rather than the number
of component-MST edges: three pads in two copper islands still require multipoint
recovery. Deferred island-source taps complete inside the same recovery transaction.
Previously closed victims must remain physically connected or the enclosing tap
is abandoned and the saved route restored under the existing collision guards.
This covers single-ended victims too; restored closed victims are protected
from subsequent cascade rip-ups during rollback.

Via-backed multipoint terminals resolve through the graph's real via on its
supported layers. A valid cap overlap can place a segment endpoint more than
0.01 mm from that via; this must not create a zero-copper island or a spurious
zero-length MST edge. Other free-end coincidence tolerances remain unchanged.

## Adapter contract

`kicraft/autoplacer/kicad_routing_tools.py` owns the KRT process boundary. It:

- requires distinct input and output board paths;
- removes stale output before launch;
- stages the authoritative sibling `.kicad_pro` and optional `.kicad_dru`;
- adds temporary via-only rule areas around ordinary assembly pad mask openings, including same-net and plated through-hole pads, using pcbnew's transformed pad shapes and mask expansion; KRT inflates those areas by via copper radius plus clearance, so the via drill remains outside the opening;
- exempts only the reviewed Espressif thermal lands shared with ground finishing, strips only its uniquely named temporary areas after routing, and rejects any remaining via-drill/opening intersections (including pre-existing ones);
- invokes `py_router/route.py` with all nets, project rules unchanged, and input copper retained;
- disables ripping pre-existing copper and plane finalization;
- records every `JSON_SUMMARY` line and diagnostic stream;
- fingerprints input/output traces and vias and retains a routed diagnostic board when custody fails.

`kicraft/autoplacer/routing_board.py` contains router-independent pcbnew subprocess, DRC, containment, validation, and copper-import helpers.

Leaf and parent routing each enter the KRT adapter once; the adapter routes width groups widest first under one shared per-stage deadline. Escape/breakout/array stamping, GND and power planes, signal/geometry/strand repair, DRC/connectivity acceptance, copper accounting, and stale-artifact guards remain router-independent pipeline stages.

Guarded breakout and repair tracks respect track-blocking rule areas on their
own copper layer, not the opposite face. Through vias still check via-blocking
areas on both faces. Ground reservations and authored keepouts retain their
declared layer restrictions.

Signal repair screens out paths crossing applicable track keepouts before
spending its guarded-stamping attempt budget. Blocked simple paths therefore
do not prevent the existing multi-layer mesh fallback from being attempted.

Leaf preparation protects contracted ground-return witnesses before routing.
Optional ground escapes on packages carrying cross-leaf signals wait for the
existing post-route ground finisher: early bonds can cage the signal landings.
Power-only interfaces retain their ordinary early ground escapes. Final native
connectivity, finite-width ground, local-return and thermal checks remain required.

## Runtime options

- `kicad_routing_tools_path`
- `kicad_routing_tools_python`
- `kicad_routing_tools_timeout_s` (leaf default `300`)
- `kicad_routing_tools_parent_timeout_s` (parent default `420`)
- `kicad_routing_tools_max_iterations` (default `200000`)
- `kicad_routing_tools_max_ripup` (default `3`)
- `kicad_routing_tools_ordering` (leaf/default adapter ordering, default `mps`)
- `kicad_routing_tools_parent_ordering` (parent override, default `inside_out`)
- `kicad_routing_tools_clearance_mm` (default `null`, use project rules)
- `kicad_routing_tools_layers` (default `null`, use board copper layers)

For bounded two-phase builds, `KICRAFT_BUILD_MAX_WALL_S` reserves 5% for
promotion/export. Leaves initially receive 65%; the parent search receives the
remaining combined 95% after the leaves finish, including unused leaf time.
The overall deadline, per-leaf limit, configured search-round count, routing
checks, and fabrication gates are unchanged. If the combined search deadline
is already exhausted, parent composition does not start.

## Engineering review without fabrication approval

USB review routing remains unapproved for manufacturing. Native local-link
qualification and a full-board build verdict are separate: neither a successful
pair route nor passing contract tests establishes fabrication readiness.

`build` and `replay` default to engineering review for a USB design when no
manufacturer approval is supplied. They warn and calculate provisional
geometry rather than rejecting the design before placement. `--engineering-review`
explicitly selects draft mode; `--require-fabrication-approval` requires approval
before layout. An invalid approval is not a fallback to automatic draft mode.

```sh
.venv/bin/python -m kicraft.design.cli_app replay STATE.json OUT_DIR \
  --quality good --seed 0 --engineering-review --no-fab --no-archive
```

Declared USB routing requires the patched KiCad Routing Tools differential-pair
frontend. Strict mode disables whole-pair and whole-member fallback. The optional
bounded policy admits short, uncontrolled local links, but the USB chain still
needs a full-profile coupled main span. The patch does not change the native
binary. Apply it once to a separate checkout of the bundled commit above:

```sh
patch --batch --forward --fuzz=0 -d "$KRT_REVIEW_CHECKOUT" -p1 \
  < deploy/tuning-i7/kicad-routing-tools-strict-coupled.patch
export KICRAFT_KICAD_ROUTING_TOOLS_PATH="$KRT_REVIEW_CHECKOUT"
```

The application checks `--require-coupled` before placement. The adapter checks
again before routing USB, including `--usb-bounded-launch-policy` when bounded
launches are enabled. It rejects partial, deferred or member-incomplete results
and single-ended member fallback. Physical copper verification remains separate.

USB-only passes default to a 0.025 mm grid when no explicit routing-grid setting
exists. The ordinary-net grid and clearance floors are unchanged. The finer grid
can represent channels between fine-pitch contacts that the 0.1 mm grid misses.
For bounded terminal routing, rectangular partner pads use pad-shaped obstacles
instead of the conservative capsule approximation.

In the November 17 project 1167 engineering-review replay (`good`, seed 0),
`PROGRAM USB` was accepted as a leaf. `POWER INPUT` did not qualify: its
buck-switch net narrowed beyond the 1.5 mm pad-escape allowance. The replay
ended at exit 6 with no selected routed parent. The promoted PCB was a partial
diagnostic, not a verified draft or release. Local evidence:
`/tmp/kicraft-krt-qualified-8WW7FT/replay.log`, `artifacts.json` in that
directory, and the replay project's `.experiments` diagnostics.

The derived sidecar records `usb_geometry_mode: engineering_review`; ordinary
invocations can reuse that input without repeating the flag. USB width/gap come
from the pinned nominal stackup solver on the actual board, with
`calculated_unconfirmed` provenance. The model excludes outer soldermask and
fabrication tolerances. It does not substitute a different layer count,
thickness, or copper weight.

Calculated review geometry defaults to bounded local pad launches and
same-component duplicate-contact joins. `--usb-bounded-launches` explicitly
requests the same policy; a saved `usb_review_allow_bounded_launches: false`
preserves a full-width-only request unless explicitly overridden. The derived
sidecar records the resolved policy. Fabrication mode rejects bounded launches.
Narrow copper uses the fabrication width floor, not a fictitious 90 Ω rating.
Endpoint launches are bounded by two calculated main-pair pitches. A local link
may use both endpoint allowances and is reported as `bounded_local`, not coupled
90 Ω copper. Contact join length and envelope bounds come from the declared
contact pads, clearance and launch width. The complete USB chain must still
contain full-profile coupled main copper. Verification lists uncontrolled copper, including alias
branches, and retains reference, skew, transition, connectivity and clearance
checks. The bounded policy does not enable whole-net independent routing fallback
or fabrication release.

Coupled routing, copper custody, physical connectivity, skew, reference copper,
return transitions, thermal and ordinary DRC checks remain required. Independent
USB-member fallback is not accepted. Missing manufacturer confirmation alone
does not make a physically valid draft a failed build.

A physically verified engineering-review result returns **9** (`review_required`),
records board-bound engineering-review and verification receipts, and produces
`<stem>_review.zip`: editable sources with a draft-only notice and no Gerber or
drill files. Fabrication approval stays false. Physical failures retain their own
exit codes; a partial preview is not a draft-ready result. Exit **0** remains
success outside engineering-review mode, including a verified `--no-fab` replay.
The web project reports **Draft ready — USB fabrication confirmation needed**,
not Failed or fabrication-ready. A strict fabrication invocation requires genuine,
stackup-bound approval.

## Verification

The production verifier is the `replay` CLI against a cold scratch copy, followed by the `artifacts` resolver and a fresh `kicad-cli pcb drc` on the promoted board. The verdict must show a fresh routed parent and promoted board from one run, zero shorts/unconnected, and zero missing authoritative input traces or vias.

### Post-cutover replay

Cold `USB_PD_TRIGGER`, `quality=good`, seed `0`, no fab completed with exit `0`,
`REPLAY COMPLETE`, and promoted `shorts=0`, `unconnected=0`. The independent
DRC then exposed three genuine hole-clearance errors in the generic signal
repair path. After adding hole-aware PTH/NPTH margins and fixing the DRC
classifier to ignore indented rule metadata, a real parent-stage rerun was
accepted and the shared promotion tail returned `0`.

The corrected artifact has:

- parent adapter custody: 117/117 input traces and 13/13 input vias matched,
  with zero missing;
- fresh DRC: eight USB-C footprint-internal pad clearances and four intrinsic
  connector annular-width items; zero genuine routed-copper clearances.

Two pre-existing artifact-accounting defects remain separate from router
custody: the per-child transformed-copper manifest reports 0/117 traces and
0/13 vias even while the adapter's authoritative input/output fingerprints
match all 117/13, and the post-promotion silk save changes promoted bytes after
provenance records the routed source hash. Neither defect changes KRT copper or
the electrical DRC verdict.
