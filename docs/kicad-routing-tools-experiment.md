# KiCad Routing Tools integration

KiCraft has one autorouter: KiCad Routing Tools (KRT). There is no backend selector or compatibility dispatch.

## Pinned router

- Repository: `https://github.com/drandyhaas/KiCadRoutingTools.git`
- Source version: `0.20.2`
- Commit: `8eeee55fd4ba4ed600b11bd8214de9994e8f5396`
- Native router reported by startup checks: `0.20.1`
- KiCraft's bundled connectivity repair atop upstream `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`; no native binary changes.

Provision from the KiCraft repository root; the repaired commit is carried in the
bundle, not assumed to exist in the upstream GitHub repository:

```sh
git clone https://github.com/drandyhaas/KiCadRoutingTools.git /opt/KiCadRoutingTools
git -C /opt/KiCadRoutingTools checkout 3ceb773722bea67aa3685e7ee430c0c0d17ef38d
git -C /opt/KiCadRoutingTools fetch "$PWD/deploy/tuning-i7/kicad-routing-tools-connectivity.bundle" main
git -C /opt/KiCadRoutingTools checkout 8eeee55fd4ba4ed600b11bd8214de9994e8f5396
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

**Experimental:** the bounded-launch native path has not completed a full
`good`/seed-0 replay. Its current USB smoke is blocked. Do not deploy the native
patch or interpret passing contract tests as board-routing qualification.

`build` and `replay` accept `--engineering-review`. This is an explicit routing
mode, not manufacturing approval:

```sh
.venv/bin/python -m kicraft.design.cli_app replay STATE.json OUT_DIR \
  --quality good --seed 0 --engineering-review --no-fab --no-archive
```

Declared USB routing requires the strict-coupled Python frontend patch on the
pinned source, without replacing the native binary. Apply it to a separate
checkout and scope the configured backend path to that checkout:

```sh
patch -d "$KRT_REVIEW_CHECKOUT" -p1 \
  < deploy/tuning-i7/kicad-routing-tools-strict-coupled.patch
export KICRAFT_KICAD_ROUTING_TOOLS_PATH="$KRT_REVIEW_CHECKOUT"
```

The adapter rejects a frontend lacking `--require-coupled`. This policy attempts
short links as coupled routes instead of automatically deferring them to
independent member routing; physically unroutable copper remains a failure.

The derived sidecar records `usb_geometry_mode: engineering_review`; subsequent
invocations must repeat the flag. USB width/gap come from the pinned nominal
stackup solver on the actual board, with `calculated_unconfirmed` provenance.
The model excludes outer soldermask and fabrication tolerances. It does not
substitute a different layer count, thickness, or copper weight.

`--usb-bounded-launches` additionally permits explicitly uncontrolled local pad
launches and same-component duplicate-contact joins in engineering review only.
The generated sidecar records `usb_review_allow_bounded_launches: true`; ordinary
fabrication mode rejects that policy. Narrow copper uses the real fabrication
floor, not a fictitious 90 Ω rating. Endpoint launches are bounded by two calculated
main-pair pitches; a short local link may consume both endpoint allowances and is
reported as `bounded_local`, not coupled 90 Ω copper. Contact joins have separate
length/envelope bounds derived from the declared contact pads, clearance and
launch width. The complete USB chain must still contain genuinely coupled,
full-profile main copper. Verification lists uncontrolled copper, including alias
branches, and retains reference, skew, transition, connectivity and clearance
checks. The bounded policy does not enable whole-net independent routing fallback
or fabrication release.

Coupled routing, copper custody, physical connectivity, skew, reference copper,
return transitions, thermal and ordinary DRC checks remain required. Independent
USB-member fallback is not accepted. A routed review candidate is still
unapproved: the final verification gate blocks fabrication acceptance and export,
writes an engineering-review receipt, and returns `7`. Inspect that receipt and
the full verification receipt to distinguish the intentional release block from
real physical failures. An ordinary fabrication invocation still requires genuine,
stackup-bound fabrication approval.

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
