# KiCad Routing Tools integration

KiCraft has one autorouter: KiCad Routing Tools (KRT). There is no backend selector or compatibility dispatch.

## Pinned upstream

- Repository: `https://github.com/drandyhaas/KiCadRoutingTools.git`
- Source version: `0.20.2`
- Commit: `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`
- Native router reported by startup checks: `0.20.1`

Provision exactly that revision:

```sh
git clone https://github.com/drandyhaas/KiCadRoutingTools.git /opt/KiCadRoutingTools
git -C /opt/KiCadRoutingTools checkout 3ceb773722bea67aa3685e7ee430c0c0d17ef38d
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

## Adapter contract

`kicraft/autoplacer/kicad_routing_tools.py` owns the KRT process boundary. It:

- requires distinct input and output board paths;
- removes stale output before launch;
- stages the authoritative sibling `.kicad_pro` and optional `.kicad_dru`;
- invokes `py_router/route.py` with all nets, project rules unchanged, and input copper retained;
- disables ripping pre-existing copper and plane finalization;
- records every `JSON_SUMMARY` line and diagnostic stream;
- fingerprints input/output traces and vias and retains a routed diagnostic board when custody fails.

`kicraft/autoplacer/routing_board.py` contains router-independent pcbnew subprocess, DRC, containment, validation, and copper-import helpers.

Leaf and parent routing each call KRT directly once. Escape/breakout/array stamping, GND and power planes, signal/geometry/strand repair, DRC/connectivity acceptance, copper accounting, and stale-artifact guards remain router-independent pipeline stages.

## Runtime options

- `kicad_routing_tools_path`
- `kicad_routing_tools_python`
- `kicad_routing_tools_timeout_s` (default `120`)
- `kicad_routing_tools_max_iterations` (default `200000`)
- `kicad_routing_tools_max_ripup` (default `3`)
- `kicad_routing_tools_ordering` (default `mps`)
- `kicad_routing_tools_clearance_mm` (default `null`, use project rules)
- `kicad_routing_tools_layers` (default `null`, use board copper layers)

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
