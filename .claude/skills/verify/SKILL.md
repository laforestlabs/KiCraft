---
name: verify
description: Verify a KiCraft place/route/promote code change end-to-end by replaying a frozen project workspace through the real build tail (no LLM, no cost) and reading the honest artifact verdict.
---

# Verifying a pipeline change (place / route / compose / promote / verify)

The runtime surface is the build CLI. `replay` drives the exact `build` tail
(`_layout_route_fab`: autoexperiment leaf+parent phases → promote → verify)
against an already-synthesized workspace — no LLM, deterministic placement
(PYTHONHASHSEED pinned by the command itself).

```bash
# 1. Stage a scratch copy of a real run's workspace (never replay in place —
#    replay regenerates .experiments; keep the evidence). Runs live under
#    ~/.kicraft/projects/<uid>/<pid>/generated/<STEM>/.
cp -a ~/.kicraft/projects/<uid>/<pid>/generated/<STEM> /tmp/.../proj
rm -rf /tmp/.../proj/.experiments        # cold replay

# 2. Drive the real flow (minutes; run in background).
.venv/bin/python -m kicraft.design.cli_app replay --project /tmp/.../proj \
    --quality good --seed 0              # match the original build's quality=

# 3. Read the outcome from the app's own outputs:
#    - replay log tail: "[build] 4/5 verify: shorts=... unconnected=..." + rc
#    - .venv/bin/python -m kicraft.design.cli_app artifacts --project /tmp/.../proj
#      (the honest board-path + freshness verdict; never glob)
#    - what shipped vs what was selected:
#      md5sum proj/.experiments/best/parent_routed.kicad_pcb \
#             proj/.experiments/subcircuits/subcircuit__*/parent_routed.kicad_pcb \
#             proj/<STEM>.kicad_pcb
```

Gotchas:
- `--project` mode without a state.json skips fab export (fine — promote and
  verify still run); copy the run's `.kicraft/state.json` next to the project
  and use `replay STATE.json OUT_DIR` if you need the fab zip.
- Routing is best-effort-stable: run-to-run DRC deltas cross grade buckets, so
  never compare artifacts across two separate replays — assert invariants
  within ONE replay (e.g. promoted bytes == best-round bytes), or replay 2–3×
  before claiming a verdict delta.
- `kicad-cli pcb drc <board> --format json --severity-error` gives a fresh,
  authoritative DRC on any single board. PCB coords are real mm (never ×100;
  only ERC-report `pos` is ×100).
- Quality presets: web free tier uses `draft`, default is `good`
  (`grep 'quality=' <run>/.kicraft/build.log` to match the original).
