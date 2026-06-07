---
description: Investigate why a KiCraft web/synthesis run failed — locates the run, prints the verdict and the ERC errors (already ×100-corrected to real mm), and points at the likely root cause (code bug vs model output).
argument-hint: "[uid/pid | pid] (optional; default: most recent run)"
---

Investigate a failed KiCraft run and hand back a fast, accurate picture of **why** it failed and **whether it is a synthesis-code bug or the model's wiring output**. Target run: `$ARGUMENTS` (may be empty → use the most recent run).

A "run" is one web design the pipeline produced under `<projects_dir>/<uid>/<pid>/`. Each leaves `events.jsonl` (the per-stage + build event stream), `state.json` (committed design state incl. the BOM), `kicraft/synthesis_check.json` (the gate verdict), and a generated KiCad tree with an ERC report `generated/<stem>/<stem>_erc.rpt`.

Print a one-line note before each step (visible reasoning); keep tool fan-out small.

## 1. Locate the run + verdict + ERC errors — one block

Resolves the *same* projects dir the server uses, picks the run, and prints the failure picture. Note the printed `RUN` / `PROJECTS` / `PY` — shell vars do **not** persist between Bash calls, so paste them literally into the later steps.

```bash
REPO=$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/KiCraft"); PY="$REPO/.venv/bin/python"
PROJECTS="${KICRAFT_PROJECTS_DIR:-}"
[ -z "$PROJECTS" ] && [ -f "$REPO/.env" ] && PROJECTS=$(grep -E '^KICRAFT_PROJECTS_DIR=' "$REPO/.env" | tail -1 | cut -d= -f2- | tr -d "\"' ")
[ -z "$PROJECTS" ] && PROJECTS="$HOME/.kicraft/projects"
ARG="$ARGUMENTS"
if   [ -n "$ARG" ] && [ -d "$PROJECTS/$ARG" ]; then RUN="$PROJECTS/$ARG"
elif [ -n "$ARG" ]; then RUN=$(find "$PROJECTS" -mindepth 2 -maxdepth 2 -type d -path "*/$ARG" 2>/dev/null | head -1)
else RUN=$(find "$PROJECTS" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-); fi
echo "RUN=$RUN"; echo "PROJECTS=$PROJECTS"; echo "PY=$PY"
"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
print(f"\n=== {run.name}   ({run}) ===")
ev = run / "events.jsonl"
if ev.is_file():
    lines = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
    stages = [e for e in lines if e.get("kind") == "stage_done"]
    bd     = [e for e in lines if e.get("kind") == "build_done"]
    blog   = [e for e in lines if e.get("kind") == "build_log"]
    if stages:
        print("stages:", [(e.get("stage"), "ok" if e.get("ok") else "FAIL") for e in stages])
    print("build_done:", bd[-1] if bd else "(no build step reached)")
    for e in blog[-10:]:
        print("   build_log:", e.get("text"))
else:
    print("no events.jsonl (the run may not have started the pipeline)")
sc = next(run.rglob("synthesis_check.json"), None)
if sc:
    d = json.loads(sc.read_text())
    print(f"synthesis_check: status={d.get('status')}  failed_checks={d.get('failed_checks')}")
erc = next(run.rglob("*_erc.rpt"), None)
if erc:
    d = json.loads(erc.read_text())
    errs = [(sh["path"], v) for sh in d.get("sheets", [])
            for v in sh.get("violations", []) if v.get("severity") == "error"]
    print(f"\nERC errors: {len(errs)}   ({erc})")
    for path, v in errs:
        print(f"  [{path}] {v['type']}: {v['description']}")
        for it in v.get("items", []):
            pos = it.get("pos") or {}
            xy = f"  @ x100=({pos.get('x',0)*100:.2f}, {pos.get('y',0)*100:.2f}) mm" if pos else ""
            print(f"       - {it.get('description','')}{xy}")
else:
    print("\nNo ERC report — synthesis likely crashed before ERC ran. Check the journal (step 3).")
PY
```

Read off: did the build fail (`build_done {"ok": false}`)? which stage? what ERC errors?
- **ERC coordinates are reported at 1/100 of real mm** — this block already multiplies `pos`/length by 100 so they match the `.kicad_sch`. (Memory: `kicad-erc-report-coords-x100`. Don't fall for the "everything is a sub-mm micro-stub at the origin" trap — that's the report's scale, not a real bug.)
- A `stage = FAIL` that the journal shows as *"parked: awaiting a clarifying answer"* is the pipeline **waiting on the user**, not a crash.

## 2. Classify: systematic code bug vs per-design model gap

**(a) Cross-run scan** — is this ERC error type unique to this design (a wiring/model gap) or hitting many runs (a synthesis-code bug)? Reuse `<PY>` / `<PROJECTS>` from step 1:

```bash
"<PY>" - "<PROJECTS>" <<'PY'
import json, sys, collections
from pathlib import Path
root = Path(sys.argv[1]); by_type = collections.Counter(); eg = collections.defaultdict(list)
for erc in root.rglob("*_erc.rpt"):
    try: d = json.loads(erc.read_text())
    except Exception: continue
    seen = {v["type"] for sh in d.get("sheets", []) for v in sh.get("violations", []) if v.get("severity") == "error"}
    for t in seen:
        by_type[t] += 1; eg[t].append(str(erc).split("/projects/")[-1].split("/generated")[0])
print("ERC error type -> #runs affected (>1 = systematic synthesis-code bug; 1 = likely this design):")
for t, n in by_type.most_common():
    print(f"  {t}: {n} run(s)   e.g. {eg[t][:4]}")
PY
```

**(b) Power-net errors** (`power_output_conflict` / `power_pin_not_driven`) — resolve which nets have a real driver. A net with a `power_out` pin is already driven and must **not** get a PWR_FLAG; a net with none **needs** one. Reuse `<PY>` / `<RUN>`:

```bash
"<PY>" - "<RUN>" <<'PY'
import json, sys
from pathlib import Path
from collections import defaultdict
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.design.models import is_power_or_ground_name
run = Path(sys.argv[1])
sf = run / "kicraft" / "state.json"
if not sf.is_file(): sf = next(run.rglob("state.json"), None)
bom = json.loads(sf.read_text())["bom"]; parts = {p["ref"]: p for p in bom["parts"]}
def ptype(ref, pin):
    p = parts.get(ref)
    if not p: return "?"
    try: pins = lookup_pins(p["symbol"])["pins"]
    except Exception: return "?"
    return next((q["electrical_type"] for q in pins if q["number"] == str(pin)), "?")
nets = defaultdict(list)
for c in bom["connections"]:
    if is_power_or_ground_name(c["net_name"]): nets[(c.get("sheet"), c["net_name"])].append(c)
print("power/ground nets — DRIVEN iff a power_out pin is present, else a PWR_FLAG is required:")
for (sheet, net), cs in sorted(nets.items()):
    eps = [(ep["ref"], ep["pin"], ptype(ep["ref"], ep["pin"])) for c in cs for ep in c["endpoints"]]
    drv = [f"{r}.{pin}" for r, pin, t in eps if t == "power_out"]
    print(f"  [{sheet}] {net}: {'DRIVEN by ' + ', '.join(drv) if drv else 'no driver -> needs PWR_FLAG'}")
    for r, pin, t in eps: print(f"       {r}.{pin} = {t}")
PY
```

## 3. Server log around the run (deployed path: systemd journal)

Per-stage `[ok]/[FAIL]` cost lines and any Python traceback. The run dir mtime is when it *finished*, so look at a window ending there. Reuse `<RUN>`:

```bash
T=$(stat -c %Y "<RUN>")
S=$(date -d "@$((T-900))" '+%Y-%m-%d %H:%M:%S'); U=$(date -d "@$((T+60))" '+%Y-%m-%d %H:%M:%S')
journalctl -u kicraft-web --no-pager -S "$S" -U "$U" 2>/dev/null | tail -60 \
  || echo "no journal access (try sudo; or if the app runs manually, check its stdout/log)"
```

Noise to ignore: `... kicraft.io/$$:0:$$ not found` and other `/robots.txt`, `/ads.txt`, `cmd_sco` 404s are unrelated frontend/crawler hits, not the run's failure.

## 4. Report — failing stage, errors, and the suspect

Summarise crisply: the failing stage, the ERC errors (with ×100 coords), **code bug vs model output**, the suspect module, and a recommended next step. Map ERC error types to their usual cause:

(These are the real KiCad `type` strings that steps 1 and 2a print.)

| ERC error type | Usual root cause | Where to look |
|---|---|---|
| `pin_to_pin` ("Power output and Power output are connected") | PWR_FLAG added to a net already driven by a `power_out` pin | `emitter.py:_power_nets_with_driver` / `emit_schematic` (made driver-aware — if it recurs, a driver pin wasn't classified `power_out`; confirm with step 2b) |
| `power_pin_not_driven` | undriven rail missing a PWR_FLAG, **or** the LLM left the rail's feed pin unwired | step 2b; if the net isn't in `bom.connections` at all → wiring stage (model) |
| `wire_dangling` ("Wires not connected to anything") | WIP band/trunk router emits a 2-endpoint trunk KiCad doesn't net to its end pins | `router.py:_draw_trunk` (systematic — check the cross-run count in 2a) |
| `label_dangling` | a net/hierarchical label whose stub doesn't land on a wire or pin | `router.py` stub+label fallback and `emitter.py:_emit_root` sheet-pin stubs (systematic — usually the most common; investigate) |
| `pin_not_connected`, or gate `9.11 net coverage` on one run | the wiring stage (LLM) left a pin unwired | `state.json` `bom.connections` — model output, not a code bug |

Rule of thumb: **same error across many runs (step 2a) = synthesis-code bug; on one design only = the model's wiring.** Quote the ERC `pos` at ×100 so the next agent can open the `.kicad_sch` straight to the spot.
