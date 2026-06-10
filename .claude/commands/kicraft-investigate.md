---
description: Investigate why a KiCraft run failed — across BOTH the schematic (ERC) and the PCB placement/routing (DRC). Locates the run, prints the build verdict, localizes the ERC/DRC failures, classifies systematic code/footprint bugs vs per-design model gaps across all runs, and recommends a generalizable fix.
argument-hint: "[uid/pid | pid | /path/to/run] (optional; default: most recent run)"
---

Investigate a failed KiCraft run and hand back a fast, accurate picture of **why** it failed and **whether the fix is generalizable** (a synthesis/layout-code or footprint-library bug that hits *every* design) **or per-design** (this design's model output). Target run: `$ARGUMENTS` (may be empty → most recent run).

A KiCraft `build` is sequential: **synthesize+ERC → place leaves → compose+route parent → verify (DRC)**. The exit code tells you the stage it died at, so the investigation **branches**:

| build rc | label | died at | investigate |
|---|---|---|---|
| 3 / 4 | incomplete / synth input | before ERC | §2 schematic |
| 5 | ERC errors | the schematic gate | §2 schematic |
| 6 | route/infra failed | placement/compose/route (no routable board produced) | §3 PCB |
| 7 | not fab-ready (DRC) | the routed board is dirty (shorts/unconnected/clearance) | §3 PCB |
| 0 | fab-ready | — | optional §3 to audit placement quality |

A "run" lives under `<projects_dir>/<uid>/<pid>/` (web) **or** any self-eval `run_NN_*` dir (you can pass its full path). Each leaves `events.jsonl` (per-stage + build events), `state.json` (committed design incl. BOM), `kicraft/synthesis_check.json` (the gate verdict), a generated KiCad tree with an ERC report `generated/<stem>/<stem>_erc.rpt`, and — once the build reaches layout — a rich `generated/<stem>/.experiments/` tree (per-leaf solves, per-round parent compose/route, DRC sidecars, the routed board).

**Two coordinate conventions — do not mix them up:** ERC report `pos` is reported at **1/100 of real mm** (multiply by 100 — see `kicad-erc-report-coords-x100`). PCB/DRC coordinates (`x_mm`/`y_mm`, `inspect_parent` issue coords) are **already real mm — never ×100 them.**

Print a one-line note before each step (visible reasoning); keep tool fan-out small.

## 1. Locate the run + unified triage — one block

Resolves the *same* projects dir the server uses, picks the run, and prints the full failure picture across schematic **and** PCB. Note the printed `RUN` / `PROJECTS` / `PY` — shell vars do **not** persist between Bash calls, so paste them literally into later steps. The last line routes you to §2 or §3.

```bash
REPO=$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/KiCraft"); PY="$REPO/.venv/bin/python"
PROJECTS="${KICRAFT_PROJECTS_DIR:-}"
[ -z "$PROJECTS" ] && [ -f "$REPO/.env" ] && PROJECTS=$(grep -E '^KICRAFT_PROJECTS_DIR=' "$REPO/.env" | tail -1 | cut -d= -f2- | tr -d "\"' ")
[ -z "$PROJECTS" ] && PROJECTS="$HOME/.kicraft/projects"
ARG="$ARGUMENTS"
if   [ -n "$ARG" ] && [ -d "$ARG" ]; then RUN="$ARG"                       # explicit path (web or self-eval run)
elif [ -n "$ARG" ] && [ -d "$PROJECTS/$ARG" ]; then RUN="$PROJECTS/$ARG"   # uid/pid
elif [ -n "$ARG" ]; then RUN=$(find "$PROJECTS" -mindepth 2 -maxdepth 2 -type d -path "*/$ARG" 2>/dev/null | head -1)
else RUN=$(find "$PROJECTS" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-); fi
echo "RUN=$RUN"; echo "PROJECTS=$PROJECTS"; echo "PY=$PY"
"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
print(f"\n=== {run.name}   ({run}) ===")
# --- pipeline + build verdict ---
ev = run / "events.jsonl"
if ev.is_file():
    lines = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
    stages = [e for e in lines if e.get("kind") == "stage_done"]
    bd     = [e for e in lines if e.get("kind") == "build_done"]
    blog   = [e for e in lines if e.get("kind") == "build_log"]
    if stages:
        print("stages:", [(e.get("stage"), "ok" if e.get("ok") else "FAIL") for e in stages])
    print("build_done:", bd[-1] if bd else "(no build step reached)")
    for e in blog[-12:]:
        print("   build_log:", e.get("text"))
else:
    print("no events.jsonl (the run may not have started the pipeline)")
sc = next(run.rglob("synthesis_check.json"), None)
if sc:
    d = json.loads(sc.read_text())
    print(f"synthesis_check: status={d.get('status')}  failed_checks={d.get('failed_checks')}")
# --- schematic: ERC errors (pos shown ×100 to match the .kicad_sch) ---
erc = next(run.rglob("*_erc.rpt"), None)
if erc:
    d = json.loads(erc.read_text())
    errs = [(sh["path"], v) for sh in d.get("sheets", [])
            for v in sh.get("violations", []) if v.get("severity") == "error"]
    print(f"\nERC errors: {len(errs)}   ({erc})")
    for path, v in errs[:20]:
        print(f"  [{path}] {v['type']}: {v['description']}")
        for it in v.get("items", []):
            pos = it.get("pos") or {}
            xy = f"  @ x100=({pos.get('x',0)*100:.2f}, {pos.get('y',0)*100:.2f}) mm" if pos else ""
            print(f"       - {it.get('description','')}{xy}")
else:
    print("\nNo ERC report — synthesis likely crashed before ERC ran (or never ran).")
# --- PCB: placement/route triage (only present once the build reaches layout) ---
exp = next((run/"generated").glob("*/.experiments"), None) if (run/"generated").is_dir() else None
if exp:
    def L(p):
        try: return json.loads(Path(p).read_text())
        except Exception: return {}
    rs = L(exp/"run_status.json"); hs = L(exp/"hierarchical_summary.json")
    routed = sorted(exp.glob("**/parent_routed.kicad_pcb"))
    leaves = []
    for dbg in sorted(exp.glob("subcircuits/*/debug.json")):
        ex = L(dbg).get("extra", {})
        la = ex.get("leaf_acceptance_structured") or ex.get("leaf_acceptance") or {}
        leaves.append(bool(la.get("accepted")))
    nacc = sum(leaves)
    print(f"\nLAYOUT: top_level={rs.get('top_level_status')}  kept_a_fab_ready_round={hs.get('best_round') is not None}"
          f"  leaves_accepted={nacc}/{len(leaves)}")
    if routed:
        print(f"   routed parent board EXISTS -> rc7 family (routed but DRC-dirty). Deep-dive: §3a then §3b.")
        print(f"   {routed[-1]}")
    else:
        print(f"   NO routed parent board -> rc6 family (parent never produced a routable board). Deep-dive: §3a + build log (§5).")
else:
    print("\nLAYOUT: no .experiments tree -> build never reached layout (rc<=5). Investigate the schematic (§2).")
PY
```

Read off: which stage failed, the ERC errors (×100 coords), and the LAYOUT line. **Route yourself:** ERC errors / rc≤5 → **§2**. `LAYOUT … routed board EXISTS` or `NO routed parent board` → **§3**. A `stage = FAIL` the journal shows as *"parked: awaiting a clarifying answer"* is the pipeline **waiting on the user**, not a crash.

## 2. Schematic deep-dive (ERC) — when the build died at/before the ERC gate

**(a) Cross-run ERC scan** — is this error type unique to this design (a wiring/model gap) or hitting many runs (a synthesis-code bug)? Reuse `<PY>` / `<PROJECTS>`:

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

Then jump to §6 (the ERC root-cause table).

## 3. PCB deep-dive (placement + routing) — when rc=6/7

The board is built bottom-up: each sheet is solved+routed as its own **leaf** mini-board, then all leaves are **composed** into a parent and the parent is routed, then a **verify** gate (no shorts, no unconnected, DRC clean) decides fab-ready. Failures live at one of those three layers — this step localizes which.

**(a) Layout verdict — per-leaf + parent.** Reuse `<PY>` / `<RUN>`:

```bash
"<PY>" - "<RUN>" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
exp = next((run/"generated").glob("*/.experiments"), None)
def L(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}
if not exp:
    print("no .experiments -> build never reached layout (rc<=5); investigate the schematic (§2)."); sys.exit()

print("LEAVES (each sheet solved+routed as its own mini-board):")
for dbg in sorted(exp.glob("subcircuits/*/debug.json")):
    d = L(dbg); ex = d.get("extra", {})
    la = ex.get("leaf_acceptance_structured") or ex.get("leaf_acceptance") or {}
    fs = ex.get("failure_summary") or {}
    nm = (ex.get("solve_summary", {}) or {}).get("sheet_name") or d.get("metadata", {}).get("sheet_name", "?")
    gates = la.get("gate_results") or {}
    failed = [g for g, r in gates.items() if isinstance(r, dict) and r.get("passed") is False]
    print(f"  [{nm}] accepted={la.get('accepted')} reject={la.get('rejection_reasons')} "
          f"failed_gates={failed or '-'} round_reasons={fs.get('unique_reasons')}")

# Parent: the round that produced a routed board (if any), else the last attempted.
pps = sorted(exp.glob("hierarchical_autoexperiment/round_*/parent_pipeline.json"))
routed = [(pp, L(pp).get("state", {})) for pp in pps if (L(pp).get("state", {}) or {}).get("routed_validation") is not None]
pp, st = routed[-1] if routed else ((pps[-1], L(pps[-1]).get("state", {})) if pps else (None, {}))
print(f"\nPARENT (compose+route of all leaves) [{pp.parent.name if pp else 'none'}]:")
cs = st.get("candidate_search") or {}; sd = st.get("stamp_drc") or {}; gv = st.get("geometry_validation") or {}
print(f"  candidate_search: tried={cs.get('tried')} placement_accepted={cs.get('accepted')} rejected_for_drc={cs.get('rejected_drc')}")
print(f"  stamp_drc (composer, PRE-route): shorts={sd.get('shorts')} clearance={sd.get('clearance')}"
      f"   <- shorts>0 here = the composer STAMPED overlapping copper (composer bug, not the router)")
print(f"  geometry: components_outside_outline={gv.get('outside_component_count')} pads_outside={gv.get('outside_pad_count')}")
rv = st.get("routed_validation")
if rv:
    drc = rv.get("drc", {})
    print(f"  routed_validation: accepted={rv.get('accepted')} reasons={rv.get('rejection_reasons')}")
    print(f"     DRC (already real mm): shorts={drc.get('shorts')} unconnected={drc.get('unconnected')} "
          f"clearance={drc.get('clearance')} annular={drc.get('annular_width')} padstack={drc.get('padstack')} "
          f"on_footprints={drc.get('clearance_footprint_refs')}")
    print(f"     unconnected nets: {drc.get('unconnected_nets')}")
else:
    print("  routed_validation: NONE -> the parent NEVER produced a routed board (rc6).")
    print("     => freerouting could not route the composed board as placed, or compose/route raised.")
    print("        Read stamp_drc + candidate_search above and the build log (§5) for an exception/SIGSEGV.")
rb = sorted(exp.glob("**/parent_routed.kicad_pcb"))
if rb:
    print(f"\n  ROUTED BOARD: {rb[-1]}\n  -> localize its DRC freshly with inspect_parent (§3b).")
PY
```

**(b) Localize the dirty routed board (rc=7).** When a `parent_routed.kicad_pcb` exists, re-run the *authoritative* gate + a localized DRC: `inspect_parent` re-runs `kicad-cli` DRC and **clusters every violation by footprint ref with real-mm coords**, prints a `BROKEN/WORKS` verdict, and flags packing/stacking waste. (The fresh DRC is authoritative — it can differ from the persisted `routed_validation`.) Reuse `<PY>` / `<RUN>`:

```bash
RB=$(find "<RUN>" -name parent_routed.kicad_pcb 2>/dev/null | sort | tail -1)
if [ -n "$RB" ]; then
  OUT=$(mktemp -d)
  "<PY>" -m kicraft.cli.inspect_parent "$RB" --output-dir "$OUT" >/dev/null 2>&1 \
    && sed -n '1,45p' "$OUT/summary.md" \
    && echo "(full: $OUT/summary.md · annotated_top.png · report.json)"
else
  echo "no parent_routed.kicad_pcb -> rc6 (parent never routed); the failure is in §3a's parent block + the build log (§5)."
fi
```

A DRC error clustered on one ref (e.g. `J1 … annular width … actual -0.0032 mm` / `PTH pad hole leaves no copper`) is a **footprint-library** problem, not a per-design one — confirm it recurs in §4. **Shorts/unconnected** are the only fab-blockers the verify gate enforces; `silk_over_copper` / `silk_overlap` are cosmetic warnings that do **not** fail the build.

## 4. Systematic vs per-design — the generalizable-fix engine (cross-run PCB scan)

This is the heart of the request: scan **every** run's layout artifacts and rank each failure mode by **how many distinct designs it hits**. >1 design = a **systematic** code/footprint bug whose one fix generalizes; exactly 1 = this design's model output. Reads only persisted JSON (no `kicad-cli`), so it's fast. Reuse `<PY>` / `<PROJECTS>`:

```bash
"<PY>" - "<PROJECTS>" "$HOME/.kicraft/self_eval" "$HOME/KiCraft/logs/self_eval" <<'PY'
import json, sys, collections
from pathlib import Path
roots = [Path(a) for a in sys.argv[1:]]
def L(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}
def verdict(exp):
    pps = sorted(exp.glob("hierarchical_autoexperiment/round_*/parent_pipeline.json"))
    routed = [L(pp).get("state", {}) for pp in pps if (L(pp).get("state", {}) or {}).get("routed_validation") is not None]
    st = routed[-1] if routed else (L(pps[-1]).get("state", {}) if pps else {})
    rb = sorted(exp.glob("**/parent_routed.kicad_pcb"))
    leaves = []
    for dbg in exp.glob("subcircuits/*/debug.json"):
        ex = L(dbg).get("extra", {}); la = ex.get("leaf_acceptance_structured") or ex.get("leaf_acceptance") or {}
        fs = ex.get("failure_summary") or {}
        leaves.append((bool(la.get("accepted")), fs.get("unique_reasons") or []))
    return rb, st.get("routed_validation"), leaves
runs = sorted({exp.parent.parent.parent for root in roots if root.is_dir() for exp in root.rglob(".experiments")})
tier = collections.Counter()
reject = collections.defaultdict(set); drc = collections.defaultdict(set); fp = collections.defaultdict(set)
nets = collections.defaultdict(set); lreason = collections.defaultdict(set)
for run in runs:
    exp = next((run/"generated").glob("*/.experiments"), None)
    if not exp: continue
    tag = run.name; rb, rv, leaves = verdict(exp)
    tier["route_fail (no parent board, rc6)" if not rb else
         ("dirty (routed, not fab-ready, rc7)" if rv and rv.get("accepted") is False else
          ("clean (fab-ready)" if rv and rv.get("accepted") else "unknown"))] += 1
    if rv:
        for r in rv.get("rejection_reasons") or []: reject[r].add(tag)
        d = rv.get("drc", {})
        for k in ("shorts", "unconnected", "clearance", "annular_width", "padstack"):
            if (d.get(k) or 0) > 0: drc[k].add(tag)
        for ref in d.get("clearance_footprint_refs") or []: fp[ref].add(tag)
        for net in d.get("unconnected_nets") or []: nets[net].add(tag)
    for _acc, reasons in leaves:
        for r in reasons: lreason[r].add(tag)
print(f"=== CROSS-RUN PCB SCAN: {len(runs)} runs with layout artifacts ===")
print("tiers:", dict(tier))
def show(title, d):
    rows = sorted(((k, len(v), sorted(v)[:3]) for k, v in d.items()), key=lambda x: -x[1])
    print(f"\n{title}  (-> #designs; >1 = SYSTEMATIC, fix generalizes; 1 = this design):")
    for k, n, eg in rows: print(f"  {k}: {n}  e.g. {eg}")
show("parent rejection reasons", reject)
show("parent DRC error types", drc)
show("clearance footprint refs (a recurring ref = footprint-library bug, one .kicad_mod fix)", fp)
show("unconnected nets (a recurring net = a missing tie/pour rule for that net family)", nets)
show("leaf failure reasons", lreason)
PY
```

The output tells you, for each failure mode, the exact set of designs it hits. **A ref like `J1` or a net like `CC2`/`GND` recurring across many designs is your generalizable fix** — change the one footprint / add the one tie-or-pour rule and every affected board improves.

## 5. Build log around the run (deployed: systemd journal)

The `[build] N/5 …` stage lines, `error: …` (the exact rc cause), and any Python traceback / `freerouting` stderr / `SIGSEGV`. The run dir mtime is when it *finished*, so look at a window ending there. Reuse `<RUN>`:

```bash
T=$(stat -c %Y "<RUN>")
S=$(date -d "@$((T-1500))" '+%Y-%m-%d %H:%M:%S'); U=$(date -d "@$((T+60))" '+%Y-%m-%d %H:%M:%S')
journalctl -u kicraft-web --no-pager -S "$S" -U "$U" 2>/dev/null | grep -iE 'build|error|freerout|segv|trace|route|drc' | tail -80 \
  || echo "no journal access (try sudo; or for a self-eval run read its run.log / the batch summary.json)"
```

Noise to ignore: `... kicraft.io/$$:0:$$ not found`, `/robots.txt`, `/ads.txt`, `cmd_sco` 404s are crawler hits, not the run's failure.

## 6. Report — failing stage, true root cause, and a GENERALIZABLE fix

Summarise crisply: the failing stage, the specific failure (with the right coords — ERC ×100, DRC real-mm), **code-bug vs footprint-library-bug vs model output**, the suspect module, and a recommended fix. **Lead with §4's #-designs-affected count** — if a failure hits many designs, name the *one* change that fixes the class.

**Schematic (ERC) root causes** — the real KiCad `type` strings §1/§2a print:

| ERC error type | Usual root cause | Where to look |
|---|---|---|
| `pin_to_pin` ("Power output … Power output are connected") | PWR_FLAG added to a net already driven by a `power_out` pin | `emitter.py:_power_nets_with_driver` (driver-aware; if it recurs, a driver pin wasn't classified `power_out` — confirm with §2b) |
| `power_pin_not_driven` | undriven rail missing a PWR_FLAG, **or** the LLM left the feed pin unwired | §2b; if the net isn't in `bom.connections` at all → wiring stage (model) |
| `wire_dangling` ("Wires not connected to anything") | trunk router emits a 2-endpoint trunk KiCad doesn't net to its end pins | `router.py:_draw_trunk` (systematic — check §2a) |
| `label_dangling` | a net/hierarchical label whose stub doesn't land on a wire or pin | `router.py` stub+label fallback / `emitter.py:_emit_root` sheet-pin stubs (systematic — usually most common) |
| `pin_not_connected`, or gate `9.11 net coverage` on one run | the wiring stage (LLM) left a pin unwired | `state.json` `bom.connections` — model output, not a code bug |

**PCB (placement + routing) root causes** — from §3/§4 (DRC coords are real mm):

| Symptom | Usual root cause | Bug class | Where to look / generalizable fix |
|---|---|---|---|
| rc6: `routed_validation = NONE`, leaves accepted, `candidate_search` placed but no routed board | freerouting can't route the composed parent as placed (congestion / parent interconnect), or compose/route raised | code (if §4 ≫1) | `compose_subcircuits._route_parent_board`, freerouting params; more spacing / routing channels; build log for exception/SIGSEGV (`parent-route-strip-segv`) |
| rc6: `FreeroutingUnavailableError` in the log | Java / FreeRouting jar / xvfb missing | infra | install the toolchain (`build-fail-missing-freerouting-java`) |
| rc6: degenerate hierarchy (0 leaves) | BOM chose an all-in-one SoC → architecture collapsed to no per-block sheets | model/arch | architecture/BOM partition by IC domain (`wiring-park-integrated-soc`, `reconcile-stage-plan`) |
| rc7: `reasons=['unconnected_nets']`, the SAME nets (e.g. `GND`,`CC2`) recur in §4 | parent route leaves a power/CC net family untied; a tie-or-pour rule is missing for it | code (systematic) | `_route_parent_board` power pours + `breakout_stubs.py`; add the tie/pour for that net family (`usb-c-leaf-unrouted-accepted`) |
| rc7: `clearance`/`annular_width`/`padstack` clustered on the SAME ref (e.g. `J1`) across designs | that part's **footprint** has a too-thin annular ring / pad-vs-hole / courtyard; DRC fails on every board using it | footprint-library | fix the `.kicad_mod` pad/annular/courtyard for that footprint — one fix → all boards (cf. the ESP32-mini antenna-keepout `.kicad_mod` fix) |
| rc7: `stamp_drc.shorts > 0` | the composer stamped overlapping copper (e.g. a perimeter tie across a part's own pads) | code | `breakout_stubs.py` foreign-pad guard (`dense-leaf-route-fail`) |
| leaf `reject=['no_unconnected']` / `leaf_routed_artifact_validation` recurring | the leaf couldn't route all its internal nets, or routed-artifact validation rejects it | code | `leaf_routing.py` / `leaf_acceptance.py`; the leaf's pour/stub strategy |
| leaf `leaf_pre_stamp_legality_repair` / `illegal_unrepaired_leaf_placement` | placement couldn't be legalized (overlap / keepout) before routing | code | `placement_solver` legalize + keepouts (e.g. antenna keepout) |
| `geometry: components_outside_outline > 0` | placement put parts outside the board outline | code | compose placement clamp / board-outline sizing |

**Rule of thumb:** the failure's **breadth across designs (§2a for ERC, §4 for PCB) is the verdict.** Many designs → a synthesis/layout **code** bug or a **footprint-library** bug whose single fix generalizes (name it). One design → that design's **model** output (wiring/part choice). Quote the offending ref + real-mm DRC coord (or ×100 ERC pos) so the next agent can open the board/schematic straight to the spot.
