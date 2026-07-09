# Placement streamline — handoff (2026-07-09)

> **⚠️ SUPERSEDED (2026-07-09) →
> [`placement-reconsider-connectivity-first-handoff.md`](./placement-reconsider-connectivity-first-handoff.md).**
> Visual inspection of the shipped soft-tidiness layouts showed the tidiness is cosmetic and
> often electrically wrong (decoupling caps 6–20 mm from the pins they bridge). The direction
> below (soft-tidiness scorer term + tune/delete) is now **history**; the current plan is a
> discrete anchor-relative placement grid with SA doing assignment. **Start at the new doc.**

**(History)** Deep design + decision detail is in
[`placement-pipeline-streamline.md`](./placement-pipeline-streamline.md); this was the resume map.

## TL;DR — where we landed

The goal: make leaf layouts look less random (passives in neat rows, consistent orientation,
tighter) **and** streamline the accreted place/route pipeline. After trying three approaches, the
validated answer is **soft tidiness**: a term in the placement objective the existing SA
co-optimizes with routing — *not* a constraint, gate, or post-pass.

- **Validated:** routing does **not** regress on the dense canary (RP2040 MCU: classic 21 vs soft
  18 unconnected, N-of-3 median), and orientation consensus jumps where routing has headroom
  (1A_LED_DRIVER leaf: 50% → 100%, residual/fill at parity). One objective, zero per-leaf
  conditionals, deletes the competing systems.
- **Pending:** corpus-wide A/B (run the harness below), tune `psw_tidiness` (ideally via the
  CMA-ES tuner), then delete the superseded systems (the LOC win).

## The decision log — do NOT re-try the falsified approaches

Three approaches were built and measured (N-of-3 routing sweeps, `scripts/phase1_routing_parity.py`):

| Approach | Shape | Dense RP2040 routing | Verdict |
|---|---|---|---|
| **Packer** (`leaf_structured_layout.py`) | post-pass: tidy rows after SA | 19 → **24** unconnected | ❌ regressed. HPWL guard didn't help (harm is congestion, not wirelength). |
| **Group-as-unit** (`leaf_group_rigid.py` + `_group_rigid_sa`) | rigid tidy groups; SA moves anchors only | 23 → **27** | ❌ regressed. A rigid 9-cap row around a 43-net MCU over-constrains (1–2 movable DOF). |
| **Soft tidiness** (`PlacementScore.tidiness`) | scoring term, SA co-optimizes | 21 → **18** | ✅ **no regression.** The one to build on. |

**The lesson (irreducible):** crisp tidiness and best routability genuinely conflict on a
congested leaf — the tidy arrangement space doesn't contain the best-routable layout. So *any*
hard-imposed tidiness (constraint / post-pass / rigid group) pays for it in dense routing, and no
cheap pre-route guard fixes it. Only a **soft** term that yields to routing generalizes. Don't add
a congestion gate or any per-leaf `if dense:` branch — that's the patch disease this work removes.

## The current approach (soft tidiness) — what's in the tree

All changes are uncommitted on `main`. Nothing is deployed.

**Live (leaf-path default):**
- `autoplacer/brain/types.py` — `PlacementScore.tidiness` field + `"tidiness"` weight (0 by
  default, so default/parent scoring is byte-identical).
- `autoplacer/brain/placement_scorer.py` — `_score_tidiness()`: per functional group, `0.5 *
  orientation-consensus + 0.5 * alignment`, where alignment = `100 * exp(-residual / ref_mm)`
  (smooth — a linear clamp saturated to 0 above 3mm and starved SA of the alignment gradient; that
  was a real bug, now fixed). Grouping is net-based → memoized once; short-circuits to neutral (no
  cost) when `psw_tidiness <= 0`, so parent/default solves pay nothing.
- `autoplacer/brain/leaf_size_reduction.py` (`local_solver_config`) — sets `psw_tidiness = 0.15`
  and `tidiness_residual_ref_mm = 3.0` for leaves. Also sets `leaf_group_rigid = False` and
  `leaf_structured_local_layout = False` (both hard-tidiness approaches OFF by default).

**Dormant (behind flags, default OFF — keep for A/B, delete once soft tidiness is confirmed):**
- `leaf_structured_layout.py` (the packer) + Step 15.7 wiring in `placement_solver.py`.
- `leaf_group_rigid.py` + `_group_rigid_sa` + the group-rigid branch/gates in `placement_solver.py`.

**Shared foundation (keep):**
- `leaf_tidiness.py` — `assign_passive_groups` (the single grouping definition, used by the metric,
  the scorer term, and the renderer so they can't disagree) + the tidiness metric.
- `leaf_layout_svg.py` — diagnostic renderer.

**Tests (all green, 0 regressions from this work; 5 pre-existing env failures unrelated):**
`test_leaf_tidiness.py`, `test_leaf_structured_layout.py`, `test_leaf_group_rigid.py`.

## Tooling (all `$0`, no LLM)

| Script | What it does |
|---|---|
| `scripts/leaf_tidiness_report.py [CORPUS]` | Per-design + corpus tidiness metrics from `solved_layout.json`. |
| `scripts/leaf_layout_viz.py [CORPUS] --designs=A,B [--rigid]` | Annotated diagnostic SVGs + gallery. |
| `scripts/phase1_routing_parity.py` | N-of-3 routing parity, soft/classic (edit `DESIGNS`/`SEEDS`). The rigorous routing verdict. |
| **`scripts/soft_tidiness_ab.py --out DIR`** | **Corpus-wide A/B: solves classic vs soft per design, measures tidiness + unconnected, emits `DIR/index.html` with side-by-side classic\|soft renders per leaf.** |

Baseline corpus: `logs/self_eval/20260707T193651Z` (yesterday's 34-brief batch, 111 leaves).
Baseline tidiness: orientation-consensus 73.1% grouped / 81.4% leaf, residual 4.02mm, fill 47.9%.

## First A/B result (3 designs, seed 0 — `scripts/soft_tidiness_ab.py`)

| leaf | orientation cl→soft | residual cl→soft | routing (unc) cl→soft |
|---|---|---|---|
| 1A_LED_DRIVER / driver (sparse, messy) | 50 → **100%** | 4.1 → 4.2 (parity) | 0 → 0 |
| HIGH_SIDE / load switch (sparse, clean) | 75 → **100%** | 1.5 → **3.0** ⚠️ | 0 → 0 |
| RP2040 / MCU (dense) | 67 → **100%** | 3.8 → **2.8** | 20 → **17** |
| RP2040 / POWER | 83 → **100%** | 3.2 → **2.0** | — |

**Orientation → 100% on every grouped leaf** (the core win, sparse and dense alike). **Routing never
regressed** (better on the dense canary). **Residual is the open tuning item:** better on dense,
parity on LED, but *worse* on the already-tidy HIGH_SIDE — the 0.5/0.5 orient-vs-align split lets
SA perfect orientation while spreading a tight group. Fix in step 2 (tune the split / `psw_tidiness`
/ `ref_mm`, ideally via CMA-ES); the residual reward gradient is now correct, so this is a weighting
question, not a bug. Page: `soft_tidiness_ab.py --out DIR` → `DIR/index.html`.

## How to resume — the remaining work, in order

1. **Corpus-wide A/B.** `python scripts/soft_tidiness_ab.py --out /tmp/soft_ab` (all 5 designs in
   `DESIGNS`; add more for full coverage). Open `/tmp/soft_ab/index.html` — per-leaf classic|soft
   renders + a metrics table. Confirm: orientation up across sparse leaves, residual not worse,
   **routing not regressed on any design** (cross-check dense ones with
   `phase1_routing_parity.py` at N-of-3). This is the gate before deletion.
2. **Tune `psw_tidiness`** (and the 0.5/0.5 orient-vs-align split, and `ref_mm`). Prefer the
   existing CMA-ES tuner (`kicraft/tuning/`, `$0` via replay) over hand-picking — it can co-optimize
   tidiness against routability/area on the corpus. Watch for residual able to go *below* classic
   (crisper rows) where routing allows.
3. **Delete the superseded systems** — once the corpus A/B holds: remove the packer
   (`leaf_structured_layout.py` + Step 15.7), group-rigid (`leaf_group_rigid.py` + `_group_rigid_sa`
   + branch), and the legacy `_apply_orderedness` (Step 8.5) + `apply_leaf_passive_ordering`
   (`leaf_passive_ordering.py`) + `placement_alignment.py` + the `_re_snap_aligned_pairs` churn.
   That's the ~1,000-line LOC win (see streamline plan §"Why it simplifies"). Do it behind the
   test suite + a replay of the corpus.
4. **(Separable) tail simplification.** Collapsing the 7-pass overlap/clamp/restore tail into one
   legalizer is still a legibility goal but is now *decoupled* from tidiness (soft tidiness doesn't
   need a group-aware legalizer). Track independently.

## Gotchas / lessons banked

- **Never compare routing across two separate solves** — FreeRouting is only best-effort-stable;
  use N-of-3 medians. Single-run on/off deltas are noise (this bit us early).
- **Alignment reward needs a gradient everywhere** — a linear clamp that hits 0 above a threshold
  leaves SA blind to the very thing it should optimize.
- **`solved_layout.json` isn't written for leaves that fail acceptance** (dense designs) — fall
  back to the latest `round_*_solved_layout.json` (the A/B harness does this).
- **`glob('**')` skips dot-dirs** — leaves live under `.experiments/`; use `os.walk`.
- Grouping is **net-based (position-independent)** — memoize it; don't recompute per SA iteration.
