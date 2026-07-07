# KiCraft whole-codebase review — 2026-07-06

> Backlog from the `[ultra]` multi-agent review of the entire codebase (pipeline + website).
> Each item is independently actionable; check items off as they are implemented.

## Provenance and how to read this

- Produced by a 3-phase multi-agent review: **13 scoped defect-finder angles** (web app, SQLite
  store + build worker, security/authz, synthesis, placement core, routing, CLI orchestration,
  layout editor, parts/render, Python pitfalls, cross-process IPC contracts, reuse/simplify,
  efficiency) + **2 idea scouts**, then per-file **adversarial verification**, then a gap sweep.
- 78 raw candidates → 74 after dedup → **73 kept, 1 refuted** (an X-Forwarded-For spoofing claim
  whose exploit premise did not hold; see appendix).
- **Verification is partial.** The verify fan-out hit the API session limit twice: only the
  `web.py`, `accounts.py`, and `build_worker.py` batches completed (15 CONFIRMED + 2 PLAUSIBLE
  verdicts, each spot-checked). The remaining **56 findings are marked NEEDS-VERIFICATION** —
  the finder anchored them to real lines it read, but no adversarial pass ran. **Re-verify each
  against current source before implementing a fix.** The gap-sweep phase also did not run, so
  this list is thorough but not exhaustive.
- Statuses: `VERIFIED-CONFIRMED` (verifier reproduced the failure logic and quoted the lines),
  `VERIFIED-PLAUSIBLE` (mechanism real, trigger timing/env-dependent), `NEEDS-VERIFICATION`.

**Counts:** 55 bug fixes · 16 cleanups · 2 convention fixes ·
10 feature enhancements · 10 engineering improvements

## Implementation log

- **2026-07-06** — first batch (16 items): B1–B5, B16–B18, B30, B38, B41–B43, C7, C8, V1.
  Where the fix shape differs from the finding's suggestion:
  - **B2**: `enqueue_build` dedupe is scoped to (workspace, **kind**), not workspace alone —
    `test_enqueue_kind_round_trips_and_defaults` pins two same-workspace rows of different
    kinds; cross-kind concurrency is blocked web-side by the B5 `_project_run_live` guard.
    A deduped caller gets the in-flight job's id and simply tails it.
  - **B3**: `finish_build` takes `expect="running"` (reaper passes `expect="queued"` to fail
    jobs nothing will run) and returns whether it landed. The worker's `_shutdown` now
    requeues/fails BEFORE killing, so the reader thread's `done rc=-9` finalize no-ops —
    this made the previously-stale `test_worker_shutdown_requeues` pass for real.
  - **B4**: `delete_project` now **raises ValueError** while a build_jobs row is 'running'
    (web catches it and notifies); it does not try to kill the worker's build. The
    running-workspace-left-for-GC test was rewritten to assert the refusal.
  - **C8**: no contact email exists anywhere, so the profile card now files a
    `kind="data_request"` support report (admin-visible) instead of naming an address.
    `docs/legal/*.md` still carry `[CONTACT EMAIL]` placeholders — operator decision.
  - **B30**: `gnd_pour_s` timing moved into the gnd-plane block; the power block records
    its own `power_pour_s` (no consumers of either key outside leaf_routing).
- **2026-07-06** — second batch (12 items): B6, B20, B23, B34, B36, B44, B45, C6, C11, C12,
  C14, C15. Fix-shape notes:
  - **B6**: no refuse-to-start — when KICRAFT_STORAGE_SECRET is unset,
    `render_serving.storage_secret()` generates a random per-process secret (dev keeps
    working; tokens/sessions just don't survive restarts) and web.py's `ui.run` cookie
    secret now shares it. The tautological guard test was replaced with a real
    forged-with-default-secret-must-fail assertion.
  - **B20**: the in-process fallback got the worker's watchdog-thread pattern verbatim,
    plus `errors="replace"` + `start_new_session` + group kill (B1's siblings on this path).
  - **B45**: one shared `fab_export.extract_lcsc_pin` used by BOTH the §9.26 sourcing gate
    and the fab BOM readback; excludes zero-led digit runs (C0201/C0402/C0603/C01005 — real
    LCSC numbers never lead with 0) plus a deny-list of nonzero-led package codes
    (C1206/C2512/...). Four other C-number regexes exist codebase-wide but are match-only
    (id extraction from known-id fields), left alone.
  - **B34**: `_round_index()` helper replaces the five `int(x or -1)` sites.
  - **C6**: `build_worker.JOB_KIND_COMMANDS` is now the one kind→argv map; web.py's
    `_JOB_KIND_ARGS` deleted, fallback imports the worker's map + `_kill_build`.
- **2026-07-06** — third batch (12 items): B15, B37, B50–B55, C9, C10, C13, V2. Notes:
  - **B15**: contextmenu/keydown registration moved to a once-per-IIFE `bindGlobalEvents`
    called from `fireRender` (render() replaces svg CHILDREN only, so the svg element and
    document accumulated one handler per repaint).
  - **B37**: fixed at both ends — the Python panel preserves a saved `pos`, AND the canvas's
    `setMountingHoles` keeps the live pos (matched by index) when a pushed hole lacks one
    (new-hole fallback is board center, not the min corner).
  - **B50**: `parent_local` is now an opaque canvas passthrough (state + getState echo);
    `_layout_to_canvas` emits it, so web-panel saves round-trip it.
  - **B54**: `_BLOCK_RE` lookahead extended with segment/via/zone/group/dimension/image —
    deliberately NOT `\(arc`, which legitimately nests inside gr_poly `pts`.
  - **C10**: shared wrap is `lcsc_retail.attach_stock(payload, cid, nullable=)`;
    the two retail-fake test harnesses now patch the module PRIMITIVES (enabled/stock)
    so the shared wrapper stays under test.
  - **C13**: only `_build_merged_nets` got the seen-set; `_append_pad_ref` operates on
    interface-port lists (tiny N), left as-is.
- **2026-07-06** — fourth batch (2 items): B8, B40. `kicraft/fsutil.py` provides the one
  `atomic_write_text` (same-dir pid-suffixed tmp + os.replace); ALL state.json writers now
  use it (cli_app ×5, session ×2, stage_driver ×2 incl. the previously hand-rolled one).
  Read side: `session._read_state_for_update` refuses (None) on an existing-but-unparsable
  state.json — `record_answers` skips its stamp, `null_downstream` raises loud — so a torn
  read can never be written back as `{}` over the committed design.
- **2026-07-07** — fifth batch (9 items, branch `codebase-review-batch5-geometry`,
  `55b3306`): the autoplacer geometry family B9–B12, B24–B27, B46. All nine re-verified
  against source by parallel adversarial agents before fixing (8 CONFIRMED, B27 PARTIAL).
  Where the fix shape differs from the finding:
  - **B24**: the finding's suggested fix (mirror `body_center.x`) was itself built on a
    wrong model. Empirical pcbnew experiment (pinned as
    `test_flip_composition_is_local_y_mirror_then_rotate`): the stamp's
    `Flip(pos)` + `SetOrientationDegrees(θ)` composes to `R_cw(θ)·M_y·local` — a
    **local-Y mirror**, not the world-X pad mirror `_assign_layers` applied. The fix
    mirrors pads AND body_center with the local-Y reflection `R(θ0)·M_y·R(−θ0)` at
    flip time, which makes the solver model equal the stamped board at ALL later
    cardinal rotations and fixes silently swapped 2-pad identities (pad-net positions)
    on flipped THT parts.
  - **B10/B11**: consolidated — every hand-rolled rotation site (SA move,
    `_optimize_rotations`, `_place_clusters` early-IC, random scatter) now delegates to
    `geometry.rotate_component_in_place`. The SA move (a 4th desync site the findings
    split across B10/B11) consults `block_rotation_geometry` for subcircuit blocks and
    SKIPS the move when the target rotation has no geometry entry; revert restores the
    exact saved extents (block per-rotation AABBs are not always transposes). Scatter
    picks the rotation before drawing position bounds (bounds previously used
    pre-rotation extents).
  - **B25**: indent-only fix. Note for a future CMA-ES retune: with all four quadrants
    contributing, `access_score = min(100, accessible/10)` saturates ~4× sooner on
    high-pin-count parts; the /10 divisor may want rescaling, deliberately NOT done here.
  - **B26**: half-extent pitch (`sizes[k]/2 + gap + sizes[k+1]/2`); this also makes the
    packed span equal the `total_h`/`total_w` that the grow-to-fit trigger and usable-range
    clamps already assumed (the internal inconsistency that proved the intent).
  - **B27**: severity downgraded from the finding — the main leaf path self-heals (post-route
    re-extraction from the routed board rebuilds `pad.size_mm` before the final outline), so
    the harm was pre-route canvas under-reserve (~0.7 mm), metrics, and fallback
    serialization paths, not shipped copper-to-edge faults. Fix per the artifact-path model
    (`_rotate_size`): `bbox_after_rotation` on `pad.size_mm` inside
    `rotate_component_in_place`.
  - **B12**: replaced the aggregate-refs waiver with per-violation classification
    (`_classify_clearance_violations`, works on report_text so the existing mocked tests
    stay valid): waivable iff every item line in the block names the same single footprint.
    The `ignorable_footprint_refs` escape hatch is TIGHTENED: it can waive fully-named
    multi-footprint blocks but never a block containing a ref-less (routed-copper) item;
    the dormant `clearance_count <= 10` blanket elif is gone.
  - **B46**: numpy repulsion now matches the Python fallback exactly: distance clamped to
    0.1 for magnitude (strongest force when coincident), true unit direction, and the
    degenerate coincident direction resolved like `atan2(0,0)` (antisymmetric
    `sign(col−row)` on +x).
  - **B28** was fixed independently the same day by the self-eval session (`4fabc70`,
    branch `self-eval-2026-07-07-fixes`) — see the item note.
  Validation: full placement/compose/edge suite green (123 passed, 4 skipped); new
  regression tests for the B12 ride-along hole, B27 pad-AABB rotation, and the B24 flip
  composition (pcbnew-agreement, opt-in). Replay verdicts (isolated worktree replays,
  quality=good, seed 0; single-replay caveat — run-to-run noise crosses grade buckets):
  - **i2c565** (I2C_GPIO_EXPANDER; prior build failed `connector_stranded:J1@-6.59mm(left)`
    — the exact B9 signature): stranding GONE, unconnected 2→0, util 16.0→41.9%. New
    verdict rc7 `courtyards_overlap` ×2 — both are unlocked passives (C1, C2) against the
    pinned TB1, the documented pre-existing parent-compose "connector-pinned residual"
    family, NOT connector-vs-connector (which would implicate the B26 pitch change).
  - **servo566** (PCA9685_SERVO_DRIVER): unconnected 15→8, reasons=[] — the known
    walled-off routing class (deferred C1 family), no new failure modes.

## 1. Bug fixes

### 1.1 High severity (15)

- [x] **B1** `kicraft/server/build_worker.py:188` — _run_job only catches OSError; any other exception (UnicodeDecodeError from the strict text-mode stdout iteration at line 177, or sqlite3.OperationalError from build_quality_for_user at line 149) kills the job thread and wedges the row in 'running' with a live claimant pid that no reaper will ever recover.
  - **Severity:** high · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    A build tool (freerouting JVM, KiCad CLI) prints one non-UTF-8 byte -> 'for line in proc.stdout'
    raises UnicodeDecodeError -> the thread dies, the finally block pops _procs so _shutdown won't
    requeue it, and the row stays status='running' claimed_by pid:<worker> where the worker is still
    alive -- requeue_stale_builds' _pid_alive check (accounts.py:1922) skips it forever. The user's
    project is stuck 'running' until the worker process itself is restarted; the subprocess runs
    unsupervised until the watchdog deadline, and finish_build is never called. Popen should use
    errors='replace' and _run_job needs a broad except that requeues/fails the job.
  - **Verifier:**
    The try block (line 153) wraps Popen and the strict text-mode stdout read but only catches
    OSError; a UnicodeDecodeError from line 177 (Popen sets text=True with no errors= at 155-158)
    propagates, kills the thread, and the finally at 192-194 pops _procs so _shutdown never requeues
    it. build_quality_for_user (line 149) is outside the try entirely. No finish_build/requeue_build
    runs, and requeue_stale_builds skips the row because the worker pid is alive (accounts.py:1922);
    list_unfinalized_builds only covers done/failed/queued. The row wedges in 'running' until the
    worker itself restarts, exactly as described.
  - **Key line(s):** `except OSError as e:
            _log(f"job {job.id}: failed to launch build: {e}")  (build_worker.py:188-189); for line in proc.stdout or []: (line 177, text=True strict decode); accounts.py:1922: if _pid_alive(_claimant_pid(r["claimed_by"])): continue`

- [x] **B2** `kicraft/server/accounts.py:1816` — enqueue_build has no per-workspace uniqueness or serialization, so two build_jobs rows for the same project dir can run as two concurrent `cli_app build` processes that race on the same .kicraft/state.json and generated/ tree.
  - **Severity:** high · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    A finished project is open in two browser tabs; each page holds its own state dict (web.py
    open_project builds a fresh dict per page, and the state['running'] guards at web.py:5107/5043
    are per-tab). Rebuild is clicked in both tabs -> two 'build' rows for the same workspace are
    INSERTed with no dedupe. On a host with slot_count()>=2 (build_slots.py: default cpu//6, and
    BuildWorker.max_jobs = max(1, slot_count())) the worker claims and runs both jobs concurrently
    in the same cwd. Both cli_app processes rewrite .kicraft/state.json via plain truncate-then-
    write (cli_app.py:3228) and race on generated/<stem>/ and .experiments/ -> a torn state.json
    (every later _load_state at cli_app.py:884 raises JSONDecodeError, bricking stage-commit and
    rebuilds for the project) and mixed-run board artifacts. The host-wide flock in build_slots
    gates total load, not same-workspace exclusivity.
  - **Verifier:**
    enqueue_build (accounts.py:1819-1824) is a bare INSERT with no uniqueness on
    workspace/project_id and no dedupe query. The only rebuild guards are per-page: web.py:5107 and
    5043 check `state["running"]` on the page's own state dict, and open_project only shares a dict
    when the run is already in _LIVE_RUNS — a finished project opened in two tabs gets two
    independent dicts, and _rerun_build_worker (web.py:1736) blindly overwrites `_LIVE_RUNS[pid] =
    state` without checking for an existing entry. Two Rebuild clicks therefore enqueue two rows for
    the same workspace. The worker runs up to `max(1, slot_count())` jobs concurrently
    (build_worker.py:77; build_slots.py:49 default max(1, cpus//6), so >=2 on a 12+-core host or
    KICRAFT_BUILD_SLOTS>=2), and the flock slots are numbered host-wide counters (slot files, not
    per-workspace locks), so both cli_app processes run in the same cwd. Both rewrite state.json via
    non-atomic truncate-write (`state_path.write_text(...)` in cli_app._persist_artifacts) and race
    on generated/<stem>/, so a torn state.json makes _load_state's json.loads raise JSONDecodeError
    on every later read. Trigger requires >=2 slots for the concurrent variant, but that is the
    worker's designed operating mode.
  - **Key line(s):** `"INSERT INTO build_jobs (project_id, user_id, workspace, status, "
                "created_at, log_path, kind) VALUES (?, ?, ?, 'queued', ?, ?, ?)",`

- [x] **B3** `kicraft/server/accounts.py:1862` — finish_build's UPDATE has no status guard (WHERE id=? only, unlike claim_build/requeue_build), so concurrent writers clobber terminal/requeued job states; build_worker.py:195-196 even documents a 'guarded UPDATE in finish_build' that does not exist and papers over it with a racy check-then-act.
  - **Severity:** high · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    SIGTERM during a live build: _shutdown (build_worker.py:114-124) kills the proc and requeues the
    job, while the _run_job thread concurrently wakes from proc.wait() with rc=-9, reads status
    'running' at line 197 just before the requeue lands, then finish_build marks the
    freshly-'queued' row 'done' rc=-9 -- the retry the shutdown intended is silently lost and the
    project finalizes on a killed build. Symmetrically, the web janitor's requeue_stale_builds
    (accounts.py:1918-1927) reads its 'running' snapshot, the worker finishes the job 'done' rc=0
    and exits, then the janitor sees the now-dead pid and (attempts>=max) calls
    finish_build(rc=None, status='failed'), overwriting a successful build as failed. Fix at the
    source: add AND status='running' to finish_build's WHERE.
  - **Verifier:**
    finish_build (accounts.py:1860-1863) has no status guard, unlike claim_build (WHERE ... AND
    status='queued', line 1839) and requeue_build (AND status='running', line 1870).
    build_worker.py:195-196 explicitly cites 'the guarded UPDATE in finish_build' and papers over
    its absence with a check-then-act: `cur = self.store.get_build_job(job.id); if cur is not None
    and cur.status == 'running': self.store.finish_build(...)` (197-199) — a classic TOCTOU. Trigger
    1: SIGTERM -> _shutdown kills the proc (114-124); _run_job's proc.wait() returns rc=-9, reads
    status 'running' at 197 before _shutdown's requeue_build commits, then finish_build stamps the
    freshly-queued row 'done' rc=-9, losing the intended retry. Trigger 2: the web janitor
    (web.py:258 calls requeue_stale_builds) reads a 'running' snapshot (accounts.py:1919-1920), the
    worker finishes 'done' rc=0 and exits, the janitor's dead-pid check then calls
    finish_build(rc=None, status='failed') at 1925, overwriting a successful build. Both are narrow
    timing windows but concretely constructible; the missing WHERE-status guard is factually
    present.
  - **Key line(s):** `"UPDATE build_jobs SET status=?, rc=?, finished_at=? WHERE id=?",`

- [x] **B4** `kicraft/server/accounts.py:1480` — delete_project unconditionally rmtree's projects_dir/<uid>/<pid> and deletes the job's 'running' build_jobs row, but under Phase-4a build-in-place that path IS the running build's workspace (job.workspace), so deleting a project yanks a live build's cwd despite the inline comment (lines 1466-1468) promising to never do that.
  - **Severity:** high · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    Web restarts mid-build (builds intentionally survive restarts via the worker); _LIVE_RUNS in
    web.py:2687 is empty so the delete guard passes; user clicks Delete -> delete_project removes
    the 'running' build_jobs row (line 1474) and rmtrees the tree (line 1480) while the worker's
    subprocess keeps running in it for up to 30 min. The build re-creates directories/files inside
    the 'permanently deleted' tree (breaking the deletion promise and leaving an orphan tree no
    reaper owns), burns a worker slot, and with the row gone requeue_stale_builds/get_build_job can
    never track or finalize it. Also, because terminal jobs' workspace == the project dir, the
    stale_ws rmtree at 1475-1477 deletes the tree first, making line 1479's tree.exists() False so
    the function returns None instead of the purged path.
  - **Verifier:**
    The only liveness guard is web.py:2687 `if _LIVE_RUNS.get(pid) is not None`, an in-process dict
    that is empty after a web restart while the separate worker process keeps its subprocess running
    (the queue's stated purpose is that builds survive web restarts; the reaper's
    list_unfinalized_builds explicitly excludes status='running', so nothing repopulates any guard).
    delete_project then executes `DELETE FROM build_jobs WHERE project_id=?` (line 1474) with no
    status filter, deleting the 'running' row, and unconditionally rmtrees projects_dir/<uid>/<pid>
    (1478-1480) — which under Phase-4a build-in-place is exactly job.workspace (web.py _project_dir
    line 1373: `_store().projects_dir / str(uid) / str(pid)`, passed as workspace at enqueue_build
    web.py:1676-1678). This contradicts the inline comment at 1466-1468 ('a still-running worker ...
    owns its workspace, so leave that to the 2-day workspace GC rather than yanking it mid-build') —
    the comment's carve-out only skips the stale_ws loop, not the tree rmtree, which is the same
    path. The secondary defect is also real: for terminal jobs workspace == tree, so the stale_ws
    rmtree at 1475-1477 deletes the tree first and line 1479 `tree.exists()` is False, returning
    None instead of the purged path.
  - **Key line(s):** `tree = self.projects_dir / str(uid) / str(project_id)
        if tree.exists():
            shutil.rmtree(tree, ignore_errors=True)`

- [x] **B5** `kicraft/server/web.py:5107` — Rebuild/resume handlers guard only the page-local state dict (state["running"]) and never check _LIVE_RUNS[pid], so two pages that opened the same project while idle can start two concurrent builds in the same build-in-place project directory.
  - **Severity:** high · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    User opens finished project P in tab A and tab B (both get independent state dicts via
    open_project's non-live branch). Tab A clicks 'Rebuild board' -> state_A.running=True,
    _LIVE_RUNS[P]=state_A, build job 1 enqueued. Tab B's dict still has running=False, so its
    'Rebuild board' passes the guard at _start_replace_build (same hole in _do_rerun:4688,
    _continue:4728, _answer_and_resume:4585, _start_manual_route:5043) and enqueues job 2 for the
    same workspace. build_worker runs up to slot_count()=cpus//6 jobs concurrently and nothing
    dedupes by project, so two `cli_app build` processes run with cwd = the same project dir, both
    writing generated/<stem>/, state.json and the SAME .kicraft/build.log -> corrupted/interleaved
    board artifacts and logs; the two threads then both run _persist_project and race the project
    row.
  - **Verifier:**
    Every rebuild/resume guard checks only the page-local dict: _start_replace_build (5107),
    _do_rerun (4688), _continue (4728), _answer_and_resume (4585), _start_manual_route (5043).
    open_project's non-live branch creates an independent `state = _fresh_run_state()` per page
    (4802), so two tabs on the same finished project each hold running=False. Neither
    _start_replace_build nor _rerun_build_worker checks _LIVE_RUNS before starting; line 1736
    blindly overwrites the entry. accounts.enqueue_build is a plain INSERT (no workspace/project
    dedupe) and claim_next_build takes the FIFO head with no same-workspace exclusion, with worker
    max_jobs = max(1, slot_count()). So tab B's rebuild passes the guard and enqueues a second job
    for the same build-in-place dir; both threads later run _persist_project on the same project
    row. True concurrent same-dir builds additionally require slot_count() >= 2 (cpus//6 or
    KICRAFT_BUILD_SLOTS), but the guard hole, double-enqueue, _LIVE_RUNS clobber, and persist race
    exist regardless.
  - **Key line(s):** `if state["running"]:
                ui.notify("A run is already in progress.", color="warning")
                return   (web.py:5107)  ...  _LIVE_RUNS[pid] = state   (web.py:1736, unconditional)`

- [x] **B6** `kicraft/server/render_serving.py:38` — Capability-token file serving falls open to the public, well-known default secret 'kicraft-dev-secret' when KICRAFT_STORAGE_SECRET is unset, and the sole guard test is a tautology, so a misconfigured prod box lets anyone forge tokens for any project dir.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    _project_secret() (render_serving.py:38, mirrored in web.py:5606) returns b'kicraft-dev-secret'
    whenever the env var is missing. The @app.get serve routes (serve_project_file,
    serve_project_render, serve_project_board) require NO login and gate purely on this HMAC. If an
    operator deploys without exporting KICRAFT_STORAGE_SECRET, an unauthenticated attacker computes
    payload=b64(abs project dir) and sig=HMAC(default_secret,payload) themselves and GETs
    /project/<forged>/<stem>.kicad_pcb (or /render/<subpath>.png, /board/<subpath>.kicad_pcb) for
    any other tenant's project under ~/.kicraft/projects/<uid>/<pid>/generated/<stem> -> full cross-
    tenant read of private schematics/boards. The intended guard
    tests/security/test_capability_token.py:83 asserts
    os.environ.get('KICRAFT_STORAGE_SECRET','kicraft-dev-secret') is not None, which is always True,
    so it can never fail and provides false assurance. The code should refuse to start on the
    default secret rather than fall open.

- [ ] **B7** `kicraft/design/synthesis/validation.py:958` — _net_is_positive_rail classifies negative rails (VSS, VEE, -12V, VCC-) as POSITIVE because it reuses POWER_NET_PATTERNS, making the hard §9.16 polarity gate reject every correctly-wired dual-supply design as 'power pins look reversed'.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Any dual-supply design (op-amp/audio, the class POWER_NET_PATTERNS' leading-sign support was
    added for): wiring wires the op-amp's V-/VEE pin to net '-12V', 'VEE', 'VSS' or 'VCC-'.
    _net_is_positive_rail reuses POWER_NET_PATTERNS wholesale, which matches all of those (verified:
    _net_is_positive_rail('-12V')/('VSS')/('VEE')/('VCC-') all return True while _net_is_ground
    returns False), so check_power_pin_polarity (line 999) flags the CORRECT wiring as
    'ground/negative pin is wired to positive rail ... power pins look reversed' (reproduced end-to-
    end with a mocked LM358 V- pin on net -12V). §9.16 is a hard gate at wiring/stage commit
    (cli_app.py:2957-2993, exit 3) and runs inside run_validations at synthesis (build rc 5), so the
    model is bounced forever — it cannot 'fix' wiring that is already correct. The router's own
    _is_ground special-cases VSS/VEE as ground; this predicate never did.

- [x] **B8** `kicraft/design/cli_app.py:3170` — Every state.json persist (stage commit, artifact persist, review persist) is a non-atomic truncate-then-write with no tmp+rename, so a crash mid-write permanently corrupts the file three processes depend on, and concurrent cross-process readers can see torn JSON.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    state.json is the sole IPC channel between the web app, the build worker, and this CLI (per
    CLAUDE.md), yet _cmd_stage_commit — whose docstring says 'Atomic stage commit' — persists it
    with a bare truncating write_text (open 'w' truncates, then writes ~100KB of JSON): a crash,
    OOM-kill, or disk-full between truncate and completion leaves a permanently corrupted/partial
    state.json, after which every _load_state (validate/stage-prep/stage-commit/build, and the web
    app reopening the project) fails with JSONDecodeError rc 2 and the whole design session is lost.
    The same non-atomic pattern runs MID-BUILD from the worker process (_persist_artifacts line
    3228, run_post_wiring_review line 3855, _surface_build_warnings line 3892,
    _surface_review_findings line 3939) while the web process reads the same file to paint the GUI,
    so a concurrently-timed read sees truncated JSON. Fix is the standard write-to-tmp + os.replace
    in one helper.

- [x] **B9** `kicraft/autoplacer/brain/placement_solver.py:968` — Four anchor_offset_mm consumption sites (_random_in_corner lines 819-820, _connector_edge_x line 968, _connector_edge_y line 993, zone branch lines 1254-1255) rotate the block anchor with the hand-rolled math-CCW formula (x*cos - y*sin, x*sin + y*cos) instead of geometry.rotate_vector's KiCad-CW convention that the block geometry is actually placed with (parent_adapter._rotated, subcircuit_instances._transform_point, _update_pad_positions all use +rot CW).
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Parent compose with an edge/corner-constrained leaf whose zone rotation is 90 or 270 (e.g. a
    connector leaf rotated to face a left/right edge, rotation set by
    attachment_constraints_to_zones): the anchor offset is rotated with the negated angle, so the
    block is positioned such that its true world anchor lands 2*|offset| mm away from the intended
    flush coordinate (verified numerically: offset (3,10) at rot 90 puts the anchor 20 mm off the
    edge target). The connector leaf is stranded inboard or overhangs the board;
    constraint_aware_outline then stretches/squeezes the parent outline to the phantom anchor
    position, producing overlapped children or bloated boards. Agrees only at 0/180 - exactly the
    '90/270 stranding' class the geometry.py docstring says was fixed at the origin-recovery sites
    but not at these four consumers.

- [x] **B10** `kicraft/autoplacer/brain/placement_solver.py:2351` — The SA rotation move sets comp.rotation and calls _update_pad_positions but never swaps width_mm/height_mm (nor consults block_rotation_geometry), so for pad-less subcircuit blocks an accepted 90/270 move changes NOTHING the scorer or legality passes can see, yet placements_from_solved_state (parent_adapter.py:325-333) stamps the real leaf at the new rotation with transposed extents.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    An unconstrained (unlocked) non-square leaf block, e.g. 30x10 mm, gets a 90-degree SA rotation
    move accepted (Metropolis accepts on score noise since the score is provably unchanged for a
    block); all subsequent overlap resolution, courtyard separation (Step 16), and board clamps
    legalize a 30x10 box, then the composer transforms the artifact at rotation 90 so the stamped
    child occupies 10x30 - overlapping a neighbouring leaf or hanging past the board outline,
    surfacing as courtyards_overlap / copper-to-edge DRC the solver said was clean. Non-square leaf
    ICs hit the same dims-desync via the same move.

- [x] **B11** `kicraft/autoplacer/brain/placement_solver.py:1796` — _optimize_rotations (lines 1789-1796), the early IC rotation in _place_clusters (1608-1616), and random-scatter rotation (1461-1464) rotate pads to the new orientation but never swap width_mm/height_mm for 90/270 deltas and (in the first two) never rotate body_center - unlike geometry.rotate_component_in_place and _orient_for_edge (1027-1028) which both maintain the AABB.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A non-square IC (e.g. SOT-223 regulator, 7x3.5 mm courtyard) is rotated 90 degrees by
    _optimize_rotations: Component.bbox() (types.py:199-220, pure width/height about body_center)
    still reports 7 wide x 3.5 tall while the real part is 3.5x7. Every later pass -
    _resolve_overlaps, _resolve_courtyard_overlaps, _clamp_to_board, the courtyard scorer -
    separates neighbours from the wrong-shaped box, so the stamped footprint's actual courtyard
    overlaps a neighbour placed 'legally' beside its phantom short side (the recurring
    courtyards_overlap fab-blocker family), or parts are needlessly pushed apart on the transposed
    long axis.

- [x] **B12** `kicraft/autoplacer/freerouting_runner.py:1590` — validate_routed_board's footprint-internal clearance waiver waives ALL clearance/hole_clearance violations — including track-to-track ones that name no footprint — whenever the only refs named come from a single footprint, and the elif fallback (line 1596) compares a double-counted mention tally (a pad-internal violation names its footprint twice) against the violation count, so a broken board passes the acceptance gate.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Routed leaf/parent with 1 USB-C J1 pad-gap clearance violation (report lines 'Pad A5 [CC1] of
    J1' + 'Pad A6 of J1' -> refs={J1}) plus 3 genuine track-to-track clearance violations (item
    lines are 'Track [VSEL0] on F.Cu' with no 'of REF', confirmed in tests/fixtures replay
    debug.json): _extract_clearance_footprint_refs returns {J1: 2}, len(refs)<=1 is true, all 4
    violations are marked footprint_internal_clearance_count and obviously_illegal_routed_geometry
    is never set -> validation.accepted=True. The parent gate only adds unconnected>0
    (kicraft/cli/_compose_route.py:359) and the fab gate is 0 shorts/0 unconnected, so real routed-
    copper clearance faults ship in the fab zip. The dominant_count>=clearance_count branch has the
    same hole: k J1-internal violations give J1 a count of 2k, letting up to k ref-less track
    violations ride along.

- [ ] **B13** `kicraft/cli/autoexperiment.py:479` — _all_leaf_artifacts scans every dir under .experiments/subcircuits/ with no filter against the CURRENT hierarchy, and the run-start purge it depended on (gui/experiment_runner._purge_prior_run_artifacts) was deleted with the GUI (commit 3aecbdf) with no production replacement, so stale leaf artifacts from previous builds contaminate every rebuild.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    User rebuilds an existing project (or rebuilds a clone -- web.py:3183 copies .experiments/ into
    clones). _cmd_build always re-runs run_synth, which re-emits sheets with random uuid4 instance
    UUIDs (emitter.py:1004) while deliberately leaving .experiments/ untouched (synthesize.py:175),
    so ALL of build #1's leaf dirs become orphans that still carry metadata.json+solved_layout.json.
    In build #2, _all_leaf_artifacts/_accepted_leaf_artifacts return old+new dirs:
    leaf_total/leaf_accepted are double-counted in every round record and the GUI; if any stale
    artifact has accepted=false, leaf_ratio<1.0 forever, so _score_round tiers EVERY round as
    partial_leaves (score<=15) even when all real leaves accept and the parent routes -- round
    keeping and best-round selection then run on a garbage scale; and _leaf_feasible_min_board (line
    2349) sums stale leaf areas, inflating the board-size search floor ~sqrt(2) per rebuild.

- [ ] **B14** `kicraft/cli/solve_subcircuits.py:780` — The leaf solve merges the pre-existing debug.json all_rounds into this run's rounds, relying (per its own comments at lines 764 and 1371) on a run-start purge that no longer exists anywhere in the codebase, so rounds and pinnable board snapshots leak across separate runs of the same workspace.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    kicraft replay / the verify skill / the tuner re-run place+route on a fixed workspace without
    re-synthesis, so slugs match and the prior run's rounds (experiment_round>0 passes the line 786
    filter) merge into all_rounds, with round_NNNN_leaf_routed.kicad_pcb snapshots from the prior
    run still on disk. autoexperiment._auto_pin_best_leaves then picks the best round ACROSS runs by
    (accepted, routed, score): after a place/route code change, the OLD code's higher-scoring round
    can win the pin, so the replay promotes the previous run's board and the verify verdict silently
    measures pre-change output (regressions masked, improvements invisible). It also silently masks
    a total this-run leaf failure: if a leaf's solve fails every round this run, the prior run's
    metadata.json still exists so the _board_only_leaf_dirs RuntimeError net (autoexperiment.py:420)
    never fires and the previous run's board ships.

- [x] **B15** `kicraft/layout_editor/static/layout_canvas.js:826` — bindLeafEvents is called from render() on every repaint (including every mousemove during a drag) and each call registers a new document-level 'keydown' listener (line 826) and a new svg-level 'contextmenu' listener (line 807) that are never removed, so duplicate rotate handlers accumulate without bound within one canvas init.
  - **Severity:** high · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    User drags a leaf for a few seconds (~200 mousemove renders), then presses 'r' or right-clicks a
    leaf: ~200 stacked handlers each apply a further -90 deg via setRotationKeepCenter and each call
    render(), so one keypress rotates the leaf by (N mod 4)*90 deg (an unpredictable orientation),
    freezes the tab with N re-renders, and roughly doubles the listener count again; the isCurrent()
    sentinel only guards against newer IIFE inits, not same-init re-registration, so memory/CPU grow
    until the page is reloaded.

### 1.2 Medium severity (25)

- [x] **B16** `kicraft/server/build_worker.py:154` — The worker appends to .kicraft/build.log (open('a')) while the web reader tails it from byte offset 0, so every rebuild replays the ENTIRE previous build's log into the new run's event stream.
  - **Severity:** medium · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    First build of a project runs through the worker, filling ws/.kicraft/build.log. User later
    clicks Rebuild: web.py _drive_build_queue (web.py:1682) initializes offset=0 and calls
    _drain_build_log on the same log path while the job is still queued -> every line of the OLD
    build is emitted as a fresh build_log event before the new build writes anything, and the worker
    then appends the new output after it. Deterministic on every worker-path rebuild. Consequences:
    the timeline shows the previous build's '[build] wrote ...'/failure lines as if produced by the
    current run, the review-findings fallback parser (web.py:1267/1325) and _build_lines_for
    classify stale lines as current, _persist_project bakes the duplicates into events.jsonl, and
    the file grows by one full replay per rebuild. (The in-process fallback
    _execute_claimed_job_local never writes the log at all, so build.log additionally goes stale
    after any fallback build.)
  - **Verifier:**
    The worker appends to .kicraft/build.log and nothing in the codebase ever truncates or deletes
    it (grep shows no open('w')/unlink of build.log anywhere), while _drive_build_queue always tails
    from byte 0 and drains on the first loop pass while the job is still queued — so a rebuild
    through the worker deterministically replays the whole previous build's log as fresh build_log
    events before the new build writes anything, and the file grows by one full copy per rebuild.
    The parenthetical is also accurate: _execute_claimed_job_local (web.py:1642-1648) streams
    proc.stdout straight to progress and never writes the log file, leaving build.log stale after
    any in-process fallback build.
  - **Key line(s):** `with log_path.open("a", encoding="utf-8") as logf:  (build_worker.py:154); offset, tail_buf = 0, ""  (web.py:1682) followed by _drain_build_log(log_path, offset, ...) before any status check acts`

- [x] **B17** `kicraft/server/accounts.py:1164` — delete_user purges likes, FTS, support_reports, projects, and the user row, but not the user's build_jobs rows (which delete_project's own comment at lines 1463-1464 notes 'are never otherwise deleted, so the rows... would leak forever'), nor their password_resets/email_verifications token rows.
  - **Severity:** medium · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    Delete a user who has a queued build: the build_jobs row survives forever, the worker later
    claims it, finds state.json gone (the tree was rmtree'd at line 1168), and burns a claim marking
    it 'failed' -- and the deleted user's user_id, workspace paths, and reset/verification token
    hashes remain queryable in the DB even though the docstring says this 'Honors the deletion right
    the Privacy Policy promises'. A user deleted mid-build additionally hits the same live-workspace
    rmtree race as delete_project, with the orphaned 'running' row then permanently unowned.
  - **Verifier:**
    delete_user (accounts.py:1147-1170) deletes project_likes, projects_fts, support_reports,
    projects, and the users row, then rmtrees the tree — but issues no DELETE for build_jobs (which
    carry user_id and workspace paths), password_resets, or email_verifications (both tables exist,
    created at accounts.py:408/424 and keyed by user_id). delete_project's own comment (1463-1464)
    confirms 'build_jobs rows are never otherwise deleted, so the rows ... would leak forever'. A
    queued job surviving user deletion is later claimed by the worker, which finds state.json gone
    after the rmtree at 1168 and burns the claim: build_worker.py:133-136 `if not (ws / ".kicraft" /
    "state.json").is_file(): ... finish_build(job.id, rc=None, status="failed")`. The docstring's
    Privacy-Policy deletion claim (1150) is contradicted by the surviving user_id-keyed rows and
    token hashes.
  - **Key line(s):** `conn.execute("DELETE FROM projects WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))`

- [x] **B18** `kicraft/server/web.py:1407` — _persist_project unconditionally rewrites events.jsonl with open('w') from state['events'], so _finalize_orphan (which passes events=[]) permanently truncates a project's persisted design transcript.
  - **Severity:** medium · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    A completed project (events.jsonl holds the full LLM-stage transcript written at its original
    finalize) is reopened and rebuilt; the web process restarts while the worker is still running
    the job. After restart the job finishes 'done', the orphan reaper calls _finalize_orphan
    (web.py:196), which builds st = {..., 'events': []} (web.py:207) and calls _persist_project ->
    the 'w'-mode write at web.py:1407 replaces the existing events.jsonl with an empty file. Every
    later reopen shows a blank timeline/thinking stream (_load_events web.py:1559 reads nothing),
    and eval/metrics_web.py consumers of events.jsonl lose the run history. The transcript loss is
    permanent even though the artifacts survived.
  - **Verifier:**
    _persist_project unconditionally rewrites events.jsonl in 'w' mode from state['events'];
    _finalize_orphan passes events=[] so the file is truncated to zero bytes. The trigger chain
    holds: a rebuild sets the durable row to 'running' (web.py:1735) so _finalize_orphan's `p.status
    != "running"` guard (203) passes after a web restart; a done build_jobs row not in the (freshly
    empty) _ACTIVE_JOBS reaches _finalize_orphan via the reaper (263-264); build-in-place means the
    truncated events.jsonl is the very file holding the original run's full transcript, which
    _load_events (state["events"] = _load_events(p.dir_path), web.py:4816) then reads back empty on
    every later reopen. Loss is permanent.
  - **Key line(s):** `st: dict = {..., "brief": p.brief or "", "events": [], ...}   (web.py:207)  ...  with (base / "events.jsonl").open("w", encoding="utf-8") as f:
            for ev in state.get("events", []):   (web.py:1407-1408)`

- [ ] **B19** `kicraft/server/web.py:5409` — No rebuild/rerun path clears the fab tab's view slot or resets view['fab_invalid']/view['build_lines'], so stale 'do not fabricate' banners, dead download buttons, and duplicated build logs accumulate across reruns.
  - **Severity:** medium · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    Run 1 fails verification with a candidate board -> _mark_fab_invalid (line 5131) paints the red
    'Fab package invalid -- do not fabricate' banner into tabs.view_slot('fab') and sets
    view['fab_invalid']=True. User clicks 'Rebuild board' (_start_replace_build resets no view
    flags, and tabs.reset_stage in _do_rerun only covers DESIGN_STAGES, never 'fab'). Rebuild
    succeeds -> the finalize block at line 5392-5433 appends the fresh 3D render + Download button
    BELOW the still-present red banner (contradictory UI on a valid package). Reverse order: after
    ok->failed rebuild, the old Download button persists while _rerun_build_worker set
    state['zip']=None, so clicking it runs ui.download(None) and errors. Each edit-rerun with
    view['fab_done'] reset also stacks another duplicate image+button, and view['build_lines'] is
    never cleared so _build_lines_for mixes the old and new builds' log lines in the synthesize/fab
    inspectors.
  - **Verifier:**
    The only view_slot .clear() anywhere is synthesize's (web.py:5291); the fab slot is append-only
    from both _mark_fab_invalid (5134) and the success finalize (5409). _start_replace_build resets
    only state flags (5113: state.update(running=True, done=False, ok=None, status=None)) -- no view
    reset, no tabs.reset_stage; _do_rerun resets fab_done but not fab_invalid or build_lines
    (4717-4719) and reset_stage runs only over the design stages in `runs` (4708-4709), never 'fab'.
    So failed->rebuild-ok appends the fresh render+Download below the latched red banner;
    ok->rebuild-failed leaves the old Download button whose lambda calls ui.download(state["zip"])
    after _rerun_build_worker set state["zip"]=None (1762); and view["build_lines"] is initialized
    only in _reset_view (4147, called from open_project paths), so rebuild log lines append to the
    previous build's, mixing logs in _build_lines_for-driven inspectors and stacking duplicate fab
    content on each edit-rerun.
  - **Key line(s):** `with tabs.view_slot("fab"):   (web.py:5409, inside 'if not view["fab_done"]:' at 5392)  ...  _mark_fab_invalid: 'if view.get("fab_invalid"): return' then 'with tabs.view_slot("fab"):' paints the red card (5129-5136)`

- [x] **B20** `kicraft/server/web.py:1651` — The in-process build fallback's 30-minute 'hard wall-clock bound' is only checked after a stdout line arrives, so a build that hangs silently is never killed and the run/job is stuck 'running' forever.
  - **Severity:** medium · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    No standalone worker heartbeating (fallback deploy) -> _drive_build_queue self-claims the job
    and _execute_claimed_job_local iterates `for line in proc.stdout`. If the build hangs producing
    no output (e.g. a freerouting hang, a documented failure mode), the readline blocks forever: the
    deadline at line 1651 is never evaluated, proc is never killed, the thread never reaches
    finish_build or _persist_project, and _ACTIVE_JOBS keeps the job id. requeue_stale_builds skips
    it (claimant pid alive) and the orphan reaper skips it (in _ACTIVE_JOBS / _LIVE_RUNS), so the
    project row stays 'running' forever, the quota slot stays burned, and the project cannot be
    deleted (live-run guard) until the web process restarts.
  - **Verifier:**
    The deadline is evaluated only inside the loop body, i.e. only after readline returns a line; a
    build that hangs producing no output (freerouting hangs are a documented KiCraft failure mode)
    blocks the thread in readline forever and the 30m bound is never checked. The claimed dead-ends
    all hold: claim_build stamped claimed_by=f"pid:{os.getpid()}" (web.py:1698), and
    requeue_stale_builds does `if _pid_alive(_claimant_pid(r["claimed_by"])): continue`
    (accounts.py) -- the web pid is alive, so no requeue; the orphan reaper does `if job.id in
    _ACTIVE_JOBS ...: continue` (web.py:261) and _ACTIVE_JOBS.discard sits in _drive_build_queue's
    finally (1712), unreachable while _execute_claimed_job_local is blocked inside the try; project
    delete is guarded by `if _LIVE_RUNS.get(pid) is not None` (2687). The row stays 'running' until
    process restart. (Contrast: the standalone worker has a real timeout_s watchdog; only this in-
    process fallback lacks one.)
  - **Key line(s):** `for line in proc.stdout or []:
        ...
        if time.monotonic() > deadline:  # hard wall-clock bound
            proc.kill()   (web.py:1646-1652)`

- [ ] **B21** `kicraft/server/build_worker.py:115` — _shutdown snapshots self._procs, so a job claimed in run_once but not yet registered (the claim->Popen window, build_worker.py lines 131-160) is neither killed nor requeued; after the 10s thread-join timeout the worker exits, orphaning the start_new_session build subprocess while the row's dead claimant pid gets it requeued and run a second time concurrently in the same workspace.
  - **Severity:** medium · **Status:** VERIFIED-PLAUSIBLE
  - **Failure scenario:**
    SIGTERM lands while a _run_job thread is between claim_next_build and self._procs[job.id]=proc
    (state.json check + build_quality_for_user DB read): _shutdown's dict(self._procs) misses the
    job, join(timeout=10) expires because the build just started, the process exits killing the
    daemon reader thread but not the detached subprocess. The web janitor sees status='running' with
    a dead pid and requeues within 30s; the restarted worker claims it and launches a second
    `kicraft build` in the same generated/ tree while the orphan is still writing to it, corrupting
    artifacts.
  - **Verifier:**
    The mechanism is real: a job claimed in run_once (line 102) but not yet registered at line 160
    is missed by _shutdown's snapshot; threads are daemon=True, join times out after 10s, and
    start_new_session=True orphans the build subprocess on worker exit, while the dead claimant pid
    gets the row requeued by the web janitor (web.py:258) and re-run in the same workspace. But the
    claim-to-register window is only an is_file check, one DB read, mkdir, and Popen (milliseconds),
    so SIGTERM landing inside it is a narrow timing race, and the host-wide build flock
    (slot_count()==1) can serialize orphan vs re-run. Confirming would need a fault-injection test
    that delays _run_job between claim and the _procs registration while delivering SIGTERM, then
    observing the orphaned subprocess plus a concurrent second claim.
  - **Key line(s):** `with self._lock:
            live = dict(self._procs)  (build_worker.py:114-115); self._procs[job.id] = proc (line 160); t.join(timeout=10) (line 126); start_new_session=True (line 158)`

- [ ] **B22** `kicraft/server/render_serving.py:83` — Stateless capability tokens never expire and the serve handlers never re-check is_public or ownership, so a token captured while a project was public keeps serving that project's KiCad/PNG files forever after it is made private.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    _register_project_dir (web.py, e.g. line 3386/3116/4826) mints a token = b64(project_dir).HMAC
    with no timestamp/expiry. /p/{id} and the browse thumbnails embed that token in <img>/board URLs
    (e.g. /project/<token>/<stem>.kicad_pcb) visible to every logged-in community user, and the
    @app.get serve handlers (render_serving.py:83-92, 105-116, 129-140) require no session and only
    verify the HMAC + path-containment -- they never call _public_project_or_none or compare user
    ids. Scenario: user A (pro tier) publishes project 123; user B (or a crawler/browser-
    history/referrer capture) records the board URL; A later flips project 123 private via the
    /projects visibility toggle; B replays the identical URL and still downloads A's now-private
    routed board and schematic indefinitely. Making a project private does not change the durable
    dir path, so the old token stays valid.

- [x] **B23** `kicraft/design/synthesis/validation.py:2099` — check_fs_connections_mapped requires an inter_sheet_net whose endpoint-sheet set EXACTLY equals the {from,to} pair, so a correctly-declared 3+-endpoint net (a shared bus) fails the hard architecture-commit gate with a misleading 'no inter_sheet_net' rejection.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Architecture declares one inter-sheet net with 3+ endpoints (e.g. I2C_SDA across MCU, SENSOR and
    DISPLAY — the natural encoding for a shared bus; InterSheetNet only requires >=2 endpoints).
    check_fs_connections_mapped indexes nets by the frozenset of ALL endpoint sheets and tests
    'any(pair in isn_by_sheets)' with the exact 2-sheet pair, so the 3-endpoint net covers none of
    its pairwise functional-spec connections. Reproduced: FS connection MCU→SENSOR with I2C_SDA
    declared over {MCU,SENSOR,DISPLAY} → ok=False, offender "crosses sheets but has no
    inter_sheet_net". The architecture commit hard-fails (cli_app.py:2916-2936, exit 3) with
    feedback the model cannot act on (the net IS declared), forcing retry churn or an artificial
    split into redundant pairwise nets. The membership test should be a subset check (pair <=
    endpoint_sheets).

- [x] **B24** `kicraft/autoplacer/brain/placement_solver.py:3651` — _assign_layers' back-side flip mirrors each pad's X about comp.pos to match pcbnew Flip(), but does not mirror body_center.x (2*pos.x - body_center.x), so a flipped component with an origin-offset courtyard keeps its bbox on the pre-flip side.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A large THT part with courtyard center offset from the footprint origin (battery holder -
    precisely the >=50 mm^2 parts this pass auto-flips) is assigned to B.Cu: pads mirror but
    Component.bbox()/_effective_bbox stay centered on the unmirrored body_center, e.g. 10 mm to the
    RIGHT of pos while the stamped, flipped footprint's body extends 10 mm to the LEFT. Overlap
    resolution and board clamps protect empty space on the wrong side, and the real flipped
    courtyard overlaps neighbours or the board edge undetected.

- [x] **B25** `kicraft/autoplacer/brain/placement_solver.py:703` — In _score_rotation_for_routing, `accessible += dist` is indented at the level of the `for dx, dy in dirs` loop (outside it), so only the last direction (-1,-1) contributes; the other three directions' openness is computed and discarded.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Every IC/connector rotation choice in _optimize_rotations and the early-rotation pass scores pad
    accessibility only toward the top-left diagonal: a rotation whose pads open generously to the
    right/bottom scores 0 accessibility while one with modest top-left openness wins, systematically
    biasing 30% of the rotation score and picking routing-hostile orientations (e.g. pads facing a
    nearby board corner at top-left instead of open space at bottom-right).

- [x] **B26** `kicraft/autoplacer/brain/placement_solver.py:1184` — Edge-group packing advances the cursor by the CURRENT part's full extent plus gap (`cursor_y += comp.height_mm + connector_gap`, same on X at line 1220) instead of half-current + gap + half-next, while the order is sorted by AREA descending - so a taller-but-smaller-area follower overlaps its predecessor.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Two connectors pinned to the left edge: J1 barrel jack (9x11 mm, area 99) then J2 screw terminal
    (4x16 mm, area 64) with default connector_gap 2.0: J2's center is placed 13 mm below J1's,
    giving edge separation 11/2 + 2 - 16/2 = -0.5 mm - the two locked, pinned connectors overlap by
    0.5 mm; _resolve_overlaps' both-pinned branch escapes one but _restore_pinned_positions snaps it
    back, and Step 16 counts it as unresolved both-locked, shipping a courtyards_overlap to DRC.

- [x] **B27** `kicraft/autoplacer/brain/geometry.py:86` — rotate_component_in_place rotates pad.pos and swaps comp width/height but never rotates/swaps pad.size_mm, whose contract (types.py:105-116) is a WORLD-axis-aligned AABB that must be re-rotated on any placement rotation (subcircuit_instances._rotate_size does this on the artifact path).
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    array_placement rotates matrix members 90 degrees via rotate_component_in_place
    (array_placement.py:478-481): each LED's rectangular pad keeps its pre-rotation world AABB (e.g.
    1.5 wide x 0.8 tall instead of 0.8x1.5), so Pad.bbox() and hence Component.physical_bbox()
    report wrong pad copper extents; leaf outline sizing and compaction that consume physical_bbox
    (subcircuit_solver.py:197-209, leaf_compaction) under-reserve on the true long axis, yielding
    copper-to-edge clearance violations on rotated array rows.

- [x] **B28** `kicraft/autoplacer/freerouting_runner.py:1020` — run_freerouting's timeout path sends only SIGTERM to the process group and then calls proc.communicate(timeout=5) outside any try — a JVM/xvfb-run group that ignores SIGTERM or takes >5s to exit raises an uncaught second TimeoutExpired and the Java process is leaked (no SIGKILL escalation anywhere).
  - **Severity:** medium · **Status:** FIXED INDEPENDENTLY — `4fabc70` on branch `self-eval-2026-07-07-fixes`
    (self-eval FIX 1: SIGTERM→SIGKILL escalation + kill_tree; two stranded July-4 JVMs confirmed
    the failure live). Not part of the batch-5 commit; lands on main when that branch merges.
  - **Failure scenario:**
    FreeRouting 1.9.0 wedged in its known hang mode (e.g. locked wire corner outside the board,
    documented in breakout_stubs.py:386) with the leaf timeout scaled up to 1200 s:
    communicate(timeout_s) raises, killpg(SIGTERM) fires, but the JVM's shutdown (or xvfb-run's TERM
    trap cleanup) doesn't complete within 5 s -> communicate(timeout=5) raises TimeoutExpired which
    propagates uncaught through route_with_freerouting/route_local_subcircuit, so the leaf is
    reported as a python exception instead of a routing timeout AND the still-running Java process
    group survives for the rest of the build, burning a CPU core per hung leaf;
    parse_freerouting_output/retry logic is skipped entirely.

- [ ] **B29** `kicraft/autoplacer/freerouting_runner.py:1136` — _set_board_clearance_um launches its pcbnew subprocess without env=_kicad_subprocess_env() (and without the _retry_pcbnew_run sentinel/retry machinery every other pcbnew call in this file uses), so in exactly the virtualenv deployments that helper exists for, the fine-pitch clearance reconciliation always fails silently.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Build worker running in a venv where pcbnew is only importable via the PYTHONPATH additions
    _kicad_subprocess_env provides (the documented reason for that helper, lines 113-118): a fine-
    pitch board (USB-C leaf) is routed at the lowered 0.153 mm clearance, then
    `subprocess.run([sys.executable, "-c", script], check=True)` fails on `import pcbnew`, prints
    only a warning, and the routed board keeps declaring its original 0.2/0.3 mm netclass clearance
    -> post-route DRC flags every fine-pitch trace as clearance violations across multiple
    footprints -> obviously_illegal_routed_geometry -> every round of the leaf is rejected and the
    build ends 'board not routable as placed', defeating the exact failure this function was written
    to prevent (see its own docstring).

- [x] **B30** `kicraft/autoplacer/brain/leaf_routing.py:959` — route_timing['gnd_pour_s'] is computed inside the power_plane_enabled block but references gnd_pour_start, which is only assigned inside the separate gnd_plane_enabled block, causing a NameError that kills the leaf after FreeRouting succeeded.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    cfg with gnd_plane_enabled=False (a real DEFAULT_CONFIG knob, kicraft/autoplacer/config.py:373)
    and power_plane_enabled left at its default True: gnd_pour_start (defined only at line 905 under
    `if cfg.get("gnd_plane_enabled", True):`) is undefined when line 959 runs, so
    route_local_subcircuit raises NameError AFTER freerouting/pour completed, and every round of
    every routable leaf fails with a python exception instead of producing a board — the same crash
    family as the KC-V8YWN8 render-gating NameError. (Inverse config, power off + gnd on, silently
    loses the gnd_pour_s timing.) The line sits outside the adjacent try/except so nothing catches
    it.

- [ ] **B31** `kicraft/autoplacer/brain/gnd_pour.py:626` — gnd_escape_specs (pre-route GND escape stubs with via_at_end=True, default-on) emits specs with no rule-area keepout check, unlike its post-route twin which guards with _in_keepout (line 847), and add_breakout_stubs has no keepout guard either — locked stub+via copper can be stamped inside an antenna do-not-allow zone, and the resulting items_not_allowed DRC type is counted by _run_kicad_cli_drc but never gates acceptance in validate_routed_board.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Leaf with an ESP32-WROOM/S3-MINI-class module whose .kicad_mod embeds an antenna keepout
    (GetDoNotAllowVias/Tracks) adjacent to small perimeter GND pads: gnd_pre_escape
    (leaf_routing.py:545) generates a 1 mm radial stub + 0.6 mm end via for each fine-pitch GND pad
    with the escape direction pointing outward — into the keepout for antenna-end pads — and
    add_breakout_stubs stamps it (its only spatial guards are foreign pads, stamped copper, holes,
    and the board inner box). Result: items_not_allowed DRC violations (the KC-S8PC37 signature this
    guard was added post-route for), which validate_routed_board tallies at
    freerouting_runner.py:1493 but never adds to rejection_reasons, so the board is accepted and
    ships with copper in the antenna near-field.

- [ ] **B32** `kicraft/autoplacer/brain/breakout_stubs.py:768` — add_breakout_stubs records every pre-existing board track/via obstacle with the flat config floor (floor_mm) as its clearance instead of its netclass-resolved clearance, so a stamped tie can legally-to-this-guard pass a Power-netclass trace closer than the DRC pair clearance.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Parent re-tie / strand-repair pass stamping into a composed board full of leaf traces: an
    existing Power-class track requires 0.30 mm pair clearance, but stamped.append((t.GetNetCode(),
    a_mm, b_mm, t_half_w, floor_mm, t_layer)) stores 0.153 mm, so _conflicts_with_copper computes
    need = max(0.153, src_cl, 0.153)+widths and accepts a Default-class tie ~0.2 mm from the power
    trace -> hard clearance DRC error on locked copper no router pass can repair; the round is
    rejected (or, if the only named refs in the report come from one footprint, waived by the
    validate_routed_board bug and shipped). gnd_pour.py's twin obstacle collection resolves per-item
    clearance via _own_clearance_mm(t, t_layer, floor_mm) (lines 701-707) precisely to prevent this
    KC-UXASHQ-class error.

- [ ] **B33** `kicraft/cli/autoexperiment.py:3145` — autoexperiment.main() returns 0 unconditionally after the round loop (only missing-input files return 2), so the exit code lies to cli_app: the leaf-phase gate at design/cli_app.py:3403 ('if leaf_rc != 0: return leaf_rc') can never fire for solve/compose failures.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Every leaf-phase round's solve_subcircuits subprocess exits 1 (e.g. one leaf crashes in
    extraction each round, no round boards written). autoexperiment prints warnings, keeps no round,
    auto-pins nothing for that leaf, and returns 0; cli_app proceeds to the parent phase, where
    every compose round aborts with 'child subcircuit(s) produced no solved artifact' (rc=1 per
    round) -- and the parent phase ALSO returns 0. The build only fails minutes later at the promote
    freshness gate as the misleading rc6 'no routed parent / route-infra failed' verdict (the exact
    cluster documented in the 0d9ec74 memory), instead of at the phase boundary built to catch it.

- [x] **B34** `kicraft/cli/solve_subcircuits.py:794` — int(r.get("round_index", -1) or -1) collapses a legitimate round_index of 0 to -1 (the exact 'x or -1 collapses 0' pitfall), so the base_offset that keeps leaf round numbering monotonic across parent rounds (docstring lines 766-768) is computed wrong when the only prior round is round 0; the same idiom repeats at lines 1148, 1152, 1379, 1383.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A leaf solved with effective_rounds==1 (subcircuit_fast_smoke_mode with
    leaf_fast_smoke_route_rounds=1, or an explicit rounds=1 config) writes a single all_rounds
    record with round_index=0 (round_index = base_offset + len(round_results), line 856). On the
    next parent round, max_idx = max(0 or -1) = -1, so base_offset stays 0 instead of 1: the new
    solve re-numbers its rounds from 0, overwrites the prior parent round's round_0000_* snapshot
    files (leaf_routed.kicad_pcb/metadata.json/solved_layout.json), and the merge at lines 1148-1152
    drops the prior round-0 record from debug.json. A user or auto-pin pointing at snapshot 0
    (pins.json) silently starts applying a different layout than the one that was pinned.

- [ ] **B35** `kicraft/cli/compose_subcircuits.py:3100` — The stamp-time DRC gate silently no-ops when kicad-cli times out or is missing: _run_kicad_cli_drc swallows TimeoutExpired/FileNotFoundError internally and returns shorts=0 with ran=False, so the 'skip routing on composer shorts' gate and the candidate search's hard-prefer-shorts==0 selection (line 2439) both pass vacuously, despite the adjacent comment (lines 3117-3123) claiming a missing stamp-DRC result fails the round loudly.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A large stamped parent whose kicad-cli DRC exceeds the hardcoded 30 s timeout: every candidate
    in _search_best_layout reads shorts=0 (so the 'raise RuntimeError if no shorts==0 candidate'
    contract at line 2334 is trivially satisfied with zero DRC signal), and the main-path gate at
    line 3110 never detects composer-stamped overlapping leaf tracks -- FreeRouting then burns its
    200s+ budget on a known-bad board and the failure surfaces post-route attributed to routing, the
    exact misattribution the stamp gate was added to prevent. Only exceptions that escape
    _run_kicad_cli_drc reach the loud re-raise branch at line 3117, and the two most common failure
    modes never do.

- [x] **B36** `kicraft/layout_editor/runner.py:195` — run_manual_compose never kills the compose_subcircuits subprocess when the awaiting task is cancelled: asyncio.wait_for in layout_panel._on_save cancels 'await proc.wait()' on timeout but nothing calls proc.kill(), contradicting layout_panel.py:48-50 which claims the subprocess is killed.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A stamp hangs past _STAMP_TIMEOUT_S=180s (e.g. pcbnew/kicad-cli DRC wedge): wait_for raises
    TimeoutError, the UI reports failure and releases the _STAMP_SEMAPHORE slot, but the orphaned
    compose_subcircuits process keeps running with cwd=the project dir, later writing
    parent_pre_freerouting.kicad_pcb/manual_stamped.json into the workspace where it races a
    subsequent save/stamp or queued build; repeated retries stack unlimited concurrent pcbnew
    processes despite the 2-slot semaphore.

- [x] **B37** `kicraft/layout_editor/nicegui_panels.py:137` — mounting_hole_panel rebuilds its hole state from initial_holes but drops the 'pos' field, and its unconditional setMountingHoles push makes the canvas reset every corner=None hole to the outline AABB min corner (layout_canvas.js:1063-1064), which getState then persists and the composer stamps verbatim.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    User selects 'None (unpinned)' for a hole (or reopens a layout whose unpinned hole had a real
    saved pos): as soon as the panel mounts, the hole's position becomes exactly the board's top-
    left corner; Save persists pos=(min.x,min.y) to manual_layout.json and
    compose_subcircuits._move_component_to/synthesis places a real MountingHole footprint centered
    on the board corner, half off Edge.Cuts, producing copper-edge/hole-off-board DRC failures on
    the stamped board.

- [x] **B38** `kicraft/leaf_library/extractor.py:164` — _render_views imports nonexistent VIEW_DEFINITIONS from kicraft.render.views (which exports only VIEWS) and calls render_views with a nonexistent out_dir= keyword (the real parameter is output_dir), so leaf-promotion render generation fails 100% of the time and the broad except at extract_leaf line 280 masks the hard code bug as a warning.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Any call to extract_leaf(req, lib_dir) with the default render=True: _render_views raises
    ImportError at line 164 (and would raise TypeError at the out_dir= kwarg on line 169 even if the
    import existed); extract_leaf's try/except at lines 275-284 logs 'render generation failed
    (continuing without renders)' and ships the leaf bundle with no renders/ directory (no
    front_all.png/back_copper.png/copper_both.png/thumbnail.png that manifest.py lines 12-13
    document as part of a leaf), making a guaranteed code defect look like a missing-toolchain
    environment issue.

- [ ] **B39** `kicraft/design/synthesis/parts_lookup.py:52` — resolve_symbol_library_path / resolve_footprint_library_path resolve a parts-library bundle by bare is_file()/is_dir() on the tier dirs with none of the loader's validation (no manifest load, no content_hash check), so the emitter can embed files from a bundle that find_part/load_all_with_overrides consider broken and have replaced with a different tier's copy.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Remaining variant of the known vendored-hash-mismatch bug: vendored bundle <name> is hand-edited
    without `validate-part --update-hash` while a stale auto-fetched copy of <name> sits in
    ~/.kicraft/parts/ — the BOM stage's list-parts/find_part path (loader.py _load_one line 162)
    rejects the vendored dir and reports the HOME bundle's manifest (mpn, symbol_name,
    footprint_name) as active, but the emitter's resolve_symbol_library_path walks the same
    resolve_tier_dirs order and returns the VENDORED <name>.kicad_sym because the file exists — the
    schematic embeds a symbol from a different bundle than the one the BOM validated, a WS2812-class
    invisible pinout divergence that no gate ties back together.

- [x] **B40** `kicraft/server/session.py:173` — record_answers/null_downstream (and stage_driver._attach_questions, cli_app's five state writes) rewrite state.json with plain non-atomic write_text, and the read half maps ANY failed/torn read to {} — so one torn read silently truncates the committed design state to near-empty.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    All state.json writers except _stamp_stage_status (stage_driver.py:581, made atomic explicitly
    because 'the web render timer reads this file concurrently') use truncate-then-write:
    session.py:173, session.py:197, stage_driver.py:666, cli_app.py:3170/3228/3855/3892/3939. A
    reader in the other process that lands mid-truncate (web timer _read_state_json web.py:768,
    reopen read_state web.py:4799) gets {} -> stage tabs flip to all-pending and
    remaining_stages({}) offers 'Continue design' from intent, which re-drives every stage over the
    committed design. Worse, record_answers and null_downstream are read-modify-write built on
    read_state (session.py:148-150), which swallows OSError/JSONDecodeError into {}: if their read
    collides with a concurrent cli_app write (reachable via the same-workspace double-build in the
    enqueue_build finding, or any build overlapping a UI state write), they then write_text
    json.dumps({}) — wiping every committed slot with no error surfaced anywhere.

### 1.3 Low severity (15)

- [x] **B41** `kicraft/server/accounts.py:1980` — next_cycle_index's docstring claims an atomic read-modify-write 'in one transaction', but with sqlite3's default isolation the SELECT at line 1980 runs in autocommit and the implicit transaction only begins at the INSERT (line 1986), so the read and write are not atomic across connections.
  - **Severity:** low · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    Two users click 'Surprise me' simultaneously (two NiceGUI session threads, each with its own
    connection): both SELECT value=5, both compute cur=5, the INSERTs serialize but both write '6'
    -- both users receive the same corpus brief and the counter advances once instead of twice,
    exactly the outcome the docstring promises cannot happen. Needs BEGIN IMMEDIATE (or a single
    UPDATE ... RETURNING) to hold the write lock across the read.
  - **Verifier:**
    _conn (accounts.py:348-352) passes no isolation_level/autocommit, so sqlite3's legacy deferred
    mode applies: the implicit transaction begins only at DML, meaning the SELECT at 1980 runs in
    autocommit and holds no lock when the INSERT at 1986 begins. Two concurrent calls on separate
    connections (each NiceGUI session thread calls _conn) can both read value=5, both compute cur=5,
    and both INSERT '6' — both callers get index 5 and the counter advances once, exactly what the
    docstring at 1973-1974 ('Atomic read-modify-write in one transaction ... never hand out the same
    index') claims cannot happen. WAL mode (line 351) makes the concurrent read-before-write even
    easier, not harder. The defect is a race so any single run may not exhibit it, but the non-
    atomicity is a factual property of the code contradicting its documented contract.
  - **Key line(s):** `conn = sqlite3.connect(self.path, timeout=30)`

- [x] **B42** `kicraft/server/web.py:1424` — The store.finish_project(...) call that flips the durable project row to its terminal status is wrapped in `except Exception: pass` with no logging, unlike the neighboring genuinely best-effort catalog/notify blocks (lines 1430-1446) which are at least commented as such.
  - **Severity:** low · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    A transient SQLite failure (e.g. 'database is locked' outlasting the 30s connection timeout
    under heavy concurrent build/queue writes) makes finish_project raise inside _persist_project's
    finally block: the completed build's status/zip_path/dir_path/cost are never written, the
    exception vanishes without a single log line, and the row stays 'running' until a janitor later
    reaps it as 'interrupted' -- the user sees a phantom interrupted project with no download for a
    build that actually succeeded, with zero diagnostics to explain why.
  - **Verifier:**
    Factually present: the one write that flips the durable project row to its terminal status is
    swallowed with a bare `except Exception: pass` and zero logging (unlike the persist-error branch
    above it, which at least appends a 'persist error' event, and the commented best-effort
    catalog/notify blocks at 1430-1446). Given any finish_project raise (e.g. sqlite 'database is
    locked' outlasting the connection timeout under concurrent queue writes), the consequence chain
    is deterministic and matches the code: the row stays 'running', _reconcile_orphan_projects later
    flips it to 'interrupted' (web.py:235-236) since no live dict remains after the thread exits,
    and no diagnostic exists anywhere to explain the phantom interrupted project.
  - **Key line(s):** `store.finish_project(pid, status, stem=stem, cost_usd=state.get("spend"),
                                 dir_path=dir_path, zip_path=zip_path)
        except Exception:
            pass   (web.py:1422-1425)`

- [x] **B43** `kicraft/server/web.py:5184` — _live_sig() iterates the module-global _LIVE_RUNS dict via .items() on the NiceGUI event-loop thread (called from the 0.2s render timer at line 5215/5466) while background design/build threads insert (lines 1736, 1810) and pop (lines 1770, 1931) entries with no lock.
  - **Severity:** low · **Status:** VERIFIED-PLAUSIBLE
  - **Failure scenario:**
    A design run finishing (thread executes _LIVE_RUNS.pop at web.py:1931) while any open workspace
    page's render timer is mid-way through the generator inside tuple(sorted(... for pid, st in
    _LIVE_RUNS.items() ...)) raises RuntimeError('dictionary changed size during iteration') inside
    the timer callback; that tick's render aborts, so the project-list refresh and question-panel
    pickup for that tick are lost and an error is logged, recurring sporadically on a busy multi-
    user server.
  - **Verifier:**
    The mechanism is real: _live_sig runs on the event-loop render timer (5215/5448, ui.timer(0.2,
    render) at 5466) while _rerun_build_worker and _run_design run in plain threading.Thread workers
    (5064/5117, 4605) that insert (1736, 1810) and pop (1770, 1931) _LIVE_RUNS entries with no lock
    anywhere in the module -- CPython raises RuntimeError('dictionary changed size during
    iteration') if a pop/insert lands mid-iteration, aborting that render tick. However the
    vulnerable window is the microseconds sorted() spends consuming the generator, hit only if a run
    finishes in exactly that slice, so the trigger is a low-probability interleaving rather than a
    nameable state. A stress test (one thread hammering _live_sig, another adding/popping _LIVE_RUNS
    entries) would confirm it; the fix (snapshot via list(_LIVE_RUNS.items()) or a lock) is cheap
    either way.
  - **Key line(s):** `return tuple(sorted(
                (pid, bool(st.get("running")))
                for pid, st in _LIVE_RUNS.items() if st.get("user_id") == user.id))   (web.py:5182-5184)`

- [x] **B44** `kicraft/design/synthesis/validation.py:309` — check_refdes_uniqueness scans references with a regex that excludes the suffixed refdes forms the ref grammar allows (D1A, J1-PWR), leaving the §9.7 duplicate gate blind to collisions among exactly the leaf-library-renumbered refs it was built to catch.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    §9.7 exists to catch leaf-library renumber-map bugs that produce the same refdes in two
    .kicad_sch files, but its scan regex '\(property \"Reference\" \"([A-Z]+[0-9]+)\"' requires the
    closing quote immediately after the digits, while the project's ref grammar (models.REF_RE
    ^[A-Z]+[0-9]+[A-Z0-9_-]*$, and validation.py's own _REFERENCE_RE at line 44) allows suffixed
    refs like 'D1A' or 'J1-PWR'. Verified: findall over '(property "Reference" "D1A")...' returns
    only the unsuffixed refs. So two sheets both carrying 'D1A' (e.g. a renumber bug on suffixed
    leaf-library refs) sail through the uniqueness gate and surface later as a silent refdes
    collision on the board — the exact class the check was written to stop.

- [x] **B45** `kicraft/design/cli_app.py:340` — _SOURCING_LCSC_RE treats capacitor package tokens like 'C0603'/'C1206' in sourcing_note prose as an explicit LCSC part pin, causing false §9.26 commit rejections or, worse, pricing/exporting the wrong real part into the fab BOM.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    _SOURCING_LCSC_RE (\bC\d{4,}\b) is the highest-priority sourcing tier in
    _resolve_bom_mpn_sourcing (line 519: an explicit pin beats manifest and MPN search), but it
    matches capacitor PACKAGE tokens — 'C0603', 'C1206' are exactly the package strings the EasyEDA
    lookup returns (_parse_easyeda_search 'package': c_para.get('package')) and that a BOM agent
    plausibly echoes into sourcing_note ('100nF X7R, package C0603'; verified the regex matches).
    Result: the part is treated as pinned to LCSC C0603 — either the §9.26 gate bounces the commit
    with the baffling 'sourcing_note claims LCSC C0603 which is not in the offline catalog' (the
    model keeps rewriting real sourcing data and re-tripping), or, if that low-numbered C# happens
    to exist in the catalog with stock, a completely unrelated part is silently priced and exported
    into the fab BOM (fab_export.py:33 reads the C# back out with the identical regex).

- [x] **B46** `kicraft/autoplacer/brain/placement_solver.py:2720` — _accumulate_repulsion_numpy's skip_mask zeroes repulsion for pairs with dists < 0.001, whereas the pure-Python fallback (lines 2679-2681) clamps d to 0.1 and applies the STRONGEST repulsion to coincident parts - the production (numpy) path silently drops the anti-coincidence force.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Two unlocked components seeded at the exact same point (cluster placement clamps both to the
    same corner/zone coordinate, dist = 0): with numpy installed they receive zero mutual repulsion
    for the entire force loop and stay stacked until _resolve_overlaps rips them apart late with a
    full-bbox escape far from their connected nets, whereas the Python fallback would have separated
    them gradually in-place; identical configs behave differently depending on numpy availability.

- [ ] **B47** `kicraft/cli/autoexperiment.py:2959` — The kept-round preview promotion calls _discover_live_preview_paths(project_dir) with NO mtime_floor inside the round loop, contradicting that function's own docstring ('Pass round_wall_started_at as the floor inside the round loop'), so a stale render can be promoted as this round's frame.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A round routes successfully (kept=True) but its render step fails (render failures are non-fatal
    in compose: parent_routed.png is only copied when _render_parent_board_views returns front_all).
    The canonical renders dir still holds a PRIOR round's (or prior run's) PNG; lines 2964-2973 copy
    it to best_preview.png, frames/frame_NNNN.png and frame_latest.png, presenting a different
    board's image as the kept round's output -- the exact class of garbage the mtime-gated snapshot
    path 150 lines earlier (lines 2770-2802) exists to prevent.

- [ ] **B48** `kicraft/cli/autoexperiment.py:1674` — _archive_run copies EVERY rounds/round_*.json from the shared live rounds dir into the new run's archive, but run start never scrubs that dir (only log/status files are unlinked at lines 2168-2171), so higher-numbered round files from a longer previous run -- or from the same build's leaf phase -- are archived as this run's rounds.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A standard build runs the leaf phase (e.g. 6 rounds under one run_id) then the parent phase
    (e.g. 2 rounds under a different run_id) against the same .experiments/rounds/ dir: the parent
    phase overwrites round_0001-0002 but rounds 0003-0006 from the leaf phase survive, and
    _archive_run copies all six into .experiments/runs/<parent_run_id>/rounds/ -- anyone auditing
    the parent run's archive (or the GUI analysis page) sees 4 leaf-phase rounds with leaf-phase
    scores/exit-codes attributed to the parent search.

- [ ] **B49** `kicraft/cli/autoexperiment.py:451` — _pin_best_parent returns early when no round was kept (best_round None or not parent_routed), so a parents-only run where every round was gate-rejected ships the LAST-written round's board instead of the best-scoring one -- unmet in exactly the regime its own docstring targets ('ships whatever was written last instead of the round the search selected').
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Every parent round routes but is rejected (compose returns 1, e.g. unconnected_nets), so kept is
    always False (subprocesses_ok requires parent_route_rc==0) and best_round stays None.
    _route_parent_board has meanwhile overwritten the canonical parent_routed.kicad_pcb on every
    round (routed_pcb is written before acceptance, _compose_route.py:81), so promotion-by-recency
    in cli_app ships round N's board even when _score_round graded round 2 as routed_dirty (0
    shorts, 0 unconnected, score ~55) and round N as not_routed (score 20) -- the user's rc7
    inspection board is strictly worse than the one the search found, even though per-round
    snapshots (round_NNNN_parent_routed.kicad_pcb, line 2828) exist to restore it.

- [x] **B50** `kicraft/layout_editor/runner.py:370` — _layout_to_canvas discards the saved layout's parent_local list (and layout_canvas.js getState never emits one), so any manual_layout.json containing parent_local overrides has them silently replaced with [] on the next web-panel save, even though the composer still honors them (compose_subcircuits.py:1363-1367).
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A project carries a manual_layout.json with parent_local entries (written by the removed offline
    GUI or by hand to pin an edge connector): the user opens the web layout editor, moves one leaf,
    and clicks Save; save_manual_layout_json writes parent_local: [] because the canvas payload
    never contained the key, and the next manual compose snaps the connector back to its constraint-
    derived position, undoing the user's persisted override with no warning.

- [x] **B51** `kicraft/layout_editor/nicegui_panels.py:252` — view_options_panel initializes options with snap_spacing_mm=1.0 and unconditionally pushes it to the canvas via ui.timer 0.3s after mount, overriding the canvas's documented 0 mm default even though the function's own docstring says defaults match historical behavior (0 mm gap) and that opening the expansion is a no-op.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    User opens the layout editor and drags two leaves together expecting the historical flush snap:
    0.3s after page render every edge-to-edge snap targets a 1.0 mm gap instead of 0, so leaves can
    no longer be snapped flush (and the stamped parent board is correspondingly larger) unless the
    user finds View options and manually types 0 into a field they never touched.

- [x] **B52** `kicraft/layout_editor/nicegui_panels.py:111` — _push_shape coerces a cleared radius/chamfer input to 0.0 and the canvas accepts it, but OutlineSpec.from_dict rejects rounded_rect/chamfered_rect with param <= 0, so the UI renders an outline state that save_manual_layout_json can never persist.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    User picks shape 'Rounded' and clears the 'Radius/chamfer (mm)' field (on_value_change fires
    with None, bypassing the min=0.5 constraint): the canvas happily draws a 0-radius rounded rect,
    but clicking 'Save & stamp preview' fails in save_manual_layout_json with ValueError
    'rounded_rect outline requires corner_radius_mm > 0', leaving the user with a layout the UI
    accepted but the pipeline refuses until they guess the field must be repopulated.

- [x] **B53** `kicraft/server/rules_panel.py:177` — _on_anchor_change only writes the override when the new target is 'none' or when a value is later picked, so switching a component's anchor from e.g. edge to corner without selecting a corner value leaves the stale edge override in holder.component_zone_overrides while the UI shows the new target with an empty value.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A component has anchor edge=left; the user changes the Anchor dropdown to 'corner', leaves Value
    blank (intending to re-anchor later or believing the edge pin is cleared), and clicks 'Apply &
    re-place': _build_slot commits {"edge": "left"} to the placement slot, so the rebuild still pins
    the part to the left edge even though the panel displayed anchor=corner with no value.

- [x] **B54** `kicraft/render/edge_cuts.py:21` — _BLOCK_RE terminates a gr_ block only at the next (gr_ / (footprint / EOF, but KiCad writes segment/via/zone sections AFTER graphics, so the last gr_ block swallows all trailing tracks and zones (verified on 88 of 400 real generated boards) and parse_edge_cuts_aabb then folds their start/end/xy points into the 'Edge.Cuts' AABB.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A board whose last Edge.Cuts gr_line is not followed by another gr_/footprint token (no gr_text
    after the outline — 22% of scanned real boards) and that has any copper outside the outline AABB
    (freerouting wires ignore the DSN boundary per KC-WXN3SN, and walled-off strays are a documented
    occurrence): the swallowed (segment (start/(end and zone (xy points inflate the returned AABB,
    so the rendered PNG extent, the manual-layout collision box, and the verification tool's board
    extent silently exceed the physical Edge.Cuts; currently latent (0/3000 boards showed a >0.01mm
    delta) but the regex's stated invariant is falsified by real files.

- [x] **B55** `kicraft/server/parts_catalog.py:202` — symbol_svgs (and footprint_svg at lines 219-220) return sorted(out_dir.glob('*.svg')) even when _run_ok reported the kicad-cli export FAILED, so partially written SVGs from a failed/timed-out export are served as valid previews while the missing .ok sentinel re-spawns the failing 30s kicad-cli run on every subsequent catalog page view.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A multi-unit symbol bundle where `kicad-cli sym export svg` writes one unit's SVG then errors or
    hits the 30s timeout: the /parts page renders the partial unit set as the part's preview
    (partial image treated as success), and because .ok is never touched the same failing 30-second
    subprocess re-runs on every page load of that part instead of caching the failure.

## 2. Cleanups (duplication, dead code, efficiency)

- [ ] **C1** `kicraft/autoplacer/hardware/adapter.py:135` — _pcbnew_subprocess_env is a byte-identical 42-line copy of _kicad_subprocess_env (kicraft/autoplacer/freerouting_runner.py:113), and kicraft/cli/solve_subcircuits.py:55 _ensure_kicad_python_path carries a third, already-drifted copy of the same KiCad site-packages discovery block; extract one shared helper (e.g. kicraft/autoplacer/hardware/adapter._pcbnew_subprocess_env promoted to a small kicad_env module) and have all three call it.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A new KiCad install layout (e.g. an extra dist-packages path or a pcbnew.pyi-only wheel) must be
    patched in three places; the cli copy has already diverged (it mutates sys.path in-process while
    the other two build a subprocess PYTHONPATH env), so a fix applied to one leaves leaf-solve
    subprocesses and freerouting/stamp subprocesses resolving pcbnew from different locations.

- [ ] **C2** `kicraft/design/synthesis/electrical_review.py:231` — _extract_json is a byte-identical 31-line copy of kicraft/eval/judge.py:93 _extract_json, and kicraft/server/stage_driver.py:350 holds a third, behaviorally different variant of the same 'parse JSON out of an LLM reply' job; extract one fence-tolerant helper (e.g. a small kicraft/llm_json.py) and reuse it in all three.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Any parsing fix (fence variants, prose around the object, brace-depth scanning) must be re-
    applied in three files; the stage_driver variant already diverges (regex-based, raises
    json.JSONDecodeError instead of returning None, no depth-scan for trailing prose), so the stage-
    commit path rejects model replies the review/judge paths would parse.

- [ ] **C3** `kicraft/autoplacer/brain/subcircuit_composer.py:2236` — subcircuit_composer re-implements _bbox_disjoint (line 2236) and _rect_area (line 1477) that also exist in kicraft/cli/_compose_geometry.py:85/67, and can_overlap (line 2255) re-inlines the pairwise loop that _compose_geometry._rect_lists_disjoint already provides; move the shared rect primitives into autoplacer/brain/geometry.py (the stated geometry home) and import them from both sides.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    The two ends of the same compose pipeline judge overlap with different code: the composer copy
    handles None bboxes while the cli copy does not, so a semantics fix (tolerance, touching-edges
    rule) applied to one leaves parent placement and compose-time validation disagreeing on whether
    the same two rects overlap.

- [ ] **C4** `kicraft/parts_library/loader.py:162` — _load_one re-computes the full SHA-256 content hash of every bundle file (including multi-MB .step/.wrl 3D models and PDFs) on every load_all_with_overrides/find_part call, with no caching anywhere.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Measured: load_all_with_overrides(None) hashes ~575MB (253MB vendored + 322MB ~/.kicraft/parts)
    and takes 0.53s per call warm (worse cold). It runs on every /parts page view (web.py:3641
    catalog()), on every /part-preview HTTP request (render_serving.py:156 get_part -> find_part; 2+
    requests per part-detail view, 17MB for the header-male bundle), and 3-4 times inside each BOM
    stage-commit subprocess (cli_app.py:316/400/511 gates + :2580 prompt extras) plus once per LLM
    tool subprocess (cli_app.py:1062 list_parts, :1213 lookup) -- since every attempt/tool call is a
    fresh process, that is roughly 15-30s of pure re-hashing per design. Cheaper: cache the
    verification verdict keyed by a stat signature (relpath,size,mtime of all bundle files) in a
    stamp file, re-hashing only when the signature changes -- this preserves the hash-mismatch-
    drops-bundle gate exactly; or verify hashes only in validate-part/startup instead of on
    serving/gate paths.

- [ ] **C5** `kicraft/parts_library/jlcparts.py:249` — search()'s substring and ANDed-description fallbacks use LIKE '%...%' predicates that cannot use idx_jlc_mfr, full-scanning the 634k-row catalog once per probe.
  - **Severity:** medium · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Measured on the live 456MB catalog: mfr LIKE '%q%' = 0.14s, the 4-column ANDed description scan
    = 0.28s, per call. One generic-passive keyword miss costs exact->substring->ANDed (~0.4s),
    doubled by the relax_keyword retry (~0.85s for '0.1uF 25V X7R 0603'), and _widen (line 228) adds
    up to 4 more 0.14s scans for MPN-ish tokens. The §9.26 gate (cli_app.py:597/659) re-runs these
    per distinct BOM keyword/MPN per commit attempt, and the per-attempt memo dies with each fresh
    subprocess, so a 10-generic-part BOM burns ~4-9s of scans per attempt, up to ~5 attempts; web
    pricing (_fetch_price kw:/mpn: keys) pays the same per uncached key. Cheaper: build an FTS5
    index over (mfr, description, manufacturer, package) in update() right where idx_jlc_mfr is
    created (jlcparts.py:539) and query it for the substring/keyword tiers (~ms each); the _widen
    prefix probes can use LIKE 'q%' which the existing NOCASE index serves.

- [x] **C6** `kicraft/server/web.py:1625` — _JOB_KIND_ARGS duplicates the cross-process build invocation already defined as _BUILD_CMD/_MANUAL_ROUTE_CMD in build_worker.py:43-49, spelling the job-kind -> command contract in two processes.
  - **Severity:** low · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    The exact argv ['build', '.kicraft/state.json', 'generated', '--no-archive'] and the manual-
    route variant exist verbatim in both files, along with duplicated --quality handling
    (web.py:1638-1641 vs build_worker.py:148-152). A flag or path change applied to one copy only
    means a build behaves differently depending on whether the standalone worker happened to be
    heartbeating at claim time — precisely the deploy-skew failure mode the worker's own kind-map
    comment (build_worker.py:70-73) tries to defend against, but with no shared constant to keep the
    two sides equal.
  - **Verifier:**
    The duplication is factually present as described: the identical argv tails exist verbatim in
    web.py:1625-1628 and build_worker.py:43-49 with no shared constant, and the --quality tier-
    override logic is likewise duplicated (web.py:1638-1641 `quality =
    _store().build_quality_for_user(...)` vs build_worker.py:148-152 `quality =
    self.store.build_quality_for_user(job.user_id)`). Which copy executes depends on whether the
    standalone worker heartbeated within 15s at claim time (build_worker_alive, accounts.py), so a
    change applied to only one file yields builds that differ by worker liveness -- exactly the
    deploy-skew hazard the worker's own kind-map comment (build_worker.py:70-73) defends against for
    unknown kinds but cannot defend against for divergent known-kind argv.
  - **Key line(s):** `_JOB_KIND_ARGS = {
    "build": ["build", ".kicraft/state.json", "generated", "--no-archive"],
    "manual_route": ["manual-route", ".kicraft/state.json", "generated"],
}   (web.py:1625-1628)  vs  _BUILD_CMD = [... "build", ".kicraft/state.json", "generated", "--no-archive"] (build_worker.py:43-44)`

- [x] **C7** `kicraft/server/web.py:410` — _zip_generated archives the entire generated/ tree including the heavy internal .experiments/ directory, so every user-facing 'KiCad project' download ships internal placement/routing experiment artifacts.
  - **Severity:** low · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    Verified on disk: projects/1/538/kicraft_project.zip contains 155 entries of which 121 are under
    PASSIVE_RC_FILTER_BNC/.experiments/ (per-round renders, subcircuit search state).
    shutil.make_archive(root_dir=generated) has no ignore filter, so each download and each
    persisted zip is inflated with internal debris that is useless to the end user and grows with
    board complexity (the .experiments tree is called out as 'heavy' in the storage model), wasting
    bandwidth/disk on every export.
  - **Verifier:**
    make_archive has no ignore/filter capability and gen is the whole generated/ tree. Independently
    re-verified on disk: /home/kicraft/.kicraft/projects/1/538/kicraft_project.zip contains 155
    entries of which 121 are under a .experiments/ subtree. This zip is what state['zip'] and the
    user Download button serve (web.py:1753, 5431-5432); the build itself is invoked with --no-
    archive so this function is the sole packaging path.
  - **Key line(s):** `return shutil.make_archive(base, "zip", root_dir=str(gen))   (web.py:410)`

- [x] **C8** `kicraft/server/web.py:2577` — The profile page's Privacy & data card tells users to contact a literal '[CONTACT EMAIL]' placeholder for data export/deletion requests.
  - **Severity:** low · **Status:** VERIFIED-CONFIRMED
  - **Failure scenario:**
    Any logged-in user opens /profile and reads 'To export or delete all your data, contact [CONTACT
    EMAIL].' -- the template placeholder was never substituted, so the advertised (privacy-policy-
    adjacent) data export/deletion channel is a dead end for every user in production.
  - **Verifier:**
    The literal placeholder string ships to every /profile viewer; grep confirms no substitution
    mechanism exists anywhere in the package -- '[CONTACT EMAIL]' appears only at this line as a
    hardcoded label.
  - **Key line(s):** `ui.label("To export or delete all your data, contact "
                     "[CONTACT EMAIL].").classes("text-xs")   (web.py:2576-2577)`

- [x] **C9** `kicraft/server/render_serving.py:175` — Part-preview SVGs are served with Cache-Control: no-store even though they are immutable per content_hash (the on-disk cache dir is already keyed by the hash), forcing every page view to refetch and re-trigger the server-side bundle re-hash.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Each revisit of /parts/<name> refires the symbol+footprint preview requests; each request re-
    runs get_part(name) -> find_part -> compute_content_hash over the whole bundle (e.g. ~17MB for
    header-male-2-54-1x40) before serving an unchanged SVG. Cheaper: embed the content-hash slice
    (parts_catalog._content_hash_key, already computed) or an mtime ?v= in the preview URL -- the
    pattern render URLs in web.py already use -- and serve with a long max-age/immutable so
    unchanged previews never hit the server again.

- [x] **C10** `kicraft/design/cli_app.py:1164` — _attach_retail in cli_app.py duplicates web.py's _attach_retail (kicraft/server/web.py:916) -- both wrap lcsc_retail.enabled()/stock()/RetailUnavailable to pin retail_stock/retail_min_buy onto a pick dict, differing only in outage encoding (payload['retail']='unverified' vs retail_stock=None); fold into one helper in kicraft/parts_library/lcsc_retail.py with an explicit outage-marker parameter.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A change to the retail reading (e.g. surfacing min_buy multiples or a new RetailUnavailable
    subtype) must be fixed twice, and the two call sites already encode 'unverified' differently, so
    the BOM-lookup path and the pricing-cache path can disagree about whether the same part's retail
    state was checked.

- [x] **C11** `kicraft/autoplacer/freerouting_runner.py:1036` — _build_contact_sheet is dead (private, zero callers repo-wide) and duplicates the live contact-sheet implementation in kicraft/autoplacer/brain/subcircuit_render_diagnostics.build_leaf_contact_sheet (line 326); delete it.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    27 dead lines in the load-bearing routing module, including a suspicious ImageMagick invocation
    ('montage' passed as a trailing argument to `magick` before the output path) that would silently
    return False if anyone revived it instead of reusing build_leaf_contact_sheet.

- [x] **C12** `kicraft/server/session.py:144` — read_state's docstring claims _state_path 'resolves the workspace (.kicraft) or durable (kicraft) layout', but _state_path (storage.py:31-33) is single-layout — stale pre-Phase-4a dual-name documentation contradicting the one-name .kicraft/ rule.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    Three docstrings still describe the removed .kicraft/kicraft duality: session.py:142-145,
    storage.py:1-13 ('workspace<->durable mechanics ... planned collapse of the .kicraft/kicraft
    duality' — the collapse shipped in Phase 4a), and web.py:3134-3136 ('older ones have kicraft/ or
    a top-level copy' — such projects were nuked and _state_path checks only .kicraft/state.json).
    Cost: an agent or dev trusting these docstrings re-introduces a no-dot 'kicraft/' fallback that
    CLAUDE.md's storage model explicitly forbids ('One name, no fallback'), or wastes time hunting
    for a fallback path that does not exist in the code.

- [x] **C13** `kicraft/autoplacer/brain/subcircuit_composer.py:1059` — _build_merged_nets dedupes pad refs with a linear `pad_ref not in net.pad_refs` list-membership per pad, making net assembly O(pads^2) per net; _append_pad_ref (line 1244) repeats the pattern.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    On a large-array parent (e.g. the 200-LED KC-SMQ3HX class board) the GND net accumulates
    hundreds of (ref,pad) tuples, so each additional GND pad rescans the whole list -- hundreds of
    thousands of tuple comparisons per compose pass -- purely for dedup bookkeeping, while the
    interconnect merge branch 13 lines below (lines 1072-1076) already does the same job with a
    seen-set. Cheaper: keep the same seen-set alongside the list in both _build_merged_nets and
    _append_pad_ref (pure bookkeeping; does not touch placement/routing decisions).

- [x] **C14** `kicraft/cli/_compose_geometry.py:34` — Four helpers in _compose_geometry.py are dead: _bbox_size (line 13), _shift_bbox (line 17), _shift_layer_envelopes (line 34), and _shift_rects (line 60) have zero references anywhere in kicraft/, tests/, tools/ or scripts/ (compose_subcircuits imports only the other six helpers); delete them.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    About 40 of the module's 98 lines are unreachable in a file documented as 'shared by the
    outline, slide, stamp and validation code', so readers and future refactors of the compose
    pipeline must reverse-engineer which third of the 'shared' helpers is actually load-bearing.

- [x] **C15** `kicraft/cli/render_pcb.py:30` — render_all is a self-described 'Compat shim for legacy callers' with zero remaining callers anywhere (the render-pcb entry point in pyproject.toml uses main(), and all other consumers call kicraft.render.render_views directly); delete it.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    16 lines of dead wrapper invite new code to call the shim (which adds print side effects and
    swallows missing-toolchain errors into an empty dict) instead of the documented canonical API
    kicraft.render.render_views.

- [ ] **C16** `kicraft/parts_library/lcsc_retail.py:117` — _save_disk re-reads, re-serializes, and atomically rewrites the entire retail-cache JSON for every single new C# entry (and _load_disk re-parses the whole file on every in-memory miss), all while holding the global _LOCK.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    A §9.26 gate pass walking a BOM with N not-yet-cached C#s performs N whole-file read+rewrite
    cycles of a file that only grows (disk entries are never pruned by TTL, so every C# ever checked
    accumulates) -- an O(total_entries * N) I/O pattern serialized against all other retail lookups
    in the process; mpn_cache.py:88 repeats the same pattern per put(). Cheaper: load the disk cache
    into _MEM once per process and flush once per gate pass (or append JSONL entries), keeping the
    same atomic-replace only at flush time.

## 3. Convention violations

- [x] **V1** `kicraft/cli/leaf.py:4` — Docstring directs readers to the promotion wizard at kicraft/gui/pages/leaf_library.py, a module deleted with the desktop GUI on 2026-06-22 (extractor.py line 4 repeats the same stale claim).
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    An agent or developer asked to promote a leaf follows the module's own pointer, greps for
    kicraft/gui/pages/leaf_library.py, finds nothing, and burns a session re-deriving that
    extract_leaf in kicraft/leaf_library/extractor.py is the only remaining promote entry point — in
    a codebase whose CLAUDE.md explicitly maintains docs as the map for agents.

- [x] **V2** `kicraft/server/routes_admin.py:835` — _self_eval_leaf_boards hand-globs '.experiments/subcircuits/*/leaf_routed.kicad_pcb', hard-coding the artifact layout and filename instead of importing LEAF_ROUTED/artifact_root from kicraft/cli/artifact_paths.py.
  - **Severity:** low · **Status:** NEEDS-VERIFICATION
  - **Failure scenario:**
    docs/ARTIFACTS.md states the contract: filename literals are defined once in artifact_paths.py
    ('Do not hard-code these literals in new code — import them') and '.experiments' must not be
    globbed by hand. This glob re-inlines both the directory layout and the LEAF_ROUTED literal; if
    the canonical names or layout move (they are single-sourced precisely so they can), the admin
    self-eval per-leaf board viewer silently renders zero leaves with no error, and the divergence
    class the resolver module was built to end (its own header documents the multi-hour stale-board
    debugging session) is reintroduced.

## 4. Feature enhancements

- [ ] **E1** Interactive 3D viewer for the user's finished board (fab tab + community page)
  - **Area:** kicraft/server/web.py (fab view_slot ~line 5424), kicraft/design/cli_app.py fab export (~4154-4179), kicraft/server/samples.py · **Impact:** high · **Effort:** medium
  - **What:**
    Extend the fab export to also emit a board.glb (kicad-cli, same path used to generate samples'
    previews/board.glb) and render it with the already self-hosted <model-viewer> component in the
    fab tab's view slot and on /p/{id}, replacing/augmenting the static fab/board_3d.png.
  - **Why:**
    The samples explorer already ships interactive 3D (samples.py has_3d/board_glb,
    _sample_model_viewer in web.py, self-hosted /static/model-viewer.min.js), but a user's own
    finished design only gets a static PNG (render() shows fab/board_3d.png) plus a STEP file buried
    in the zip (artifacts.step_file, cli_app.py 4163). The gap between the marketing surface and the
    product deliverable is visible to every user; all building blocks exist.

- [ ] **E2** Visual DRC/ERC failure overlay in the Place/Route tab on failed builds
  - **Area:** kicraft/server/web.py (_mark_fab_invalid ~5125, _inspector_spec place_route ~1225), kicraft/cli/render_drc_overlay.py, kicraft/server/render_serving.py · **Impact:** high · **Effort:** medium
  - **What:**
    When a build ends ok=False with a board present, run the existing render-drc-overlay / render-
    failure-heatmap tooling on the failed candidate and show the annotated PNG (violation markers on
    the board) in the place_route view slot next to the red 'do not fabricate' banner, with a legend
    of shorts/unconnected/courtyard errors.
  - **Why:**
    render-drc-overlay and render-failure-heatmap already exist as CLI entry points (pyproject.toml
    lines 64-65) but nothing in the web UI uses them (grep for overlay/heatmap/drc in web.py returns
    nothing). Today a failed run shows only build-log text lines and a text banner; users must
    download nothing and can diagnose nothing visually, even though the failed board is deliberately
    kept for inspection.

- [ ] **E3** Fab-spec summary card and zip manifest on the FAB tab
  - **Area:** kicraft/server/web.py (_inspector_spec 'fab' branch, lines 1296-1308) + kicraft/design/cli_app.py fab export · **Impact:** high · **Effort:** small
  - **What:**
    Replace the bare status/pending kv rows with an ordering-ready spec card: board dimensions
    (already computed in run_status board_metrics), layer count, min track/clearance/via-drill (from
    the .kicad_pro netclasses), part count, plus a listing of the zip's contents (Gerbers, drill,
    CPL, BOM, STEP) with individual file downloads — the exact fields a JLCPCB/PCBWay order form
    asks for.
  - **Why:**
    The zip already contains Gerbers + drill + CPL + BOM + STEP (cli_app.py line 4155) and
    board_metrics with width/height/utilization already flow into the place_route inspector (web.py
    1240-1253), but the fab tab shows only 'fab package: ready / STEP: ready' and a single download
    button. Users ordering their first PCB must open KiCad to answer the fab's questionnaire;
    surfacing what's already computed closes the last mile of the product promise.

- [ ] **E4** BOM board-quantity selector and CSV export
  - **Area:** kicraft/server/web.py (_inspector_spec 'bom' branch, lines 1142-1190) + kicraft/server/stagetabs.py (_table_html) · **Impact:** medium · **Effort:** small
  - **What:**
    Add a qty input (1/5/10/custom boards) to the BOM inspector that rescales the cost column and
    total (using price_10/price_100 breaks the price fetch already returns for parts, see
    part_detail_page 3801-3805), and an 'Export BOM (CSV)' button that downloads the priced parts
    table with LCSC C-numbers without waiting for a successful full build.
  - **Why:**
    The BOM inspector prices only qty-1 unit cost ('est. unit price, cheapest in-stock LCSC match')
    and the only way to get the BOM as a file is the fab zip, which exists only after a fully
    successful build. Qty breaks are already fetched (_price_for_lcsc returns price_10/price_100),
    so multi-board costing is nearly free to add and matches how people actually order.

- [ ] **E5** Per-row 'swap this part' picker in the BOM inspector
  - **Area:** kicraft/server/web.py (BOM inspector + build_edit_panel ~4608, _do_rerun ~4687), kicraft/server/parts_catalog.py, offline jlcparts catalog · **Impact:** high · **Effort:** medium
  - **What:**
    Add a swap action on each BOM row that searches the offline jlcparts catalog (636k parts,
    already local) filtered to the same value/footprint class, shows price/stock/Basic-vs-Extended
    for candidates, and on confirm commits the edited bom slot and re-runs downstream stages via the
    existing commit_slot + null_downstream machinery.
  - **Why:**
    Today changing one part means either hand-editing the raw slot JSON or typing a freeform
    instruction in 'Edit a stage & re-run' (build_edit_panel offers form/JSON/agent modes only) —
    both re-run the whole downstream chain with no candidate guidance. The pipeline already has
    candidate-suggestion logic for BOM misses (memory: fix 48b6a01) and the pricing/stock plumbing
    (_ensure_bom_prices, parts_catalog.py); exposing it per-row turns the most common design tweak
    into one click.

- [ ] **E6** Logged-out viewing of public projects with social preview cards
  - **Area:** kicraft/server/web.py public_project_page (lines 3338-3373) + _board_thumb_url (~3093) + render_serving.py · **Impact:** high · **Effort:** medium
  - **What:**
    Let /p/{project_id} render read-only for anonymous visitors (board + schematic KiCanvas preview,
    BOM table, prominent 'Sign up to clone' CTA) instead of redirecting to /login, and add
    OpenGraph/Twitter meta tags using the existing board thumbnail so shared links unfurl with the
    actual board image.
  - **Why:**
    public_project_page returns RedirectResponse('/login') for user None (line 3344-3345), so a
    shared community link is a login wall — the like/clone/view loop (all already built) can never
    recruit new users. The file-serving layer is already token-gated and stateless
    (render_serving.py), so minting a read-only token for a verified-public project is a contained
    change with outsized acquisition value.

- [ ] **E7** REST API with personal access tokens (submit brief, poll status, fetch zip)
  - **Area:** kicraft/server/ (new routes module alongside render_serving.py), kicraft/server/accounts.py (users + build_jobs queue), stage_driver.py · **Impact:** medium · **Effort:** large
  - **What:**
    Token-authed JSON endpoints: POST /api/v1/designs (brief -> project, quota-checked), GET
    /api/v1/designs/{id} (stage statuses derived from state.json + projects row), GET
    /api/v1/designs/{id}/package (the fab zip), plus token management on /profile. Runs through the
    same _run_design/build_jobs path the UI uses.
  - **Why:**
    The only non-UI HTTP surface today is /billing/webhook (grep of @app.get/@app.post in web.py);
    everything else is NiceGUI pages. Yet the architecture is already API-shaped — files-as-IPC via
    state.json, a SQLite build queue, quota_status enforcement in accounts.py — so an API is mostly
    plumbing plus auth. It unlocks CI/scripting users and hardware-tool integrations that a chat UI
    can't serve.

- [ ] **E8** Project rename, notes, and search on the My Projects page
  - **Area:** kicraft/server/web.py projects_page (lines 2603-2800) + kicraft/server/accounts.py projects table · **Impact:** medium · **Effort:** small
  - **What:**
    Inline rename of the display name (keeping project_stem as the immutable file-set stem), an
    optional one-line note, and a search/filter box over the row list (name, board code, brief,
    status).
  - **Why:**
    Project rows show only the auto-generated project_stem, board code, status, and timestamp,
    newest-first with no filtering; grep shows no rename anywhere in web.py/accounts.py (the only
    'rename' hit is a parts-catalog comment). Accounts already has FTS infrastructure for the
    community browser (reindex_search at line 2787), and users iterating on variants of the same
    brief get indistinguishable stems.

- [ ] **E9** Design revision history with rollback after stage edits
  - **Area:** kicraft/server/stage_driver.py (stage-commit called with --no-archive, line 533), kicraft/server/web.py _do_rerun (~4687) + build_edit_panel · **Impact:** medium · **Effort:** medium
  - **What:**
    Snapshot state.json (and the artifacts pointer) before each edit-rerun/answer-resume into
    .kicraft/revisions/, and add a 'Versions' expander next to 'Edit a stage & re-run' listing prior
    commits with per-stage diff summary and a one-click 'restore this version' that recommits the
    old slots and re-runs synthesis/build.
  - **Why:**
    _do_rerun calls null_downstream and overwrites the committed slots destructively, and the web
    driver explicitly commits with --no-archive (stage_driver.py:533) even though the CLI's stage-
    commit supports archiving. An edit that makes the board worse (common with the 'Ask agent & re-
    run' freeform mode) is currently unrecoverable except by re-typing the brief — a real trust cost
    given each re-run spends quota and minutes.

- [ ] **E10** Download the KiCad project directly from community project pages
  - **Area:** kicraft/server/web.py public_project_page (3338-3434) + accounts.py (projects.zip_path) · **Impact:** medium · **Effort:** small
  - **What:**
    Add a 'Download KiCad project (.zip)' button to /p/{id} for completed public projects (served
    from the row's zip_path, same as the owner's Download in projects_page line 2752-2755), counted
    alongside views/clones as a community metric.
  - **Why:**
    The public page renders the schematic, board, BOM, likes, views, and Clone — but the only way to
    get the files is to clone into your own quota'd workspace and then download. The zip already
    exists on disk per project (p.zip_path) and the owner-facing Download button proves the serving
    path; for a 'browse and reuse' community this is the single most-expected missing action.

## 5. Engineering improvements

- [ ] **I1** Make CI exercise what actually ships: install runtime extras + KiCad 9 in the test job
  - **Area:** .github/workflows/ci.yml · **Impact:** high · **Effort:** medium
  - **What:**
    The test matrix job runs `pip install -e ".[dev]"` against a package whose base dependencies are
    empty ([project].dependencies = [] in pyproject.toml), so nicegui/requests/pydantic/stripe are
    absent: `kicraft.server.web` is unimportable and the ~20 test_web_*.py modules that import it at
    module scope (e.g. tests/test_web_browse.py line 16, `from kicraft.server import web` with no
    importorskip) cannot even collect, while 21 test files that `importorskip("pcbnew")` silently
    skip because KiCad is never installed. Fix: install `.[dev,design,server,tuning,loadtest]` in
    the test job (the security job already does exactly this, with an in-file comment admitting the
    skip problem), add the KiCad 9 PPA install that deploy/DEPLOY.md already scripts for the box so
    pcbnew-gated autoplacer tests run, and add a `pytest -rs`-based skip-count assertion so a silent
    skip regression is visible. Optionally add pytest-cov reporting in the same change (no coverage
    config exists anywhere today).
  - **Why:**
    Read ci.yml (test job installs only [dev]; security job comment: 'Full extras so the abuse tests
    actually run rather than skip'), pyproject.toml (empty base deps, stripe in `server`, matplotlib
    in `experiment`), counted 21 pcbnew importorskip files in tests/, and confirmed
    tests/test_web_browse.py has a bare top-level web import. The place/route engine that CLAUDE.md
    calls the product's actual value is never tested in CI.

- [ ] **I2** Burn the 27-failure local test baseline down to zero and retire the stash-diff regression gate
  - **Area:** tests/ + docs/plans/refactor-handoff-remaining.md · **Impact:** high · **Effort:** medium
  - **What:**
    docs/plans/refactor-handoff-remaining.md institutionalizes a red suite: the merge gate is 'new
    FAILED set ⊆ those 27' verified by a manual `git stash` diff of FAILED lines. The 27 are
    enumerated there: ModuleNotFound for matplotlib/stripe (both have extras — convert to
    pytest.importorskip or add to dev extra), test_kicraft_lookup_lcsc data-asserts ('parts-library'
    == 'easyeda', an offline-catalog difference — make the fixture hermetic), test_kicraft_stage_cli
    'unrecognized arguments' (a real CLI drift), and 2 stale tests (test_pro_and_max_limits,
    test_worker_shutdown_requeues in test_build_queue.py). Fix or explicitly skip-with-reason each
    so plain `pytest` is green. This directly unblocks roadmap Phase 3d: the cli_app.py parts_cli
    cut is frozen 'once its CLI tests are green (they fail on env-data today, so the move can't be
    verified)'.
  - **Why:**
    The handoff doc's own words make a permanently-red baseline the gate, and the roadmap names red
    tests as the blocker for the next cli_app extraction. A green suite turns every future refactor
    step's verification from a fragile diff ritual into `pytest` exit code 0, and lets CI become a
    real merge gate.

- [ ] **I3** Converge on one process manager: install the systemd units and make restart scripts systemctl wrappers
  - **Area:** deploy/ · **Impact:** high · **Effort:** small
  - **What:**
    The box currently has split-brain process management: `systemctl` shows kicraft-web.service
    ENABLED but INACTIVE, kicraft-build-worker.service exists in deploy/ but is NOT installed (is-
    enabled → not-found), while the real web+worker run as detached `setsid nohup` processes started
    by deploy/restart-web.sh / restart-build-worker.sh, which stop instances via pgrep/pkill -9
    pattern matching. Consequences: after a reboot systemd starts a web instance (possible port
    fight with a script-started one) and NO build worker — the web app then silently falls back to
    in-process builds (documented in build_worker.py's docstring). Fix: install both units, convert
    the restart scripts to `systemctl restart` + readiness wait, and delete the pkill logic (auto-
    memory already flags pkill -f kicraft.server.web as having killed the live :8080 instance). This
    also fixes the DEPLOY.md drift — its §4 documents the systemd path that ops no longer follow.
  - **Why:**
    Verified live on the box: `systemctl is-active kicraft-web` → inactive while `pgrep -af
    kicraft.server` shows both processes running detached; `systemctl is-enabled kicraft-build-
    worker` → not-found; read both restart scripts' pgrep/pkill logic and DEPLOY.md §4's systemd
    instructions.

- [ ] **I4** Journal events.jsonl incrementally instead of write-once at finalize
  - **Area:** kicraft/server/web.py (_persist_project / _finalize_orphan) · **Impact:** medium · **Effort:** small
  - **What:**
    events.jsonl — the full design timeline including LLM reasoning — is written only when a run
    finalizes: _persist_project (web.py ~1407) opens it with mode "w" and dumps state["events"] from
    memory. Synthesis runs in-process in a web-app thread, so a web crash or deploy restart mid-run
    loses the entire trace; worse, _finalize_orphan (web.py ~205) reconciles orphaned runs with
    `"events": []`, permanently overwriting the project with an EMPTY timeline. Change the event
    sink to append each event to `<project>/events.jsonl` as it is emitted (the project dir already
    exists from design start under build-in-place), making _persist_project's dump a no-op/fsync and
    making _finalize_orphan preserve whatever was journaled. This is also the durable event source
    Phase 4b needs for 'is it still running?' reconciliation.
  - **Why:**
    Read _persist_project (open("w") at finalize only), the _finalize_orphan orphan path that
    constructs state with events=[], and the reopen path (~4813) that renders the timeline purely
    from events.jsonl — so every mid-run restart today produces a blank or empty timeline for a real
    user project.

- [ ] **I5** Do Phase 4b (single source of truth for project/run state) as the next roadmap step
  - **Area:** kicraft/server/web.py + accounts.py (roadmap Phase 4b) · **Impact:** high · **Effort:** large
  - **What:**
    Pick one owner for 'what state is this project in / is it live?' — today it is smeared across
    the projects table, state.json, the in-process _LIVE_RUNS dict, and the build_jobs queue
    (roadmap Phase 4b's own words: 'the root of the reopen-is-missing-things / is-it-still-running
    bug family'). The handoff's 3b analysis shows this is also the gate for the two remaining big
    extractions: build_orchestration.py is blocked because tests rebind web._LIVE_RUNS (×9) and
    web._persist_project (×2) — '_LIVE_RUNS's one home IS the Phase 4b decision' — and
    project_view.py (3c) is deferred behind Phase 4. With Phase 4a fully done and verified, 4b is
    the highest-leverage remaining structural change: it removes a concept, fixes a recurring bug
    family, and unblocks ~500+ lines of clean extraction from the monolith.
  - **Why:**
    Read refactor-roadmap.md Phase 4b and refactor-handoff-remaining.md 3b (the AST-closure evidence
    and the explicit 'do this after Phase 4b' reversal). The orphan-reaper/_LIVE_RUNS code I read in
    web.py confirms the four-way state smear is still live.

- [ ] **I6** Add a monolith size ratchet to CI so Phase-3 gains stop silently eroding
  - **Area:** CI + kicraft/server/web.py, kicraft/design/cli_app.py · **Impact:** medium · **Effort:** small
  - **What:**
    The roadmap's hard-won reductions are regressing: web.py was cut 7,292 → 5,029/5,076 across
    Phases 3–4a but now measures 5,613 lines; cli_app.py was baselined at 3,913 and now measures
    5,304 (+36%) with its parts_cli cut still pending. Add a tiny CI step (or pre-commit check) with
    a checked-in ratchet file mapping tracked monoliths to their max allowed line counts, failing
    when a file exceeds its ratchet and auto-lowering when it shrinks. Also refresh the stale counts
    in CLAUDE.md/roadmap from the same source so the map agents load each session stays honest.
  - **Why:**
    Measured `wc -l` today (web.py 5,613, cli_app.py 5,304) against the roadmap's recorded baselines
    and post-cut counts; the roadmap's whole premise is that these files' size caused multi-hour
    agent-session failures, yet nothing guards the regained headroom.

- [ ] **I7** Gate CI security scans on NEW findings via a checked-in baseline instead of `|| true`
  - **Area:** .github/workflows/ci.yml + kicraft/security/scans · **Impact:** medium · **Effort:** small
  - **What:**
    The security job runs `python -m kicraft.security.scans || true` (comment: 'Never fails the
    build today') and gitleaks with continue-on-error — so a PR can introduce a fresh bandit
    finding, CVE, or committed secret without failing CI. Rather than waiting for the full backlog
    triage, implement the ratchet the file itself promises: have kicraft.security.scans emit a
    normalized findings list, check in the current set as a baseline, and fail the job only on
    findings not in the baseline (shrinking the baseline as /admin/security triage progresses). Same
    pattern for gitleaks: at minimum drop continue-on-error since new secrets are always
    regressions.
  - **Why:**
    Read ci.yml's security job verbatim, including its own comment 'flip to gating once the backlog
    is clean' — a baseline ratchet gets the gating benefit now without blocking on the backlog, and
    the /admin/security triage surface referenced in the comment already exists to manage the
    baseline.

- [ ] **I8** Persist build-side phase timings to a queryable store (build_runs), mirroring stage_runs
  - **Area:** kicraft/cli/autoexperiment.py + kicraft/server/spend_guard.py · **Impact:** medium · **Effort:** medium
  - **What:**
    Synthesis has real durable telemetry: spend_guard.py's stage_runs table records
    wall_s/cpu_s/rounds/tool_calls per LLM stage (exercised by
    tests/test_stage_resource_telemetry.py). The place/route side already MEASURES equivalent data —
    autoexperiment.py builds per-round timing_breakdown dicts (solve/compose/route keys,
    round_total) — but writes them only into summary.json/hierarchical_summary.json inside the heavy
    per-project .experiments/ tree, so fleet-level questions ('where do the 30-minute freerouting
    timeouts burn time?', 'what phase regressed after this deploy?') require spelunking individual
    project dirs. Add a build_runs table (or one summary row per build keyed by project/board_code)
    written at build end from the winning round's timing_breakdown + rc/verdict, and chart it on the
    existing /admin dashboards (ui.echart).
  - **Why:**
    Read the timing_breakdown plumbing in autoexperiment.py (lines ~100–160, 1729–1806, 2503–2918)
    and the stage_runs schema/insert in spend_guard.py; docs/plans/stage-resource-telemetry-and-bom-
    cost.md shows the pattern was deliberately built for synthesis and simply never extended to the
    build half, which is where the wall-clock actually goes (freerouting timeout incidents in
    memory).

- [ ] **I9** Centralize the reload-safe web test harness as a shared fixture
  - **Area:** tests/ (web harnesses) + kicraft/server/web.py registration block · **Impact:** medium · **Effort:** small
  - **What:**
    Six web test harnesses (test_web_core_components, _index_autoopen, _pricing, _projects_page,
    _layout_editor, _support_reports) each do importlib.reload(web) to get a fresh app + route
    table, and the handoff documents the trap this created: a reloaded web.py does not re-run
    already-imported submodules, so @ui.page routes registered by routes_admin 404 only in the full
    suite — fixed by an _ADMIN_ROUTES_REGISTERED flag + conditional reload inside web.py. Every
    future Phase-3 extraction of @ui.page handlers (project_view is next) re-runs this minefield.
    Extract one shared pytest fixture in tests/conftest.py (fresh_web_app) that performs the reload
    plus submodule route re-registration in one place, and have web.py expose a single re-register
    hook instead of per-module flags.
  - **Why:**
    Read handoff 3a's 'non-obvious trap' section (it cost the only red gate of the refactor) and
    confirmed tests/conftest.py currently contains only the lcsc-retail guard — the reload pattern
    is copy-pasted per harness with the workaround living as a special-case flag in web.py.

- [ ] **I10** Fix deploy-doc drift and finish the Phase 5 docs prune
  - **Area:** deploy/DEPLOY.md, README.md, docs/ · **Impact:** low · **Effort:** small
  - **What:**
    Three concrete doc gaps: (1) DEPLOY.md §4 instructs running under systemd (`systemctl enable
    --now kicraft-web`) while actual operations use deploy/restart-*.sh setsid instances — whichever
    way the process-manager convergence goes, make the doc match reality, and document the build
    worker + jlcparts timer as first-class deploy steps (the worker unit install is currently absent
    from the runbook flow). (2) Roadmap Phase 5's named leftovers: README still carries GUI-coupled
    feature prose (leaf promotion 'GUI-only', Setup tab, searchable-params tab) for a GUI deleted
    2026-06-22. (3) docs/ top level mixes specs with one-off artifacts (bug-replay-no-route-
    promotes-stale-routed-board.md, electrical_review_model_bakeoff.pdf) that Phase 5 says to prune
    or fold; stale docs are load-bearing here because agents navigate by them (the roadmap's own
    Phase 1 found a doc instructing every agent to launch a deleted module).
  - **Why:**
    Read DEPLOY.md §4 vs the restart scripts and live process state; refactor-roadmap.md Phase 5
    'Remaining' explicitly lists the README prose and loose docs; listed docs/ and found the one-off
    files at top level.

## Appendix: refuted during review

- `kicraft/server/web.py:308` — _client_ip trusts the leftmost X-Forwarded-For value, which the client fully controls, so the per-IP signup throttle and per-IP abuse counters are bypassable by header spoofing.
  - **Why refuted:**
    The code does trust the leftmost X-Forwarded-For (web.py:308-311), but the finding's exploit
    premise -- 'Caddy APPENDS the real peer, leaving <attacker-supplied>, <caddy-ip>' -- is wrong
    for the deployed proxy. Since Caddy 2.5.0 (reverseproxy #4507), incoming X-Forwarded-* headers
    from a peer NOT in trusted_proxies are discarded, and this Caddyfile configures no
    trusted_proxies; the box runs caddy v2.11.4, so the app receives X-Forwarded-For containing only
    the genuine client IP and _client_ip returns the real address. The bypass would require hitting
    :8080 directly with a forged header, which is blocked twice: the service binds
    KICRAFT_WEB_HOST=127.0.0.1 and ufw admits only 22/80/443. The signup throttle therefore keys on
    the real IP in this deploy. (The leftmost-XFF pattern remains fragile for other topologies, but
    as stated the finding's failure scenario does not occur.)

