"""Run one arm of the architecture correction-ladder experiment.

Operator harness for `docs/plans/architecture-contract-correction-ladder.md`
§3: replay the frozen pre-architecture state of a board through the REAL
provider and stage pipeline once per run, under exactly one
`KICRAFT_CONTRACT_LADDER` arm, and write every artifact the plan's metrics need.

Per run it records, into `<out>/runs.jsonl`:

  commit / failure_kind / attempts / rungs (the call-mode ladder)
  defects per rung (the blocking diagnostic codes the drive was shown)
  dropped declared content (nets/ports present in rung i, absent in rung i+1
  with no diagnostic naming them)
  cost, wall seconds, and the raw event + attempt traces (separate files)

Usage (one arm at a time; arms share the process-wide spend guard):

    .venv/bin/python tools/ladder_experiment.py \
        --arm preserving --state ~/.kicraft/projects/1/825/.kicraft/state.json \
        --runs 10 --out /tmp/ladder-exp

    .venv/bin/python tools/ladder_experiment.py --summary /tmp/ladder-exp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kicraft.design.architecture_intent import EDGE_PREFIX  # noqa: E402
from kicraft.server.config import Settings, parse_contract_ladder  # noqa: E402
from kicraft.server.stage_pipeline import drive_replay  # noqa: E402
from kicraft.server.stage_runtime import NONINTERACTIVE_DEFAULTS_INSTRUCTION  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _diagnostic_codes(event: dict) -> list[str]:
    row = event.get("diagnostic") or {}
    codes = [str(row["code"])] if row.get("code") else []
    codes.extend(
        str(item["code"])
        for item in row.get("evidence") or []
        if isinstance(item, dict) and item.get("code")
    )
    return codes


def _rejection_text(event: dict) -> str:
    return " ".join(
        str(part)
        for part in (event.get("schema_error"), json.dumps(event.get("diagnostic") or {}))
        if part
    )


def _identity_name(identity: str) -> str:
    return identity.split(":", 1)[1]


def _run_metrics(stage_result: dict, events: list[dict], trace: list[dict]) -> dict:
    retries = [event for event in events if event.get("kind") == "retry"]
    identities = [
        event.get("declared_identities") or []
        for event in retries
        if event.get("declared_identities") is not None
    ]
    dropped: list[dict] = []
    for index in range(1, len(identities)):
        previous, current = set(identities[index - 1]), set(identities[index])
        text = _rejection_text(retries[index])
        lost = sorted(
            _identity_name(item) for item in previous - current if _identity_name(item) not in text
        )
        if lost:
            dropped.append({"rung": index + 1, "items": lost})
    return {
        "commit": bool(stage_result and stage_result.get("commit_ok")),
        "failure_kind": (stage_result or {}).get("failure_kind"),
        # A parked stage asked the user instead of committing: the row has to say
        # so, or a park reads like an ordinary failure (next-steps plan §4 B2).
        "needs_input": bool((stage_result or {}).get("needs_input")),
        "questions": len((stage_result or {}).get("questions") or []),
        "attempts": (stage_result or {}).get("attempts"),
        "rungs": [
            {
                "attempt": row.get("provider_attempt"),
                "mode": row.get("call_mode"),
                "outcome": row.get("outcome"),
            }
            for row in trace
        ],
        "defects": [
            {
                "attempt": row.get("provider_attempt"),
                "call_mode": row.get("call_mode"),
                "failure_kind": row.get("failure_kind"),
                "codes": _diagnostic_codes(row),
            }
            for row in retries
        ],
        "dropped_content": dropped,
        "cost_usd": (stage_result or {}).get("cost_usd"),
        "wall_s": (stage_result or {}).get("wall_s"),
    }


def _spent_total(owner) -> float:
    guard = getattr(owner, "guard", None)
    if guard is None:
        return 0.0
    status = guard.status() if callable(getattr(guard, "status", None)) else {}
    return float((status or {}).get("spent_total_usd") or 0.0)


def run_arm(args) -> int:
    try:
        parse_contract_ladder(args.arm)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.environ["KICRAFT_CONTRACT_LADDER"] = args.arm
    label = args.label or args.arm
    state = Path(args.state).expanduser()
    board = args.board or state.parent.parent.name
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    manifest = {
        "arm": args.arm,
        "label": label,
        "board": board,
        "state": str(state),
        "runs_requested": args.runs,
        "budget_per_run_usd": args.budget,
        "max_retries": args.max_retries,
        "ceiling_usd": args.ceiling,
        "repo_head": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain")),
        "contract_ladder": os.environ["KICRAFT_CONTRACT_LADDER"],
        "settings": Settings.from_env().redacted(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / f"manifest-{label}-{board}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"arm={args.arm} board={board} runs={args.runs} budget=${args.budget} "
        f"ceiling=${args.ceiling} head={manifest['repo_head'][:12]}"
    )

    spent_start = None
    committed = 0
    completed = 0
    with runs_path.open("a", encoding="utf-8") as sink:
        for index in range(1, args.runs + 1):
            events: list[dict] = []
            trace: list[dict] = []
            from kicraft.server.stage_pipeline import make_budget_client

            client = make_budget_client(args.budget)
            if spent_start is None:
                spent_start = _spent_total(client)
            started = time.monotonic()
            # One unique artifact pair per run: an interleaved driver invokes this
            # script once per run with `--runs 1`, so a label+index name would
            # overwrite the previous run's events and trace.
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            events_path = out / f"{label}-{board}-{stamp}.events.jsonl"
            trace_path = out / f"{label}-{board}-{stamp}.trace.jsonl"
            record: dict = {
                "arm": args.arm,
                "label": label,
                "board": board,
                "run": index,
                "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "artifacts": {"events": str(events_path), "trace": str(trace_path)},
            }
            try:
                result = drive_replay(
                    state,
                    "architecture",
                    budget_usd=args.budget,
                    max_retries=args.max_retries,
                    progress=events.append,
                    attempt_observer=trace.append,
                    client=client,
                    instruction=(
                        NONINTERACTIVE_DEFAULTS_INSTRUCTION if args.unattended else None
                    ),
                )
                events_path.write_text(
                    "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
                    encoding="utf-8",
                )
                trace_path.write_text(
                    "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in trace),
                    encoding="utf-8",
                )
                record.update(_run_metrics(result.get("stage"), events, trace))
                record["error"] = result.get("error")
            except Exception as exc:  # noqa: BLE001 - every aborted run is reported
                record.update(
                    {
                        "commit": False,
                        "aborted": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "rungs": [
                            {
                                "attempt": row.get("provider_attempt"),
                                "mode": row.get("call_mode"),
                                "outcome": row.get("outcome"),
                            }
                            for row in trace
                        ],
                    }
                )
            record["wall_s_total"] = round(time.monotonic() - started, 3)
            record["spent_total_usd"] = _spent_total(client)
            record["spent_arm_usd"] = round(record["spent_total_usd"] - spent_start, 6)
            completed += 1
            committed += 1 if record.get("commit") else 0
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            sink.flush()
            print(
                f"  r{index:<3} {'COMMIT' if record.get('commit') else 'fail  '} "
                f"{record.get('failure_kind') or '-'} attempts={record.get('attempts')} "
                f"cost=${record.get('cost_usd') or 0:.4f} arm=${record['spent_arm_usd']:.4f}",
                flush=True,
            )
            if record.get("spent_arm_usd", 0.0) >= args.ceiling:
                print(f"  stop: arm reached the ${args.ceiling} ceiling")
                break
    print(f"arm {args.arm}: {committed}/{completed} committed")
    return 0


def summarise(out: Path) -> int:
    runs_path = out / "runs.jsonl"
    if not runs_path.is_file():
        print(f"no {runs_path}", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
    arms: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("label") or row["arm"])
        bucket = arms.setdefault(
            key, {"runs": 0, "commits": 0, "aborted": 0, "cost": 0.0, "boards": {}}
        )
        bucket["runs"] += 1
        bucket["commits"] += 1 if row.get("commit") else 0
        bucket["aborted"] += 1 if row.get("aborted") else 0
        bucket["cost"] += float(row.get("cost_usd") or 0.0)
        board = bucket["boards"].setdefault(row["board"], {"runs": 0, "commits": 0})
        board["runs"] += 1
        board["commits"] += 1 if row.get("commit") else 0
    print(f"{'arm':<18} {'commits/runs':>12} {'pooled':>8} {'cost':>9}  per board")
    for arm, bucket in sorted(arms.items()):
        pooled = bucket["commits"] / bucket["runs"] if bucket["runs"] else 0.0
        per_board = "  ".join(
            f"{board}:{value['commits']}/{value['runs']}"
            for board, value in sorted(bucket["boards"].items())
        )
        print(
            f"{arm:<18} {bucket['commits']:>5}/{bucket['runs']:<6} {pooled:>8.2f} "
            f"${bucket['cost']:>8.4f}  {per_board}"
        )
    return 0


def scan_corpus(roots: list[Path], since: float | None = None) -> int:
    """Count terminal-by-policy stage deaths across local event streams.

    Uses the production reader (`triage.collect_stages`) rather than a second
    detector, so the number matches what `triage stages` reports for one board.
    """
    from kicraft.cli.triage import collect_stages, resolve_projects_dir

    projects = roots or [resolve_projects_dir()]
    streams = sorted(
        path
        for root in projects
        for pattern in ("*/*/events.jsonl", "*/events.jsonl")
        for path in root.glob(pattern)
    )
    counts: dict[str, dict[str, int]] = {}
    scanned = 0
    for stream in streams:
        if since is not None and stream.stat().st_mtime < since:
            continue
        run = stream.parent
        try:
            stages = collect_stages(run)
        except Exception as exc:  # noqa: BLE001 - one unreadable run must not stop the scan
            print(f"  skipped {run}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        scanned += 1
        for row in stages.get("stages") or []:
            stage = str(row.get("stage"))
            bucket = counts.setdefault(stage, {"runs": 0, "terminal_by_policy": 0})
            bucket["runs"] += 1
            if not row.get("ok") and row.get("terminal_by_policy"):
                bucket["terminal_by_policy"] += 1
    print(f"scanned {scanned} event streams under {[str(root) for root in projects]}")
    print(f"{'stage':<18} {'terminal_by_policy':>19} {'streams':>8}")
    for stage, bucket in sorted(counts.items()):
        print(f"{stage:<18} {bucket['terminal_by_policy']:>19} {bucket['runs']:>8}")
    return 0


def run_full(args) -> int:
    """Drive all five stages + the deterministic build for a fresh brief."""
    from kicraft.server.stage_pipeline import run_pipeline
    from kicraft.server.stage_state import DESIGN_STAGES

    try:
        parse_contract_ladder(args.arm)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.environ["KICRAFT_CONTRACT_LADDER"] = args.arm
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    brief = args.brief or (Path(args.state).expanduser().parent.parent / "brief.txt").read_text(
        encoding="utf-8"
    )
    for index in range(1, args.runs + 1):
        workspace = out / f"full-{args.arm}-r{index}"
        result = run_pipeline(
            brief.strip(),
            workspace,
            stages=DESIGN_STAGES,
            budget_usd=args.budget,
            max_retries=args.max_retries,
            build=True,
        )
        record = {
            "arm": args.arm,
            "run": index,
            "workspace": str(workspace),
            "all_committed": result["all_committed"],
            "build_rc": result["build_rc"],
            "cost_usd": sum(float(stage.get("cost_usd") or 0.0) for stage in result["stages"]),
            "stages": [
                {
                    "stage": stage.get("stage"),
                    "commit_ok": stage.get("commit_ok"),
                    "attempts": stage.get("attempts"),
                    "failure_kind": stage.get("failure_kind"),
                }
                for stage in result["stages"]
            ],
        }
        (out / "full-runs.jsonl").open("a", encoding="utf-8").write(
            json.dumps(record, separators=(",", ":")) + "\n"
        )
        print(
            f"  full r{index}: all_committed={record['all_committed']} "
            f"build_rc={record['build_rc']} cost=${record['cost_usd']:.4f}"
        )
    return 0


def _draft_payloads(path: Path) -> list[dict]:
    """Every provider draft in one run's event stream, in order.

    The stream holds the model's raw answer as `answer_delta` chunks and a
    `retry` event per rejection, so the drafts are the answers between retries.
    Chunks that never completed a JSON object (an aborted loop, a truncated
    stream) are dropped — they were never a draft the reader saw.
    """
    drafts: list[dict] = []
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("kind") == "answer_delta":
            chunks.append(str(event.get("text") or ""))
        elif event.get("kind") == "retry" and chunks:
            drafts.append(_parse_draft("".join(chunks)))
            chunks = []
    if chunks:
        drafts.append(_parse_draft("".join(chunks)))
    return [draft for draft in drafts if draft is not None]


def _parse_draft(text: str) -> dict | None:
    from kicraft.server.stage_contracts import _extract_json

    try:
        payload = _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _project_to_intent(architecture: dict) -> tuple[dict, list[str]]:
    """Reverse-project a committed/explicit draft onto the intent-shaped slot.

    The corpus predates the intent slot, so the offline replay has to state the
    same design in the new shape: sheets and parts with their supply rails,
    signals with a source port and peer ports, and the rails with the port that
    generates them. It is deliberately lossy in one direction only — information
    the draft never had cannot be projected — so a draft the projection cannot
    express is reported as such, never counted as a pass.
    """
    from kicraft.design.lowering import registered_lowerers
    from kicraft.design.recipes.registry import registered_recipes
    from kicraft.design.recipes.resolver import _PORT_ALIASES, _net_identity

    recipes = {}
    for registered in registered_recipes():
        definition = registered.definition
        if definition.family:
            recipes.setdefault(definition.family, definition)
        if definition.exact_part:
            recipes.setdefault(definition.exact_part.casefold(), definition)
    lowerer_families = {family for row in registered_lowerers() for family in row.families}

    def definition_of(row: dict):
        return recipes.get(row.get("family") or "") or recipes.get(
            str(row.get("exact_part") or "").casefold()
        )

    def ports_of(row: dict) -> dict[str, str]:
        definition = definition_of(row)
        return {port.name: port.direction for port in definition.ports} if definition else {}

    sheets = architecture.get("sheets") or []
    requirements = [row for row in architecture.get("requirements") or [] if isinstance(row, dict)]
    nets = [row for row in architecture.get("inter_sheet_nets") or [] if isinstance(row, dict)]
    rail_voltages = {
        str(name): float(value) for name, value in (architecture.get("rail_voltages") or {}).items()
    }
    rails = {
        name: volts
        for name, volts in rail_voltages.items()
        if name != "GND" and name in set(architecture.get("power_nets") or [])
    }
    by_sheet: dict[str, list[dict]] = {}
    for row in requirements:
        by_sheet.setdefault(str(row.get("sheet")), []).append(row)
    endpoints_by_net = {
        str(net.get("name")): [row for row in net.get("endpoints") or [] if isinstance(row, dict)]
        for net in nets
    }

    def _key(raw: object) -> str:
        """A port key the intent slot accepts (the drafts predate the key pattern)."""
        return re.sub(r"[^a-z0-9_]", "_", str(raw).lower()).strip("_") or "pin"

    def _port_key(row: dict, port: str) -> str | None:
        """The intent-slot key for one draft port, or None when the draft has none.

        A recipe-advertised key (or any lowerer key) travels verbatim. A draft that
        named an MCU application pin bare (`r0`, `led`) — the older explicit-slot
        style — becomes the capability-prefixed key the allocator expects, chosen
        from the direction the bound net leaves that sheet. A port the selected
        recipe does not have and no capability spelling covers is NOT expressible
        in the intent slot, and the projection reports it instead of inventing one.
        """
        family = str(row.get("family") or "")
        directions = ports_of(row)
        if port in directions or family in lowerer_families or not directions:
            return _key(port)
        for canonical, aliases in _PORT_ALIASES.items():
            if canonical in directions and _net_identity(port) in {
                _net_identity(alias) for alias in (canonical, *aliases)
            }:
                return canonical
        definition = definition_of(row)
        if not (definition and definition.allocatable_pins):
            return None  # no allocatable pin can carry this key: it is not expressible
        if re.match(r"^(?:input|output|touch|parallel|pwm|adc|gpio\d)", port):
            return _key(port)  # already a capability-shaped application key
        net = (row.get("ports") or {}).get(port)
        direction = next(
            (
                str(endpoint.get("direction"))
                for endpoint in endpoints_by_net.get(str(net), [])
                if endpoint.get("sheet") == row.get("sheet")
            ),
            "output",
        )
        return _key(f"{'input' if direction == 'input' else 'output'}_{port}")

    def binding(row: dict, net: str, *, want: str | None = None) -> str | None:
        directions = ports_of(row)
        for port, value in (row.get("ports") or {}).items():
            if value != net:
                continue
            if want is None or directions.get(port) == want:
                return port
        return None

    def owner(sheet: str, net: str, *, want: str | None = None) -> dict | None:
        for row in by_sheet.get(sheet, []):
            if binding(row, net, want=want):
                return row
        return None

    power_rails: dict[str, dict] = {}
    for net in {net for row in requirements for net in (row.get("ports") or {}).values()}:
        if net not in rails:
            continue
        source = None
        for endpoint in endpoints_by_net.get(net, []):
            if endpoint.get("direction") != "output":
                continue
            row = owner(str(endpoint.get("sheet")), net)
            port = binding(row, net) if row else None
            key = _port_key(row, port) if row is not None and port else None
            if key:
                source = f"{row.get('id')}.{key}"
                break
        power_rails[net] = {"voltage": rails[net], "from": source}

    projected_requirements = []
    for row in requirements:
        directions = ports_of(row)
        known = bool(directions) or str(row.get("family") or "") in lowerer_families
        supply_port = next(
            (
                port
                for port in ("vdd", "vm", "vin", "input", "vdd_5v")
                if (row.get("ports") or {}).get(port) in power_rails
            ),
            None,
        )
        ties = {
            key: net
            for port, net in (row.get("ports") or {}).items()
            if ((net == "GND" and port != "gnd") or net in power_rails)
            and (key := _port_key(row, port)) is not None
        }
        if supply_port:
            ties.pop(_port_key(row, supply_port), None)
        projected = {
            "id": row.get("id"),
            "sheet": row.get("sheet"),
            "role": row.get("role"),
            "family": row.get("family"),
            "exact_part": row.get("exact_part"),
            "parameters": dict(row.get("parameters") or {}),
            "supply": (row.get("ports") or {}).get(supply_port) if supply_port else None,
            "interfaces": list(row.get("interfaces") or []),
            "functional_blocks": list(row.get("functional_blocks") or []),
            "ties": ties,
            "declared_ports": [],
        }
        if not known:
            # An uncurated part: the draft's own port bindings are the only interface
            # statement it carries, so they become the declared interface.
            projected["declared_ports"] = [
                {
                    "key": _port_key(row, port) or _key(port),
                    "direction": (
                        directions.get(port)
                        or next(
                            (
                                str(endpoint.get("direction"))
                                for endpoint in endpoints_by_net.get(net, [])
                                if endpoint.get("sheet") == row.get("sheet")
                            ),
                            "bidirectional",
                        )
                        or "bidirectional"
                    ),
                    "function": f"projected from the draft's {port!r} binding",
                }
                for port, net in (row.get("ports") or {}).items()
            ]
        projected_requirements.append(projected)

    signals: list[dict] = []
    dropped_ports: list[str] = []
    known_sheet_stems = {str(row.get("stem")) for row in sheets}

    def _label(net: str) -> str:
        label = re.sub(r"[^A-Z0-9_]", "_", net.upper()).strip("_") or "EDGE"
        return f"{label}_EDGE" if label in known_sheet_stems else label

    def _ref(row: dict, port: str) -> str | None:
        key = _port_key(row, port)
        if key is None:
            dropped_ports.append(f"{row.get('id')}.{port}")
        return f"{row.get('id')}.{key}" if key else None

    for net, endpoints in endpoints_by_net.items():
        if net in power_rails or net == "GND":
            continue
        owners_by_sheet = {}
        for endpoint in endpoints:
            sheet = str(endpoint.get("sheet"))
            row = owner(sheet, net)
            port = binding(row, net) if row else None
            owners_by_sheet[sheet] = (endpoint, row, port)
        source_sheet = next(
            (
                sheet
                for sheet, (endpoint, row, port) in owners_by_sheet.items()
                if row is not None and port and endpoint.get("direction") == "output"
            ),
            None,
        )
        if source_sheet is None:
            # A bidirectional or peer-typed net (USB, a bus) has no output endpoint:
            # the first sheet that owns a binding on it is the source, the rest peers.
            source_sheet = next(
                (
                    sheet
                    for sheet, (_endpoint, row, port) in owners_by_sheet.items()
                    if row is not None and port
                ),
                None,
            )
        if source_sheet is None:
            continue
        _endpoint, source_row, source_port = owners_by_sheet[source_sheet]
        source_ref = _ref(source_row, source_port)
        if source_ref is None:
            continue
        peer_refs = [
            ref
            for sheet, (_endpoint, row, port) in owners_by_sheet.items()
            if sheet != source_sheet
            for ref in [
                _ref(row, port) if row is not None and port else f"{EDGE_PREFIX}{_label(sheet)}"
            ]
            if ref is not None
        ]
        if not peer_refs:
            continue
        signals.append({"name": net, "from": source_ref, "to": peer_refs})

    # A net only one sheet binds is a signal that leaves the board: the draft's
    # dangling output (the class the O9 completion rescued) is stated as an edge.
    for row in requirements:
        for port, net in (row.get("ports") or {}).items():
            if net in endpoints_by_net or net == "GND" or net in power_rails:
                continue
            source_ref = _ref(row, port)
            if source_ref is None:
                continue
            signals.append({"name": net, "from": source_ref, "to": [f"{EDGE_PREFIX}{_label(net)}"]})

    known_sheets = {str(row.get("name")) for row in sheets}
    return {
        "topologies": dict(architecture.get("topologies") or {}),
        "comms_protocols": list(architecture.get("comms_protocols") or []),
        "mcu_present": bool(architecture.get("mcu_present")),
        "power": {"rails": power_rails},
        "sheets": [
            {
                "name": row.get("name"),
                "stem": row.get("stem"),
                "role": "interface",
                "function": row.get("function"),
            }
            for row in sheets
            if str(row.get("name")) in known_sheets
        ],
        "requirements": projected_requirements,
        "signals": signals,
        "assumptions": list(architecture.get("assumptions") or []),
    }, sorted(dropped_ports)


def _draft_codes(
    payload: dict, prompt_state: dict, slot: str = "explicit"
) -> tuple[list[str], str | None]:
    """The blocking codes one draft is rejected for, or the refusal that stopped it."""
    from kicraft.server.stage_contracts import StageSchemaError, _normalize_stage_response

    try:
        _normalize_stage_response(
            "architecture", json.loads(json.dumps(payload)), prompt_state, slot=slot
        )
    except StageSchemaError as exc:
        diagnostic = getattr(exc, "diagnostic", None) or {}
        if isinstance(diagnostic, dict) and diagnostic.get("code"):
            codes = [str(diagnostic["code"])]
            codes.extend(
                str(item["code"])
                for item in diagnostic.get("evidence") or []
                if isinstance(item, dict) and item.get("code")
            )
            return sorted(set(codes)), None
        return ["unclassified"], str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 - a projection failure is reported, never a pass
        return [], f"{type(exc).__name__}: {exc}"[:200]
    return [], None


def replay_corpus(root: Path, *, limit: int | None, state_root: Path) -> int:
    """Replay every saved first draft through the reader, before and after derivation.

    The offline half of the plan's §6 measurement: for every architecture run whose
    raw drafts are still on disk, count the blocking classes the FIRST draft hit
    under the explicit reader, project that draft onto the intent slot, derive it,
    and count the classes that remain. Classes the derivation owns disappear; a
    draft the derivation cannot rebuild is reported as skipped, never as a pass.
    """
    from kicraft.design.architecture_intent import ArchitectureIntentError, derive_architecture

    streams = sorted(root.glob("*.events.jsonl"))
    if limit is not None:
        streams = streams[:limit]
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    states: dict[str, dict] = {}
    rows: list[dict] = []
    for stream in streams:
        board = stream.name.split("-")[-2]
        state_path = state_root / board / ".kicraft" / "state.json"
        if not state_path.is_file():
            continue
        state = states.setdefault(board, json.loads(state_path.read_text(encoding="utf-8")))
        prompt_state = {
            "intent": state.get("intent"),
            "functional_spec": state.get("functional_spec"),
        }
        drafts = _draft_payloads(stream)
        if not drafts:
            continue
        first = drafts[0]
        draft_codes, unclassified = _draft_codes(first, prompt_state)
        for code in draft_codes:
            before[code] = before.get(code, 0) + 1
        skipped = None
        derived_codes: list[str] = []
        dropped: list[str] = []
        try:
            intent, dropped = _project_to_intent(first)
            derived = derive_architecture(intent).model_dump(exclude_none=True)
            derived_codes, unclassified = _draft_codes(derived, prompt_state)
        except ArchitectureIntentError as exc:
            skipped = f"refused: {sorted({row.code for row in exc.diagnostics})}"
            derived_codes = sorted({row.code for row in exc.diagnostics})
        except Exception as exc:  # noqa: BLE001 - one bad draft must not stop the replay
            skipped = f"{type(exc).__name__}: {exc}"[:200]
        for code in derived_codes:
            after[code] = after.get(code, 0) + 1
        rows.append(
            {
                "stream": stream.name,
                "drafts": len(drafts),
                "draft_codes": draft_codes,
                "derived_codes": derived_codes,
                "skipped": skipped,
                "unclassified": unclassified,
                "inexpressible_ports": dropped,
            }
        )

    def _expressible(row: dict) -> bool:
        return not row["skipped"] and not row["inexpressible_ports"]

    known = [row for row in rows if _expressible(row)]
    accepted_before = sum(1 for row in known if not row["draft_codes"] and not row["unclassified"])
    accepted_after = sum(1 for row in known if not row["derived_codes"])
    print(f"replayed {len(rows)} first drafts from {len(streams)} event streams")
    print(
        f"first drafts accepted: explicit reader {accepted_before}/{len(known)}; "
        f"after projection+derivation {accepted_after}/{len(known)} "
        f"({len(rows) - len(known)} draft(s) the projection could not express are excluded)"
    )
    print(f"{'blocking class':<42} {'explicit':>9} {'derived':>8}")
    for code in sorted(
        {*before, *after}, key=lambda name: -(before.get(name, 0) + after.get(name, 0))
    ):
        print(f"{code:<42} {before.get(code, 0):>9} {after.get(code, 0):>8}")
    inexpressible = sum(len(row["inexpressible_ports"]) for row in rows)
    if inexpressible:
        print(
            f"projection could not express {inexpressible} draft port(s) "
            "(reported per row in replay.jsonl)"
        )
    skipped = sum(1 for row in rows if row["skipped"])
    if skipped:
        print(f"{skipped} draft(s) the projection could not rebuild:")
        for row in rows:
            if row["skipped"]:
                print(f"  {row['stream']}: {row['skipped']}")
    (root / "replay.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", help="the KICRAFT_CONTRACT_LADDER arm to run")
    parser.add_argument("--state", help="frozen state.json (intent+functional_spec committed)")
    parser.add_argument("--board", help="board label for the artifact names (default: project id)")
    parser.add_argument(
        "--label",
        help="scoreboard label for these runs (default: the arm name); use it to keep a "
        "re-measurement or a knob variant separate from the arm's round-1 rows",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--budget", type=float, default=0.25, help="per-run USD cap")
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="declare the drive non-interactive (the caller's defaults instruction): a stage may "
        "not park for a user answer, so a brief with no user to ask measures the stage itself",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="the production caller value")
    parser.add_argument(
        "--ceiling", type=float, default=10.0, help="hard stop on the arm's cumulative spend"
    )
    parser.add_argument("--out", default="/tmp/ladder-exp", help="artifact directory")
    parser.add_argument("--summary", action="store_true", help="print the arm scoreboard and exit")
    parser.add_argument(
        "--full",
        action="store_true",
        help="drive all five stages + build (L2) instead of the architecture replay",
    )
    parser.add_argument("--brief", help="brief text for --full (default: the board's brief.txt)")
    parser.add_argument(
        "--since",
        help="--scan only: ignore event streams older than this ISO-8601 timestamp",
    )
    parser.add_argument(
        "--scan",
        nargs="*",
        metavar="PROJECTS_DIR",
        help="count terminal-by-policy stage deaths across local event streams and exit",
    )
    parser.add_argument(
        "--replay-corpus",
        nargs="?",
        const="/tmp/ladder-exp",
        metavar="EVENTS_DIR",
        help="replay every saved architecture draft through the reader, before and after the "
        "intent-slot derivation, and print the per-class counts",
    )
    parser.add_argument("--limit", type=int, help="--replay-corpus: only the first N streams")
    parser.add_argument(
        "--state-root",
        default=str(Path.home() / ".kicraft" / "projects" / "1"),
        help="--replay-corpus: where the frozen <board>/state.json files live",
    )
    args = parser.parse_args(argv)
    if args.replay_corpus is not None:
        return replay_corpus(
            Path(args.replay_corpus).expanduser(),
            limit=args.limit,
            state_root=Path(args.state_root).expanduser(),
        )
    if args.scan is not None:
        since = datetime.fromisoformat(args.since).timestamp() if args.since else None
        return scan_corpus([Path(item).expanduser() for item in args.scan], since)
    if args.summary:
        return summarise(Path(args.out).expanduser())
    if not args.arm:
        parser.error("--arm is required unless --summary/--scan is given")
    if args.full:
        return run_full(args)
    if not args.state:
        parser.error("--state is required for an architecture replay")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
