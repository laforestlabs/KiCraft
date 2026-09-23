"""Artifact-backed product acceptance for independently supplied briefs.

This evaluator deliberately has no knowledge of the built-in benchmark corpus.  Its
obligations are frozen by the external-brief manifest and are evaluated only from
facts extracted from the run workspace.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile
from typing import Any

from .artifact_evidence import _result_status, extract_artifact_facts
from .external_briefs import stable_hash
from .design_acceptance import evaluate_obligation

_AUDIT_PATH = Path("eval/product_acceptance.json")
_DRC_REPORT_PATH = Path("eval/product_drc_report.json")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUIRED_POLICY = (
    "max_cost_usd",
    "max_duration_s",
    "max_park_rounds",
    "build_timeout_s",
)


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _valid_policy(policy: Any) -> list[str]:
    if not isinstance(policy, Mapping):
        return ["invalid_policy: frozen policy is absent"]
    errors: list[str] = []
    for key in _REQUIRED_POLICY:
        value = policy.get(key)
        if key == "max_park_rounds":
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        else:
            valid = _finite_number(value, positive=True)
        if not valid:
            errors.append(f"invalid_policy: {key} must be a positive finite number")
    return errors


def _run_local_file(rundir: Path, relative_path: Any) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path:
        return None
    path = Path(relative_path)
    if path.is_absolute():
        return None
    candidate = (rundir / path).resolve()
    try:
        candidate.relative_to(rundir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _relative(rundir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(rundir.resolve()))


def _artifact_inventory(rundir: Path) -> list[str]:
    """Capture additions that can change delivered-board selection or DRC inputs."""
    generated = rundir / "generated"
    suffixes = {".zip", ".kicad_pro", ".kicad_dru", ".kicad_sch"}
    return sorted(
        str(path.relative_to(rundir))
        for path in generated.glob("*/*")
        if path.is_file()
        and (
            path.suffix in suffixes
            or path.name.endswith(".receipt.json")
            or (path.suffix == ".kicad_pcb" and path.stem == path.parent.name)
        )
    )


def _audit_context(record: Mapping[str, Any], obligations, policy) -> dict:
    return {
        "slug": record.get("slug"),
        "original_brief_hash": record.get("original_brief_hash"),
        "execution_brief_hash": record.get("execution_brief_hash"),
        "obligations_hash": stable_hash(obligations),
        "policy_hash": stable_hash(policy),
    }


def _audit_artifacts(
    rundir: Path, artifacts: Sequence[Mapping[str, Any]], facts: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Hash every run-local source artifact used by the product verdict."""
    paths: set[str] = set()
    for item in artifacts:
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            paths.add(item["path"])
    for key in ("drc_certification", "fabrication_archive_certification"):
        certification = facts.get(key)
        if isinstance(certification, Mapping) and isinstance(certification.get("report_path"), str):
            paths.add(certification["report_path"])
    exports = facts.get("verified_export_paths")
    if isinstance(exports, list):
        paths.update(path for path in exports if isinstance(path, str))
    # ``synthesis_check`` is a manufacturing source even though the extractor
    # intentionally exposes only build_gate in its generic evidence list.
    paths.add(".kicraft/synthesis_check.json")
    paths.update(_artifact_inventory(rundir))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for relative_path in sorted(paths):
        path = _run_local_file(rundir, relative_path)
        if path is None:
            errors.append(f"unverified_evidence: missing or non-local artifact {relative_path!r}")
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative_path,
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return rows, errors


def verify_product_audit(
    rundir: Path, *, record: Mapping[str, Any], obligations, policy
) -> list[str]:
    """Verify the exact successful verdict, frozen inputs, and current deliverables."""
    rundir = Path(rundir).resolve()
    path = _run_local_file(rundir, record.get("product_evidence_path"))
    if path is None:
        return ["unverified_evidence: product acceptance audit is missing or non-local"]
    payload = path.read_bytes()
    if record.get("product_evidence_sha256") != "sha256:" + hashlib.sha256(payload).hexdigest():
        return ["audit_changed: saved success is not bound to this acceptance audit"]
    try:
        audit = json.loads(payload)
    except (ValueError, UnicodeError):
        return ["unverified_evidence: product acceptance audit is unreadable"]
    if not isinstance(audit, Mapping) or audit.get("schema_version") != 2:
        return ["unverified_evidence: product acceptance audit schema is invalid"]
    errors: list[str] = []
    if audit.get("context") != _audit_context(record, obligations, policy):
        errors.append("audit_context_changed: brief, obligations or policy differ")
    verdicts = audit.get("obligations")
    if (
        audit.get("product_success") is not True
        or audit.get("errors") != []
        or not isinstance(verdicts, dict)
        or not verdicts
        or any(
            not isinstance(row, dict)
            or row.get("status") != "pass"
            or row.get("reason") is not None
            for row in verdicts.values()
        )
    ):
        errors.append("unverified_evidence: audit does not certify successful fulfillment")
    if audit.get("artifact_inventory") != _artifact_inventory(rundir):
        errors.append("artifact_inventory_changed: delivered artifact selection differs")
    rows = audit.get("artifacts")
    if not isinstance(rows, list) or not rows:
        return [*errors, "unverified_evidence: product acceptance audit has no artifact hashes"]
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append(
                "unverified_evidence: product acceptance audit has an invalid artifact row"
            )
            continue
        relative_path = row.get("path")
        artifact = _run_local_file(rundir, relative_path)
        if artifact is None:
            errors.append(
                f"unverified_evidence: audited artifact {relative_path!r} is missing or non-local"
            )
            continue
        digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        if row.get("sha256") != digest:
            errors.append(f"artifact_changed: audited artifact {relative_path!r} hash differs")
    return errors


def _record_preconditions(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """Return prerequisite failures before evaluating artifact evidence."""
    errors: list[tuple[str, str]] = []

    failure_kind = next(
        (
            str(record[key])
            for key in ("failure_kind", "design_failure_kind")
            if isinstance(record.get(key), str) and record[key]
        ),
        None,
    )
    if failure_kind is not None:
        errors.append((failure_kind, f"{failure_kind}: terminal design failure"))
    elif any(record.get(key) not in (None, "", False) for key in ("error", "design_error")):
        errors.append(("terminal_design_failure", "terminal_design_failure: terminal design error"))
    status = record.get("design_status")
    if status is not None and status != "ok":
        errors.append(
            ("terminal_design_failure", f"terminal_design_failure: design status is {status!r}")
        )
    if record.get("design_committed") is not True:
        errors.append(("design_not_committed", "design_not_committed: design is not committed"))
    original_hash = record.get("original_brief_hash")
    execution_hash = record.get("execution_brief_hash")
    if (
        not isinstance(original_hash, str)
        or not _HASH.fullmatch(original_hash)
        or not isinstance(execution_hash, str)
        or not _HASH.fullmatch(execution_hash)
    ):
        errors.append(
            (
                "unverified_evidence",
                "unverified_evidence: original and execution brief hashes are missing or invalid",
            )
        )
    elif original_hash != execution_hash:
        errors.append(
            (
                "brief_identity_mismatch",
                "brief_identity_mismatch: original and execution brief hashes differ",
            )
        )
    manual = record.get("manual_intervention")
    if manual is True:
        errors.append(
            ("manual_intervention", "manual_intervention: run required manual intervention")
        )
    elif manual is not False:
        errors.append(
            (
                "unverified_evidence",
                "unverified_evidence: no-manual-intervention evidence is absent",
            )
        )
    clarification = record.get("clarification_assisted")
    if clarification is True:
        errors.append(
            (
                "clarification_assisted",
                "clarification_assisted: run required clarification assistance",
            )
        )
    elif clarification is not False:
        errors.append(
            ("unverified_evidence", "unverified_evidence: no-clarification evidence is absent")
        )

    cost = record.get("ledger_cost_usd")
    if cost is None and record.get("design_cost_source") == "spend_ledger":
        cost = record.get("design_cost_usd")
    exposure = record.get("budget_exposure") or {}
    if isinstance(exposure, Mapping) and exposure.get("active_exposure_usd", 0) > 0:
        errors.append(
            ("unverified_spend", "unverified_spend: a dispatched call lacks settled usage evidence")
        )
    if not _finite_number(cost):
        errors.append(("unverified_evidence", "unverified_evidence: numeric ledger cost is absent"))
    elif cost > policy["max_cost_usd"]:
        errors.append(
            ("cost_limit_exceeded", "cost_limit_exceeded: ledger cost exceeds frozen policy")
        )
    duration = record.get("duration_s")
    if not _finite_number(duration):
        errors.append(("unverified_evidence", "unverified_evidence: numeric duration is absent"))
    elif duration > policy["max_duration_s"]:
        errors.append(
            ("duration_limit_exceeded", "duration_limit_exceeded: duration exceeds frozen policy")
        )
    rounds = record.get("park_rounds", record.get("rounds"))
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        errors.append(
            ("unverified_evidence", "unverified_evidence: actual park-round count is absent")
        )
    elif rounds > policy["max_park_rounds"]:
        errors.append(
            ("park_limit_exceeded", "park_limit_exceeded: park rounds exceed frozen policy")
        )
    return errors


def _delivered_board(
    rundir: Path, facts: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]
) -> Path | None:
    candidate = _run_local_file(rundir, facts.get("delivered_board_path"))
    if candidate is not None and candidate.suffix == ".kicad_pcb":
        return candidate
    boards = sorted(
        path
        for item in artifacts
        if isinstance(item, Mapping)
        for path in [_run_local_file(rundir, item.get("path"))]
        if path is not None and path.suffix == ".kicad_pcb"
    )
    return boards[0] if len(boards) == 1 else None


def _certify_drc(
    rundir: Path, facts: dict[str, Any], artifacts: list[dict[str, Any]], timeout_s: float
) -> list[tuple[str, str]]:
    """Run and preserve a fresh DRC on the exact board used for extraction."""
    board = _delivered_board(rundir, facts, artifacts)
    if board is None:
        return [
            (
                "unverified_evidence",
                "unverified_evidence: delivered board artifact is absent or ambiguous",
            )
        ]
    for suffix in (".kicad_pro", ".kicad_dru"):
        configuration = board.with_suffix(suffix)
        if configuration.is_file():
            artifacts.append(
                {
                    "kind": "artifact",
                    "path": str(configuration.relative_to(rundir.resolve())),
                }
            )
    report = rundir / _DRC_REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    certification: dict[str, Any] = {
        "board_path": _relative(rundir, board),
        "board_sha256": "sha256:" + hashlib.sha256(board.read_bytes()).hexdigest(),
        "report_path": str(_DRC_REPORT_PATH),
    }
    try:
        completed = subprocess.run(
            [
                "kicad-cli",
                "pcb",
                "drc",
                "-o",
                str(report),
                "--format",
                "json",
                "--severity-error",
                "--severity-exclusions",
                "--all-track-errors",
                "--exit-code-violations",
                str(board),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        certification["status"] = "unverified"
        facts["drc_certification"] = certification
        return [("unverified_evidence", "unverified_evidence: delivered-board DRC timed out")]
    except FileNotFoundError:
        certification["status"] = "unverified"
        facts["drc_certification"] = certification
        return [
            (
                "unverified_evidence",
                "unverified_evidence: kicad-cli is unavailable for delivered-board DRC",
            )
        ]
    if not report.is_file():
        certification["status"] = "unverified"
        facts["drc_certification"] = certification
        return [
            ("unverified_evidence", "unverified_evidence: delivered-board DRC produced no report")
        ]
    try:
        report_data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report_data = None
    artifacts.append({"kind": "tool-report", "path": str(_DRC_REPORT_PATH)})
    certification["report_sha256"] = "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest()
    certification["returncode"] = completed.returncode
    facts["drc_certification"] = certification
    if not isinstance(report_data, dict) or any(
        not isinstance(report_data.get(key), list) for key in ("violations", "unconnected_items")
    ):
        certification["status"] = "unverified"
        return [
            ("unverified_evidence", "unverified_evidence: delivered-board DRC report is invalid")
        ]
    if completed.returncode not in (0, 5):
        certification["status"] = "unverified"
        return [
            ("unverified_evidence", "unverified_evidence: delivered-board DRC invocation failed")
        ]
    if completed.returncode != 0 or report_data["violations"] or report_data["unconnected_items"]:
        certification["status"] = "fail"
        return [
            (
                "design_defect",
                "design_defect: delivered-board DRC reports errors or unconnected items",
            )
        ]
    certification["status"] = "pass"
    facts.setdefault("gates", {})["drc"] = "pass"
    return []


def _certify_fabrication_archives(
    rundir: Path, facts: dict[str, Any], artifacts: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Check actual zip integrity, nonempty Gerber/drill members, and board pairing."""
    board = _delivered_board(rundir, facts, artifacts)
    exports = facts.get("verified_export_paths")
    if board is None or not isinstance(exports, list) or not exports:
        return [
            (
                "unverified_evidence",
                "unverified_evidence: Gerber/drill export is not independently verified",
            )
        ]
    receipts: list[dict[str, Any]] = []
    for relative_path in exports:
        archive = _run_local_file(rundir, relative_path)
        if archive is None:
            return [
                (
                    "unverified_evidence",
                    "unverified_evidence: fabrication archive is missing or non-local",
                )
            ]
        receipt_path = archive.with_suffix(".receipt.json")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return [
                (
                    "unverified_evidence",
                    "unverified_evidence: fabrication export receipt is absent or invalid",
                )
            ]
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("board") != board.name
            or receipt.get("board_sha256") != hashlib.sha256(board.read_bytes()).hexdigest()
            or receipt.get("archive") != archive.name
            or receipt.get("archive_sha256") != hashlib.sha256(archive.read_bytes()).hexdigest()
        ):
            return [
                (
                    "unverified_evidence",
                    "unverified_evidence: fabrication package is not bound to this board",
                )
            ]
        artifacts.append({"kind": "artifact", "path": _relative(rundir, receipt_path)})
        try:
            with zipfile.ZipFile(archive) as contents:
                infos = [info for info in contents.infolist() if not info.is_dir()]
                names = [info.filename.casefold() for info in infos]
                corrupt = contents.testzip()
                if receipt.get("files") != {
                    info.filename: hashlib.sha256(contents.read(info)).hexdigest() for info in infos
                }:
                    return [
                        (
                            "unverified_evidence",
                            "unverified_evidence: fabrication members differ from the export receipt",
                        )
                    ]
                job_names = [info.filename for info in infos if info.filename.endswith(".gbrjob")]
                if len(job_names) != 1:
                    raise ValueError("expected one Gerber job manifest")
                job = json.loads(contents.read(job_names[0]))
                outputs = job["FilesAttributes"]
                copper = [row for row in outputs if row["FileFunction"].startswith("Copper,")]
                if (
                    job["GeneralSpecs"]["ProjectId"]["Name"] != board.stem
                    or len(copper) != job["GeneralSpecs"]["LayerNumber"]
                    or len(copper) < 2
                    or not any(row["FileFunction"] == "Profile" for row in outputs)
                ):
                    raise ValueError("Gerber job does not cover the board stack and outline")
                if len({row["Path"] for row in outputs}) != len(outputs):
                    raise ValueError("duplicate Gerber job outputs")
                for row in outputs:
                    content = contents.read(row["Path"])
                    if not content.strip() or b"%FS" not in content or b"M02*" not in content:
                        raise ValueError("invalid or empty Gerber output")
                for info in infos:
                    if info.filename.endswith((".drl", ".xln")):
                        content = contents.read(info)
                        if b"M48" not in content or b"M30" not in content:
                            raise ValueError("invalid drill output")
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
            zipfile.BadZipFile,
        ):
            return [
                ("unverified_evidence", "unverified_evidence: fabrication archive is unreadable")
            ]
        gerbers = [
            info for info, name in zip(infos, names) if name.endswith((".gbr", ".gtl", ".gbl"))
        ]
        drills = [info for info, name in zip(infos, names) if name.endswith((".drl", ".xln"))]
        jobs = [info for info, name in zip(infos, names) if name.endswith(".gbrjob")]
        if (
            corrupt is not None
            or not gerbers
            or not drills
            or not jobs
            or any(info.file_size <= 0 for info in [*gerbers, *drills, *jobs])
            or archive.parent != board.parent
            or not archive.name.startswith(f"{board.stem}_fab_")
        ):
            return [
                (
                    "unverified_evidence",
                    "unverified_evidence: fabrication archive is inconsistent with the delivered board",
                )
            ]
        receipts.append(
            {
                "path": str(relative_path),
                "sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
                "members": len(infos),
            }
        )
    facts["fabrication_archive_certification"] = {
        "board_path": _relative(rundir, board),
        "board_sha256": "sha256:" + hashlib.sha256(board.read_bytes()).hexdigest(),
        "archives": receipts,
    }
    return []


def _manufacturing_errors(
    record: Mapping[str, Any], facts: Mapping[str, Any]
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    if record.get("build_rc") != 0:
        errors.append(("build_failed", "build_failed: full build did not return rc0"))
    if facts.get("board_loaded") is not True:
        errors.append(
            ("unverified_evidence", "unverified_evidence: delivered board could not be loaded")
        )
    gate = facts.get("build_gate")
    if not isinstance(gate, Mapping):
        errors.append(
            ("unverified_evidence", "unverified_evidence: structured manufacturing gate is absent")
        )
    elif gate.get("fab_acceptable") is not True:
        errors.append(
            ("design_defect", "design_defect: manufacturing gate is not fabrication acceptable")
        )
    gates = facts.get("gates")
    if not isinstance(gates, Mapping) or gates.get("erc") not in {"pass", "fail"}:
        errors.append(("unverified_evidence", "unverified_evidence: ERC verdict is absent"))
    elif gates["erc"] != "pass":
        errors.append(("design_defect", "design_defect: ERC verdict failed"))
    return errors


def _baseline_obligations(
    facts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Required board-level evidence independent of an external brief's wording."""
    checks = (
        ("sourceable-parts", {"kind": "part_inventory"}),
        ("pin-footprint-mapping", {"kind": "pin_mapping"}),
        (
            "complete-required-connections",
            {"kind": "gate", "gate": "complete_required_connections"},
        ),
        ("erc", {"kind": "gate", "gate": "erc"}),
        ("drc", {"kind": "gate", "gate": "drc"}),
        ("closed-board-outline", {"kind": "gate", "gate": "geometry"}),
        ("fabrication-export", {"kind": "artifacts"}),
    )
    obligations = [{"id": f"product.{suffix}", "check": check} for suffix, check in checks]
    errors: list[tuple[str, str]] = []
    classes = facts.get("part_classes")
    microcontrollers = classes.get("microcontroller") if isinstance(classes, Mapping) else None
    if not isinstance(microcontrollers, int) or isinstance(microcontrollers, bool):
        errors.append(
            (
                "unverified_evidence",
                "unverified_evidence: programming applicability is not evidenced by the delivered BOM",
            )
        )
    elif microcontrollers > 0:
        obligations.append(
            {
                "id": "product.programming",
                "check": {"kind": "gate", "gate": "programming"},
            }
        )
    return obligations, errors


def _obligation_results(
    obligations: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    rundir: Path,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    rows: dict[str, dict[str, Any]] = {}
    errors: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, obligation in enumerate(obligations):
        obligation_id = obligation.get("id") if isinstance(obligation, Mapping) else None
        label = (
            obligation_id
            if isinstance(obligation_id, str) and obligation_id
            else f"obligation[{index}]"
        )
        if label in seen:
            errors.append(
                (
                    "unverified_evidence",
                    f"unverified_evidence: duplicate required obligation {label!r}",
                )
            )
            continue
        seen.add(label)
        if not isinstance(obligation, Mapping) or not isinstance(obligation.get("check"), Mapping):
            errors.append(
                (
                    "unverified_evidence",
                    f"unverified_evidence: {label} has no valid acceptance check",
                )
            )
            continue
        status = _result_status(dict(obligation["check"]), dict(facts))
        result = {"status": status, "evidence": list(artifacts)}
        reason = evaluate_obligation(obligation, result, artifact_root=rundir, facts_override=facts)
        rows[label] = {"status": status, "reason": reason}
        if reason:
            kind = "unverified_evidence" if status == "unverified" else "design_defect"
            errors.append((kind, f"{kind}: {label}: {reason}"))
    return rows, errors


def _write_audit(
    rundir: Path,
    record: Mapping[str, Any],
    facts: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    obligation_results: Mapping[str, Any],
    errors: Sequence[str],
    obligations,
    policy,
) -> tuple[str, str, list[str]]:
    audit_rows, audit_errors = _audit_artifacts(rundir, artifacts, facts)
    audit = {
        "schema_version": 2,
        "context": _audit_context(record, obligations, policy),
        "product_success": not errors and not audit_errors,
        "artifact_inventory": _artifact_inventory(rundir),
        "artifact_facts_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "artifacts": audit_rows,
        "obligations": dict(obligation_results),
        "errors": [*errors, *audit_errors],
    }
    path = rundir / _AUDIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return str(_AUDIT_PATH), "sha256:" + hashlib.sha256(payload).hexdigest(), audit_errors


def evaluate_product(rundir: Path, record: dict, obligations: list[dict], policy: dict) -> dict:
    """Evaluate one run's conjunctive product-success predicate.

    ``record`` is only used for run metadata and frozen-policy observations;
    artifact facts and obligation status are always recomputed from ``rundir``.
    The returned mapping is safe to merge into that record.
    """
    rundir = Path(rundir).resolve()
    all_errors: list[tuple[str, str]] = []
    policy_errors = _valid_policy(policy)
    if policy_errors:
        all_errors.extend(("invalid_policy", error) for error in policy_errors)
        facts: dict[str, Any] = {}
        artifacts: list[dict[str, Any]] = []
    else:
        all_errors.extend(_record_preconditions(record, policy))
        try:
            facts, artifacts = extract_artifact_facts(
                rundir, {"slug": record.get("slug"), "obligations": obligations}
            )
        except Exception as exc:  # An unreadable build must not abort its own verdict.
            facts, artifacts = {}, []
            all_errors.append(
                (
                    "unverified_evidence",
                    f"unverified_evidence: artifact extraction failed: {type(exc).__name__}",
                )
            )
        all_errors.extend(_certify_drc(rundir, facts, artifacts, float(policy["build_timeout_s"])))
        all_errors.extend(_certify_fabrication_archives(rundir, facts, artifacts))
        all_errors.extend(_manufacturing_errors(record, facts))

    if not isinstance(obligations, list) or not obligations:
        all_errors.append(
            (
                "unverified_evidence",
                "unverified_evidence: at least one external obligation is required",
            )
        )
        obligation_results: dict[str, dict[str, Any]] = {}
    else:
        baseline, baseline_errors = _baseline_obligations(facts)
        all_errors.extend(baseline_errors)
        obligation_results, obligation_errors = _obligation_results(
            [*baseline, *obligations], facts, artifacts, rundir
        )
        all_errors.extend(obligation_errors)

    error_messages = [message for _, message in all_errors]
    evidence_path, evidence_hash, audit_errors = _write_audit(
        rundir, record, facts, artifacts, obligation_results, error_messages, obligations, policy
    )
    all_errors.extend(("unverified_evidence", error) for error in audit_errors)
    error_messages = [message for _, message in all_errors]
    return {
        "product_success": not all_errors,
        "product_failure_kind": all_errors[0][0] if all_errors else None,
        "product_errors": error_messages,
        "product_evidence_path": evidence_path,
        "product_evidence_sha256": evidence_hash,
        "product_obligations": obligation_results,
    }


def _capability_keys(check: Mapping[str, Any]) -> list[str]:
    """Name the requested capabilities an obligation measures, independent of its id.

    A combined check (several semantic paths in one obligation) expands so a
    capability missing from many briefs is ranked by that frequency rather than
    hidden inside a unique combination.
    """
    kind = str(check.get("kind"))
    for field in ("fact", "channel", "feature", "part_class", "gate"):
        if isinstance(check.get(field), str) and check[field]:
            return [f"{kind}:{check[field]}"]
    if kind == "net_paths":
        paths = check.get("paths")
        return [f"{kind}:{path}" for path in paths] if isinstance(paths, list) and paths else [kind]
    if kind == "connector_map":
        return [f"{kind}:{check.get('connector')}"]
    if kind == "outline":
        return [f"{kind}:{check.get('shape')}"]
    return [kind]


def _capability_gaps(
    records: Sequence[Mapping[str, Any]],
    obligations_by_slug: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Inventory which requested capabilities were defect vs unproven, by check.

    A ``fail`` is the delivered board violating a requirement; ``unverified`` is
    absent evidence.  Ranking the two separately keeps a missing evaluator from
    being mistaken for a missing circuit, and vice versa.
    """
    gaps: dict[str, dict[str, int]] = defaultdict(
        lambda: {"defect_runs": 0, "unverified_runs": 0, "passed_runs": 0}
    )
    checks: dict[str, list[str]] = {}
    for slug, obligations in obligations_by_slug.items():
        for obligation in obligations or []:
            if isinstance(obligation, Mapping) and isinstance(obligation.get("check"), Mapping):
                checks[f"{slug}:{obligation.get('id')}"] = _capability_keys(obligation["check"])
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("slug"), str):
            continue
        results = record.get("product_obligations")
        if not isinstance(results, Mapping):
            continue
        for obligation_id, row in results.items():
            keys = checks.get(f"{record['slug']}:{obligation_id}")
            if keys is None or not isinstance(row, Mapping):
                continue
            status = row.get("status")
            field = (
                "passed_runs"
                if status == "pass"
                else "defect_runs"
                if status == "fail"
                else "unverified_runs"
            )
            for key in keys:
                gaps[key][field] += 1
    return dict(
        sorted(
            gaps.items(),
            key=lambda item: (
                -(item[1]["defect_runs"] + item[1]["unverified_runs"]),
                item[0],
            ),
        )
    )


def summarize_product(
    records: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    obligations_by_slug: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the full eligible denominator without dropping absent runs."""
    eligible = {
        str(entry["slug"]): entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("eligible") is True
        and isinstance(entry.get("slug"), str)
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if (
            isinstance(record, Mapping)
            and isinstance(record.get("slug"), str)
            and record["slug"] in eligible
        ):
            grouped[record["slug"]].append(record)
    missing = sorted(slug for slug in eligible if not grouped[slug])
    duplicate = sorted(slug for slug, rows in grouped.items() if len(rows) > 1)
    pending = sorted(
        slug
        for slug, rows in grouped.items()
        if len(rows) == 1 and not isinstance(rows[0].get("product_success"), bool)
    )
    success_slugs = {
        slug
        for slug, rows in grouped.items()
        if len(rows) == 1 and rows[0].get("product_success") is True
    }
    failures: Counter[str] = Counter(
        {
            name: count
            for name, count in (
                ("missing_run", len(missing)),
                ("duplicate_run", len(duplicate)),
                ("pending", len(pending)),
            )
            if count
        }
    )
    for slug, rows in grouped.items():
        if len(rows) != 1 or slug in pending or slug in success_slugs:
            continue
        kind = rows[0].get("product_failure_kind")
        failures[str(kind) if isinstance(kind, str) and kind else "unclassified_failure"] += 1

    family_members: dict[str, set[str]] = defaultdict(set)
    for slug, entry in eligible.items():
        families = entry.get("families")
        if isinstance(families, list):
            for family in families:
                if isinstance(family, str) and family:
                    family_members[family].add(slug)
    family_stats = {}
    for family in sorted(family_members):
        members = family_members[family]
        family_success = len(members & success_slugs)
        family_pending = len(members & set(pending))
        family_missing = len(members & set(missing))
        family_duplicate = len(members & set(duplicate))
        denominator = len(members)
        family_stats[family] = {
            "eligible_n": denominator,
            "product_success_n": family_success,
            "product_yield": family_success / denominator if denominator else None,
            "missing_n": family_missing,
            "pending_n": family_pending,
            "duplicate_n": family_duplicate,
        }
    denominator = len(eligible)
    return {
        "schema_version": 1,
        "eligible_n": denominator,
        "product_success_n": len(success_slugs),
        "product_yield": len(success_slugs) / denominator if denominator else None,
        "missing_slugs": missing,
        "pending_slugs": pending,
        "duplicate_slugs": duplicate,
        "family_stats": family_stats,
        "failure_pareto": dict(sorted(failures.items(), key=lambda item: (-item[1], item[0]))),
        "capability_gaps": _capability_gaps(records, obligations_by_slug or {}),
    }
