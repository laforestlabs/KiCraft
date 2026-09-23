"""Fail-closed loading for immutable, externally supplied evaluation corpora.

The corpus and its acceptance obligations deliberately travel in separate JSON
files.  Briefs are the only material handed to design; obligations remain an
evaluator-side contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


_SCHEMA_VERSION = 1
_SLUG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.:-]{0,127})")
_CHECK_KINDS = frozenset(
    {
        "part_class_count",
        "part_class_exact",
        "part_classes",
        "part_identity",
        "part_identities",
        "part_inventory",
        "pin_mapping",
        "net_paths",
        "numeric_range",
        "connector_map",
        "set_members",
        "channel_count",
        "geometry_count",
        "outline",
        "gate",
        "applicable_gate",
        "artifacts",
    }
)


def stable_hash(value: Any) -> str:
    """Return the canonical JSON SHA-256 identity for a validated value."""
    _ensure_json(value, "hash input")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fail(where: str, message: str) -> None:
    raise ValueError(f"external corpus {where}: {message}")


def _ensure_json(value: Any, where: str) -> None:
    """Reject Python-only values and non-finite numbers before hashing or storing."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(where, "contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json(item, f"{where}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(where, "contains a non-string object key")
            _ensure_json(item, f"{where}.{key}")
        return
    _fail(where, f"contains non-JSON value {type(value).__name__}")


def _object(
    value: Any,
    where: str,
    *,
    keys: set[str] | None = None,
    required_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    if keys is not None:
        unknown = set(value) - keys
        if unknown:
            _fail(where, f"unknown keys {sorted(unknown)}")
    if required_keys is not None:
        missing = required_keys - set(value)
        if missing:
            _fail(where, f"missing keys {sorted(missing)}")
    return value


def _nonempty_text(value: Any, where: str, *, natural: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "must be a nonempty string")
    if "\x00" in value:
        _fail(where, "must not contain NUL")
    if natural and not any(char.isalpha() for char in value):
        _fail(where, "must contain natural-language text")
    return value


def _finite_number(value: Any, where: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(where, "must be a finite number")
    if positive and value <= 0:
        _fail(where, "must be positive")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(where, "must be a non-negative integer")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(where, "must be a positive integer")
    return value


def _string_list(value: Any, where: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        _fail(where, "must be a nonempty list")
    for index, item in enumerate(value):
        _nonempty_text(item, f"{where}[{index}]")
    if len(value) != len(set(value)):
        _fail(where, "must not contain duplicates")
    return value


def _validate_check(check: Any, where: str) -> dict[str, Any]:
    value = _object(check, where)
    kind = value.get("kind")
    if kind not in _CHECK_KINDS:
        _fail(where + ".kind", f"must be one of {sorted(_CHECK_KINDS)}")

    def exact(*allowed: str) -> None:
        expected = {"kind", *allowed}
        unknown = set(value) - expected
        if unknown:
            _fail(where, f"unknown keys {sorted(unknown)}")

    if kind == "part_class_count":
        exact("part_class", "minimum")
        _nonempty_text(value.get("part_class"), where + ".part_class")
        _nonnegative_int(value.get("minimum"), where + ".minimum")
    elif kind == "part_class_exact":
        exact("part_class", "value")
        _nonempty_text(value.get("part_class"), where + ".part_class")
        _nonnegative_int(value.get("value"), where + ".value")
    elif kind == "part_classes":
        exact("required")
        required = _object(value.get("required"), where + ".required")
        if not required:
            _fail(where + ".required", "must not be empty")
        for name, minimum in required.items():
            _nonempty_text(name, where + ".required key")
            _nonnegative_int(minimum, f"{where}.required.{name}")
    elif kind == "part_identity":
        exact("identity", "package")
        _nonempty_text(value.get("identity"), where + ".identity")
        if "package" in value:
            _nonempty_text(value["package"], where + ".package")
    elif kind == "part_identities":
        exact("identities")
        _string_list(value.get("identities"), where + ".identities")
    elif kind in {"part_inventory", "pin_mapping", "artifacts"}:
        exact()
    elif kind == "net_paths":
        exact("paths")
        _string_list(value.get("paths"), where + ".paths")
    elif kind == "numeric_range":
        exact("fact", "minimum", "maximum")
        _nonempty_text(value.get("fact"), where + ".fact")
        if "minimum" in value:
            _finite_number(value["minimum"], where + ".minimum")
        if "maximum" in value:
            _finite_number(value["maximum"], where + ".maximum")
        if "minimum" in value and "maximum" in value and value["minimum"] > value["maximum"]:
            _fail(where, "minimum must be less than or equal to maximum")
    elif kind == "connector_map":
        exact("connector", "required", "distinct", "source_distinct")
        _nonempty_text(value.get("connector"), where + ".connector")
        _string_list(value.get("required"), where + ".required")
        for name in ("distinct", "source_distinct"):
            if name in value and not isinstance(value[name], bool):
                _fail(where + "." + name, "must be a boolean")
    elif kind == "set_members":
        exact("fact", "required")
        _nonempty_text(value.get("fact"), where + ".fact")
        required = value.get("required")
        if not isinstance(required, list) or not required:
            _fail(where + ".required", "must be a nonempty list")
        for index, member in enumerate(required):
            if isinstance(member, (dict, list)):
                _fail(f"{where}.required[{index}]", "must be a scalar JSON value")
            _ensure_json(member, f"{where}.required[{index}]")
        if len(
            {json.dumps(member, sort_keys=True, separators=(",", ":")) for member in required}
        ) != len(required):
            _fail(where + ".required", "must not contain duplicates")
    elif kind == "channel_count":
        exact("channel", "minimum")
        _nonempty_text(value.get("channel"), where + ".channel")
        _nonnegative_int(value.get("minimum"), where + ".minimum")
    elif kind == "geometry_count":
        exact("feature", "minimum")
        _nonempty_text(value.get("feature"), where + ".feature")
        _nonnegative_int(value.get("minimum"), where + ".minimum")
    elif kind == "outline":
        exact("shape", "diameter_mm")
        _nonempty_text(value.get("shape"), where + ".shape")
        if "diameter_mm" in value:
            _finite_number(value["diameter_mm"], where + ".diameter_mm", positive=True)
    elif kind in {"gate", "applicable_gate"}:
        exact("gate")
        _nonempty_text(value.get("gate"), where + ".gate")
    return value


def _validate_obligations(
    payload: Any, corpus_id: str, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    root = _object(
        payload,
        "obligations",
        keys={"schema_version", "corpus_id", "briefs"},
        required_keys={"schema_version", "corpus_id", "briefs"},
    )
    if root["schema_version"] != _SCHEMA_VERSION or isinstance(root["schema_version"], bool):
        _fail("obligations.schema_version", "must equal 1")
    if _nonempty_text(root["corpus_id"], "obligations.corpus_id") != corpus_id:
        _fail("obligations.corpus_id", "must match manifest corpus_id")
    by_slug = _object(root["briefs"], "obligations.briefs")
    slugs = {entry["slug"] for entry in entries}
    if set(by_slug) != slugs:
        _fail("obligations.briefs", "must contain exactly the manifest slugs")

    ids: set[str] = set()
    eligible_by_slug = {entry["slug"]: entry["eligible"] for entry in entries}
    for slug, rows in by_slug.items():
        where = f"obligations.briefs.{slug}"
        if not isinstance(rows, list):
            _fail(where, "must be a list")
        if eligible_by_slug[slug] and not rows:
            _fail(where, "must contain an obligation for an eligible brief")
        for index, obligation in enumerate(rows):
            obligation_where = f"{where}[{index}]"
            row = _object(
                obligation,
                obligation_where,
                keys={"id", "check", "statement", "owner", "evidence_source"},
                required_keys={"id", "check"},
            )
            obligation_id = _nonempty_text(row["id"], obligation_where + ".id")
            if not _ID_RE.fullmatch(obligation_id):
                _fail(obligation_where + ".id", "must be a safe stable identifier")
            if obligation_id in ids:
                _fail(obligation_where + ".id", "must be unique across the corpus")
            ids.add(obligation_id)
            _validate_check(row["check"], obligation_where + ".check")
            for key in ("statement", "owner", "evidence_source"):
                if key in row:
                    _nonempty_text(
                        row[key], obligation_where + "." + key, natural=key == "statement"
                    )
    return root


def _validate_manifest(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _object(
        payload,
        "manifest",
        keys={"schema_version", "corpus_id", "sampling", "capability_envelope", "policy", "briefs"},
        required_keys={
            "schema_version",
            "corpus_id",
            "sampling",
            "capability_envelope",
            "policy",
            "briefs",
        },
    )
    if root["schema_version"] != _SCHEMA_VERSION or isinstance(root["schema_version"], bool):
        _fail("manifest.schema_version", "must equal 1")
    _nonempty_text(root["corpus_id"], "manifest.corpus_id")

    sampling = _object(
        root["sampling"],
        "manifest.sampling",
        keys={"kind", "description"},
        required_keys={"kind", "description"},
    )
    if sampling["kind"] not in {"synthetic", "independent", "user"}:
        _fail("manifest.sampling.kind", "must be synthetic, independent, or user")
    _nonempty_text(sampling["description"], "manifest.sampling.description", natural=True)

    envelope = _object(root["capability_envelope"], "manifest.capability_envelope")
    if not envelope:
        _fail("manifest.capability_envelope", "must not be empty")

    policy = _object(
        root["policy"],
        "manifest.policy",
        keys={"max_cost_usd", "max_duration_s", "max_park_rounds", "build_timeout_s"},
        required_keys={"max_cost_usd", "max_duration_s", "max_park_rounds", "build_timeout_s"},
    )
    _finite_number(policy["max_cost_usd"], "manifest.policy.max_cost_usd", positive=True)
    _finite_number(policy["max_duration_s"], "manifest.policy.max_duration_s", positive=True)
    _positive_int(policy["max_park_rounds"], "manifest.policy.max_park_rounds")
    _finite_number(policy["build_timeout_s"], "manifest.policy.build_timeout_s", positive=True)

    briefs = root["briefs"]
    if not isinstance(briefs, list) or not briefs:
        _fail("manifest.briefs", "must be a nonempty list")
    entries: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for index, brief in enumerate(briefs, 1):
        where = f"manifest.briefs[{index - 1}]"
        row = _object(
            brief,
            where,
            keys={"slug", "brief", "archetype", "families", "eligible", "provenance"},
            required_keys={"slug", "brief", "archetype", "families", "eligible", "provenance"},
        )
        slug = _nonempty_text(row["slug"], where + ".slug")
        if not _SLUG_RE.fullmatch(slug):
            _fail(where + ".slug", "must be a safe lowercase slug")
        if slug in seen_slugs:
            _fail(where + ".slug", "must be unique")
        seen_slugs.add(slug)
        _nonempty_text(row["brief"], where + ".brief", natural=True)
        _nonempty_text(row["archetype"], where + ".archetype")
        _string_list(row["families"], where + ".families")
        if not isinstance(row["eligible"], bool):
            _fail(where + ".eligible", "must be a boolean")
        provenance = _object(row["provenance"], where + ".provenance")
        if not provenance:
            _fail(where + ".provenance", "must not be empty")
        entries.append({"index": index, **deepcopy(row)})
    return root, entries


def _bundle(
    manifest: dict[str, Any], obligations: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    result = {
        "manifest": deepcopy(manifest),
        "obligations": deepcopy(obligations),
        "entries": deepcopy(entries),
    }
    result["hash"] = stable_hash(
        {"manifest": result["manifest"], "obligations": result["obligations"]}
    )
    return result


def validate_external_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-stored external bundle before resume or execution.

    The returned value is a deep-copied, JSON-serializable canonical bundle; a
    stored ``entries`` projection or digest cannot silently substitute briefs.
    """
    root = _object(
        bundle,
        "bundle",
        keys={"manifest", "obligations", "entries", "hash"},
        required_keys={"manifest", "obligations", "entries", "hash"},
    )
    _ensure_json(root, "bundle")
    manifest, entries = _validate_manifest(root["manifest"])
    obligations = _validate_obligations(root["obligations"], manifest["corpus_id"], entries)
    expected = _bundle(manifest, obligations, entries)
    if root["entries"] != expected["entries"]:
        _fail("bundle.entries", "does not match the normalized manifest briefs")
    if root["hash"] != expected["hash"]:
        _fail("bundle.hash", "does not match manifest and obligations")
    return expected


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"external corpus {label} is unreadable: {exc}") from exc
    _ensure_json(value, label)
    return _object(value, label)


def load_external_manifest(path: Path, obligations_path: Path) -> dict[str, Any]:
    """Load separate immutable corpus and evaluator-obligation JSON documents.

    No files are written and no supplied object is mutated.  The resulting bundle
    is self-contained so it can be checkpointed and revalidated on resume.
    """
    manifest, entries = _validate_manifest(_read_json(path, "manifest"))
    obligations = _validate_obligations(
        _read_json(obligations_path, "obligations"), manifest["corpus_id"], entries
    )
    return _bundle(manifest, obligations, entries)
