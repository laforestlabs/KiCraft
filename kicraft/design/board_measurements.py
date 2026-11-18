"""Conservative removal of component placeholders for board measurements."""

from kicraft.design.models import board_measurement_block
from kicraft.design.part_identity import (
    canonical_physical_features,
    quantity_subject_binds,
    realizable_physical_features,
)


def _field(value: object, key: str, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _references(value: object, name: str) -> bool:
    if isinstance(value, dict):
        return any(_references(item, name) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_references(item, name) for item in value)
    return isinstance(value, str) and (value == name or value.startswith(name + "."))


def _could_realize_obligation(family: str, obligation: object) -> bool:
    kind = _field(obligation, "kind")
    if kind == "quantity":
        subject = str(_field(obligation, "subject", ""))
        if quantity_subject_binds(subject, family):
            return True
        # A count of pins/contacts on a component is not a count of components,
        # but still makes removal of that component unsafe.
        words = set(subject.casefold().replace("_", "-").split())
        return any(feature in words for feature in canonical_physical_features(family))
    field = {
        "physical": "component_class",
        "external_interface": "protocol",
        "fabrication": "feature",
    }.get(kind)
    if field is None:
        return False
    demand = str(_field(obligation, field, "") or "")
    return bool(canonical_physical_features(family) & canonical_physical_features(demand))


def prune_board_measurement_requirements(
    payload: dict, *, intent: object | None = None, functional_spec: object | None = None
) -> None:
    """Remove only isolated placeholders that exclusively own board-measurement blocks.

    The caller supplies a copied payload and the committed upstream context. Missing or
    ambiguous evidence leaves the candidate untouched for ordinary validation.
    """
    blocks = _field(functional_spec, "blocks")
    obligations = _field(intent, "obligations")
    if obligations is None:
        obligations = _field(functional_spec, "obligations")
    if obligations is None:
        obligations = payload.get("obligations")
    requirements = payload.get("requirements")
    sheets = payload.get("sheets")
    if not isinstance(blocks, (list, tuple)) or not isinstance(obligations, (list, tuple)):
        return
    if not isinstance(requirements, list) or not isinstance(sheets, list):
        return
    by_name = {}
    for block in blocks:
        by_name.setdefault(_field(block, "name"), []).append(block)
    block_connections = _field(functional_spec, "connections", ()) or ()
    candidates = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        identifier = requirement.get("id")
        names = requirement.get("functional_blocks")
        family = requirement.get("family")
        if not isinstance(identifier, str) or not identifier or not isinstance(family, str):
            continue
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            continue
        if any(
            len(by_name.get(name, ())) != 1
            or not board_measurement_block(by_name[name][0], obligations)
            for name in names
        ):
            continue
        if any(
            _field(connection, "from_block") in names or _field(connection, "to_block") in names
            for connection in block_connections
        ):
            continue
        if any(
            isinstance(other, dict) and other is not requirement
            and set(names).intersection(other.get("functional_blocks") or ())
            for other in requirements
        ):
            continue
        allowed = {
            "id", "sheet", "role", "family", "functional_blocks", "obligation_ids", "parameters",
        }
        if any(value for key, value in requirement.items() if key not in allowed):
            continue
        owned_ids = requirement.get("obligation_ids") or []
        block_ids = {
            block_id
            for name in names
            for block_id in _field(by_name[name][0], "obligation_ids", ())
        }
        if owned_ids and (
            not board_measurement_block({"obligation_ids": owned_ids}, obligations)
            or not set(owned_ids).issubset(block_ids)
        ):
            continue
        if not realizable_physical_features(family):
            continue
        if any(_could_realize_obligation(family, row) for row in obligations):
            continue
        if _references(payload.get("signals", ()), identifier) or _references(payload.get("power", {}), identifier):
            continue
        if sum(isinstance(row, dict) and row.get("id") == identifier for row in requirements) != 1:
            continue
        candidates.append(requirement)

    removed_sheets = {row.get("sheet") for row in candidates}
    if not candidates:
        return
    payload["requirements"] = [row for row in requirements if all(row is not item for item in candidates)]
    for sheet_name in removed_sheets:
        matching = [row for row in sheets if isinstance(row, dict) and row.get("name") == sheet_name]
        if len(matching) != 1:
            continue
        sheet = matching[0]
        if any(sheet.get(key) is not None for key in ("from_library", "library_instance", "replication_group", "replication_instance")):
            continue
        if any(_field(row, "sheet") == sheet_name for row in payload["requirements"]):
            continue
        if any(_references(row, sheet_name) for row in sheets if row is not sheet):
            continue
        if any(_references(value, sheet_name) for key, value in payload.items() if key != "sheets"):
            continue
        sheets.remove(sheet)
