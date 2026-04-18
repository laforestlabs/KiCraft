"""JSONL → SQLite importer for legacy experiment data."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..db import Database, ExperimentRun


def import_jsonl(db: Database, jsonl_path: str | Path,
                 experiment_name: str | None = None) -> int:
    """Import a JSONL experiment file into the database.

    Returns the experiment_id of the created experiment.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    records = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {line_num}: {e}")

    if not records:
        print(f"  No records found in {jsonl_path}")
        return -1

    # Derive experiment name
    if experiment_name is None:
        experiment_name = jsonl_path.stem.replace("_", " ").title()

    # Check if already imported
    existing = db.get_experiments()
    for exp in existing:
        if exp.source_jsonl == str(jsonl_path):
            print(f"  Already imported: {jsonl_path} → experiment #{exp.id}")
            return exp.id

    # Best score from records
    best_score = max((r.get("score", 0) for r in records), default=0)

    exp = db.create_experiment(
        name=experiment_name,
        total_rounds=len(records),
        config=records[0].get("config_delta", {}),
        source_jsonl=str(jsonl_path),
    )
    db.update_experiment(exp.id, status="done", best_score=best_score,
                         completed_rounds=len(records))

    # Bulk insert rounds
    for rec in records:
        db.add_round(exp.id, rec)

    print(f"  Imported {len(records)} rounds → experiment #{exp.id} "
          f"'{experiment_name}' (best={best_score:.2f})")
    return exp.id


def import_all_jsonl(db: Database, experiments_dir: str | Path) -> list[int]:
    """Import all JSONL files from the experiments directory."""
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return []

    jsonl_files = sorted(experiments_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {experiments_dir}")
        return []

    print(f"Found {len(jsonl_files)} JSONL files in {experiments_dir}")
    ids = []
    for f in jsonl_files:
        try:
            exp_id = import_jsonl(db, f)
            if exp_id > 0:
                ids.append(exp_id)
        except Exception as e:
            print(f"  Error importing {f}: {e}")

    return ids
