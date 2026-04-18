"""SQLite database layer for experiment tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {})


def _json_loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _json_loads_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


class ExperimentRun(Base):
    """Top-level experiment run (one per autoexperiment invocation)."""

    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), default="idle")  # idle, running, done, error
    pcb_file = Column(String(500))
    total_rounds = Column(Integer, default=0)
    completed_rounds = Column(Integer, default=0)
    best_score = Column(Float, default=0.0)
    config_json = Column(Text, default="{}")
    source_jsonl = Column(String(500))  # original JSONL file if imported
    notes = Column(Text, default="")

    @property
    def config(self) -> dict[str, Any]:
        return _json_loads_dict(self.config_json)


class Round(Base):
    """One round within a hierarchical subcircuit experiment run."""

    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False, index=True)
    round_num = Column(Integer, nullable=False)
    seed = Column(Integer)
    mode = Column(String(40))  # legacy: minor/major/explore/elite, new: hierarchical
    score = Column(Float, default=0.0)
    kept = Column(Boolean, default=False)

    # Legacy score breakdown
    placement_score = Column(Float)
    route_completion = Column(Float)
    trace_efficiency = Column(Float)
    via_score = Column(Float)
    courtyard_overlap = Column(Float)
    board_containment = Column(Float)

    # DRC / legacy routing
    drc_shorts = Column(Integer, default=0)
    drc_unconnected = Column(Integer, default=0)
    drc_clearance = Column(Integer, default=0)
    drc_courtyard = Column(Integer, default=0)
    drc_total = Column(Integer, default=0)
    duration_s = Column(Float)
    placement_ms = Column(Float)
    routing_ms = Column(Float)
    nets_routed = Column(Integer)
    failed_net_names_json = Column(Text, default="[]")
    config_delta_json = Column(Text, default="{}")
    board_width_mm = Column(Float)
    board_height_mm = Column(Float)

    # Hierarchical experiment fields
    leaf_total = Column(Integer)
    leaf_accepted = Column(Integer)
    parent_composed = Column(Boolean)
    parent_routed = Column(Boolean)
    accepted_trace_count = Column(Integer)
    accepted_via_count = Column(Integer)
    latest_stage = Column(String(80))
    artifact_root = Column(Text)
    composition_json = Column(Text)
    parent_output_json = Column(Text)
    leaf_names_json = Column(Text, default="[]")
    accepted_leaf_names_json = Column(Text, default="[]")
    hierarchy_json = Column(Text, default="{}")
    artifacts_json = Column(Text, default="{}")
    commands_json = Column(Text, default="{}")
    timing_breakdown_json = Column(Text, default="{}")
    leaf_timing_summary_json = Column(Text, default="{}")

    # Shared free-form detail
    details = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def config_delta(self) -> dict[str, Any]:
        return _json_loads_dict(self.config_delta_json)

    @property
    def failed_net_names(self) -> list[Any]:
        return _json_loads_list(self.failed_net_names_json)

    @property
    def leaf_names(self) -> list[Any]:
        return _json_loads_list(self.leaf_names_json)

    @property
    def accepted_leaf_names(self) -> list[Any]:
        return _json_loads_list(self.accepted_leaf_names_json)

    @property
    def hierarchy(self) -> dict[str, Any]:
        return _json_loads_dict(self.hierarchy_json)

    @property
    def artifacts(self) -> dict[str, Any]:
        return _json_loads_dict(self.artifacts_json)

    @property
    def commands(self) -> dict[str, Any]:
        return _json_loads_dict(self.commands_json)

    @property
    def timing_breakdown(self) -> dict[str, Any]:
        return _json_loads_dict(self.timing_breakdown_json)

    @property
    def leaf_timing_summary(self) -> dict[str, Any]:
        return _json_loads_dict(self.leaf_timing_summary_json)


class Preset(Base):
    """Saved configuration preset."""

    __tablename__ = "presets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, unique=True)
    config_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    notes = Column(Text, default="")

    @property
    def config(self) -> dict[str, Any]:
        return _json_loads_dict(self.config_json)


class Database:
    """Manages SQLite connection and provides query helpers."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = self._default_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)

        @event.listens_for(self.engine, "connect")
        def _set_wal(dbapi_conn, _rec):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        Base.metadata.create_all(self.engine)
        self._ensure_round_columns()
        self._Session = sessionmaker(bind=self.engine)

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).resolve().parent.parent / ".experiments" / "experiments.db"

    def _ensure_round_columns(self) -> None:
        """Best-effort additive migration for older local DBs."""
        required_columns: dict[str, str] = {
            "leaf_total": "INTEGER",
            "leaf_accepted": "INTEGER",
            "parent_composed": "BOOLEAN",
            "parent_routed": "BOOLEAN",
            "accepted_trace_count": "INTEGER",
            "accepted_via_count": "INTEGER",
            "latest_stage": "VARCHAR(80)",
            "artifact_root": "TEXT",
            "composition_json": "TEXT",
            "parent_output_json": "TEXT",
            "leaf_names_json": "TEXT DEFAULT '[]'",
            "accepted_leaf_names_json": "TEXT DEFAULT '[]'",
            "hierarchy_json": "TEXT DEFAULT '{}'",
            "artifacts_json": "TEXT DEFAULT '{}'",
            "commands_json": "TEXT DEFAULT '{}'",
            "timing_breakdown_json": "TEXT DEFAULT '{}'",
            "leaf_timing_summary_json": "TEXT DEFAULT '{}'",
        }

        with self.engine.begin() as conn:
            rows = conn.exec_driver_sql("PRAGMA table_info(rounds)").fetchall()
            existing = {str(row[1]) for row in rows}
            for name, sql_type in required_columns.items():
                if name in existing:
                    continue
                conn.exec_driver_sql(f"ALTER TABLE rounds ADD COLUMN {name} {sql_type}")

    def session(self) -> Session:
        return self._Session()

    # -- Experiment runs ---------------------------------------------------

    def create_experiment(
        self,
        name: str,
        pcb_file: str = "",
        total_rounds: int = 0,
        config: dict[str, Any] | None = None,
        source_jsonl: str = "",
    ) -> ExperimentRun:
        with self.session() as s:
            exp = ExperimentRun(
                name=name,
                pcb_file=pcb_file,
                total_rounds=total_rounds,
                config_json=_json_dumps(config or {}),
                source_jsonl=source_jsonl,
            )
            s.add(exp)
            s.commit()
            s.refresh(exp)
            return exp

    def get_experiments(self) -> list[ExperimentRun]:
        with self.session() as s:
            return (
                s.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).all()
            )

    def get_experiment(self, exp_id: int) -> ExperimentRun | None:
        with self.session() as s:
            return s.get(ExperimentRun, exp_id)

    def update_experiment(self, exp_id: int, **kwargs) -> None:
        with self.session() as s:
            exp = s.get(ExperimentRun, exp_id)
            if exp:
                for k, v in kwargs.items():
                    setattr(exp, k, v)
                s.commit()

    # -- Rounds ------------------------------------------------------------

    def add_round(self, experiment_id: int, data: dict[str, Any]) -> Round:
        hierarchy = data.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            hierarchy = {}

        artifacts = data.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        commands = data.get("commands", {})
        if not isinstance(commands, dict):
            commands = {}

        timing_breakdown = data.get("timing_breakdown", {})
        if not isinstance(timing_breakdown, dict):
            timing_breakdown = {}

        leaf_timing_summary = data.get("leaf_timing_summary", {})
        if not isinstance(leaf_timing_summary, dict):
            leaf_timing_summary = {}

        leaf_names = data.get("leaf_names", hierarchy.get("leaf_names", []))
        if not isinstance(leaf_names, list):
            leaf_names = []

        accepted_leaf_names = data.get(
            "accepted_leaf_names",
            hierarchy.get("accepted_leaf_names", []),
        )
        if not isinstance(accepted_leaf_names, list):
            accepted_leaf_names = []

        with self.session() as s:
            r = Round(
                experiment_id=experiment_id,
                round_num=int(data.get("round_num", data.get("round", 0)) or 0),
                seed=data.get("seed"),
                mode=data.get("mode"),
                score=float(data.get("score", 0) or 0),
                kept=bool(data.get("kept", False)),
                placement_score=data.get("placement_score"),
                route_completion=data.get("route_completion"),
                trace_efficiency=data.get("trace_efficiency"),
                via_score=data.get("via_score"),
                courtyard_overlap=data.get("courtyard_overlap"),
                board_containment=data.get("board_containment"),
                drc_shorts=int(data.get("drc_shorts", 0) or 0),
                drc_unconnected=int(data.get("drc_unconnected", 0) or 0),
                drc_clearance=int(data.get("drc_clearance", 0) or 0),
                drc_courtyard=int(data.get("drc_courtyard", 0) or 0),
                drc_total=int(data.get("drc_total", 0) or 0),
                duration_s=data.get("duration_s"),
                placement_ms=data.get("placement_ms"),
                routing_ms=data.get("routing_ms"),
                nets_routed=data.get("nets_routed"),
                failed_net_names_json=_json_dumps(data.get("failed_net_names", [])),
                config_delta_json=_json_dumps(data.get("config_delta", {})),
                board_width_mm=data.get("board_width_mm"),
                board_height_mm=data.get("board_height_mm"),
                leaf_total=data.get("leaf_total", hierarchy.get("leaf_total")),
                leaf_accepted=data.get("leaf_accepted", hierarchy.get("leaf_accepted")),
                parent_composed=data.get(
                    "parent_composed", hierarchy.get("parent_composed")
                ),
                parent_routed=data.get("parent_routed", hierarchy.get("parent_routed")),
                accepted_trace_count=data.get(
                    "accepted_trace_count", hierarchy.get("accepted_trace_count")
                ),
                accepted_via_count=data.get(
                    "accepted_via_count", hierarchy.get("accepted_via_count")
                ),
                latest_stage=data.get("latest_stage", data.get("stage")),
                artifact_root=data.get("artifact_root", artifacts.get("artifact_root")),
                composition_json=data.get(
                    "composition_json", artifacts.get("composition_json")
                ),
                parent_output_json=data.get(
                    "parent_output_json", artifacts.get("parent_output_json")
                ),
                leaf_names_json=_json_dumps(leaf_names),
                accepted_leaf_names_json=_json_dumps(accepted_leaf_names),
                hierarchy_json=_json_dumps(hierarchy),
                artifacts_json=_json_dumps(artifacts),
                commands_json=_json_dumps(commands),
                timing_breakdown_json=_json_dumps(timing_breakdown),
                leaf_timing_summary_json=_json_dumps(leaf_timing_summary),
                details=data.get("details"),
            )
            s.add(r)
            s.commit()
            s.refresh(r)
            return r

    def get_rounds(self, experiment_id: int) -> list[Round]:
        with self.session() as s:
            return (
                s.query(Round)
                .filter(Round.experiment_id == experiment_id)
                .order_by(Round.round_num)
                .all()
            )

    def get_round_dicts(self, experiment_id: int) -> list[dict[str, Any]]:
        """Return rounds as plain dicts for charting and tables."""
        rounds = self.get_rounds(experiment_id)
        result: list[dict[str, Any]] = []
        for r in rounds:
            result.append(
                {
                    "round_num": r.round_num,
                    "seed": r.seed,
                    "mode": r.mode,
                    "score": r.score,
                    "kept": r.kept,
                    "placement_score": r.placement_score,
                    "route_completion": r.route_completion,
                    "trace_efficiency": r.trace_efficiency,
                    "via_score": r.via_score,
                    "courtyard_overlap": r.courtyard_overlap,
                    "board_containment": r.board_containment,
                    "drc_shorts": r.drc_shorts,
                    "drc_unconnected": r.drc_unconnected,
                    "drc_clearance": r.drc_clearance,
                    "drc_courtyard": r.drc_courtyard,
                    "drc_total": r.drc_total,
                    "duration_s": r.duration_s,
                    "placement_ms": r.placement_ms,
                    "routing_ms": r.routing_ms,
                    "nets_routed": r.nets_routed,
                    "failed_net_names": r.failed_net_names,
                    "config_delta": r.config_delta,
                    "board_width_mm": r.board_width_mm,
                    "board_height_mm": r.board_height_mm,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    # Hierarchical fields
                    "leaf_total": r.leaf_total,
                    "leaf_accepted": r.leaf_accepted,
                    "parent_composed": r.parent_composed,
                    "parent_routed": r.parent_routed,
                    "accepted_trace_count": r.accepted_trace_count,
                    "accepted_via_count": r.accepted_via_count,
                    "latest_stage": r.latest_stage,
                    "artifact_root": r.artifact_root,
                    "composition_json": r.composition_json,
                    "parent_output_json": r.parent_output_json,
                    "leaf_names": r.leaf_names,
                    "accepted_leaf_names": r.accepted_leaf_names,
                    "hierarchy": r.hierarchy,
                    "artifacts": r.artifacts,
                    "commands": r.commands,
                    "timing_breakdown": r.timing_breakdown,
                    "leaf_timing_summary": r.leaf_timing_summary,
                }
            )
        return result

    # -- Presets -----------------------------------------------------------

    def save_preset(self, name: str, config: dict[str, Any], notes: str = "") -> Preset:
        with self.session() as s:
            existing = s.query(Preset).filter(Preset.name == name).first()
            if existing:
                existing.config_json = _json_dumps(config)
                existing.notes = notes
                s.commit()
                s.refresh(existing)
                return existing
            p = Preset(name=name, config_json=_json_dumps(config), notes=notes)
            s.add(p)
            s.commit()
            s.refresh(p)
            return p

    def get_presets(self) -> list[Preset]:
        with self.session() as s:
            return s.query(Preset).order_by(Preset.name).all()

    def load_preset(self, name: str) -> dict[str, Any] | None:
        with self.session() as s:
            p = s.query(Preset).filter(Preset.name == name).first()
            return p.config if p else None

    def delete_preset(self, name: str) -> bool:
        with self.session() as s:
            p = s.query(Preset).filter(Preset.name == name).first()
            if p:
                s.delete(p)
                s.commit()
                return True
            return False
