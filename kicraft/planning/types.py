"""Data types for the KiCraft project planning layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class Requirement:
    kind: str
    description: str
    measurable: bool = False
    value: float | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Constraint:
    kind: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PowerRail:
    name: str
    nominal_voltage_v: float
    max_current_a: float
    source_block_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitBlock:
    block_id: str
    name: str
    category: str
    function: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    notes: str = ""
    candidate_part_families: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockConnection:
    from_block: str
    to_block: str
    signal: str
    kind: str = "power"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpenQuestion:
    topic: str
    question: str
    blocks_progress: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectSpec:
    raw_text: str
    name: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    explicit_voltages_v: list[float] = field(default_factory=list)
    explicit_currents_a: list[float] = field(default_factory=list)
    explicit_part_numbers: list[str] = field(default_factory=list)
    explicit_requirements: list[Requirement] = field(default_factory=list)
    explicit_constraints: list[Constraint] = field(default_factory=list)
    input_power_hints: list[str] = field(default_factory=list)
    load_hints: list[str] = field(default_factory=list)
    interface_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["explicit_requirements"] = [r.to_dict() for r in self.explicit_requirements]
        data["explicit_constraints"] = [c.to_dict() for c in self.explicit_constraints]
        return data


@dataclass
class ProjectPlan:
    schema_version: str
    name: str
    slug: str
    spec: ProjectSpec
    summary: str
    requirements: list[Requirement] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    power_rails: list[PowerRail] = field(default_factory=list)
    blocks: list[CircuitBlock] = field(default_factory=list)
    connections: list[BlockConnection] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    research_notes: list[str] = field(default_factory=list)
    research_source: str = "rule-based"
    next_layers: list[str] = field(
        default_factory=lambda: [
            "formalize-design",
            "select-parts",
            "generate-schematic",
            "autoplace-and-route",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "slug": self.slug,
            "spec": self.spec.to_dict(),
            "summary": self.summary,
            "requirements": [r.to_dict() for r in self.requirements],
            "constraints": [c.to_dict() for c in self.constraints],
            "power_rails": [r.to_dict() for r in self.power_rails],
            "blocks": [b.to_dict() for b in self.blocks],
            "connections": [c.to_dict() for c in self.connections],
            "open_questions": [q.to_dict() for q in self.open_questions],
            "research_notes": list(self.research_notes),
            "research_source": self.research_source,
            "next_layers": list(self.next_layers),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectPlan:
        spec_data = data["spec"]
        spec = ProjectSpec(
            raw_text=spec_data.get("raw_text", ""),
            name=spec_data.get("name", data.get("name", "")),
            summary=spec_data.get("summary", ""),
            keywords=list(spec_data.get("keywords", [])),
            explicit_voltages_v=list(spec_data.get("explicit_voltages_v", [])),
            explicit_currents_a=list(spec_data.get("explicit_currents_a", [])),
            explicit_part_numbers=list(spec_data.get("explicit_part_numbers", [])),
            explicit_requirements=[
                Requirement(**r) for r in spec_data.get("explicit_requirements", [])
            ],
            explicit_constraints=[
                Constraint(**c) for c in spec_data.get("explicit_constraints", [])
            ],
            input_power_hints=list(spec_data.get("input_power_hints", [])),
            load_hints=list(spec_data.get("load_hints", [])),
            interface_hints=list(spec_data.get("interface_hints", [])),
        )
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            name=data["name"],
            slug=data["slug"],
            spec=spec,
            summary=data.get("summary", ""),
            requirements=[Requirement(**r) for r in data.get("requirements", [])],
            constraints=[Constraint(**c) for c in data.get("constraints", [])],
            power_rails=[PowerRail(**r) for r in data.get("power_rails", [])],
            blocks=[CircuitBlock(**b) for b in data.get("blocks", [])],
            connections=[BlockConnection(**c) for c in data.get("connections", [])],
            open_questions=[OpenQuestion(**q) for q in data.get("open_questions", [])],
            research_notes=list(data.get("research_notes", [])),
            research_source=data.get("research_source", "rule-based"),
            next_layers=list(
                data.get(
                    "next_layers",
                    [
                        "formalize-design",
                        "select-parts",
                        "generate-schematic",
                        "autoplace-and-route",
                    ],
                )
            ),
        )

    def to_markdown(self) -> str:
        from .formatter import plan_to_markdown

        return plan_to_markdown(self)
