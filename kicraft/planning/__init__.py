"""KiCraft project planning layer.

This is the top-most layer of KiCraft. It takes a user's freeform text
description of a new (greenfield) electronics project and produces a
formalized project plan including circuit topology at the block level.

It intentionally stops BEFORE:

- selecting exact part numbers
- producing a real schematic
- producing a real PCB layout

Those are the responsibility of later layers (a future schematic-formalization
layer, a future part-selection / sourcing layer, and the existing autoplacer
pipeline under ``kicraft.autoplacer``).

This package has zero dependency on ``pcbnew`` or any other KiCad runtime,
and zero dependency on the autoplacer / placement / routing modules. It is
pure Python and safe to import in any environment.

Public API:

    from kicraft.planning import plan_project, ProjectPlan, ProjectSpec

    plan = plan_project("USB-C powered ESP32 dev board with one 3.3V rail "
                        "at 500mA and a status LED")
    print(plan.to_markdown())
"""

from __future__ import annotations

from .formatter import plan_to_markdown
from .planner import PlanResult, plan_project
from .types import (
    BlockConnection,
    CircuitBlock,
    Constraint,
    OpenQuestion,
    PowerRail,
    ProjectPlan,
    ProjectSpec,
    Requirement,
)

__all__ = [
    "BlockConnection",
    "CircuitBlock",
    "Constraint",
    "OpenQuestion",
    "PlanResult",
    "PowerRail",
    "ProjectPlan",
    "ProjectSpec",
    "Requirement",
    "plan_project",
    "plan_to_markdown",
]
