"""Stage modules. Each exports run(state) -> (slot, [Question])."""
from __future__ import annotations

from . import architecture, bom, functional_spec, intent
from . import synthesis as synthesis_stage

# Registry the orchestrator uses to dispatch by name.
STAGES = {
    "intent": intent,
    "functional_spec": functional_spec,
    "architecture": architecture,
    "bom": bom,
}

__all__ = [
    "STAGES",
    "intent",
    "functional_spec",
    "architecture",
    "bom",
    "synthesis_stage",
]
