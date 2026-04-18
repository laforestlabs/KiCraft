"""Base classes for layout scoring checks."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Issue:
    severity: str  # "error", "warning", "info"
    message: str
    location: Optional[dict] = None  # {"x": mm, "y": mm, "ref": str, "net": str}


@dataclass
class CheckResult:
    score: float  # 0-100
    issues: list[Issue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    summary: str = ""


class LayoutCheck:
    name: str = ""
    display_name: str = ""
    weight: float = 0.0

    def run(self, board, config: dict) -> CheckResult:
        raise NotImplementedError
