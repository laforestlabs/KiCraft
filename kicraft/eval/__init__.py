"""KiCraft run-evaluation core.

The shippable home of the scoring contract (``rubric.yaml``), the deterministic
Class-C scorers, the script gates, and the finalize math. It is shared by two
front-ends:

  * the offline skill-eval harness in ``tests/skill-eval/bin/`` (portable subject
    agent plus observer, with an optional agent transcript), and
  * the in-app web self-evaluation (``kicraft.eval.run_web``), where the Class-C
    half scores from the web run's own artifacts and the Class-J half is graded
    by an automated LLM judge (``kicraft.eval.judge``) against the same rubric.

Nothing here imports the server, so the harness can use it without a web stack;
the web caller injects its OpenRouter client into the judge.
"""
from __future__ import annotations

from .rubric import RUBRIC_PATH, compute_hash, load_rubric, write_stored_hash
from .scoring import (
    CLASS_C_SCORERS,
    compute_latency_min,
    dim_by_id,
    eval_script_gates,
    finalize_report,
    grade_for,
    metrics_block,
    score_class_c_dims,
)

__all__ = [
    "RUBRIC_PATH", "load_rubric", "compute_hash", "write_stored_hash",
    "CLASS_C_SCORERS", "eval_script_gates", "score_class_c_dims", "metrics_block",
    "finalize_report", "grade_for", "dim_by_id", "compute_latency_min",
]
