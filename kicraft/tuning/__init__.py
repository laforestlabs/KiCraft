"""Cross-corpus auto-tuning of KiCraft's default placement/routing config.

KiCraft already searches placement parameters *per build* (kicraft/cli/
autoexperiment.py keeps the best layout by ``PlacementScore``). This package
adds the missing *outer* loop: it tunes the global ``DEFAULT_CONFIG``
(kicraft/autoplacer/config.py) to maximize aggregate *routed* outcomes across a
frozen corpus of synthesized boards — a signal nothing else targets.

The premise is spare CPU: ``replay``/compose re-run place+route on an already
synthesized project with **$0 LLM cost**, placement byte-deterministic, only
FreeRouting stochastic (handled by seed replication). See the approved plan at
.claude/plans/declarative-juggling-avalanche.md.

Module map:
  space        param-overlay <-> normalized vector over CONFIG_SEARCH_SPACE
  optimizer    CMA-ES (cma pkg) ask/tell — hard dependency, no fallback
  workspace    scratch-copy + detokenize + config-overlay injection
  corpus       discover/dedupe synthesized workspaces; brief-level train/holdout
  evaluate     evaluate_config(overlay, board, seed) -> EvalResult  (subprocess)
  reward       seed/board aggregation; 3-axis Pareto; scalarization
  store        sqlite cache keyed (config_hash, board, seed, mode)
  screen       sensitivity screening -> active dims
  orchestrator the daemon loop (ask -> eval batch -> tell), slot-aware, resumable
  cli          python -m kicraft.tuning.cli {corpus-stats,screen,run,resume,report,promote}
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
