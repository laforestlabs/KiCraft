"""Subprocess manager for hierarchical autoexperiment.py."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class ExperimentRunner:
    """Launches and monitors hierarchical autoexperiment.py as a subprocess."""

    def __init__(self, project_root: Path, scripts_dir: Path, experiments_dir: Path):
        self.project_root = project_root
        self.scripts_dir = scripts_dir
        self.experiments_dir = experiments_dir
        self._process: subprocess.Popen | None = None
        self._pid_file = experiments_dir / "experiment.pid"

    @property
    def is_running(self) -> bool:
        if self._process is not None:
            if self._process.poll() is None:
                return True
            self._process = None
            self._cleanup_stale_state()
            return False
        if self._pid_file.exists():
            try:
                pid = int(self._pid_file.read_text().strip())
                os.kill(pid, 0)
                return True
            except (ValueError, ProcessLookupError, PermissionError):
                self._cleanup_stale_state()
        return False

    def _purge_prior_run_artifacts(self, phase: str | None = None) -> None:
        """Clear per-run outputs so the Monitor tab reflects the new run.

        Removes per-round metadata, the parent-round log, the
        run-status files, and per-leaf round snapshots. Keeps:

        * Pinned canonical leaf artifacts (``leaf_routed.kicad_pcb``,
          ``solved_layout.json``, ``metadata.json``, ``renders/``) so
          ``pins.json`` references survive between leaves-only runs and
          ``--parents-only`` reads stay valid.
        * User/config files (presets DB, program.md, config overlays).

        Per-leaf ``round_NNNN_*`` snapshot files and ``debug.json``
        are wiped on every run (regardless of phase). These accumulate
        across runs and are the source of the "rounds 1..14 in a 3x3
        run" GUI confusion: the score plot is built from
        ``debug.json``'s ``all_rounds`` list, and prior runs' entries
        used to linger.

        Phase-aware: in ``--parents-only`` and ``--leaves-only`` modes
        the ``subcircuits/`` dir's per-leaf canonical files are
        preserved (only round-snapshot files inside each leaf dir are
        removed). On a full run the dir is wiped wholesale because
        leaves are about to be re-solved from scratch.
        """
        exp = self.experiments_dir
        if not exp.exists():
            return

        # subcircuits/ contains canonical leaf artifacts that pins
        # depend on. Cleanup is phase-aware:
        #
        # * full run (phase=None): wipe subcircuits/ wholesale -- leaves
        #   are about to be re-solved from scratch anyway.
        # * leaves-only: keep canonical (so existing pins survive) but
        #   drop per-leaf debug.json + round_NNNN_* snapshots so the
        #   new run produces clean round indices [0..N-1] in the GUI's
        #   score plot. This is the fix for "rounds 1..14 in a 3x3 run".
        # * parents-only: do NOT touch subcircuits/. Leaves were
        #   pinned from a previous leaves-only run; their debug.json
        #   is the source of truth for the GUI's leaf round timeline
        #   AND for selected_round filtering. Wiping it makes every
        #   leaf card flip to "FAILED" because the round filter sees
        #   rounds=[] (the parents-only run produces no leaf rounds).
        purge_subcircuits_wholesale = phase is None
        purge_per_leaf_snapshots = phase == "leaves_only"
        round_dirs = ["rounds", "frames", "hierarchical_autoexperiment"]
        if purge_subcircuits_wholesale:
            round_dirs.insert(0, "subcircuits")

        for sub in round_dirs:
            path = exp / sub
            if path.exists():
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except OSError:
                    pass

        if purge_per_leaf_snapshots:
            self._purge_per_leaf_round_snapshots(exp / "subcircuits")

        for name in (
            "experiments.jsonl",
            "run_status.json",
            "run_status.txt",
            "parent_composition_routed.json",
            "hierarchical_summary.json",
            "experiment.log",
        ):
            (exp / name).unlink(missing_ok=True)

        exp.mkdir(parents=True, exist_ok=True)

    def _purge_per_leaf_round_snapshots(self, sub_root: Path) -> None:
        """Drop ``round_NNNN_*`` snapshots and ``debug.json`` per leaf.

        Pin canonical files (``leaf_routed.kicad_pcb``,
        ``solved_layout.json``, ``metadata.json``, ``renders/``) are
        preserved so an ongoing pin keeps applying. The per-attempt
        snapshots and the cumulative-history ``debug.json`` are
        ephemeral run-state and have no business lingering once a new
        run starts solving the same leaves.
        """
        if not sub_root.is_dir():
            return
        for leaf_dir in sub_root.iterdir():
            if not leaf_dir.is_dir():
                continue
            for snapshot in leaf_dir.glob("round_*"):
                # Top-level files only; leaf renders live under
                # leaf_dir / "renders" / round_NNNN_*.png and we leave
                # those in place because they're cheap to overwrite
                # by the next run.
                if snapshot.is_file():
                    try:
                        snapshot.unlink()
                    except OSError:
                        pass
            debug_json = leaf_dir / "debug.json"
            if debug_json.exists():
                try:
                    debug_json.unlink()
                except OSError:
                    pass

    def _cleanup_stale_state(self) -> None:
        """Remove stale PID/stop files and mark status as done."""
        self._pid_file.unlink(missing_ok=True)
        (self.experiments_dir / "stop.now").unlink(missing_ok=True)
        status_path = self.experiments_dir / "run_status.json"
        if status_path.exists():
            try:
                with open(status_path) as f:
                    status = json.load(f)
                if status.get("phase") in ("running", "stopping"):
                    status["phase"] = "done"
                    with open(status_path, "w") as f:
                        json.dump(status, f, indent=2)
            except (json.JSONDecodeError, OSError):
                pass

    def start(
        self,
        pcb_file: str,
        rounds: int,
        workers: int = 0,
        seed: int | None = None,
        param_ranges: dict | None = None,
        score_weights: dict | None = None,
        extra_config: dict | None = None,
        phase: str | None = None,
    ) -> int:
        """Start a hierarchical experiment subprocess. Returns PID."""
        if self.is_running:
            raise RuntimeError("Experiment already running")

        stop_file = self.experiments_dir / "stop.now"
        stop_file.unlink(missing_ok=True)

        self._purge_prior_run_artifacts(phase=phase)

        autoexp = self.scripts_dir / "autoexperiment.py"
        pcb_path = self.project_root / pcb_file
        if extra_config and extra_config.get("schematic_file"):
            schematic_file = str(extra_config["schematic_file"])
        else:
            # Derive schematic from pcb_file by swapping the extension
            schematic_file = pcb_file.replace(".kicad_pcb", ".kicad_sch")
        schematic_path = self.project_root / schematic_file

        hierarchical_workers = workers
        if extra_config and extra_config.get("leaf_workers") is not None:
            hierarchical_workers = int(extra_config["leaf_workers"])

        cmd = [
            sys.executable,
            str(autoexp),
            str(pcb_path),
            "--schematic",
            str(schematic_path),
            "--rounds",
            str(rounds),
            "--workers",
            str(hierarchical_workers),
            "--status-file",
            str(self.experiments_dir / "run_status.json"),
            "--log",
            str(self.experiments_dir / "experiments.jsonl"),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if phase == "leaves_only":
            cmd += ["--leaves-only"]
        elif phase == "parents_only":
            cmd += ["--parents-only"]
        if extra_config:
            parent = extra_config.get("parent")
            if parent:
                cmd += ["--parent", str(parent)]

            only = extra_config.get("only", [])
            if isinstance(only, list):
                for selector in only:
                    cmd += ["--only", str(selector)]

            leaf_rounds = extra_config.get("leaf_rounds")
            if leaf_rounds is not None:
                cmd += ["--leaf-rounds", str(leaf_rounds)]

            # Write config overlay merging project config + GUI overrides.
            # This ensures project-specific settings (ic_groups, component_zones,
            # power_nets, etc.) are preserved when GUI overrides are active.
            placement_cfg = extra_config.get("placement_config")
            if placement_cfg and isinstance(placement_cfg, dict):
                from kicraft.autoplacer.config import (
                    discover_project_config,
                    load_project_config,
                )

                merged: dict[str, Any] = {}
                discovered = discover_project_config(self.project_root)
                if discovered:
                    merged.update(load_project_config(str(discovered)))
                merged.update(placement_cfg)

                overlay_path = self.experiments_dir / "gui_config_overlay.json"
                self.experiments_dir.mkdir(parents=True, exist_ok=True)
                serializable = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in merged.items()
                }
                with open(overlay_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2)
                    f.write("\n")
                cmd += ["--config", str(overlay_path)]

        if param_ranges:
            ranges_path = self.experiments_dir / "gui_param_ranges.json"
            self.experiments_dir.mkdir(parents=True, exist_ok=True)
            with open(ranges_path, "w", encoding="utf-8") as f:
                json.dump(param_ranges, f, indent=2)
                f.write("\n")
            cmd += ["--param-ranges", str(ranges_path)]

        program_path = self.scripts_dir / "program.md"
        program_data: dict[str, Any] = {
            "param_ranges": param_ranges or {},
            "score_weights": score_weights or {},
            "hierarchical_workers": hierarchical_workers,
        }
        if extra_config:
            program_data.update(extra_config)

        with open(program_path, "w") as f:
            f.write("# Hierarchical Experiment Program (auto-generated by GUI)\n\n")
            f.write("```json\n")
            f.write(json.dumps(program_data, indent=2))
            f.write("\n```\n")

        log_path = self.experiments_dir / "experiment.log"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", buffering=1)
        self._process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        self._pid_file.write_text(str(self._process.pid))
        return self._process.pid

    def stop(self) -> None:
        """Request graceful stop via signal file."""
        if not self.is_running:
            self._cleanup_stale_state()
            return
        stop_file = self.experiments_dir / "stop.now"
        stop_file.touch()

    def kill(self) -> None:
        """Force kill the subprocess and its entire process group."""
        pid = None
        if self._process and self._process.poll() is None:
            pid = self._process.pid
        elif self._pid_file.exists():
            try:
                pid = int(self._pid_file.read_text().strip())
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError, PermissionError):
                pid = None

        if pid is not None:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
            else:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass

        if self._process:
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._cleanup_stale_state()

    def read_status(self) -> dict:
        """Read current run_status.json."""
        status_path = self.experiments_dir / "run_status.json"
        if not status_path.exists():
            return {
                "phase": "idle",
                "round": 0,
                "total_rounds": 0,
                "progress_percent": 0,
                "workers": {"total": 0, "in_flight": 0, "idle": 0},
                "kept_count": 0,
                "best_score": 0,
                "latest_score": None,
                "elapsed_s": 0,
                "eta_s": 0,
                "maybe_stuck": False,
                "hierarchy": {
                    "leaf_total": 0,
                    "leaf_completed": 0,
                    "parent_total": 0,
                    "parent_completed": 0,
                    "current_node": None,
                },
            }
        try:
            with open(status_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"phase": "error", "error": "Failed to read status file"}

    def read_latest_rounds(self, since_round: int = 0) -> list[dict]:
        """Read new records from experiments.jsonl since a given round number."""
        jsonl_path = self.experiments_dir / "experiments.jsonl"
        if not jsonl_path.exists():
            return []
        rounds = []
        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("round_num", 0) > since_round:
                            rounds.append(rec)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return rounds

    def get_stdout_tail(self, n_lines: int = 50) -> str:
        """Read last N lines of subprocess stdout/stderr from experiment.log."""
        log_path = self.experiments_dir / "experiment.log"
        if not log_path.exists():
            return ""
        try:
            with open(log_path) as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except OSError:
            return ""
