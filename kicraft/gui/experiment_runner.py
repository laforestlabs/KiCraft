"""Subprocess manager for hierarchical autoexperiment.py."""

from __future__ import annotations

import json
import os
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
        plateau: int = 3,
        seed: int | None = None,
        param_ranges: dict | None = None,
        score_weights: dict | None = None,
        extra_config: dict | None = None,
    ) -> int:
        """Start a hierarchical experiment subprocess. Returns PID."""
        if self.is_running:
            raise RuntimeError("Experiment already running")

        stop_file = self.experiments_dir / "stop.now"
        stop_file.unlink(missing_ok=True)

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
            "--plateau",
            str(plateau),
            "--status-file",
            str(self.experiments_dir / "run_status.json"),
            "--log",
            str(self.experiments_dir / "experiments.jsonl"),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
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

            # Write placement config overlay as JSON and pass via --config
            placement_cfg = extra_config.get("placement_config")
            if placement_cfg and isinstance(placement_cfg, dict):
                overlay_path = self.experiments_dir / "gui_config_overlay.json"
                self.experiments_dir.mkdir(parents=True, exist_ok=True)
                with open(overlay_path, "w", encoding="utf-8") as f:
                    json.dump(placement_cfg, f, indent=2)
                    f.write("\n")
                cmd += ["--config", str(overlay_path)]

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

        self._process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
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
        """Read last N lines of subprocess stdout if available."""
        if self._process and self._process.stdout:
            pass
        return ""
