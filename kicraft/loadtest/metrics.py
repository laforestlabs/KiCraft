"""Host / process / queue sampler for load-test runs.

A background thread samples at a fixed cadence (1 Hz by default) and appends rows
to a LoadResultStore, so a whole push-to-failure is reconstructable afterward and
the /admin/loadtest dashboard can chart it live. Uses ``psutil`` when present and
falls back to ``/proc`` + stdlib otherwise (the prod box is Linux), so the sampler
runs before the ``loadtest`` extra is installed.

Captured per sample: host CPU%/mem/loadavg/disk-free, summed RSS of the web and
build-worker processes, build-queue depth/running, SQLite WAL bytes, and an
optional ``BEGIN IMMEDIATE`` lock-acquire latency probe (write-contention signal).
The pid/queue/wal sources are injected as callables so this module stays decoupled
from accounts.py and is unit-testable with stubs.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

try:  # optional fast path
    import psutil  # type: ignore
except Exception:  # pragma: no cover - exercised on boxes without the extra
    psutil = None

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_MB = 1024.0 * 1024.0


def find_pids(token: str) -> list[int]:
    """PIDs whose cmdline contains ``token`` (e.g. 'kicraft.server.web')."""
    if psutil is not None:
        out = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                if token in " ".join(p.info["cmdline"] or []):
                    out.append(p.info["pid"])
            except Exception:
                continue
        return out
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", "replace")
        except OSError:
            continue
        if token in cmd:
            out.append(int(entry.name))
    return out


def rss_mb(pids) -> float:
    """Summed resident-set size (MB) of the given pids; dead pids are skipped."""
    total = 0
    for pid in pids or ():
        try:
            if psutil is not None:
                total += psutil.Process(pid).memory_info().rss
            else:
                fields = (Path(f"/proc/{pid}/statm").read_text().split())
                total += int(fields[1]) * _PAGE_SIZE  # resident pages
        except (OSError, ValueError, IndexError):
            continue
        except Exception:  # pragma: no cover - psutil.NoSuchProcess etc.
            continue
    return round(total / _MB, 2)


def wal_bytes(paths) -> int:
    """Summed size of the given SQLite ``-wal`` sidecar files (0 if absent)."""
    total = 0
    for p in paths or ():
        try:
            total += Path(p).stat().st_size
        except OSError:
            continue
    return total


def lock_latency_ms(db_path: str | Path, timeout_s: float = 5.0) -> float | None:
    """Time to grab a write lock (``BEGIN IMMEDIATE``) on the DB, in ms.

    A rising value under load is the SQLite write-contention signal that tells you
    when to move the accounts/queue DB to Postgres. Returns None if the DB is
    missing or the lock could not be taken within ``timeout_s``.
    """
    p = Path(db_path)
    if not p.exists():
        return None
    db = sqlite3.connect(str(p), timeout=timeout_s)
    try:
        t0 = time.perf_counter()
        db.execute("BEGIN IMMEDIATE")
        dt = (time.perf_counter() - t0) * 1000.0
        db.execute("ROLLBACK")
        return round(dt, 3)
    except sqlite3.OperationalError:
        return None
    finally:
        db.close()


class _HostProbe:
    """CPU / memory / loadavg / disk via psutil or /proc."""

    def __init__(self, disk_path: str | Path) -> None:
        self.disk_path = str(disk_path)
        self._prev = None  # (total, idle) for the /proc cpu delta
        if psutil is not None:
            psutil.cpu_percent(None)  # prime the psutil delta baseline

    def _cpu_pct(self) -> float | None:
        if psutil is not None:
            return psutil.cpu_percent(None)
        try:
            parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            vals = [int(x) for x in parts]
        except (OSError, ValueError):
            return None
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        prev, self._prev = self._prev, (total, idle)
        if prev is None:
            return None  # first sample has no delta baseline
        dt, di = total - prev[0], idle - prev[1]
        return round(100.0 * (1.0 - di / dt), 2) if dt > 0 else None

    def _mem(self) -> tuple[float | None, float | None]:
        if psutil is not None:
            vm = psutil.virtual_memory()
            return round(vm.used / _MB, 1), round(vm.percent, 1)
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
        except (OSError, ValueError):
            return None, None
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - avail
        return round(used / _MB, 1), (round(100.0 * used / total, 1) if total else None)

    def sample(self) -> dict:
        used_mb, pct = self._mem()
        try:
            loadavg = os.getloadavg()[0]
        except (OSError, AttributeError):
            loadavg = None
        try:
            free_mb = round(shutil.disk_usage(self.disk_path).free / _MB, 1)
        except OSError:
            free_mb = None
        return {"cpu_pct": self._cpu_pct(), "mem_used_mb": used_mb, "mem_pct": pct,
                "loadavg": loadavg, "disk_free_mb": free_mb}


class MetricsSampler(threading.Thread):
    """Background sampler: append a host/process/queue row to ``store`` every
    ``interval_s`` until stopped. Usable as a context manager."""

    def __init__(self, store, run_id: str, *, interval_s: float = 1.0,
                 disk_path: str | Path | None = None, web_pids_probe=None,
                 worker_pids_probe=None, queue_probe=None, wal_paths=(),
                 lock_db: str | Path | None = None) -> None:
        super().__init__(daemon=True, name=f"loadtest-metrics-{run_id}")
        self.store = store
        self.run_id = run_id
        self.interval_s = interval_s
        self.disk_path = disk_path or (Path.home() / ".kicraft")
        self._host = _HostProbe(self.disk_path)
        self._web_pids = web_pids_probe
        self._worker_pids = worker_pids_probe
        self._queue = queue_probe
        self._wal_paths = list(wal_paths)
        self._lock_db = lock_db
        self._stop_evt = threading.Event()

    def sample(self) -> dict:
        s = {"ts": time.time(), **self._host.sample()}
        s["web_rss_mb"] = rss_mb(self._web_pids()) if self._web_pids else None
        s["worker_rss_mb"] = rss_mb(self._worker_pids()) if self._worker_pids else None
        if self._queue:
            depth, running = self._queue()
            s["queue_depth"], s["queue_running"] = depth, running
        else:
            s["queue_depth"] = s["queue_running"] = None
        s["wal_bytes"] = wal_bytes(self._wal_paths) if self._wal_paths else None
        s["lock_ms"] = lock_latency_ms(self._lock_db) if self._lock_db else None
        return s

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self.store.add_sample(self.run_id, self.sample())
            except Exception:  # never let a sampling hiccup kill the load run
                pass
            self._stop_evt.wait(self.interval_s)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def __enter__(self) -> "MetricsSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
