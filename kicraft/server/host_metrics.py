"""Always-on host-resource sampler for the /admin dashboard.

A single daemon thread samples CPU / RAM / drive usage every ``interval_s``
(default 30 s) and appends a row to a small SQLite store, so the admin
overview can chart host-resource *trends over time* -- not just the
per-loadtest-run snapshots that ``kicraft/loadtest/metrics.py`` captures.

The prod ``server`` extra has no psutil, so the probe falls back to ``/proc``
+ stdlib (``/proc/stat`` for CPU, ``/proc/meminfo`` for RAM,
``shutil.disk_usage`` for the drive); psutil is used when the optional
``loadtest`` extra is installed. The store is append-only under WAL with one
connection per op (the same pattern as accounts.py / spend_guard.py / the
loadtest store), so the sampler thread and the admin page never block each
other. The store path is independent of ``Settings`` (mirrors the loadtest
store) so reading it never requires an OPENROUTER_API_KEY.
"""
from __future__ import annotations

import math
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

try:  # optional fast path (the `loadtest` extra); /proc fallback covers the prod box
    import psutil  # type: ignore
except Exception:  # pragma: no cover - exercised on boxes without the extra
    psutil = None

_MB = 1024.0 * 1024.0
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# Sampling cadence. 30 s keeps a year of rows well under 1.1 M (sqlite handles
# that comfortably) while a 7-day default window plots ~20k points, which we
# downsample in the chart layer before sending to the browser.
DEFAULT_INTERVAL_S = 30.0
# Purge rows older than this on sampler start (keeps the DB from growing
# without bound across a long-lived process / many restarts).
DEFAULT_RETENTION_DAYS = 365

_SCHEMA = """
CREATE TABLE IF NOT EXISTS host_samples (
    ts           REAL PRIMARY KEY,   -- unix seconds, floored to the interval
    cpu_pct      REAL,
    mem_used_mb  REAL,
    mem_pct      REAL,
    disk_used_mb REAL,
    disk_total_mb REAL,
    disk_pct     REAL
);
CREATE INDEX IF NOT EXISTS idx_host_samples_ts ON host_samples(ts);
"""


def default_store_path() -> Path:
    """Host-metrics DB location -- ``~/.kicraft/host_metrics.db`` unless
    ``KICRAFT_HOST_METRICS_DIR`` overrides the directory. Decoupled from
    ``Settings`` so the admin page can read it without an API key."""
    root = os.environ.get("KICRAFT_HOST_METRICS_DIR", "").strip()
    base = Path(root) if root else Path.home() / ".kicraft"
    return base / "host_metrics.db"


def default_disk_path() -> Path:
    """The filesystem the drive-space chart reports on -- the KiCraft data
    root by default (where projects/live), overridable via env."""
    p = os.environ.get("KICRAFT_HOST_METRICS_DISK", "").strip()
    return Path(p) if p else Path.home() / ".kicraft"


class HostMetricsStore:
    """SQLite-backed time series of host CPU/RAM/drive samples."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as db:
            db.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=30.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def record(self, sample: dict) -> None:
        """Insert/replace one sample row. ``ts`` is the PRIMARY KEY -- callers
        floor it to the interval so a quick restart never double-writes."""
        with self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO host_samples "
                "(ts, cpu_pct, mem_used_mb, mem_pct, disk_used_mb, disk_total_mb, disk_pct) "
                "VALUES (:ts, :cpu_pct, :mem_used_mb, :mem_pct, "
                ":disk_used_mb, :disk_total_mb, :disk_pct)",
                {
                    "ts": sample["ts"],
                    "cpu_pct": sample.get("cpu_pct"),
                    "mem_used_mb": sample.get("mem_used_mb"),
                    "mem_pct": sample.get("mem_pct"),
                    "disk_used_mb": sample.get("disk_used_mb"),
                    "disk_total_mb": sample.get("disk_total_mb"),
                    "disk_pct": sample.get("disk_pct"),
                },
            )

    def series(self, since: float | None = None,
               until: float | None = None) -> list[dict]:
        """Ascending samples in ``[since, until]`` (open-ended when None)."""
        sql = "SELECT * FROM host_samples"
        clauses, params = [], []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("ts <= ?")
            params.append(float(until))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts ASC"
        with self._conn() as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as db:
            return int(db.execute("SELECT COUNT(*) FROM host_samples").fetchone()[0])

    def purge_before(self, ts: float) -> int:
        """Delete rows older than ``ts``; return the number removed."""
        with self._conn() as db:
            cur = db.execute("DELETE FROM host_samples WHERE ts < ?", (float(ts),))
            return int(cur.rowcount or 0)


class _HostSampleProbe:
    """CPU / RAM / drive via psutil or ``/proc`` + stdlib (mirrors the loadtest
    ``_HostProbe``, but reports drive *used/total/pct* -- the dashboard's drive
    chart wants usage, not just free)."""

    def __init__(self, disk_path: str | Path) -> None:
        self.disk_path = str(disk_path)
        self._cpu_prev: tuple[int, int] | None = None
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
        prev, self._cpu_prev = self._cpu_prev, (total, idle)
        if prev is None:
            return None  # first sample has no delta baseline
        dt, di = total - prev[0], idle - prev[1]
        return round(100.0 * (1.0 - di / dt), 2) if dt > 0 else None

    def _mem(self) -> tuple[float | None, float | None]:
        if psutil is not None:
            vm = psutil.virtual_memory()
            return round(vm.used / _MB, 1), round(vm.percent, 1)
        try:
            info: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
        except (OSError, ValueError):
            return None, None
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - avail
        pct = round(100.0 * used / total, 1) if total else None
        return (round(used / _MB, 1), pct)

    def _disk(self) -> tuple[float | None, float | None, float | None]:
        try:
            du = shutil.disk_usage(self.disk_path)
        except OSError:
            return None, None, None
        used_mb = round(du.used / _MB, 1)
        total_mb = round(du.total / _MB, 1)
        pct = round(100.0 * du.used / du.total, 1) if du.total else None
        return used_mb, total_mb, pct

    def sample(self) -> dict:
        used_mb, mem_pct = self._mem()
        disk_used, disk_total, disk_pct = self._disk()
        return {
            "cpu_pct": self._cpu_pct(),
            "mem_used_mb": used_mb,
            "mem_pct": mem_pct,
            "disk_used_mb": disk_used,
            "disk_total_mb": disk_total,
            "disk_pct": disk_pct,
        }


class HostMetricsSampler(threading.Thread):
    """Background sampler: append one host row to ``store`` every
    ``interval_s`` until stopped. Usable as a context manager. Never raises --
    a sampling hiccup must not kill the dashboard's data feed."""

    def __init__(self, store: HostMetricsStore, *,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 disk_path: str | Path | None = None) -> None:
        super().__init__(daemon=True, name="kicraft-host-metrics")
        self.store = store
        self.interval_s = float(interval_s)
        self.disk_path = str(disk_path) if disk_path else str(default_disk_path())
        self._probe = _HostSampleProbe(self.disk_path)
        self._stop_evt = threading.Event()

    def _floored_ts(self) -> float:
        return math.floor(time.time() / self.interval_s) * self.interval_s

    def sample(self) -> dict:
        return {"ts": self._floored_ts(), **self._probe.sample()}

    def run(self) -> None:
        # Bounded retention: cull ancient rows once at start so a long-lived
        # box does not accumulate years of points.
        try:
            self.store.purge_before(time.time() - DEFAULT_RETENTION_DAYS * 86400)
        except Exception:  # pragma: no cover - never fatal
            pass
        while not self._stop_evt.is_set():
            try:
                self.store.record(self.sample())
            except Exception:  # pragma: no cover - never kill the feed
                pass
            self._stop_evt.wait(self.interval_s)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def __enter__(self) -> "HostMetricsSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# --- process-singleton starter (the web process starts exactly one) --------- #
_SAMPLER: HostMetricsSampler | None = None
_SAMPLER_LOCK = threading.Lock()


def get_store(path: str | Path | None = None) -> HostMetricsStore:
    """Convenience accessor the admin route uses to read the series."""
    return HostMetricsStore(path)


def start_host_metrics_sampler(*, interval_s: float | None = None,
                               disk_path: str | Path | None = None,
                               store: HostMetricsStore | None = None) -> HostMetricsSampler:
    """Start the one background sampler for this process (idempotent)."""
    global _SAMPLER
    with _SAMPLER_LOCK:
        if _SAMPLER is not None and _SAMPLER.is_alive():
            return _SAMPLER
        store = store or HostMetricsStore()
        interval = float(interval_s) if interval_s is not None else DEFAULT_INTERVAL_S
        disk = disk_path or default_disk_path()
        _SAMPLER = HostMetricsSampler(store, interval_s=interval, disk_path=disk)
        _SAMPLER.start()
        return _SAMPLER


def stop_host_metrics_sampler(timeout: float = 2.0) -> None:
    """Test helper: stop and forget the process sampler."""
    global _SAMPLER
    with _SAMPLER_LOCK:
        s = _SAMPLER
        _SAMPLER = None
    if s is not None:
        s.stop(timeout=timeout)