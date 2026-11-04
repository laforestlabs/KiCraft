import os
from pathlib import Path
import shutil
import sqlite3
import subprocess


REPO = Path(__file__).resolve().parents[1]


def test_snapshot_restore_preserves_literal_database_paths(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("phase0_snapshot.sh", "phase0_restore.sh"):
        shutil.copy2(REPO / "scripts" / name, scripts / name)
    bins = tmp_path / "bin"
    bins.mkdir()
    systemctl = bins / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 0\n")
    systemctl.chmod(0o755)
    account = tmp_path / "accounts' $HOME.db"
    ledger = tmp_path / "ledger' $HOME.db"
    for db in (account, ledger):
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE marker (value TEXT)")
            conn.execute("INSERT INTO marker VALUES ('before')")
    with sqlite3.connect(ledger) as conn:
        conn.execute("CREATE TABLE spend (cost_usd REAL)")
        conn.execute("INSERT INTO spend VALUES (1.25)")
    env = {**os.environ, "HOME": str(tmp_path), "KICRAFT_USERS_DB": str(account),
           "KICRAFT_SPEND_LEDGER": str(ledger), "PYTHONPATH": str(REPO),
           "PATH": str(bins) + os.pathsep + os.environ["PATH"]}
    snapshot = tmp_path / "snapshot"
    subprocess.run(["bash", str(scripts / "phase0_snapshot.sh"), str(snapshot)],
                   env=env, check=True, capture_output=True, text=True)
    assert "spent_total_usd=1.2500" in (snapshot / "meta.txt").read_text()
    for db in (account, ledger):
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE marker SET value='after'")
    subprocess.run(["bash", str(scripts / "phase0_restore.sh"), str(snapshot)],
                   env=env, check=True, capture_output=True, text=True)
    for db in (account, ledger):
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT value FROM marker").fetchone() == ("before",)


def test_snapshot_and_restore_reject_colliding_basenames(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("phase0_snapshot.sh", "phase0_restore.sh"):
        shutil.copy2(REPO / "scripts" / name, scripts / name)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(REPO),
           "KICRAFT_USERS_DB": str(tmp_path / "a" / "state.db"),
           "KICRAFT_SPEND_LEDGER": str(tmp_path / "b" / "state.db")}
    for name in ("phase0_snapshot.sh", "phase0_restore.sh"):
        result = subprocess.run(["bash", str(scripts / name), str(snapshot)],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 2
        assert "basenames must differ" in result.stderr
