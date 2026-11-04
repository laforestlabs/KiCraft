"""Exercise watcher trips in an isolated fake workspace and fake systemctl."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("stop_worker", [False, True])
@pytest.mark.parametrize("fail_stop", [False, True])
def test_trip_sets_live_guard_and_optionally_requests_worker_stop(tmp_path, stop_worker, fail_stop):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).parents[1] / "scripts/phase0_watch.sh"
    shutil.copy(source, scripts / "phase0_watch.sh")
    (tmp_path / ".env").write_text("OTHER=keep\nKICRAFT_KILL_SWITCH=0\n")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    ctl = fakebin / "systemctl"
    ctl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        'if [ "$FAIL_STOP" = 1 ]; then exit 1; fi\n'
    )
    ctl.chmod(0o755)
    calls = tmp_path / "calls"
    env = dict(os.environ, PATH=f"{fakebin}:{os.environ['PATH']}", CALLS=str(calls),
               FAIL_STOP="1" if fail_stop else "0")
    args = ["bash", str(scripts / "phase0_watch.sh"), "--watch-path", str(tmp_path),
            "--min-disk-gib", "999999999", "--abort-file", str(tmp_path / "abort")]
    if stop_worker:
        args.append("--stop-worker")
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert (calls.read_text().splitlines() if calls.exists() else []) == (
        ["stop kicraft-build-worker"] if stop_worker else []
    )
    assert (tmp_path / "abort").read_text().strip() == "abort"
    assert (tmp_path / ".env").read_text() == "OTHER=keep\nKICRAFT_KILL_SWITCH=0\n\nKICRAFT_KILL_SWITCH=1\n"
    assert "already admitted may still complete" in result.stdout
    assert ("operator intervention required" in result.stderr) == (stop_worker and fail_stop)
    if stop_worker and not fail_stop:
        assert "detached processes are unaffected" in result.stdout
