"""Deployment scripts are exercised in temporary checkouts with fake commands."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(os.environ.get("KICRAFT_DEPLOY_TEST_SOURCE", Path(__file__).resolve().parents[1]))


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


def checkout(tmp_path, script):
    dest = tmp_path / "repo" / script
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / script, dest)
    if script == "deploy/check-build-slots.sh":
        shutil.copyfile(Path(__file__).resolve().parents[1] / "deploy/build-slot-process.py",
                        dest.parent / "build-slot-process.py")
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    executable(fake / "pgrep", "exit 1")
    executable(fake / "sleep", "exit 0")
    executable(fake / "setsid", 'exec "$@"')
    executable(fake / "nohup", 'exec "$@"')
    env = os.environ.copy()
    env.pop("KICRAFT_WEB_LOG", None)
    env.pop("KICRAFT_BUILD_WORKER_LOG", None)
    env["PATH"] = str(fake) + os.pathsep + env["PATH"]
    return dest, fake, env


def run(script, env, *args):
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=8)


def test_worker_rejects_ready_line_from_previous_run(tmp_path):
    script, _, env = checkout(tmp_path, "deploy/restart-build-worker.sh")
    repo = script.parent.parent
    (repo / "logs").mkdir()
    (repo / "logs/kicraft_build_worker.log").write_text("[build-worker] ready (old)\n")
    executable(repo / ".venv/bin/python", "exec /bin/sleep 2")
    result = run(script, env)
    assert result.returncode != 0, result.stdout


def test_web_rejects_http_from_other_process_when_new_server_exits(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/restart-web.sh")
    # Port lookup must succeed, but the server process exits immediately.
    executable(script.parent.parent / ".venv/bin/python", 'if [ "${1:-}" = -c ]; then echo 8080; exit 0; fi\nexit 1')
    executable(fake / "curl", "/bin/sleep 0.1\nexit 0")
    result = run(script, env)
    assert result.returncode != 0, result.stdout


def test_web_rejects_unrelated_http_listener(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/restart-web.sh")
    executable(script.parent.parent / ".venv/bin/python",
               'if [ "${1:-}" = -c ]; then echo 8080; exit 0; fi\nexec /bin/sleep 2')
    executable(fake / "curl", "exit 0")
    result = run(script, env)
    assert result.returncode != 0, result.stdout


@pytest.mark.parametrize(("override", "expected"), [(None, "127.0.0.1"), ("127.0.0.2", "127.0.0.2")])
def test_web_restart_binds_loopback_despite_legacy_dotenv(tmp_path, override, expected):
    script, fake, env = checkout(tmp_path, "deploy/restart-web.sh")
    repo = script.parent.parent
    (repo / ".env").write_text("KICRAFT_WEB_HOST=0.0.0.0\n")
    fixture = tmp_path / "web_fixture.py"
    fixture.write_text(
        "import os, socket, time\n"
        "from pathlib import Path\n"
        "for line in Path('.env').read_text().splitlines():\n"
        "    key, value = line.split('=', 1); os.environ.setdefault(key, value)\n"
        "sock = socket.socket(); sock.bind((os.environ['KICRAFT_WEB_HOST'], 0))\n"
        "Path(os.environ['BIND_RECORD']).write_text(sock.getsockname()[0])\n"
        "time.sleep(0.2)\n"
    )
    executable(repo / ".venv/bin/python",
               'if [ "${1:-}" = -c ]; then echo 8080; exit 0; fi\n'
               'exec "$REAL_PYTHON" "$WEB_FIXTURE"')
    executable(fake / "curl", "exit 1")
    executable(fake / "sleep", "/bin/sleep 0.02")
    record = tmp_path / "bound"
    env.update(REAL_PYTHON=sys.executable, WEB_FIXTURE=str(fixture), BIND_RECORD=str(record))
    env.pop("KICRAFT_WEB_HOST", None)
    if override is not None:
        env["KICRAFT_WEB_HOST"] = override
    run(script, env)
    assert record.read_text() == expected


def test_merge_rejects_non_numeric_pr_before_curl(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/merge-pr.sh")
    marker = tmp_path / "called"
    executable(fake / "curl", f'touch "{marker}"\nprintf 200')
    result = run(script, env, "../unsafe")
    assert result.returncode != 0
    assert not marker.exists()


def test_sync_rejects_remote_shell_metacharacters(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/tuning-i7/sync-to-admin.sh")
    (tmp_path / "progress.json").write_text("{}")
    marker = tmp_path / "called"
    executable(fake / "ssh", f'touch "{marker}"')
    executable(fake / "rsync", f'touch "{marker}"')
    env.update(RUN_DIR=str(tmp_path), CLOUD="user@example.com", CLOUD_DIR="x';touch /tmp/bad;#")
    result = run(script, env)
    assert result.returncode != 0
    assert not marker.exists()


def test_slot_monitor_json_and_environment_override(tmp_path):
    script, _, env = checkout(tmp_path, "deploy/check-build-slots.sh")
    slot_dir = tmp_path / 'absent "slots"'
    env.update(KICRAFT_BUILD_SLOTS="0", KICRAFT_BUILD_SLOTS_DIR=str(slot_dir))
    result = run(script, env, "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["slots_total"] == 0
    assert report["slots_dir"] == str(slot_dir)


def test_slot_monitor_reports_kernel_lock_owner_not_unrelated_opener(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/check-build-slots.sh")
    slots = tmp_path / "slots"
    slots.mkdir()
    lock = slots / "slot_0.lock"
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,sys,time; f=open(sys.argv[1], 'w'); "
         "fcntl.flock(f, fcntl.LOCK_EX); print('ready', flush=True); time.sleep(3)",
         str(lock)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        # lsof used to report every opener, not the actual flock owner.
        executable(fake / "lsof", f"echo {os.getpid()}")
        executable(fake / "ps", 'case "$1 $2" in "-o ppid=") echo 2;; '
                   '"-o etimes=") echo 10;; "-o rss=") echo 100;; esac')
        env.update(KICRAFT_BUILD_SLOTS="1", KICRAFT_BUILD_SLOTS_DIR=str(slots))
        result = run(script, env)
        assert result.returncode == 0, result.stderr
        assert f"pid={child.pid}" in result.stdout
        assert f"pid={os.getpid()}" not in result.stdout
    finally:
        child.wait(timeout=5)


def test_reaper_fails_closed_when_lock_ownership_unavailable(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/check-build-slots.sh")
    slots = tmp_path / "slots"
    slots.mkdir()
    (slots / "slot_0.lock").touch()
    executable(fake / "python3", "exit 1")
    executable(fake / "ps", "exit 0")
    env.update(KICRAFT_BUILD_SLOTS="1", KICRAFT_BUILD_SLOTS_DIR=str(slots))
    result = run(script, env, "--reap")
    assert result.returncode == 1
    assert "owner unknown" in result.stdout


@pytest.mark.parametrize(("stable_reads", "change_start"), [(1, False), (2, False), (2, True), (3, False)])
def test_reaper_refuses_changed_identity_without_signalling_process(tmp_path, stable_reads, change_start):
    script, fake, env = checkout(tmp_path, "deploy/check-build-slots.sh")
    slots = tmp_path / "slots"
    slots.mkdir()
    signals = tmp_path / "signals"
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(f'kill() {{ printf "%s\\n" "$*" >> "{signals}"; return 0; }}\n')
    uid_seen = tmp_path / "uid_seen"
    executable(fake / "stat", f'n=0; [ ! -f "{uid_seen}" ] || read -r n < "{uid_seen}"; '
               f'n=$((n + 1)); echo "$n" > "{uid_seen}"; '
               f'if [ "$n" -le {stable_reads} ]; then echo {os.getuid()}; else echo {os.getuid() + 1}; fi')
    # A real /proc entry supplies kernel start time, but all signals are
    # intercepted above. The child exits by itself; no process is killed.
    child = subprocess.Popen(
        ["python -m kicraft.design.cli_app build", "3"],
        executable="/bin/sleep",
    )
    try:
        # A fake pidfd and intercepted signal make real signalling impossible.
        # The shim also presents the temporary child as an orphan.
        (tmp_path / "sitecustomize.py").write_text(
            "import os, pathlib, signal\n"
            "os.pidfd_open = lambda pid, flags=0: os.open('/dev/null', os.O_RDONLY)\n"
            "_read = pathlib.Path.read_text\n"
            "def read(self, *args, **kwargs):\n"
            "    text = _read(self, *args, **kwargs)\n"
            f"    if str(self) == '/proc/{child.pid}/stat':\n"
            "        prefix, rest = text.rsplit(') ', 1)\n"
            "        fields = rest.split(); fields[1] = '1'\n"
            f"        if {change_start!r}: fields[19] = str(int(fields[19]) + 1)\n"
            "        return prefix + ') ' + ' '.join(fields)\n"
            "    return text\n"
            "pathlib.Path.read_text = read\n"
            "def send(fd, sig, *args):\n"
            f"    with open({str(signals)!r}, 'a') as f: f.write('-' + signal.Signals(sig).name[3:] + ' {child.pid}\\n')\n"
            "signal.pidfd_send_signal = send\n"
        )
        executable(fake / "ps", f'case "$1 $2" in '
                   f'"-eo pid=,ppid=") echo "{child.pid} 1";; '
                   f'"-o ppid=") echo 1;; "-o etimes=") echo 1000;; '
                   f'"-o rss=") echo 100;; esac')
        env.update(BASH_ENV=str(bash_env), PYTHONPATH=str(tmp_path),
                   KICRAFT_BUILD_SLOTS="0", KICRAFT_BUILD_SLOTS_DIR=str(slots))
        result = run(script, env, "--reap", "--min-age-s", "0")
        assert "orphan build process" in result.stdout
        calls = signals.read_text().splitlines() if signals.exists() else []
        expected = [] if stable_reads == 1 or change_start else [f"-TERM {child.pid}"]
        if stable_reads == 3:
            expected.append(f"-KILL {child.pid}")
        assert calls == expected
    finally:
        child.wait(timeout=5)


def test_tuning_entrypoint_refuses_dirty_checkout(tmp_path):
    script, fake, env = checkout(tmp_path, "deploy/tuning-i7/entrypoint.sh")
    data = tmp_path / "data"
    (data / "KiCraft/.git").mkdir(parents=True)
    marker = tmp_path / "git_calls"
    executable(fake / "git", f'printf "%s\\n" "$*" >> "{marker}"\n'
               'case " $* " in *" diff --quiet "*|*" fetch "*) exit 1;; esac\nexit 0')
    env["KICRAFT_DATA"] = str(data)
    result = run(script, env)
    assert result.returncode != 0
    calls = marker.read_text()
    assert "fetch" not in calls and "reset" not in calls and "merge" not in calls


def test_packaged_assets(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(ROOT / "pyproject.toml", project / "pyproject.toml")
    (project / "README.md").write_text("fixture")
    for name in (
        "kicraft/__init__.py",
        "kicraft/server/__init__.py",
        "kicraft/server/static/kc_landing.css",
        "kicraft/parts_library/__init__.py",
        "kicraft/parts_library/example/manifest.json",
        "kicraft/parts_library/example/example.kicad_sym",
        "kicraft/parts_library/example/example.pretty/example.kicad_mod",
    ):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    wheels = tmp_path / "wheels"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(wheels), str(project)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    with zipfile.ZipFile(next(wheels.glob("*.whl"))) as wheel:
        names = set(wheel.namelist())
    assert "kicraft/server/static/kc_landing.css" in names
    assert "kicraft/parts_library/example/manifest.json" in names
    assert "kicraft/parts_library/example/example.kicad_sym" in names
    assert "kicraft/parts_library/example/example.pretty/example.kicad_mod" in names
