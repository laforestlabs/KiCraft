#!/usr/bin/env python3
"""Dashboard Flask application for monitoring autoexperiments.

Runs as a standalone daemon that monitors and controls experiment runs.
Read-only from experiment files - no performance impact on running experiments.

Usage:
    python3 dashboard_app.py [--port 5000] [--experiments-dir .experiments]
    python3 dashboard_app.py --stop  # Stop running experiment gracefully
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Flask is optional - show helpful error if missing
try:
    from flask import Flask, jsonify, render_template_string, request, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Error: Flask is required. Install with: pip install flask")
    print("       Or run: pip install -r requirements.txt")
    sys.exit(1)


app = Flask(__name__)

# Configuration
EXPERIMENTS_DIR = Path(".experiments")
POLL_INTERVAL = 2  # seconds


def _detect_default_pcb():
    """Auto-detect the PCB filename from the project root."""
    root = _find_project_root()
    pro_files = list(root.glob("*.kicad_pro"))
    if pro_files:
        return f"{pro_files[0].stem}.kicad_pcb"
    return None


def _detect_project_name():
    """Auto-detect project name from the .kicad_pro file, fallback to 'KiCad'."""
    root = _find_project_root()
    pro_files = list(root.glob("*.kicad_pro"))
    if pro_files:
        return pro_files[0].stem
    return "KiCad"


def _find_project_root() -> Path:
    """Find KiCad project root directory by locating a *.kicad_pro file."""
    # Path structure: .../<project>/.claude/skills/KiCraft/scripts/dashboard_app.py
    script_dir = Path(__file__).resolve().parent
    # Up 4 levels: scripts -> KiCraft -> skills -> .claude -> project root
    project_root = script_dir.parent.parent.parent.parent
    if list(project_root.glob("*.kicad_pro")):
        return project_root
    # Fallback: try cwd if running from project root
    cwd = Path.cwd()
    if list(cwd.glob("*.kicad_pro")):
        return cwd
    return project_root  # Last resort


def _find_experiments_dir() -> Path:
    """Find .experiments directory - check for nested duplicate too."""
    root = _find_project_root()
    exp_dir = root / ".experiments"
    # Check for nested .experiments/.experiments that may have newer data
    nested = exp_dir / ".experiments"
    if nested.exists() and (nested / "run_status.json").exists():
        return nested
    # Ensure the canonical one exists
    exp_dir.mkdir(exist_ok=True)
    return exp_dir


# Global state
experiment_process: subprocess.Popen | None = None
DEFAULT_PCB = _detect_default_pcb()
_PROJECT_NAME = _detect_project_name()
current_pcb = DEFAULT_PCB


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_HTML, project_name=_PROJECT_NAME)


@app.route("/api/status")
def api_status():
    """Get current run status."""
    exp_dir = _find_experiments_dir()
    status_file = exp_dir / "run_status.json"
    
    if not status_file.exists():
        return jsonify({
            "phase": "idle",
            "round": 0,
            "total_rounds": 0,
            "progress_percent": 0.0,
            "best_score": 0.0,
            "latest_score": None,
            "kept_count": 0,
            "elapsed_s": 0.0,
            "eta_s": 0.0,
        })
    
    try:
        with open(status_file) as f:
            status = json.load(f)
        return jsonify(status)
    except (json.JSONDecodeError, OSError):
        return jsonify({"phase": "error", "message": "Could not read status"})


@app.route("/api/history")
def api_history():
    """Get experiment history from JSONL."""
    exp_dir = _find_experiments_dir()
    jsonl_file = exp_dir / "experiments.jsonl"
    
    if not jsonl_file.exists():
        return jsonify([])
    
    history = []
    try:
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return jsonify(history)
    except OSError:
        return jsonify([])


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start an experiment run."""
    global experiment_process, current_pcb
    
    root = _find_project_root()
    exp_dir = root / ".experiments"
    
    # Get parameters from request
    data = request.get_json() or {}
    pcb = data.get("pcb", DEFAULT_PCB)
    rounds = data.get("rounds", 50)
    workers = data.get("workers", 0)
    program = data.get("program", "program.md")
    no_render = data.get("no_render", False)
    
    # Resolve full path to PCB relative to project root
    pcb_path = root / pcb if not Path(pcb).is_absolute() else Path(pcb)
    
    if not pcb_path.exists():
        return jsonify({"error": f"PCB file not found: {pcb}"}), 400
    
    current_pcb = str(pcb_path)
    
    # Clean up old stop signal
    stop_file = exp_dir / "stop.now"
    if stop_file.exists():
        stop_file.unlink()
    
    # Build command - run from project root
    script_dir = Path(__file__).parent.resolve()
    script_path = script_dir / "autoexperiment.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        str(pcb_path),
        "--rounds", str(rounds),
        "--program", str(script_dir / program),
    ]
    if workers > 0:
        cmd.extend(["--workers", str(workers)])
    if no_render:
        cmd.append("--no-render")
    
    # Start subprocess from project root
    try:
        experiment_process = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Write PID
        pid_file = exp_dir / "experiment.pid"
        with open(pid_file, "w") as f:
            f.write(str(experiment_process.pid))
        
        return jsonify({
            "status": "started",
            "pid": experiment_process.pid,
            "pcb": str(pcb_path),
            "rounds": rounds,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Signal experiment to stop gracefully."""
    exp_dir = _find_experiments_dir()
    
    # Create stop signal file
    stop_file = exp_dir / "stop.now"
    stop_file.touch()
    
    return jsonify({"status": "stop_requested"})


@app.route("/api/log")
def api_log():
    """Get tail of debug log."""
    exp_dir = _find_experiments_dir()
    log_file = exp_dir / "debug.log"
    
    if not log_file.exists():
        return jsonify({"lines": [], "total_size": 0})
    
    try:
        with open(log_file) as f:
            lines = f.readlines()
        
        # Return last 100 lines
        tail = lines[-100:] if len(lines) > 100 else lines
        
        return jsonify({
            "lines": [l.rstrip() for l in tail],
            "total_size": len(lines),
        })
    except OSError:
        return jsonify({"lines": [], "total_size": 0, "error": str(OSError)})


@app.route("/api/best.png")
def api_best_png():
    """Get best board preview image."""
    exp_dir = _find_experiments_dir()
    best_dir = exp_dir / "best"
    best_pcb = best_dir / "best.kicad_pcb"
    
    if not best_pcb.exists():
        return "No best board yet", 404
    
    # Check if PNG exists
    png_path = best_dir / "best.png"
    if png_path.exists():
        return send_file(png_path, mimetype="image/png")
    
    # Try to render on-the-fly (lightweight)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        subprocess.run([
            "kicad-cli", "pcb", "export", "svg",
            "--layers", "F.Cu,B.Cu,F.SilkS,Edge.Cuts",
            "--mode-single", "--fit-page-to-board",
            "-o", tmp_path.replace(".png", ".svg"),
            str(best_pcb),
        ], capture_output=True)
        
        svg_path = tmp_path.replace(".png", ".svg")
        if Path(svg_path).exists():
            subprocess.run([
                "magick", svg_path, "-background", "white", "-flatten",
                "-resize", "400x300!", tmp_path
            ], capture_output=True)
            Path(svg_path).unlink()
        
        if Path(tmp_path).exists():
            return send_file(tmp_path, mimetype="image/png")
    except Exception:
        pass
    
    return "No preview available", 404


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/api/round/<int:round_num>")
def api_round_detail(round_num):
    """Get full round detail JSON."""
    exp_dir = _find_experiments_dir()
    rounds_dir = exp_dir / "rounds"
    path = rounds_dir / f"round_{round_num:04d}.json"
    if not path.exists():
        return jsonify({"error": "Round not found"}), 404
    try:
        with open(path) as f:
            return jsonify(json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nets")
def api_nets():
    """Get per-net routing success rates across rounds."""
    exp_dir = _find_experiments_dir()
    rounds_dir = exp_dir / "rounds"
    if not rounds_dir.exists():
        return jsonify({})

    net_stats = {}
    for path in sorted(rounds_dir.glob("round_*.json")):
        try:
            with open(path) as f:
                rd = json.load(f)
            for nr in rd.get("per_net", []):
                name = nr.get("net", "")
                if not name:
                    continue
                if name not in net_stats:
                    net_stats[name] = {"total": 0, "failures": 0}
                net_stats[name]["total"] += 1
                if not nr.get("success", True):
                    net_stats[name]["failures"] += 1
        except (json.JSONDecodeError, OSError):
            continue

    for s in net_stats.values():
        s["success_rate"] = (
            (s["total"] - s["failures"]) / s["total"]
            if s["total"] > 0 else 1.0
        )
    return jsonify(net_stats)


# Embedded dashboard HTML
DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>{{ project_name }} Autoexperiment Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 { font-size: 1.5rem; color: #00d9ff; }
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .status-idle { background: #4a5568; }
        .status-running { background: #48bb78; color: #1a1a2e; }
        .status-done { background: #00d9ff; color: #1a1a2e; }
        .status-error { background: #f56565; }
        
        .main { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        
        .controls {
            background: #16213e;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }
        .controls label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #888;
            font-size: 0.875rem;
        }
        .controls input, .controls select {
            background: #0f3460;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
        }
        .controls input[type="number"] {
            width: 80px;
        }
        .controls button {
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 0.25rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .controls button:hover { background: #00b8d9; }
        .controls button.stop { background: #f56565; color: white; }
        .controls button.stop:hover { background: #e53e3e; }
        
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        
        .card {
            background: #16213e;
            padding: 1.5rem;
            border-radius: 0.5rem;
        }
        .card h2 {
            font-size: 1rem;
            color: #888;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
        .stat { text-align: center; }
        .stat-value { font-size: 2rem; font-weight: 700; color: #00d9ff; }
        .stat-label { font-size: 0.75rem; color: #888; }
        
        .chart-container {
            height: 250px;
            position: relative;
        }
        canvas { width: 100%; height: 100%; }
        
        .table-container { max-height: 400px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td {
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }
        th { color: #888; font-weight: 500; font-size: 0.75rem; }
        .kept { color: #48bb78; }
        .shorts { color: #f56565; font-weight: bold; }
        .stagnation-warn {
            background: #744210;
            color: #fefcbf;
            padding: 0.75rem 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
            display: none;
        }
        .stagnation-warn.visible { display: block; }
        
        .log-container {
            height: 300px;
            overflow-y: auto;
            background: #0a0a15;
            padding: 1rem;
            border-radius: 0.25rem;
            font-family: monospace;
            font-size: 0.75rem;
            line-height: 1.5;
        }
        .log-line { white-space: pre-wrap; }
        .log-info { color: #888; }
        .log-warn { color: #ecc94b; }
        .log-error { color: #f56565; }
        .log-best { color: #48bb78; }
        
        .error { color: #f56565; padding: 1rem; background: #2d1b1b; border-radius: 0.25rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ project_name }} Autoexperiment</h1>
        <span id="status-badge" class="status-badge status-idle">Idle</span>
    </div>
    
    <div class="main">
        <div id="stagnation-warn" class="stagnation-warn">
            ⚠ Possible stagnation detected — no recent completions
        </div>
        <div class="controls">
            <label>Rounds: <input type="number" id="rounds" value="50" min="1" max="1000" style="width:60px"></label>
            <label>Workers: <input type="number" id="workers" value="0" min="0" max="16" style="width:60px"></label>
            <label><input type="checkbox" id="no_render"> Skip render</label>
            <button id="start-btn" onclick="startExperiment()">Start</button>
            <button id="stop-btn" class="stop" onclick="stopExperiment()" disabled>Stop</button>
            <span id="error-msg" style="color:#f56565;margin-left:1rem"></span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value" id="progress">-/-</div>
                        <div class="stat-label">Round</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="best-score">-</div>
                        <div class="stat-label">Best Score</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="kept-count">0</div>
                        <div class="stat-label">Kept</div>
                    </div>
                </div>
                <div class="stats" style="margin-top: 1rem;">
                    <div class="stat">
                        <div class="stat-value" id="shorts-count" style="color:#f56565">0</div>
                        <div class="stat-label">Latest Shorts</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="drc-count" style="color:#ecc94b">0</div>
                        <div class="stat-label">Latest DRC</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="eta">-</div>
                        <div class="stat-label">ETA</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Score History</h2>
                <div class="chart-container">
                    <canvas id="chart"></canvas>
                </div>
            </div>
            
            <div class="card" style="grid-column: 1 / -1;">
                <h2>Rounds</h2>
                <div class="table-container">
                    <table id="rounds-table">
                        <thead>
                            <tr>
                                <th>Round</th>
                                <th>Score</th>
                                <th>Mode</th>
                                <th>Duration</th>
                                <th>Shorts</th>
                                <th>DRC</th>
                                <th>Kept?</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
            
            <div class="card" style="grid-column: 1 / -1;">
                <h2>Log</h2>
                <div class="log-container" id="log"></div>
            </div>
        </div>
    </div>
    
    <script>
        const POLL_MS = 2000;
        let paused = false;
        
        function $(id) { return document.getElementById(id); }
        
        async function startExperiment() {
            const rounds = $('rounds').value;
            const workers = $('workers').value;
            const noRender = $('no_render').checked;
            
            $('error-msg').textContent = '';
            $('start-btn').disabled = true;
            $('stop-btn').disabled = false;
            
            try {
                console.log('Starting experiment:', {rounds, workers, noRender});
                const res = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        rounds: parseInt(rounds),
                        workers: parseInt(workers),
                        no_render: noRender
                    })
                });
                const data = await res.json();
                console.log('Response:', data);
                if (data.error) {
                    $('error-msg').textContent = data.error;
                    $('start-btn').disabled = false;
                    $('stop-btn').disabled = true;
                } else {
                    // Success - show feedback and start polling
                    $('error-msg').textContent = 'Started! (PID: ' + data.pid + ')';
                    $('error-msg').style.color = '#48bb78';
                    // Trigger immediate status update
                    updateStatus();
                    // Re-enable stop button
                    $('stop-btn').disabled = false;
                }
            } catch (e) {
                $('error-msg').textContent = 'Error: ' + e;
                console.error('Start error:', e);
                $('start-btn').disabled = false;
                $('stop-btn').disabled = true;
            }
        }
        
        async function stopExperiment() {
            $('stop-btn').disabled = true;
            await fetch('/api/stop', {method: 'POST'});
        }
        
        async function updateStatus() {
            if (paused) return;
            
            try {
                const res = await fetch('/api/status');
                const status = await res.json();
                
                // Update status badge
                const badge = $('status-badge');
                badge.className = 'status-badge status-' + status.phase;
                badge.textContent = status.phase.charAt(0).toUpperCase() + status.phase.slice(1);
                
                // Update stats
                $('progress').textContent = status.round + '/' + status.total_rounds;
                $('best-score').textContent = status.best_score.toFixed(1);
                $('kept-count').textContent = status.kept_count;
                
                // ETA
                if (status.eta_s > 0) {
                    const m = Math.floor(status.eta_s / 60);
                    const s = Math.floor(status.eta_s % 60);
                    $('eta').textContent = m + 'm' + String(s).padStart(2, '0') + 's';
                }
                
                // Stagnation warning
                const warn = $('stagnation-warn');
                if (status.maybe_stuck) {
                    warn.classList.add('visible');
                } else {
                    warn.classList.remove('visible');
                }
                
                // Update buttons
                if (status.phase === 'running') {
                    $('start-btn').disabled = true;
                    $('stop-btn').disabled = false;
                } else if (status.phase === 'done' || status.phase === 'idle') {
                    $('start-btn').disabled = false;
                    $('stop-btn').disabled = true;
                }
            } catch (e) {
                console.error('Status error:', e);
            }
        }
        
        async function updateHistory() {
            if (paused) return;
            
            try {
                const res = await fetch('/api/history');
                const history = await res.json();
                
                // Update chart
                updateChart(history);
                
                // Update table
                const tbody = $('rounds-table').querySelector('tbody');
                tbody.innerHTML = '';
                
                // Show last 20 rounds
                const recent = history.slice(-20);
                for (const exp of recent) {
                    const tr = document.createElement('tr');
                    const shortsClass = exp.drc_shorts > 0 ? 'shorts' : '';
                    tr.innerHTML = '<td>' + exp.round_num + '</td>' +
                        '<td>' + exp.score.toFixed(1) + '</td>' +
                        '<td>' + exp.mode + '</td>' +
                        '<td>' + exp.duration_s + 's</td>' +
                        '<td class="' + shortsClass + '">' + (exp.drc_shorts || 0) + '</td>' +
                        '<td>' + (exp.drc_total || 0) + '</td>' +
                        '<td class="kept">' + (exp.kept ? '✓' : '-') + '</td>';
                    tbody.appendChild(tr);
                }
                
                // Update latest shorts/DRC counts from most recent entry
                if (history.length > 0) {
                    const latest = history[history.length - 1];
                    $('shorts-count').textContent = latest.drc_shorts || 0;
                    $('drc-count').textContent = latest.drc_total || 0;
                }
            } catch (e) {
                console.error('History error:', e);
            }
        }
        
        let chartData = { labels: [], best: [], latest: [] };
        
        function updateChart(history) {
            chartData.labels = history.map(h => h.round_num);
            chartData.best = history.map(h => h.score);
            chartData.latest = history.map(h => h.score);
            
            const canvas = $('chart');
            const ctx = canvas.getContext('2d');
            const w = canvas.width = canvas.offsetWidth * 2;
            const h = canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);
            
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, w, h);
            
            if (history.length === 0) return;
            
            const scores = history.map(h => h.score);
            const maxScore = Math.max(...scores, 100);
            const minScore = Math.min(...scores, 0);
            const range = maxScore - minScore || 1;
            
            // Draw lines
            ctx.strokeStyle = '#00d9ff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            history.forEach((h, i) => {
                const x = (i / (history.length - 1 || 1)) * w;
                const y = h - minScore;
                const yPos = h - (y / range) * h;
                
                if (i === 0) ctx.moveTo(x, h - yPos);
                else ctx.lineTo(x, h - yPos);
            });
            ctx.stroke();
            
            // Draw points for kept
            history.forEach((h, i) => {
                if (!h.kept) return;
                const x = (i / (history.length - 1 || 1)) * w;
                const y = h - minScore;
                const yPos = h - (y / range) * h;
                
                ctx.fillStyle = '#48bb78';
                ctx.beginPath();
                ctx.arc(x, h - yPos, 4, 0, Math.PI * 2);
                ctx.fill();
            });
        }
        
        async function updateLog() {
            if (paused) return;
            
            try {
                const res = await fetch('/api/log');
                const data = await res.json();
                
                const container = $('log');
                container.innerHTML = '';
                
                for (const line of data.lines) {
                    const div = document.createElement('div');
                    div.className = 'log-line';
                    
                    if (line.includes('BEST') || line.includes('NEW')) {
                        div.className += ' log-best';
                    } else if (line.includes('ERROR') || line.includes('FAIL')) {
                        div.className += ' log-error';
                    } else if (line.includes('WARN')) {
                        div.className += ' log-warn';
                    } else {
                        div.className += ' log-info';
                    }
                    
                    div.textContent = line;
                    container.appendChild(div);
                }
                
                container.scrollTop = container.scrollHeight;
            } catch (e) {
                console.error('Log error:', e);
            }
        }
        
        // Poll all data
        updateStatus();
        updateHistory();
        updateLog();
        
        setInterval(updateStatus, POLL_MS);
        setInterval(updateHistory, POLL_MS);
        setInterval(updateLog, POLL_MS);
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Experiment dashboard")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run on")
    parser.add_argument("--experiments-dir", "-e", default=".experiments",
                       help="Experiments directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    global EXPERIMENTS_DIR
    EXPERIMENTS_DIR = Path(args.experiments_dir)
    
    if not EXPERIMENTS_DIR.exists():
        EXPERIMENTS_DIR.mkdir(parents=True)
        print(f"Created: {EXPERIMENTS_DIR}")
    
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    print(f"Monitoring: {EXPERIMENTS_DIR.absolute()}")
    print(f"Use --stop to signal a running experiment to stop gracefully")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()