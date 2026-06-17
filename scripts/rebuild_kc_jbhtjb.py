"""Headless rebuild of KC-JBHTJB against the corrected parts library.

Re-runs the LLM design chain (so BOM re-picks the genuine 1.5mm sk6805-1515 and
wiring derives a correct DIN/DOUT chain from the fixed symbol) then the
deterministic build. Prints a summary; the board is left in the rundir for
inspection. Spends real LLM $ (small board).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/kicraft/KiCraft")

from kicraft.server.client import CappedOpenRouterClient
from kicraft.server.config import Settings
from kicraft.eval.self_eval import _event_writer, run_build, run_design

BRIEF = ("5 x 9 array of 1515 RGB LEDs arranged at a pitch of 3mm, "
         "with a header for power and data")
RUNDIR = Path("/home/kicraft/.kicraft/work/rebuild_kc_jbhtjb")


def main() -> int:
    if RUNDIR.exists():
        shutil.rmtree(RUNDIR)
    (RUNDIR / ".kicraft").mkdir(parents=True, exist_ok=True)
    (RUNDIR / "brief.txt").write_text(BRIEF + "\n", encoding="utf-8")
    progress = _event_writer(RUNDIR / "events.jsonl")

    s = Settings.from_env()
    client = CappedOpenRouterClient(s)

    print("=== DESIGN (LLM chain) ===", flush=True)
    d = run_design(client, BRIEF, RUNDIR, progress, run_id="rebuild-kc-jbhtjb")
    print("design result:", d, flush=True)
    if d["status"] != "ok":
        print("DESIGN did not complete -> stopping before build.", flush=True)
        return 1

    print("=== BUILD (synth+place+route+fab) ===", flush=True)
    rc = run_build(RUNDIR, progress, timeout_s=2400)
    print(f"BUILD rc={rc}", flush=True)
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
