"""`kicraft-new` — stdin/stdout chat loop driving the CircuitChat pipeline.

Useful for scripting and end-to-end tests; the same orchestrator and
state model power the GUI page. The CLI deliberately keeps zero
dependencies beyond the circuitchat package itself.

Usage:
    kicraft-new                       # interactive chat loop
    kicraft-new --synthesize PATH     # synthesize current state (no chat)
    kicraft-new --load STATE.json     # resume from a saved state file
    kicraft-new --save STATE.json     # write state on exit / after synth

State files are plain JSON dumps of `ConversationState.model_dump()`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import ConversationState
from .orchestrator import run_turn


_BANNER = """KiCraft CircuitChat -- describe the project you want to build.
Type ':help' for commands, ':quit' to exit.
"""


def _print_state_summary(state: ConversationState) -> None:
    bits = []
    bits.append(f"project_stem={state.project_stem!r}")
    bits.append(f"intent={'set' if state.intent else 'none'}")
    bits.append(f"functional_spec={'set' if state.functional_spec else 'none'}")
    bits.append(f"architecture={'set' if state.architecture else 'none'}")
    bits.append(f"bom={'set' if state.bom else 'none'}")
    bits.append(f"open_questions={len(state.open_questions)}")
    print("[state] " + " ".join(bits))


def _synthesize(state: ConversationState, project_dir: Path, smoke: bool) -> int:
    from .stages.synthesis import SynthesisInputError, run as run_synth
    from .synthesis.validation import SynthesisValidationError

    try:
        artifacts, results = run_synth(state, project_dir, smoke=smoke)
    except SynthesisInputError as e:
        print(f"synthesis input error: {e}", file=sys.stderr)
        return 2
    except SynthesisValidationError as e:
        print(str(e), file=sys.stderr)
        return 3

    print(f"synthesized to {artifacts.project_dir}")
    for r in results:
        print(f"  [{r.name}] {'ok' if r.ok else 'FAIL'}: {r.message}")
    return 0


def _load_state(path: Path) -> ConversationState:
    data = json.loads(path.read_text())
    return ConversationState.model_validate(data)


def _save_state(state: ConversationState, path: Path) -> None:
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kicraft-new", description=__doc__)
    ap.add_argument("--load", type=Path, help="resume from a saved state JSON")
    ap.add_argument("--save", type=Path, help="write state JSON on exit")
    ap.add_argument(
        "--synthesize",
        type=Path,
        metavar="DIR",
        help="synthesize current state into DIR and exit (no chat)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="run the SS9.7 solve-subcircuits smoke check (slow; needs PCB)",
    )
    ap.add_argument(
        "--expert", action="store_true", help="show structured state output each turn"
    )
    args = ap.parse_args(argv)

    state = _load_state(args.load) if args.load else ConversationState()
    state.expert_mode = args.expert

    if args.synthesize:
        rc = _synthesize(state, args.synthesize, smoke=args.smoke)
        if args.save:
            _save_state(state, args.save)
        return rc

    print(_BANNER)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == ":quit" or line == ":q":
            break
        if line == ":help":
            print(
                ":help            -- this message\n"
                ":state           -- print slot summary\n"
                ":dump            -- print full state as JSON\n"
                ":synth DIR       -- synthesize project to DIR\n"
                ":expert on|off   -- toggle expert mode\n"
                ":quit / :q       -- exit\n"
            )
            continue
        if line == ":state":
            _print_state_summary(state)
            continue
        if line == ":dump":
            print(json.dumps(state.model_dump(mode="json"), indent=2))
            continue
        if line.startswith(":synth "):
            target = Path(line[len(":synth "):].strip())
            rc = _synthesize(state, target, smoke=args.smoke)
            if rc != 0:
                print(f"(synthesis returned {rc})")
            continue
        if line.startswith(":expert "):
            val = line[len(":expert "):].strip()
            state.expert_mode = val.lower() in ("on", "true", "1", "yes")
            print(f"expert_mode={state.expert_mode}")
            continue

        try:
            state = run_turn(state, line)
        except Exception as e:  # noqa: BLE001 — surface any failure to the user
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        last = state.history[-1]
        print(f"\n{last.content}\n")
        if state.expert_mode:
            _print_state_summary(state)

    if args.save:
        _save_state(state, args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
