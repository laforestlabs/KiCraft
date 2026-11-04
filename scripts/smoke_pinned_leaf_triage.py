#!/usr/bin/env python3
"""Exercise pinned-leaf diagnostic custody against the retained 9102 witness.

This is deliberately artifact-only: it copies the retained leaf and pin manifest
into a temporary directory, re-pins R2, then asks the production triage reader
for the leaf's reported open count. It neither builds nor contacts a provider.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

from kicraft.autoplacer.brain import pins
from kicraft.cli.triage import collect_leaves


WITNESS_EXPERIMENTS = Path(
    "logs/self_eval/surprise_completion/20261010T204651Z/development/"
    "physical_candidate_v5/9102_trial_1/generated/RP2040_MOTOR_CONTROLLER/"
    ".experiments"
)
WITNESS_LEAF = "b1807844-5c2e-48a4-9af3-0431eb18947b__859487ba1d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        type=Path,
        default=WITNESS_EXPERIMENTS,
        help="retained witness .experiments directory",
    )
    args = parser.parse_args()
    source = args.experiments
    source_leaf = source / "subcircuits" / WITNESS_LEAF
    if not source_leaf.is_dir() or not (source / "pins.json").is_file():
        raise SystemExit(f"missing retained witness artifacts under {source}")

    originals = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [source / "pins.json", *source_leaf.rglob("*")]
        if path.is_file()
    }
    with tempfile.TemporaryDirectory(prefix="kicraft-pinned-triage-") as temp:
        copied_experiments = Path(temp) / ".experiments"
        copied_leaf = copied_experiments / "subcircuits" / WITNESS_LEAF
        copied_leaf.parent.mkdir(parents=True)
        shutil.copytree(source_leaf, copied_leaf)
        shutil.copy2(source / "pins.json", copied_experiments / "pins.json")

        # Reapply the artifact selected by the manifest; this is the physical
        # R2 board whose DRC reports six opens, not last invocation's R5.
        pins.pin_leaf(copied_experiments, WITNESS_LEAF, 2, source="smoke")
        leaves = collect_leaves(copied_experiments)
        leaf = next((entry for entry in leaves if entry["name"] == WITNESS_LEAF), None)
        if leaf is None:
            leaf = next((entry for entry in leaves if entry["kind"] == "leaf"), None)
        assert leaf is not None
        assert leaf["raw_drc"]["unconnected"] == 6, leaf
        assert leaf["unconnected"] is None, leaf
        assert leaf["accepted"] is False, leaf
        assert leaf["rejection_reasons"] == ["pinned_round_acceptance_unavailable"], leaf
        assert originals == {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals
        }, "original manifest/leaf bytes changed"
        print("pinned R2: raw DRC=6; acceptance unavailable/rejected; originals unchanged")


if __name__ == "__main__":
    main()
