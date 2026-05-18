"""``kicraft-leaf`` command-line tool — read-only views over the leaf library.

Promotion and removal are GUI-driven (the wizard in ``kicraft/gui/pages/
leaf_library.py``); the CLI exists for scripting and inspection only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kicraft.leaf_library import (
    LeafLibrary,
    resolve_library_dir,
)


def _format_list_row(name: str, version: str, hash_: str, tags: list[str], desc: str) -> str:
    short_hash = hash_.removeprefix("sha256:")[:12]
    first_line = (desc.splitlines() or [""])[0]
    tag_str = ",".join(tags) if tags else "-"
    return f"{name}@{version}\t{short_hash}\t{tag_str}\t{first_line}"


def _cmd_list(args: argparse.Namespace) -> int:
    lib = LeafLibrary(_base_dir(args))
    loaded, broken = lib.load_all()
    if not loaded and not broken:
        print(f"# library is empty: {lib.base_dir}", file=sys.stderr)
        return 0
    if loaded:
        print("# name@version\thash\ttags\tdescription")
    for leaf in loaded:
        m = leaf.manifest
        print(_format_list_row(m.name, m.version, m.content_hash, m.tags, m.description))
    if broken:
        print("\n# broken leaves (excluded from reuse):", file=sys.stderr)
        for b in broken:
            print(f"# {b.dir.name}: {b.reason}", file=sys.stderr)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    lib = LeafLibrary(_base_dir(args))
    leaf_dir = lib.base_dir / args.name
    if not leaf_dir.is_dir():
        print(f"error: leaf {args.name!r} not found in {lib.base_dir}", file=sys.stderr)
        return 1
    result = lib.load_one(leaf_dir)
    if not hasattr(result, "manifest"):
        # BrokenLeaf
        print(f"error: leaf {args.name!r} is broken: {result.reason}", file=sys.stderr)
        return 1
    print(json.dumps(result.manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _cmd_path(args: argparse.Namespace) -> int:
    print(_base_dir(args))
    return 0


def _base_dir(args: argparse.Namespace) -> Path:
    if args.library:
        return Path(args.library).expanduser()
    return resolve_library_dir()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kicraft-leaf",
        description="Read-only views over the KiCraft leaf library "
        "($KICRAFT_LEAF_LIB, default ~/.kicraft/leaves/)",
    )
    parser.add_argument(
        "--library",
        help="Override the library directory (default: $KICRAFT_LEAF_LIB)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List every leaf in the library")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Pretty-print a leaf's manifest")
    p_show.add_argument("name", help="Leaf name (directory name in the library)")
    p_show.set_defaults(func=_cmd_show)

    p_path = sub.add_parser("path", help="Print the resolved library path")
    p_path.set_defaults(func=_cmd_path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
