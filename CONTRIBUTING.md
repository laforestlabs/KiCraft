# Contributing to KiCraft

KiCraft is a KiCad PCB automation toolkit. Contributions are welcome — here’s how to get started.

## Development Setup

```bash
git clone https://github.com/laforestlabs/KiCraft.git
cd KiCraft
pip install -e ".[dev]"
```

The `dev` extra pulls in `ruff`, `pytest`, and other development dependencies.
Other optional groups (`gui`, `scoring`, `experiment`) can be installed as needed:

```bash
pip install -e ".[dev,gui,scoring,experiment]"
```

**Python 3.10+** is required. CI tests against 3.10, 3.12, and 3.13.

## Running Tests

```bash
pytest -v
```

Tests that depend on KiCad 9’s `pcbnew` Python module will be **automatically skipped**
if `pcbnew` is not available in your environment. This is expected — those tests run in
CI environments where KiCad is installed, or locally if you have KiCad 9 on your system.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check kicraft/
```

Auto-fix what it can:

```bash
ruff check --fix kicraft/
```

Please ensure `ruff check` is clean before submitting a PR.

## Code Style

- **Line length:** 100 characters
- **Linter:** ruff (configuration in `pyproject.toml`)
- **Intentional ignores:**
  - `E402` — module-level imports not at top of file (needed for `pcbnew` path setup)
  - `E702` — multiple statements on one line
  - `E741` — ambiguous variable names (used intentionally in geometry code)

Follow existing conventions in the file you’re editing. No need to reformat unrelated code.

## Pull Requests

1. **Branch from `main`** — use a descriptive branch name (e.g. `fix-placement-scoring`, `add-cli-export`).
2. **Ensure tests pass** — run `pytest -v` locally.
3. **Ensure lint is clean** — run `ruff check kicraft/`.
4. **Write descriptive commit messages** — explain *what* and *why*, not just *how*.
5. Keep PRs focused. One logical change per PR is easier to review.

## Package Structure

```
kicraft/
├── autoplacer/          # Placement and routing engine
│   ├── brain/           #   Solver logic, force simulation, types, scoring weights
│   │   ├── placement.py #   Core placement solver
│   │   ├── types.py     #   Data types and scoring weights
│   │   └── subcircuit_* #   Hierarchical subcircuit decomposition
│   ├── hardware/        #   KiCad pcbnew API adapter layer
│   │   └── adapter.py   #   Board read/write interface
│   └── config.py        #   Default config + project config loader
├── scoring/             # Layout quality scoring checks
├── gui/                 # NiceGUI experiment manager
└── cli/                 # CLI entry points
    ├── autoexperiment.py
    ├── solve_subcircuits.py
    └── compose_subcircuits.py
```

## Adding CLI Commands

1. **Create the module** in `kicraft/cli/` with a `main()` function that uses `argparse`.
2. **Register the entry point** in `pyproject.toml` under `[project.scripts]`:
   ```toml
   [project.scripts]
   my-command = "kicraft.cli.my_module:main"
   ```
3. **Re-install** the package so the entry point is picked up:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Add a `--help` smoke test** in `tests/` to verify the command is wired up:
   ```python
   import subprocess

   def test_my_command_help():
       result = subprocess.run(
           ["my-command", "--help"],
           capture_output=True, text=True,
       )
       assert result.returncode == 0
       assert "usage:" in result.stdout.lower()
   ```

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](https://opensource.org/licenses/MIT).
