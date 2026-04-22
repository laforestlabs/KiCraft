# KiCraft opencode plugin -- start-new-project layer

The topmost layer of the [KiCraft](../) pipeline, packaged as an
[opencode](https://opencode.ai) plugin.

## What it does

You open opencode in (or next to) a KiCad project directory and run:

```
/kicraft-new
```

The host LLM (running through your existing opencode installation -- no API
keys to manage) walks you through a short clarifying conversation about the
electronics project you want to build, performs background research on the
problem domain, then writes two files into the project directory:

- `project_plan.json` -- machine-readable, schema v1
- `project_plan.md`   -- human-readable rendering of the same plan

The plan is **topology level only**: functional blocks, signal flow, power
tree, generic candidate part *classes* (e.g. "synchronous boost converter
IC"). Exact part numbers, schematic capture, and PCB layout are the job of
later KiCraft layers, which are not yet implemented.

State that survives the conversation lives in `<project>/.kicraft/spec.json`
so an interrupted `/kicraft-new` can be resumed. Resume is real: the
`kicraft_load_state` tool returns the verbatim original spec text plus every
recorded clarification Q/A and the full saved plan body if one exists.

## Pieces

```
opencode-plugin/
  src/index.ts                       # The Plugin: 5 LLM-callable tools + plan validator
  command/kicraft-new.md             # The /kicraft-new slash command (workflow prompt)
  schema/project_plan.schema.json    # JSON Schema (draft 2020-12) for project_plan.json
  schema/example_plan.json           # Reference plan (used by the self-test)
  install.sh                         # Installs the plugin into a target KiCad project
  qa-e2e.ts                          # End-to-end QA: validator, MPN guard, install layout
  package.json / tsconfig.json       # TS dev tooling (typecheck + self-test only)
```

The five plugin tools the LLM can call:

| Tool | Purpose |
|------|---------|
| `kicraft_inspect_project`     | Detect existing spec/plan -- decides resume vs fresh start. |
| `kicraft_load_state`          | Resume: return verbatim spec text, every clarification Q/A, and the full saved plan body. |
| `kicraft_capture_spec`        | Save the user's verbatim project description. |
| `kicraft_record_clarification`| Append one Q/A pair from the clarifying conversation. |
| `kicraft_save_plan`           | Validate the topology-level plan and write `project_plan.json` + `.md`. |

## Install into a KiCad project

```bash
./install.sh /path/to/your/kicad-project
```

This installs the plugin as a single flat file at
`<target>/.opencode/plugins/kicraft-start-project.ts` and the slash command
at `<target>/.opencode/command/kicraft-new.md`. The flat layout is required
by opencode's plugin loader, which globs `{plugin,plugins}/*.{ts,js}`
(direct files only, no nested directories). Opencode auto-loads the plugin
on its next start in that directory.

## Develop / verify

```bash
cd KiCraft/opencode-plugin
npm install            # or: bun install
npm test               # typecheck + selftest + qa-e2e
```

`npm test` runs:

1. `npx tsc --noEmit` -- typecheck
2. `node --experimental-strip-types src/index.ts` -- selftest: loads
   `schema/example_plan.json` and runs the in-plugin validator against it
3. `node --experimental-strip-types qa-e2e.ts` -- end-to-end QA:
   - happy-path validator + plan rendering
   - 11 negative tests proving the validator rejects out-of-scope keys
     (`schematic`, `netlist`, `pcb`, ...), unknown keys at every level,
     MPN strings in `candidate_part_classes`, and missing/empty
     `research_notes`
   - MPN heuristic guards: 8 benign strings (USB-C, 0.1uF, 18650, RS-485,
     I2C, 0805, 1S2P, 3V3) must NOT match; 6 MPN strings (BQ24072,
     TPS5430, STM32F405, ...) MUST match
   - install.sh integration test: runs `install.sh` into a temp dir and
     asserts the output matches the opencode loader glob (direct `*.ts`
     file present, no nested directories)

## Scope guardrails (intentional, do not relax)

- **No exact part numbers** anywhere in `candidate_part_classes` --
  enforced by the `looksLikeMPN` regex `/[A-Z]{2,}[0-9]{3,}/` in the
  validator. Pinned MPNs the user explicitly requires go in
  `requirements.must_use_parts` only.
- **No schematic, no netlist, no PCB** -- enforced by the
  `FORBIDDEN_TOP_KEYS` allowlist in the validator
  (`schematic`, `netlist`, `pcb`, `layout`, `footprints`, `parts`,
  `bom`, `components`).
- **Closed-world key allowlists** at every nesting level reject any
  unexpected key, preventing scope creep into later KiCraft layers.
- **Background research is required** -- `research_notes` must be a
  non-empty array; the slash command's Step 4 forces the LLM to perform
  research before the plan can be saved.
- The LLM drives all conversation; the plugin only persists and validates.
- Plan schema is versioned (`schema_version: 1`) so later KiCraft layers
  can rely on a stable input contract.
