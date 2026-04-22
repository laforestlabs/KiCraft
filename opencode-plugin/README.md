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
electronics project you want to build, then writes two files into the project
directory:

- `project_plan.json` -- machine-readable, schema v1
- `project_plan.md`   -- human-readable rendering of the same plan

The plan is **topology level only**: functional blocks, signal flow, power
tree, generic candidate part *classes* (e.g. "synchronous boost converter
IC"). Exact part numbers, schematic capture, and PCB layout are the job of
later KiCraft layers, which are not yet implemented.

State that survives the conversation lives in `<project>/.kicraft/spec.json`
so an interrupted `/kicraft-new` can be resumed.

## Pieces

```
opencode-plugin/
  src/index.ts                       # The Plugin: 4 LLM-callable tools + plan validator
  command/kicraft-new.md             # The /kicraft-new slash command (workflow prompt)
  schema/project_plan.schema.json    # JSON Schema (draft 2020-12) for project_plan.json
  schema/example_plan.json           # Reference plan (used by the self-test)
  install.sh                         # Installs the plugin into a target KiCad project
  package.json / tsconfig.json       # TS dev tooling (typecheck + self-test only)
```

The four plugin tools the LLM can call:

| Tool | Purpose |
|------|---------|
| `kicraft_inspect_project`     | Detect existing spec/plan -- decides resume vs fresh start. |
| `kicraft_capture_spec`        | Save the user's verbatim project description. |
| `kicraft_record_clarification`| Append one Q/A pair from the clarifying conversation. |
| `kicraft_save_plan`           | Validate the topology-level plan and write `project_plan.json` + `.md`. |

## Install into a KiCad project

```bash
./install.sh /path/to/your/kicad-project
```

This copies the plugin source and slash command into `.opencode/plugins/` and
`.opencode/commands/` inside the target project. Opencode auto-loads them on
its next start in that directory.

## Develop / verify

```bash
cd KiCraft/opencode-plugin
npm install            # or: bun install
npx tsc --noEmit       # typecheck
node --experimental-strip-types src/index.ts   # or: bun src/index.ts
```

The self-test loads `schema/example_plan.json`, runs the in-plugin validator
against it, and prints `OK: ...` plus a markdown preview.

## Scope guardrails (intentional, do not relax)

- No exact part numbers anywhere in the plan output (block-level only).
- No schematic, no netlist, no PCB.
- The LLM drives all conversation; the plugin only persists and validates.
- Plan schema is versioned (`schema_version: 1`) so later KiCraft layers can
  rely on a stable input contract.
