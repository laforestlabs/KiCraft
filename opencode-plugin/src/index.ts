/**
 * KiCraft -- start-new-project layer (opencode plugin).
 *
 * This is the topmost layer of the KiCraft pipeline. It turns a natural
 * language project description into a structured project_plan.json (machine
 * readable) and project_plan.md (human readable) inside the user's KiCad
 * project directory.
 *
 * Scope (do not exceed):
 *   - Capture the user's raw spec.
 *   - Capture clarifying Q/A pairs supplied by the LLM/user.
 *   - Produce a topology-level plan: functional blocks, signal flow,
 *     power tree, generic candidate part *classes* (e.g. "synchronous
 *     buck IC"), open questions.
 *   - NEVER pick exact part numbers (MPNs).
 *   - NEVER produce a schematic, netlist, or PCB.
 *
 * Conversation orchestration (asking the user clarifying questions, doing
 * background reasoning) is handled by the host LLM running inside opencode,
 * driven by the `/kicraft-new` slash command. This plugin only exposes
 * deterministic, LLM-callable tools that read the spec, validate the
 * produced plan, and write artifacts to disk.
 *
 * Self-containment: this file MUST NOT depend on sibling files at runtime.
 * The opencode plugin loader globs `{plugin,plugins}/*.{ts,js}` and loads
 * each match as a single module; nested folders / relative reads are not
 * supported. The example_plan.json used by the dev-time self-test is read
 * relative to this source file in repo only when the file is run directly
 * via `node --experimental-strip-types src/index.ts`.
 */

import { type Plugin, tool } from "@opencode-ai/plugin"
import { mkdir, readFile, writeFile, stat } from "node:fs/promises"
import { dirname, isAbsolute, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const SCHEMA_VERSION = 1
const PLAN_BASENAME = "project_plan"
const STATE_DIRNAME = ".kicraft"
const SPEC_FILENAME = "spec.json"

// Plan types mirror schema/project_plan.schema.json. They are duplicated
// here so the plugin does not need to load the JSON schema at runtime to
// produce well-shaped output.
type Clarification = { question: string; answer: string }

type UserSpec = {
  raw_text: string
  captured_at: string
  clarifications?: Clarification[]
}

type MustUsePart = { name: string; reason?: string }

type Requirements = {
  functional: string[]
  electrical?: string[]
  mechanical?: string[]
  environmental?: string[]
  interfaces?: string[]
  compliance?: string[]
  must_use_parts?: MustUsePart[]
  explicitly_excluded?: string[]
}

type TopologyBlock = {
  id: string
  role: string
  description?: string
  rationale?: string
  candidate_part_classes?: string[]
}

type SignalEdge = {
  from: string
  to: string
  kind:
    | "power"
    | "ground"
    | "analog"
    | "digital"
    | "comm_bus"
    | "rf"
    | "control"
    | "feedback"
  label?: string
}

type PowerRail = {
  rail: string
  source_block?: string
  consumer_blocks?: string[]
  estimated_current_a?: number
}

type Topology = {
  blocks: TopologyBlock[]
  signal_flow: SignalEdge[]
  power_tree?: PowerRail[]
}

type ResearchNote = { topic: string; note: string; source?: string }

type NextLayerHints = {
  preferred_topology_variant?: string
  design_priorities?: string[]
}

export type ProjectPlan = {
  schema_version: 1
  kicraft_layer: "start_new_project"
  name: string
  summary: string
  user_spec: UserSpec
  requirements: Requirements
  topology: Topology
  open_questions?: string[]
  research_notes?: ResearchNote[]
  next_layer_hints?: NextLayerHints
}

function nowIso(): string {
  return new Date().toISOString()
}

const SLUG_RE = /^[a-z0-9][a-z0-9_-]*$/
const BLOCK_ID_RE = /^[a-z0-9_]+$/

// Allowlist of top-level plan keys. Anything else is rejected so that the
// "topology layer only" guarantee cannot be silently broken by accidentally
// including schematic / netlist / PCB / parts data.
const ALLOWED_TOP_KEYS = new Set([
  "schema_version",
  "kicraft_layer",
  "name",
  "summary",
  "user_spec",
  "requirements",
  "topology",
  "open_questions",
  "research_notes",
  "next_layer_hints",
])

const ALLOWED_REQUIREMENTS_KEYS = new Set([
  "functional",
  "electrical",
  "mechanical",
  "environmental",
  "interfaces",
  "compliance",
  "must_use_parts",
  "explicitly_excluded",
])

const ALLOWED_BLOCK_KEYS = new Set([
  "id",
  "role",
  "description",
  "rationale",
  "candidate_part_classes",
])

const ALLOWED_EDGE_KEYS = new Set(["from", "to", "kind", "label"])

const ALLOWED_RAIL_KEYS = new Set([
  "rail",
  "source_block",
  "consumer_blocks",
  "estimated_current_a",
])

const ALLOWED_TOPOLOGY_KEYS = new Set(["blocks", "signal_flow", "power_tree"])

const ALLOWED_USER_SPEC_KEYS = new Set([
  "raw_text",
  "captured_at",
  "clarifications",
])

const ALLOWED_NEXT_LAYER_HINTS_KEYS = new Set([
  "preferred_topology_variant",
  "design_priorities",
])

const ALLOWED_RESEARCH_NOTE_KEYS = new Set(["topic", "note", "source"])

const ALLOWED_MUST_USE_PART_KEYS = new Set(["name", "reason"])

// Forbidden top-level keys that signal scope creep from the topology layer
// into later layers (formalize-design, select-parts, layout). Listed
// explicitly for clearer error messages than a generic "unknown key".
const FORBIDDEN_TOP_KEYS = new Set([
  "schematic",
  "netlist",
  "pcb",
  "layout",
  "footprints",
  "parts",
  "bom",
  "components",
])

function resolveProjectDir(directory: string, relOrAbs?: string): string {
  if (!relOrAbs || relOrAbs.trim() === "" || relOrAbs === ".") {
    return directory
  }
  return isAbsolute(relOrAbs) ? relOrAbs : resolve(directory, relOrAbs)
}

async function ensureDirExists(p: string): Promise<void> {
  await mkdir(p, { recursive: true })
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await stat(p)
    return true
  } catch {
    return false
  }
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

function checkAllowedKeys(
  obj: Record<string, unknown>,
  allowed: Set<string>,
  where: string,
): void {
  for (const key of Object.keys(obj)) {
    if (!allowed.has(key)) {
      throw new Error(
        `${where}: unexpected key ${JSON.stringify(key)} -- allowed keys are: ${[...allowed].join(", ")}`,
      )
    }
  }
}

/**
 * Heuristic MPN (manufacturer part number) detector. Returns true if the
 * string looks like a specific part number rather than a generic part class.
 *
 * Rules (any one matches => looks like an MPN):
 *   - Contains a token that is >= 2 uppercase letters immediately followed
 *     by >= 3 digits (e.g. "BQ24072", "TPS5430", "STM32F405", "AP2112K").
 *   - Contains a token starting with >= 2 uppercase letters, then 2+ digits,
 *     then a hyphen and more chars (e.g. "MAX17048-T").
 *
 * Common false positives explicitly NOT matched:
 *   - "USB-C", "USB3", "RS-485", "I2C", "SPI", "0805", "0603", "1S2P",
 *     "18650", "0.1uF", "3V3", "5V" -- all lack the "letters then digits"
 *     adjacency or have too few of one or the other.
 */
export function looksLikeMPN(s: string): boolean {
  // Primary rule: 2+ uppercase letters glued to 3+ digits.
  // Catches: BQ24072, TPS5430, AP2112K, MAX17048, LM7805, AT89C52
  if (/[A-Z]{2,}[0-9]{3,}/.test(s)) return true
  // Secondary rule: alternating letters/digits with embedded family char,
  // total 7+ chars and starting with 2+ uppercase letters. Catches part
  // numbers like STM32F405 (STM + 32 + F + 405) where the digit run after
  // the leading letters is too short for the primary rule but the overall
  // shape (LL+D+L+D...) is unmistakably an MPN.
  // The leading [A-Z]{2,} requirement and 7-char minimum keep benign
  // strings like "USB3", "I2C", "0805", "1S2P", "RS-485", "3V3" out.
  if (/[A-Z]{2,}[0-9]+[A-Z]+[0-9]+/.test(s)) return true
  return false
}

/**
 * Validate a plan object as a defense-in-depth check. The LLM is asked to
 * produce schema-compliant JSON, but we still verify before writing it to
 * disk so the next KiCraft layer can rely on the contract.
 *
 * Rejects:
 *   - Unknown top-level keys (closed world).
 *   - Forbidden scope-creep keys (schematic, netlist, pcb, ...).
 *   - candidate_part_classes entries that look like MPNs.
 *   - Plans without research_notes (the LLM must have done research).
 */
export function validatePlan(plan: unknown): asserts plan is ProjectPlan {
  assert(plan && typeof plan === "object", "plan must be an object")
  const p = plan as Record<string, unknown>

  // Closed-world top-level keys.
  for (const key of Object.keys(p)) {
    if (FORBIDDEN_TOP_KEYS.has(key)) {
      throw new Error(
        `plan contains forbidden key ${JSON.stringify(key)} -- this layer is topology-only; ${key} belongs to a later KiCraft layer`,
      )
    }
    if (!ALLOWED_TOP_KEYS.has(key)) {
      throw new Error(
        `plan contains unknown top-level key ${JSON.stringify(key)} -- allowed keys are: ${[...ALLOWED_TOP_KEYS].join(", ")}`,
      )
    }
  }

  assert(p.schema_version === SCHEMA_VERSION, `schema_version must be ${SCHEMA_VERSION}`)
  assert(
    p.kicraft_layer === "start_new_project",
    `kicraft_layer must be "start_new_project"`,
  )
  assert(typeof p.name === "string" && SLUG_RE.test(p.name), "name must be a slug matching /^[a-z0-9][a-z0-9_-]*$/")
  assert(typeof p.summary === "string" && p.summary.trim().length > 0, "summary required")

  const spec = p.user_spec as Record<string, unknown> | undefined
  assert(spec && typeof spec === "object", "user_spec required")
  checkAllowedKeys(spec, ALLOWED_USER_SPEC_KEYS, "user_spec")
  assert(typeof spec.raw_text === "string" && spec.raw_text.length > 0, "user_spec.raw_text required")
  assert(typeof spec.captured_at === "string", "user_spec.captured_at required (ISO 8601)")
  if (spec.clarifications !== undefined) {
    assert(Array.isArray(spec.clarifications), "user_spec.clarifications must be an array")
    for (const c of spec.clarifications as Array<Record<string, unknown>>) {
      assert(c && typeof c === "object", "user_spec.clarifications[] must be objects")
      assert(typeof c.question === "string" && c.question.length > 0, "clarifications[].question required")
      assert(typeof c.answer === "string" && c.answer.length > 0, "clarifications[].answer required")
    }
  }

  const req = p.requirements as Record<string, unknown> | undefined
  assert(req && typeof req === "object", "requirements required")
  checkAllowedKeys(req, ALLOWED_REQUIREMENTS_KEYS, "requirements")
  assert(
    Array.isArray(req.functional) && req.functional.length > 0 && req.functional.every((s) => typeof s === "string"),
    "requirements.functional must be a non-empty string[]",
  )
  if (req.must_use_parts !== undefined) {
    assert(Array.isArray(req.must_use_parts), "requirements.must_use_parts must be an array")
    for (const m of req.must_use_parts as Array<Record<string, unknown>>) {
      assert(m && typeof m === "object", "must_use_parts[] must be objects")
      checkAllowedKeys(m, ALLOWED_MUST_USE_PART_KEYS, "must_use_parts[]")
      assert(typeof m.name === "string" && (m.name as string).length > 0, "must_use_parts[].name required")
    }
  }

  const top = p.topology as Record<string, unknown> | undefined
  assert(top && typeof top === "object", "topology required")
  checkAllowedKeys(top, ALLOWED_TOPOLOGY_KEYS, "topology")
  assert(Array.isArray(top.blocks) && top.blocks.length > 0, "topology.blocks must be a non-empty array")
  const blockIds = new Set<string>()
  for (const b of top.blocks as Array<Record<string, unknown>>) {
    checkAllowedKeys(b, ALLOWED_BLOCK_KEYS, `topology.blocks[id=${JSON.stringify(b.id)}]`)
    assert(typeof b.id === "string" && BLOCK_ID_RE.test(b.id), `topology.blocks[].id must match /^[a-z0-9_]+$/, got ${JSON.stringify(b.id)}`)
    assert(!blockIds.has(b.id), `topology.blocks[].id must be unique, duplicate: ${b.id}`)
    blockIds.add(b.id)
    assert(typeof b.role === "string" && (b.role as string).length > 0, `topology.blocks[id=${b.id}].role required`)
    if (b.candidate_part_classes !== undefined) {
      assert(
        Array.isArray(b.candidate_part_classes) && (b.candidate_part_classes as unknown[]).every((s) => typeof s === "string"),
        "candidate_part_classes must be string[] (generic classes only -- no specific part numbers)",
      )
      for (const c of b.candidate_part_classes as string[]) {
        if (looksLikeMPN(c)) {
          throw new Error(
            `topology.blocks[id=${b.id}].candidate_part_classes contains what looks like a specific part number: ${JSON.stringify(c)}. ` +
              `This layer is topology-only -- describe the part by its generic class (e.g. "Li-ion charger IC with integrated power-path"). ` +
              `If the user pinned a specific part by name, record it under requirements.must_use_parts instead.`,
          )
        }
      }
    }
  }

  assert(Array.isArray(top.signal_flow), "topology.signal_flow must be an array")
  const validKinds = new Set(["power", "ground", "analog", "digital", "comm_bus", "rf", "control", "feedback"])
  for (const e of top.signal_flow as Array<Record<string, unknown>>) {
    checkAllowedKeys(e, ALLOWED_EDGE_KEYS, "topology.signal_flow[]")
    assert(typeof e.from === "string" && blockIds.has(e.from), `signal_flow.from must reference a block id, got ${JSON.stringify(e.from)}`)
    assert(typeof e.to === "string" && blockIds.has(e.to), `signal_flow.to must reference a block id, got ${JSON.stringify(e.to)}`)
    assert(typeof e.kind === "string" && validKinds.has(e.kind as string), `signal_flow.kind invalid: ${JSON.stringify(e.kind)}`)
  }

  if (top.power_tree !== undefined) {
    assert(Array.isArray(top.power_tree), "topology.power_tree must be an array")
    for (const r of top.power_tree as Array<Record<string, unknown>>) {
      checkAllowedKeys(r, ALLOWED_RAIL_KEYS, "topology.power_tree[]")
      assert(typeof r.rail === "string" && (r.rail as string).length > 0, "power_tree[].rail required")
      if (r.source_block !== undefined) {
        assert(typeof r.source_block === "string" && blockIds.has(r.source_block), `power_tree[].source_block must reference a block id, got ${JSON.stringify(r.source_block)}`)
      }
      if (r.consumer_blocks !== undefined) {
        assert(Array.isArray(r.consumer_blocks), "power_tree[].consumer_blocks must be an array")
        for (const c of r.consumer_blocks as unknown[]) {
          assert(typeof c === "string" && blockIds.has(c), `power_tree[].consumer_blocks[] must reference a block id, got ${JSON.stringify(c)}`)
        }
      }
    }
  }

  if (p.open_questions !== undefined) {
    assert(Array.isArray(p.open_questions) && (p.open_questions as unknown[]).every((s) => typeof s === "string"),
      "open_questions must be string[]")
  }

  // research_notes is REQUIRED (non-empty) -- the LLM must have done at
  // least one research step before producing a plan. This enforces the
  // workflow rule from kicraft-new.md.
  assert(
    Array.isArray(p.research_notes) && (p.research_notes as unknown[]).length > 0,
    "research_notes is required and must contain at least one entry -- do background research before drafting the plan",
  )
  for (const n of p.research_notes as Array<Record<string, unknown>>) {
    checkAllowedKeys(n, ALLOWED_RESEARCH_NOTE_KEYS, "research_notes[]")
    assert(typeof n.topic === "string" && (n.topic as string).length > 0, "research_notes[].topic required")
    assert(typeof n.note === "string" && (n.note as string).length > 0, "research_notes[].note required")
    if (n.source !== undefined) {
      assert(typeof n.source === "string", "research_notes[].source must be a string when present")
    }
  }

  if (p.next_layer_hints !== undefined) {
    const h = p.next_layer_hints as Record<string, unknown>
    checkAllowedKeys(h, ALLOWED_NEXT_LAYER_HINTS_KEYS, "next_layer_hints")
  }
}

export function renderPlanMarkdown(plan: ProjectPlan): string {
  const lines: string[] = []
  lines.push(`# ${plan.name}`)
  lines.push("")
  lines.push(`> KiCraft project plan -- topology layer (schema v${plan.schema_version}).`)
  lines.push(`> Generated by the KiCraft \`start_new_project\` opencode plugin.`)
  lines.push(`> This file is regenerated alongside \`${PLAN_BASENAME}.json\`. Edit the JSON, not this file.`)
  lines.push("")
  lines.push("## Summary")
  lines.push("")
  lines.push(plan.summary)
  lines.push("")

  lines.push("## Original user spec")
  lines.push("")
  lines.push("```")
  lines.push(plan.user_spec.raw_text)
  lines.push("```")
  lines.push("")
  if (plan.user_spec.clarifications && plan.user_spec.clarifications.length > 0) {
    lines.push("### Clarifications captured")
    lines.push("")
    for (const c of plan.user_spec.clarifications) {
      lines.push(`- **Q:** ${c.question}`)
      lines.push(`  **A:** ${c.answer}`)
    }
    lines.push("")
  }

  lines.push("## Requirements")
  lines.push("")
  const r = plan.requirements
  const renderList = (title: string, items: string[] | undefined) => {
    if (!items || items.length === 0) return
    lines.push(`### ${title}`)
    lines.push("")
    for (const it of items) lines.push(`- ${it}`)
    lines.push("")
  }
  renderList("Functional", r.functional)
  renderList("Electrical", r.electrical)
  renderList("Mechanical", r.mechanical)
  renderList("Environmental", r.environmental)
  renderList("Interfaces", r.interfaces)
  renderList("Compliance", r.compliance)
  renderList("Explicitly excluded", r.explicitly_excluded)
  if (r.must_use_parts && r.must_use_parts.length > 0) {
    lines.push("### Must-use parts (user-pinned by name)")
    lines.push("")
    for (const m of r.must_use_parts) {
      lines.push(`- **${m.name}**${m.reason ? ` -- ${m.reason}` : ""}`)
    }
    lines.push("")
  }

  lines.push("## Topology")
  lines.push("")
  lines.push("### Functional blocks")
  lines.push("")
  for (const b of plan.topology.blocks) {
    lines.push(`#### \`${b.id}\` -- ${b.role}`)
    lines.push("")
    if (b.description) {
      lines.push(b.description)
      lines.push("")
    }
    if (b.rationale) {
      lines.push(`*Why:* ${b.rationale}`)
      lines.push("")
    }
    if (b.candidate_part_classes && b.candidate_part_classes.length > 0) {
      lines.push("Candidate part classes (generic, no exact PNs):")
      for (const c of b.candidate_part_classes) lines.push(`  - ${c}`)
      lines.push("")
    }
  }

  if (plan.topology.signal_flow.length > 0) {
    lines.push("### Signal flow")
    lines.push("")
    lines.push("| From | To | Kind | Label |")
    lines.push("|------|-----|------|-------|")
    for (const e of plan.topology.signal_flow) {
      lines.push(`| \`${e.from}\` | \`${e.to}\` | ${e.kind} | ${e.label ?? ""} |`)
    }
    lines.push("")
  }

  if (plan.topology.power_tree && plan.topology.power_tree.length > 0) {
    lines.push("### Power tree")
    lines.push("")
    for (const rail of plan.topology.power_tree) {
      const cur = rail.estimated_current_a !== undefined ? ` (~${rail.estimated_current_a} A)` : ""
      lines.push(`- **${rail.rail}**${cur}`)
      if (rail.source_block) lines.push(`  - source: \`${rail.source_block}\``)
      if (rail.consumer_blocks && rail.consumer_blocks.length > 0) {
        lines.push(`  - consumers: ${rail.consumer_blocks.map((c) => `\`${c}\``).join(", ")}`)
      }
    }
    lines.push("")
  }

  if (plan.open_questions && plan.open_questions.length > 0) {
    lines.push("## Open questions for the next layer")
    lines.push("")
    for (const q of plan.open_questions) lines.push(`- ${q}`)
    lines.push("")
  }

  if (plan.research_notes && plan.research_notes.length > 0) {
    lines.push("## Research notes")
    lines.push("")
    for (const n of plan.research_notes) {
      lines.push(`- **${n.topic}**: ${n.note}${n.source ? `  _(source: ${n.source})_` : ""}`)
    }
    lines.push("")
  }

  if (plan.next_layer_hints) {
    lines.push("## Hints for the formalize-design layer")
    lines.push("")
    if (plan.next_layer_hints.preferred_topology_variant) {
      lines.push(`- Preferred topology variant: ${plan.next_layer_hints.preferred_topology_variant}`)
    }
    if (plan.next_layer_hints.design_priorities && plan.next_layer_hints.design_priorities.length > 0) {
      lines.push(`- Design priorities (ordered): ${plan.next_layer_hints.design_priorities.join(", ")}`)
    }
    lines.push("")
  }

  return lines.join("\n")
}

export const KiCraftStartProjectPlugin: Plugin = async ({ directory, worktree, client }) => {
  const log = (level: "debug" | "info" | "warn" | "error", message: string, extra?: Record<string, unknown>) => {
    // Best-effort structured logging; never throw.
    try {
      void client.app.log({
        body: { service: "kicraft-start-project", level, message, ...(extra ? { extra } : {}) },
      })
    } catch {
      /* ignore logging failures */
    }
  }

  log("info", "kicraft-start-project plugin initialized", { directory, worktree })

  return {
    tool: {
      kicraft_capture_spec: tool({
        description:
          "Record the user's verbatim natural-language project description for a new KiCraft project. " +
          "Call this ONCE at the very start of /kicraft-new, before asking any clarifying questions. " +
          "The text is stored under <project_dir>/.kicraft/spec.json and survives across sessions. " +
          "Returns the absolute path of the saved spec and the project directory.",
        args: {
          project_dir: tool.schema
            .string()
            .describe(
              "Path to the KiCad project directory the user wants this plan to live in. " +
                'Use "." for the current directory. Relative paths resolve against the opencode session directory.',
            ),
          raw_text: tool.schema
            .string()
            .min(1)
            .describe(
              "The user's verbatim project description, copied as exactly as possible. Do not paraphrase.",
            ),
        },
        async execute(args, ctx) {
          const target = resolveProjectDir(directory, args.project_dir)
          await ensureDirExists(target)
          const stateDir = join(target, STATE_DIRNAME)
          await ensureDirExists(stateDir)
          const specPath = join(stateDir, SPEC_FILENAME)
          const spec: UserSpec = {
            raw_text: args.raw_text,
            captured_at: nowIso(),
            clarifications: [],
          }
          await writeFile(specPath, JSON.stringify(spec, null, 2) + "\n", "utf8")
          log("info", "captured user spec", { specPath })
          ctx.metadata({
            title: "kicraft: captured spec",
            metadata: { project_dir: target, spec_path: specPath },
          })
          return (
            `Saved verbatim user spec to ${specPath}.\n\n` +
            `Project directory: ${target}\n\n` +
            `Next: do background research on the problem domain, then ask clarifying questions ` +
            `(one at a time), recording each Q/A pair via kicraft_record_clarification. ` +
            `Then call kicraft_save_plan with the full plan including research_notes.`
          )
        },
      }),

      kicraft_record_clarification: tool({
        description:
          "Append a single clarifying Q/A pair to the captured spec. Call this every time the LLM asks " +
          "the user a clarifying question about the project and the user replies. The clarifications end up " +
          "in the final project_plan.json under user_spec.clarifications.",
        args: {
          project_dir: tool.schema.string().describe('Same project directory used in kicraft_capture_spec. Use "." for cwd.'),
          question: tool.schema.string().min(1).describe("The exact clarifying question that was asked."),
          answer: tool.schema.string().min(1).describe("The user's exact answer."),
        },
        async execute(args, ctx) {
          const target = resolveProjectDir(directory, args.project_dir)
          const specPath = join(target, STATE_DIRNAME, SPEC_FILENAME)
          assert(await pathExists(specPath), `No spec found at ${specPath}. Call kicraft_capture_spec first.`)
          const raw = await readFile(specPath, "utf8")
          const spec = JSON.parse(raw) as UserSpec
          spec.clarifications = spec.clarifications ?? []
          spec.clarifications.push({ question: args.question, answer: args.answer })
          await writeFile(specPath, JSON.stringify(spec, null, 2) + "\n", "utf8")
          log("info", "recorded clarification", { specPath, count: spec.clarifications.length })
          ctx.metadata({
            title: "kicraft: clarification recorded",
            metadata: { count: spec.clarifications.length },
          })
          return `Recorded clarification #${spec.clarifications.length} in ${specPath}.`
        },
      }),

      kicraft_save_plan: tool({
        description:
          "Validate and persist the final topology-level project plan. Writes both project_plan.json " +
          "(machine-readable, schema v1) and project_plan.md (human-readable) into the project directory. " +
          "The plan MUST stay at the topology level: functional blocks, signal flow, power tree, generic " +
          "candidate part *classes* (e.g. 'synchronous buck IC'). The validator REJECTS any of: top-level " +
          "schematic/netlist/pcb/layout/parts keys, candidate_part_classes entries that look like specific " +
          "part numbers (e.g. 'BQ24072'), or plans with no research_notes. " +
          "If user_spec.raw_text or user_spec.clarifications is omitted in the input, the plugin will fill " +
          "them in from the previously captured .kicraft/spec.json.",
        args: {
          project_dir: tool.schema.string().describe('Same project directory used in kicraft_capture_spec. Use "." for cwd.'),
          plan: tool.schema
            .any()
            .describe(
              "The full project plan object. schema_version MUST be 1 and kicraft_layer MUST be " +
                "'start_new_project'. Allowed top-level keys: schema_version, kicraft_layer, name, summary, " +
                "user_spec, requirements, topology, open_questions, research_notes (REQUIRED, non-empty), " +
                "next_layer_hints. Any other top-level key is rejected.",
            ),
          overwrite: tool.schema
            .boolean()
            .optional()
            .describe("If true, overwrite an existing project_plan.json. Defaults to true."),
        },
        async execute(args, ctx) {
          const target = resolveProjectDir(directory, args.project_dir)
          await ensureDirExists(target)

          const planObj = (args.plan ?? {}) as Record<string, unknown>
          const specPath = join(target, STATE_DIRNAME, SPEC_FILENAME)
          if (await pathExists(specPath)) {
            const captured = JSON.parse(await readFile(specPath, "utf8")) as UserSpec
            const incoming = (planObj.user_spec ?? {}) as Partial<UserSpec>
            planObj.user_spec = {
              raw_text: incoming.raw_text ?? captured.raw_text,
              captured_at: incoming.captured_at ?? captured.captured_at,
              clarifications: incoming.clarifications ?? captured.clarifications ?? [],
            }
          }
          if (planObj.schema_version === undefined) planObj.schema_version = SCHEMA_VERSION
          if (planObj.kicraft_layer === undefined) planObj.kicraft_layer = "start_new_project"

          validatePlan(planObj)
          const plan = planObj as ProjectPlan

          const jsonPath = join(target, `${PLAN_BASENAME}.json`)
          const mdPath = join(target, `${PLAN_BASENAME}.md`)
          const overwrite = args.overwrite ?? true
          if (!overwrite && (await pathExists(jsonPath))) {
            throw new Error(`Refusing to overwrite existing ${jsonPath} (overwrite=false). Pass overwrite=true to replace.`)
          }
          await writeFile(jsonPath, JSON.stringify(plan, null, 2) + "\n", "utf8")
          await writeFile(mdPath, renderPlanMarkdown(plan) + "\n", "utf8")
          log("info", "saved project plan", { jsonPath, mdPath, blocks: plan.topology.blocks.length })

          ctx.metadata({
            title: `kicraft: saved plan ${plan.name}`,
            metadata: {
              json_path: jsonPath,
              md_path: mdPath,
              block_count: plan.topology.blocks.length,
              edge_count: plan.topology.signal_flow.length,
            },
          })

          return (
            `Wrote project plan:\n` +
            `  - ${jsonPath}\n` +
            `  - ${mdPath}\n\n` +
            `Topology: ${plan.topology.blocks.length} block(s), ${plan.topology.signal_flow.length} signal-flow edge(s).\n` +
            `Open questions: ${(plan.open_questions ?? []).length}.\n` +
            `Research notes: ${(plan.research_notes ?? []).length}.\n\n` +
            `This plan is the input to the (not-yet-implemented) formalize-design layer. ` +
            `No exact part numbers and no schematic were generated -- those belong to later layers.`
          )
        },
      }),

      kicraft_inspect_project: tool({
        description:
          "Inspect a project directory and report whether a KiCraft start-new-project artifact already " +
          "exists there (spec.json and/or project_plan.json). Returns presence flags and basic counts. " +
          "Use this at the very start of /kicraft-new to decide whether to resume or to start fresh. " +
          "If you need the full saved spec text or the full saved plan body to actually resume the " +
          "conversation, follow this with kicraft_load_state.",
        args: {
          project_dir: tool.schema.string().describe('Project directory to inspect. Use "." for cwd.'),
        },
        async execute(args, ctx) {
          const target = resolveProjectDir(directory, args.project_dir)
          const specPath = join(target, STATE_DIRNAME, SPEC_FILENAME)
          const planPath = join(target, `${PLAN_BASENAME}.json`)
          const hasSpec = await pathExists(specPath)
          const hasPlan = await pathExists(planPath)
          let summary = `Inspected ${target}:\n  spec: ${hasSpec ? specPath : "absent"}\n  plan: ${hasPlan ? planPath : "absent"}`
          let plan_name: string | undefined
          if (hasPlan) {
            try {
              const p = JSON.parse(await readFile(planPath, "utf8")) as ProjectPlan
              plan_name = p.name
              summary += `\n  plan.name: ${p.name}\n  plan.blocks: ${p.topology.blocks.length}`
            } catch {
              summary += `\n  (plan exists but failed to parse)`
            }
          }
          ctx.metadata({
            title: "kicraft: project inspected",
            metadata: { project_dir: target, has_spec: hasSpec, has_plan: hasPlan, plan_name },
          })
          return summary
        },
      }),

      kicraft_load_state: tool({
        description:
          "Load and return the FULL saved KiCraft state for a project directory: the verbatim user spec " +
          "(raw_text, captured_at, every clarification Q/A pair) and, if present, the full saved project " +
          "plan body. This is what you call to actually resume an interrupted /kicraft-new session -- " +
          "kicraft_inspect_project only tells you presence, this returns the content. " +
          "Returns a structured text block; the calling LLM should read it to understand where the prior " +
          "session left off before asking any new questions.",
        args: {
          project_dir: tool.schema.string().describe('Project directory to load. Use "." for cwd.'),
        },
        async execute(args, ctx) {
          const target = resolveProjectDir(directory, args.project_dir)
          const specPath = join(target, STATE_DIRNAME, SPEC_FILENAME)
          const planPath = join(target, `${PLAN_BASENAME}.json`)
          const hasSpec = await pathExists(specPath)
          const hasPlan = await pathExists(planPath)

          const sections: string[] = []
          sections.push(`KiCraft saved state for ${target}:`)
          sections.push("")

          if (hasSpec) {
            const spec = JSON.parse(await readFile(specPath, "utf8")) as UserSpec
            sections.push(`## .kicraft/spec.json`)
            sections.push(`captured_at: ${spec.captured_at}`)
            sections.push(``)
            sections.push(`### raw user spec (verbatim)`)
            sections.push(``)
            sections.push("```")
            sections.push(spec.raw_text)
            sections.push("```")
            sections.push(``)
            const cl = spec.clarifications ?? []
            sections.push(`### clarifications captured: ${cl.length}`)
            sections.push(``)
            for (let i = 0; i < cl.length; i++) {
              sections.push(`${i + 1}. Q: ${cl[i].question}`)
              sections.push(`   A: ${cl[i].answer}`)
            }
            sections.push(``)
          } else {
            sections.push(`## .kicraft/spec.json -- absent (no prior session)`)
            sections.push(``)
          }

          if (hasPlan) {
            const planRaw = await readFile(planPath, "utf8")
            sections.push(`## project_plan.json`)
            sections.push(``)
            sections.push("```json")
            sections.push(planRaw.trimEnd())
            sections.push("```")
            sections.push(``)
          } else {
            sections.push(`## project_plan.json -- absent (plan never saved)`)
          }

          ctx.metadata({
            title: "kicraft: state loaded",
            metadata: { project_dir: target, has_spec: hasSpec, has_plan: hasPlan },
          })
          return sections.join("\n")
        },
      }),
    },
  }
}

export default KiCraftStartProjectPlugin

// ----------------------------------------------------------------------------
// Dev-time self-test. Only runs when this file is executed directly via
// `node --experimental-strip-types src/index.ts`. Loads the example plan
// from the sibling schema/ folder; this path does NOT exist in installed
// (flat) form, which is fine because the installed plugin never executes
// the self-test (process.argv[1] is not this file when loaded as a plugin).
// ----------------------------------------------------------------------------

const isDirect =
  typeof process !== "undefined" &&
  process.argv[1] &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))

if (isDirect) {
  const here = dirname(fileURLToPath(import.meta.url))
  const examplePath = resolve(here, "..", "schema", "example_plan.json")
  const example = JSON.parse(await readFile(examplePath, "utf8"))
  validatePlan(example)
  process.stdout.write(`OK: example_plan.json validates against the in-plugin validator.\n`)
  process.stdout.write(renderPlanMarkdown(example as ProjectPlan).slice(0, 400) + "\n... (truncated)\n")
}
