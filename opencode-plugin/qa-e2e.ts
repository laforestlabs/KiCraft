// E2E QA harness for the KiCraft start-new-project opencode plugin.
//
// What this verifies (in order):
//   1. The plugin module loads, instantiates, and exposes 5 tools (the
//      original 4 plus the new kicraft_load_state).
//   2. Happy path: capture spec -> record clarifications -> save plan ->
//      re-inspect -> load_state -> re-validate written plan.
//   3. validatePlan is exported and rejects the negative cases Oracle
//      flagged (top-level `schematic` key, MPN in candidate_part_classes,
//      missing research_notes, unknown top-level keys).
//   4. install.sh produces a layout that matches opencode's plugin loader
//      glob (.opencode/{plugin,plugins}/*.{ts,js} -- direct files only,
//      no nested folders).
//
// This harness deliberately drops the optional-chain (`?.`) guard on
// validatePlan: validatePlan MUST be exported, and a missing export is a
// real test failure, not a silent skip.

import { execFileSync } from "node:child_process"
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

const HERE = dirname(fileURLToPath(import.meta.url))
const PLUGIN_PATH = pathToFileURL(join(HERE, "src", "index.ts")).href
const mod = await import(PLUGIN_PATH)
const Plugin = mod.KiCraftStartProjectPlugin

if (typeof mod.validatePlan !== "function") {
  throw new Error("FAIL: validatePlan is not exported from src/index.ts")
}
if (typeof mod.looksLikeMPN !== "function") {
  throw new Error("FAIL: looksLikeMPN is not exported from src/index.ts")
}

const projectDir = await mkdtemp(join(tmpdir(), "kicraft-e2e-"))
console.log(`[qa] using project dir: ${projectDir}`)

const mockClient = {
  app: {
    log: async (_input: any) => { /* swallow */ },
  },
}

const hooks = await Plugin({
  client: mockClient,
  project: { id: "qa", worktree: projectDir, vcs: undefined },
  directory: projectDir,
  worktree: projectDir,
  $: undefined as any,
  serverUrl: "http://localhost:0",
  experimental_workspace: undefined,
})

const tools = hooks.tool!
const expectedTools = [
  "kicraft_capture_spec",
  "kicraft_record_clarification",
  "kicraft_save_plan",
  "kicraft_inspect_project",
  "kicraft_load_state",
]
for (const t of expectedTools) {
  if (!tools[t]) throw new Error(`FAIL: expected tool ${t} missing from plugin`)
}
console.log(`[qa] all 5 expected tools present`)

const ctx = {
  sessionID: "qa",
  messageID: "qa",
  agent: "qa",
  directory: projectDir,
  worktree: projectDir,
  abort: new AbortController().signal,
  metadata: (_: any) => {},
  ask: () => ({}) as any,
}

console.log("[qa] step 1: inspect (should be empty)")
const r1 = await tools.kicraft_inspect_project.execute({ project_dir: "." }, ctx)
console.log(r1)

console.log("[qa] step 2: capture_spec")
const r2 = await tools.kicraft_capture_spec.execute(
  { project_dir: ".", raw_text: "I want a tiny battery-backed 5V/3V3 power module from a 18650 pair." },
  ctx,
)
console.log(r2)

console.log("[qa] step 3: record_clarification x2")
await tools.kicraft_record_clarification.execute(
  { project_dir: ".", question: "Max 5V current?", answer: "1A continuous" },
  ctx,
)
await tools.kicraft_record_clarification.execute(
  { project_dir: ".", question: "Charging IC family?", answer: "BQ24072" },
  ctx,
)

console.log("[qa] step 4: load_state (should show 2 clarifications, no plan)")
const rLoad1 = await tools.kicraft_load_state.execute({ project_dir: "." }, ctx)
if (!rLoad1.includes("clarifications captured: 2")) {
  throw new Error(`FAIL: load_state did not report 2 clarifications. Got:\n${rLoad1}`)
}
if (!rLoad1.includes("18650")) {
  throw new Error(`FAIL: load_state did not return raw spec text`)
}
if (!rLoad1.includes("project_plan.json -- absent")) {
  throw new Error(`FAIL: load_state should report plan absent at this stage`)
}
console.log(`[qa] load_state contains spec + 2 clarifications`)

console.log("[qa] step 5: save_plan (load example, swap name, persist)")
const examplePath = join(HERE, "schema", "example_plan.json")
const example = JSON.parse(await readFile(examplePath, "utf8"))
example.name = "qa_e2e_demo"
delete example.user_spec
const r5 = await tools.kicraft_save_plan.execute(
  { project_dir: ".", plan: example, overwrite: true },
  ctx,
)
console.log(r5)

console.log("[qa] step 6: re-inspect (should now report plan present)")
const r6 = await tools.kicraft_inspect_project.execute({ project_dir: "." }, ctx)
console.log(r6)

console.log("[qa] step 7: re-validate written plan via the exported validator")
const written = JSON.parse(await readFile(join(projectDir, "project_plan.json"), "utf8"))
mod.validatePlan(written)
if (written.user_spec.clarifications.length !== 2) {
  throw new Error(`FAIL: expected 2 clarifications backfilled, got ${written.user_spec.clarifications.length}`)
}
if (written.user_spec.raw_text.indexOf("18650") === -1) {
  throw new Error(`FAIL: raw_text did not survive backfill`)
}
const md = await readFile(join(projectDir, "project_plan.md"), "utf8")
if (md.indexOf("# qa_e2e_demo") === -1) {
  throw new Error(`FAIL: markdown header missing`)
}
console.log(`[qa] OK -- plan re-validates, clarifications backfilled, markdown rendered`)

console.log("[qa] step 8: load_state after save (should now embed full plan body)")
const rLoad2 = await tools.kicraft_load_state.execute({ project_dir: "." }, ctx)
if (!rLoad2.includes(`"name": "qa_e2e_demo"`)) {
  throw new Error(`FAIL: load_state after save did not embed plan body`)
}
console.log(`[qa] load_state after save embeds full plan body`)

// ---------------------------------------------------------------------------
// Negative tests against the EXPORTED validatePlan -- these are the cases
// Oracle flagged as silently passing in the previous version.
// ---------------------------------------------------------------------------

function expectReject(plan: any, expectedSubstring: string, label: string): void {
  let threw = false
  let msg = ""
  try {
    mod.validatePlan(plan)
  } catch (e: any) {
    threw = true
    msg = String(e?.message ?? e)
  }
  if (!threw) {
    throw new Error(`FAIL [${label}]: validatePlan was supposed to REJECT this plan but accepted it`)
  }
  if (!msg.toLowerCase().includes(expectedSubstring.toLowerCase())) {
    throw new Error(`FAIL [${label}]: rejection message did not mention ${JSON.stringify(expectedSubstring)}. Got: ${msg}`)
  }
  console.log(`[qa]   reject [${label}]: ${msg.split("\n")[0]}`)
}

console.log("[qa] step 9: negative tests")

const goodSkeleton = () => JSON.parse(JSON.stringify({
  schema_version: 1,
  kicraft_layer: "start_new_project",
  name: "neg",
  summary: "negative test base",
  user_spec: {
    raw_text: "x",
    captured_at: "2025-01-01T00:00:00.000Z",
    clarifications: [],
  },
  requirements: { functional: ["do a thing"] },
  topology: {
    blocks: [{ id: "a", role: "thing" }],
    signal_flow: [],
  },
  research_notes: [{ topic: "t", note: "n" }],
}))

const negSchematic = goodSkeleton()
negSchematic.schematic = { sheets: [] }
expectReject(negSchematic, "forbidden key", "top-level schematic key")

const negNetlist = goodSkeleton()
negNetlist.netlist = []
expectReject(negNetlist, "forbidden key", "top-level netlist key")

const negPcb = goodSkeleton()
negPcb.pcb = {}
expectReject(negPcb, "forbidden key", "top-level pcb key")

const negUnknown = goodSkeleton()
negUnknown.bonus_field = "nope"
expectReject(negUnknown, "unknown top-level key", "unknown top-level key")

const negMpn = goodSkeleton()
negMpn.topology.blocks[0].candidate_part_classes = ["TPS5430 buck regulator"]
expectReject(negMpn, "specific part number", "MPN in candidate_part_classes (TPS5430)")

const negMpn2 = goodSkeleton()
negMpn2.topology.blocks[0].candidate_part_classes = ["BQ24072-class power-path charger"]
expectReject(negMpn2, "specific part number", "MPN in candidate_part_classes (BQ24072)")

const negMpn3 = goodSkeleton()
negMpn3.topology.blocks[0].candidate_part_classes = ["STM32F405 microcontroller"]
expectReject(negMpn3, "specific part number", "MPN in candidate_part_classes (STM32F405)")

const negNoResearch = goodSkeleton()
delete negNoResearch.research_notes
expectReject(negNoResearch, "research_notes is required", "missing research_notes")

const negEmptyResearch = goodSkeleton()
negEmptyResearch.research_notes = []
expectReject(negEmptyResearch, "research_notes is required", "empty research_notes")

const negUnknownInBlock = goodSkeleton()
negUnknownInBlock.topology.blocks[0].footprint = "0805"
expectReject(negUnknownInBlock, "unexpected key", "unknown key inside block")

const negUnknownInRequirements = goodSkeleton()
negUnknownInRequirements.requirements.bom = []
expectReject(negUnknownInRequirements, "unexpected key", "unknown key inside requirements")

// MPN false-positive guards: these strings must NOT trigger the heuristic.
const okStrings = [
  "USB-C receptacle (16-pin, power-only acceptable)",
  "0.1uF X7R decoupling cap",
  "18650 cell holder (through-hole, 2x)",
  "RS-485 transceiver",
  "I2C EEPROM",
  "0805 chip resistor",
  "1S2P battery pack",
  "3V3 LDO",
]
for (const s of okStrings) {
  if (mod.looksLikeMPN(s)) {
    throw new Error(`FAIL: MPN heuristic false positive on benign string: ${JSON.stringify(s)}`)
  }
}
console.log(`[qa]   MPN false-positive guard: all ${okStrings.length} benign strings pass`)

// MPN true-positive guards: these MUST trigger the heuristic.
const mpnStrings = [
  "BQ24072",
  "TPS5430",
  "STM32F405",
  "AP2112K",
  "MAX17048",
  "Li-ion charger IC with integrated power-path (BQ24072-class)",
]
for (const s of mpnStrings) {
  if (!mod.looksLikeMPN(s)) {
    throw new Error(`FAIL: MPN heuristic missed real MPN: ${JSON.stringify(s)}`)
  }
}
console.log(`[qa]   MPN true-positive guard: all ${mpnStrings.length} MPN strings detected`)

// ---------------------------------------------------------------------------
// Integration test: run install.sh into a temp project and verify the
// resulting layout matches opencode's plugin loader glob exactly:
//   .opencode/{plugin,plugins}/*.{ts,js}
// (direct files only -- nested folders are NOT discovered by opencode).
// ---------------------------------------------------------------------------

console.log("[qa] step 10: install.sh integration test")
const installTarget = await mkdtemp(join(tmpdir(), "kicraft-install-"))
const installScript = join(HERE, "install.sh")
execFileSync("bash", [installScript, installTarget], { stdio: "inherit" })

const pluginsDir = join(installTarget, ".opencode", "plugins")
const entries = await readdir(pluginsDir, { withFileTypes: true })
const directTsFiles = entries.filter((e) => e.isFile() && e.name.endsWith(".ts"))
if (directTsFiles.length === 0) {
  throw new Error(`FAIL: install produced no direct *.ts files under ${pluginsDir}`)
}
const nestedDirs = entries.filter((e) => e.isDirectory())
if (nestedDirs.length > 0) {
  throw new Error(`FAIL: install produced nested directories under ${pluginsDir} -- opencode loader will NOT discover plugins inside them: ${nestedDirs.map((d) => d.name).join(", ")}`)
}
const expectedFile = join(pluginsDir, "kicraft-start-project.ts")
const installedSrc = await readFile(expectedFile, "utf8")
if (!installedSrc.includes("KiCraftStartProjectPlugin")) {
  throw new Error(`FAIL: installed plugin file does not contain expected export`)
}
console.log(`[qa]   installed plugin at ${expectedFile} -- matches loader glob`)

const cmdFile = join(installTarget, ".opencode", "command", "kicraft-new.md")
const cmdSrc = await readFile(cmdFile, "utf8")
if (!cmdSrc.includes("kicraft_load_state")) {
  throw new Error(`FAIL: installed command does not reference kicraft_load_state`)
}
if (cmdSrc.includes("KiCraft/opencode-plugin/schema/")) {
  throw new Error(`FAIL: installed command still references the dev-time schema path that does not exist in greenfield projects`)
}
console.log(`[qa]   installed command at ${cmdFile} -- references load_state, no bad path refs`)

await rm(installTarget, { recursive: true, force: true })

await rm(projectDir, { recursive: true, force: true })
console.log("[qa] cleaned up; PASS")
