// E2E QA harness: imports the plugin module, instantiates it with a mock
// plugin context that simulates opencode's loader, and drives the four tools
// in the same order /kicraft-new would. Verifies that artifacts land on disk
// and re-validate against the in-plugin validator.

import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"

const PLUGIN_PATH = new URL("./src/index.ts", pathToFileURL(process.cwd() + "/")).href
const mod = await import(PLUGIN_PATH)
const Plugin = mod.KiCraftStartProjectPlugin

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

console.log("[qa] step 4: save_plan (load example, swap name, persist)")
const examplePath = "schema/example_plan.json"
const example = JSON.parse(await readFile(examplePath, "utf8"))
example.name = "qa_e2e_demo"
delete example.user_spec
const r4 = await tools.kicraft_save_plan.execute(
  { project_dir: ".", plan: example, overwrite: true },
  ctx,
)
console.log(r4)

console.log("[qa] step 5: re-inspect (should now report plan present)")
const r5 = await tools.kicraft_inspect_project.execute({ project_dir: "." }, ctx)
console.log(r5)

console.log("[qa] step 6: re-validate written plan via the same validator")
const written = JSON.parse(await readFile(join(projectDir, "project_plan.json"), "utf8"))
mod.validatePlan?.(written)
if (written.user_spec.clarifications.length !== 2) {
  throw new Error(`expected 2 clarifications backfilled, got ${written.user_spec.clarifications.length}`)
}
if (written.user_spec.raw_text.indexOf("18650") === -1) {
  throw new Error(`raw_text did not survive backfill`)
}
const md = await readFile(join(projectDir, "project_plan.md"), "utf8")
if (md.indexOf("# qa_e2e_demo") === -1) {
  throw new Error(`markdown header missing`)
}
console.log(`[qa] OK -- plan re-validates, clarifications backfilled, markdown rendered`)

await rm(projectDir, { recursive: true, force: true })
console.log("[qa] cleaned up; PASS")
