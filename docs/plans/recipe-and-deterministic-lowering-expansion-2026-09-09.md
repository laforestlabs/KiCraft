# Recipe and deterministic-lowering expansion plan

**Status:** implementation handoff; no implementation in this document  
**Primary objective:** move common, reviewed circuits out of BOM and wiring model generation without weakening architecture, ERC, DRC, electrical-review, or fabrication gates.  
**Release objective:** an ESP32, RP2040, STM32F103, or supported tiny-MCU core must never be authored by the BOM or wiring LLM.

## Decision

Treat common circuits as compiled assets, not prompt examples.

The LLM may decide that a design needs Wi-Fi, CAN, a motor driver, an I2C sensor, or an application-specific signal. It must not recreate a supported MCU power/boot/programming circuit, USB-C sink, standard regulator application, bus transceiver support network, or other registered common block.

The execution order is:

1. Architecture emits typed circuit requirements and external interfaces.
2. A deterministic resolver selects exact, versioned recipes for every supported common circuit.
3. Recipes expand exact parts, owned pins, internal nets, constraints, and placement intent.
4. Deterministic lowerers generate algebraic/passive structures from typed parameters.
5. A deterministic pin allocator binds application signals to recipe-declared capable pins.
6. BOM and wiring model work units receive only unresolved roles and unowned pins.
7. Aggregate validation, ERC, DRC, electrical review, and fabrication gates remain unchanged and strict.

A supported common family that cannot be resolved safely must stop with a specific blocking question or unsupported-variant diagnostic. It must never silently fall through to model-authored BOM or wiring.

## Non-negotiable invariants

1. **Protected-family invariant:** model-authored BOM groups cannot contain a protected identity such as an ESP32 module after that family is registered.
2. **Owned-pin invariant:** model wiring cannot write recipe-owned or deterministically allocated pins.
3. **Exact-variant invariant:** recipes bind one exact symbol, footprint, package, and pin map. Do not create a generic “ESP32” or generic “STM32” recipe.
4. **Immutable-version invariant:** changing parts, pin ownership, values, defaults, or electrical behavior creates a new `@N` recipe. Existing versions never mutate in meaning.
5. **Typed-match invariant:** new lowerers match typed architecture requirements, not increasingly broad regular expressions over prose.
6. **No permissive fallback:** unknown variant, invalid parameters, missing port bindings, or unavailable verified library assets produce an attributable failure—not an LLM attempt.
7. **Instance isolation:** every internal recipe net is namespaced by recipe instance. Two instances cannot accidentally share `EN`, `BOOT`, `FB`, or other internal nets.
8. **Validation parity:** recipe/lowered candidates pass the same schema, symbol, footprint, net-coverage, programming-access, ERC, DRC, electrical-review, and fab gates as model-authored candidates.
9. **Evidence invariant:** every production recipe cites manufacturer documentation and has a frozen expansion fixture plus a real build-tail verification workspace.
10. **Observable execution:** provenance reports the recipe/lowerer ID, version, parameters, owned parts/pins, unresolved remainder, and calls avoided.

## Current state and gaps

### What exists

- One registered circuit recipe: `rp2040-minimal@1`.
- Recipe parts carry `recipe_id`, `recipe_instance`, and `recipe_role` provenance.
- Recipe wiring is reconstructed and locked before model wiring normalization.
- Recipe-owned pins are excluded from wiring work units and overwrite attempts are rejected.
- BOM normalization expands recipe parts before model groups and rejects duplicate recipe identities.
- Deterministic sheet-level BOM lowerers exist for:
  - generic pin headers;
  - MCP1700 3.3 V LDO blocks;
  - MCP6001 low-voltage voltage followers;
  - dimensioned/defaultable R-2R ladders.
- Deterministic wiring lowerers exist for:
  - generic connectors;
  - MCP6001 followers;
  - R-2R ladders.
- User-visible provenance now distinguishes recipe, deterministic, recipe-plus-LLM, reused, and LLM paths and persists them in `provenance.jsonl`.
- The curated parts library already contains many likely recipe ingredients, including ESP32-S3 modules, ESP32-WROOM-32E, STM32F103C8T6, CH32V003J4M6, CH340, USB connectors/protection, fixed regulators, switchers, sensors, motor drivers, and interface devices.

### What prevents safe scale today

1. Recipe selection is optional and model-authored. A matching common part can still reach BOM generation without a recipe.
2. `RecipeDefinition.parameters` are validated, then discarded by generic expansion. Parameters cannot currently alter parts, pins, values, or optional features.
3. Recipe nets are fixed strings. Internal nets are not instance-scoped and external ports have no explicit binding contract.
4. Recipes have no declared protected aliases, capability metadata, allocatable pins, source-document metadata, maturity, or compatibility tests.
5. BOM work units are sheet scoped. A mixed sheet with one recipe block and unrelated circuitry still receives one model call for the whole residual sheet.
6. Current deterministic lowering is a central ordered `if` chain over prose-derived sheet descriptions. Adding dozens of regex lowerers would become ambiguous and unsafe.
7. There is no deterministic peripheral-pin allocator. Even with a recipe MCU core, application GPIO assignment can still be model-authored.
8. There is no hard gate rejecting a model-authored supported MCU when architecture omitted its recipe.
9. `rp2040-minimal@1` exposes fixed GPIO nets and should be audited under the new port/internal-net rules before serving as the pattern for more MCU recipes.
10. Coverage is not yet measured as percentages of BOM parts, pins, sheets, or model calls avoided.

## Target architecture

### 1. Typed architecture requirements

Extend architecture with bounded implementation requirements. Keep the fields narrow enough to validate deterministically:

```python
class CircuitRequirement(BaseModel):
    id: str
    sheet: str
    role: Literal[
        "mcu_core", "power_input", "regulator", "programming",
        "bus_interface", "sensor", "driver", "analog_block",
        "user_io", "connector"
    ]
    family: str                 # canonical family, e.g. esp32-s3-module
    exact_part: str | None      # explicit user choice when present
    parameters: dict[str, JsonScalar]
    ports: dict[str, str]       # logical recipe port -> architecture net
    interfaces: list[str]       # usb_device, i2c_controller, pwm, adc, etc.
```

Architecture remains responsible for requirements and inter-sheet topology, not pin numbers or support-component invention.

Add normalization that derives requirements conservatively from existing `topologies`, sheet functions, `intent.named_parts`, and declared protocols so legacy architecture candidates remain loadable. Persist the normalized requirements in `state.json`.

### 2. Deterministic recipe resolver

Add `kicraft/design/recipes/resolver.py` with one entry point:

```python
resolve_architecture_recipes(architecture, intent, registry) -> ResolutionResult
```

`ResolutionResult` contains:

- selected recipes;
- normalized exact part identities;
- defaulted parameters and assumptions;
- unresolved requirements;
- blocking unsupported variants;
- the protected family/identity set for downstream gates.

Resolution rules:

1. Exact user-named supported part wins.
2. Exact architecture part wins when compatible with user constraints.
3. A bounded family request may choose the registry’s explicit default variant and record an assumption.
4. Conflicting exact part, voltage, package, programming, or interface requirements block.
5. Unsupported members of a protected family block; they never fall through to BOM/wiring generation.
6. Explicit user constraints beat recipe defaults.
7. Resolver output is stable under ordering and repeated execution.

Run resolution during architecture normalization/commit, before BOM work-unit planning. Do not rely on prompt compliance.

### 3. Composable recipe contract

Extend recipe models rather than cloning static pin tables ad hoc.

Required fields:

- `recipe`, `maturity`, and immutable version;
- canonical family and protected symbol/value/MPN aliases;
- official source-document URLs and reviewed revision/date;
- exact parts and optional/conditional parts;
- typed external ports with direction and required/optional status;
- internal nets marked explicitly and instance-namespaced during expansion;
- fixed owned pins and explicit no-connects;
- allocatable application pins with capabilities and exclusions;
- parameter schema, defaults, and allowed combinations;
- placement constraints such as antenna keepout, connector edge, decoupling proximity, and thermal intent;
- electrical assertions evaluated after expansion.

Prefer boring, per-recipe pure expansion functions over a general-purpose templating language. The registry should call a typed deterministic builder and then validate the returned ordinary `RecipeExpansion`. Do not invent a string-expression DSL for conditional parts or calculations.

Suggested interface:

```python
@dataclass(frozen=True)
class RegisteredRecipe:
    definition: RecipeDefinition
    expand: Callable[[ResolvedRecipeSelection], RecipeExpansion]
```

Static recipes may use the existing generic expander. Parameterized recipes use a small pure function in their own module. Every builder must be deterministic and side-effect free.

### 4. External ports and internal-net scoping

Add explicit recipe ports and `RecipeSelection.port_bindings`.

- External port example: `usb_dp -> USB_DP`.
- Power port example: `vdd -> +3V3`.
- Internal net example: `boot_rc`, expanded to a collision-proof name derived from `recipe_instance`.
- Exposed application GPIO is not treated as a fixed internal net.

Validate that:

- required ports are bound;
- bindings reference declared architecture power/inter-sheet/local nets;
- direction is compatible;
- two output-only ports are not shorted;
- internal nets never leak into inter-sheet contracts;
- two recipe instances cannot collide.

### 5. Deterministic MCU pin allocator

Add `kicraft/design/recipes/pin_allocator.py`.

Each MCU recipe declares allocatable pins and capabilities:

```text
GPIO, input, output, ADC, PWM, I2C-SDA, I2C-SCL,
SPI-SCLK, SPI-MOSI, SPI-MISO, SPI-CS, UART-TX, UART-RX,
interrupt, touch, strapping, reserved, input-only
```

The allocator receives typed application interfaces and returns stable pin assignments. Requirements:

- allocate buses atomically;
- reserve flash/PSRAM/native-USB/crystal/programming pins;
- avoid strapping pins unless a requirement explicitly permits them;
- respect input-only and ADC-domain restrictions;
- prefer contiguous/nearby GPIOs for parallel buses such as HUB75;
- preserve an existing valid allocation across downstream reruns;
- return a concrete unsatisfied-capability diagnostic instead of calling the LLM;
- use stable ordering and deterministic tie-breaking.

This is what makes the “LLM never writes an ESP32 MCU sheet” guarantee complete: recipes own core support and the allocator owns application GPIO binding.

### 6. Role-scoped work units

Move BOM/wiring ownership below the sheet level.

Extend `StageWorkUnit` with:

- `requirement_ids`;
- `owned_roles`;
- `excluded_refs` and `excluded_pins`;
- planned resolution source;
- recipe/lowerer IDs contributing to the unit.

Planning order:

1. Expand recipes.
2. Apply deterministic lowerers.
3. Run deterministic pin allocation.
4. Subtract all owned roles, refs, and pins.
5. Omit empty work units completely.
6. Create bounded model work units only for remaining requirements.

A mixed sheet may therefore contain a fully recipe-owned ESP32 core plus an application-specific analog section without giving the model any opportunity to recreate or modify the ESP32 core.

### 7. Registered deterministic lowerers

Replace the central ordered lowerer chain with a registry whose members implement:

```python
class Lowerer(Protocol):
    id: str
    version: int
    def match(requirement, context) -> MatchResult: ...
    def lower_bom(requirement, context) -> dict: ...
    def lower_wiring(requirement, bom, context) -> dict: ...
    def validate(candidate, context) -> list[Diagnostic]: ...
```

`MatchResult` must explain every required precondition and why it matched. Ambiguous or partial matches return no candidate. Do not use priority order to hide overlapping matches; registry initialization should reject overlapping exact selectors, and tests should exercise ambiguous prose/requirements.

Use lowerers for algebraic or mechanically derived structures. Use recipes for exact IC application circuits and safety/layout-sensitive blocks.

## Prioritized recipe program

The order below is driven by the public example set, the 34-brief benchmark corpus, existing bundled parts, and current production failure patterns. The benchmark contains 12 MCU-bearing briefs; all 12 must eventually have deterministic MCU-core coverage.

### Wave A — MCU cores: hard guarantee first

Implement and production-verify these before broadening lowerers:

| Recipe | Exact scope | Required variants | Expected model work removed |
|---|---|---|---|
| `esp32-s3-mini-1-minimal@1` | ESP32-S3-MINI-1 module, 3V3 decoupling, EN/reset, BOOT, native USB, antenna placement contract | native USB; optional USB connector owned separately | Complete ESP32-S3 MCU BOM and wiring unit |
| `esp32-s3-wroom-1-minimal@1` | ESP32-S3-WROOM-1 N8/N8R8-compatible reviewed module variant | exact flash/PSRAM identity; native USB | Complete ESP32-S3 MCU BOM and wiring unit |
| `esp32-wroom-32e-minimal@1` | ESP32-WROOM-32E module, EN/IO0 straps and decoupling | external UART programming port or paired USB-UART recipe | Complete classic ESP32 MCU core |
| `esp32-c3-mini-1-minimal@1` | Exact ESP32-C3 module used by rounded dev-board corpus | native USB/JTAG and boot/reset | Complete C3 MCU core |
| `stm32f103c8t6-minimal@1` | LQFP-48 MCU, supply/analog supply, reset, BOOT0/BOOT1 policy, SWD, 8 MHz crystal and load network | USB-device optional only when clock/pin requirements are satisfied | Complete `stm32-min` MCU sheet |
| `attiny402-updi-minimal@1` / `attiny412-updi-minimal@1` | Exact ATtiny402 and ATtiny412 package/pin-map recipes, decoupling, reset/UPDI access | one recipe per exact orderable/package; share reviewed builder helpers only where pin tables are identical | MCU sheets for LED ornaments/ring |
| `attiny1614-updi-minimal@1` | Exact SOIC/QFN variant used by badge corpus | UPDI and decoupling | Badge MCU sheet |
| `ch32v003j4m6-minimal@1` | SOP-8 MCU, VDD decoupling, SWIO programming access | exact SOP-8 pin map | BMP280-reader MCU sheet |

Also migrate and audit `rp2040-minimal@1` under the new port, internal-net, allocator, source-document, and protected-family contracts. Create `rp2040-minimal@2` if semantics change; do not silently mutate `@1`.

Wave-A policy gates:

- Add all supported MCU aliases/MPNs/symbol identities to protected identities.
- Architecture automatically resolves a supported MCU requirement to a recipe.
- BOM model output containing a protected MCU or its recipe-owned support roles fails with `model_authored_protected_identity`.
- Wiring model output touching recipe/allocator-owned MCU pins fails with the existing recipe-owned gate or a more specific stable code.
- Unsupported ESP32/STM32/ATtiny variants stop before BOM with `unsupported_protected_variant`.

### Wave B — power entry and regulation

| Recipe | Scope |
|---|---|
| `usb-c-5v-sink@1` | Exact receptacle, both CC pulldowns, VBUS/GND grouping, optional PTC/TVS parameter variants, shield policy |
| `usb-c-usb2-device@1` | USB-C receptacle, CC pulldowns, D+/D− routing ownership, ESD device, optional series resistors, shield policy |
| `ch224k-pd-trigger@1` | Exact CH224K application for one selected fixed PDO; selector variant only after switch truth table is verified |
| `tp4056-1s-charger@1` | TP4056, programming resistor, input/output capacitors, status outputs, thermal/application constraints |
| `me6211-3v3@1` | Curated <=500 mA 3.3 V LDO application with exact capacitor requirements |
| `mcp1700-3v3@1` | Exact MCP1700 3.3 V LDO application, replacing the current prose-matched lowerer |
| `mcp6001-follower@1` | Exact unity-gain MCP6001 application and bypassing, replacing the current prose-matched lowerer |
| `ams1117-3v3@1` | Curated <=1 A application with dropout and dissipation constraints |
| `tlv62569-3v3@1` | Fixed/common 5 V to 3.3 V synchronous buck application |
| `ap63203-3v3@1` / `ap63205-5v@1` | Exact fixed-output wide-input buck applications |
| `tps54331-adjustable@1` | Adjustable 3 A buck with deterministically calculated divider, inductor, compensation, and voltage-rated capacitors within reviewed bounds |

Do not create one generic regulator recipe. Each controller/output/package/reference circuit is versioned separately. Parameter bounds must enforce input voltage, output current, thermal, inductor saturation, capacitor voltage, and divider tolerance limits.

### Wave C — buses and common drivers

| Recipe | Scope |
|---|---|
| `sn65hvd230-can-node@1` | 3.3 V transceiver, local decoupling, TX/RX ports, CANH/CANL, optional switchable 120 Ω termination, TVS/connector port |
| `max3485-rs485-node@1` | 3.3 V RS-485 transceiver, DE/RE mode, termination/bias options, terminal port |
| `drv8833-dual-motor@1` | Exact motor driver, decoupling/bulk, sleep/fault policy, two motor outputs |
| `a4988-stepper@1` | A4988 support, current-sense values, microstep straps, charge-pump caps, motor/power ports |
| `pca9685-servo-bank@1` | PCA9685, oscillator/support, address straps, I2C pull-up ownership policy, parameterized channel count/connectors |
| `tca9555-gpio-expander@1` | TCA9555 support, address straps, interrupt, I2C ports |
| `ads1115-i2c-adc@1` | ADC support, address selection, decoupling, bounded input-interface options |
| `hub75-sn74hct245-interface@1` | HUB75 connector plus reviewed level-shifter banks, OE policy, power/ground pins |
| `ws2812-output@1` | Logic-level policy, series resistor, local bulk/decoupling, connector or LED-chain port |
| `ch340c-usb-uart@1` | USB-UART bridge, USB protection, VIO binding, UART port; optional MCU-specific auto-reset companion only when reviewed |

### Wave D — sensor/application support

Create recipes only for parts whose reference application is stable and fully reviewed:

- BME280/BMP388 sensor with decoupling, interface selection, address straps, and I2C pull-up ownership policy;
- MCP23017 if explicitly named despite TCA9555 being the curated default;
- INA226 current monitor with deterministic shunt selection under bounded current/power constraints;
- MAX31855 thermocouple interface with connector and filtering policy;
- AHT20, VEML7700, VL53L0X, and common Qwiic sensor ports;
- analog/I2S audio amplifier blocks already represented in the curated parts catalog.

Do not recipe every catalog part. A recipe is earned when the part has nontrivial required support, pin straps, safety behavior, or repeated model failure. Simple stand-alone parts remain catalog resolution plus deterministic lowerers.

## Prioritized deterministic-lowering program

Implement only after typed `CircuitRequirement` matching exists.

### Tier 1 — exact algebra and mechanical mappings

1. Generic 1xN and 2xN headers from explicit signal/power/ground port lists.
2. Screw-terminal blocks from declared wire ports and current class.
3. FPC-to-header one-to-one breakouts with exact pin count and pitch.
4. Test points for explicitly requested named nets.
5. Repeated connector banks, such as N identical 3-pin servo outputs.
6. R-2R ladders using explicit bit count and resistor pair.
7. Pure passive crossovers only when component values are supplied; never synthesize acoustic design requirements from prose.

### Tier 2 — bounded passive calculations

1. LED current-limiting resistor from rail voltage, declared LED forward-voltage range, and target current.
2. Voltage divider from explicit input range, target voltage, impedance/current constraint, and tolerance.
3. RC low/high-pass from explicit cutoff plus one fixed or bounded component choice.
4. I2C pull-ups from bus voltage, speed, and bounded bus-capacitance class.
5. Open-drain/pull-up straps from explicit voltage and logic role.
6. Switch/button input using an explicitly selected internal or external pull policy.
7. Bulk/local decoupling only when a recipe or typed requirement states the count/value/voltage rule; never infer arbitrary IC decoupling from a symbol alone.
8. Repeated status LEDs and repeated identical channels.

Every calculation records inputs, equation, chosen standard value, tolerance, and assumption provenance. Reject out-of-range calculations instead of clipping to a convenient value.

### Tier 3 — deterministic wiring composition

1. Direct connector port-to-pin maps.
2. Series component chains where input/output ports are typed.
3. Divider, RC, LED/resistor, button, pull-up, and test-point nets generated from the same object that generated the BOM.
4. Same-name bus stitching only across explicitly declared compatible ports.
5. Recipe-to-recipe port binding without a model call.
6. Repeated-channel wiring generated from stable instance indices.

BOM and wiring must come from one lowering result object. Do not independently rediscover topology in separate BOM and wiring regex functions.

## Implementation phases

### Phase 0 — baseline and frozen corpus

**Change**

1. Add a coverage analyzer over the 34 benchmark briefs and four public example boards.
2. Record per run:
   - total BOM parts;
   - recipe-owned parts/pins;
   - deterministically lowered parts/pins;
   - model-authored parts/pins;
   - BOM/wiring calls and cost by work unit;
   - protected-family violations.
3. Freeze canonical architecture candidates and build-tail workspaces for the first recipe wave. Remove provider text/reasoning and user-identifying content.
4. Capture the pre-change baseline before enabling mandatory resolution.

**Files**

- `kicraft/eval/` coverage reporter
- `kicraft/server/stage_runtime.py`
- `kicraft/server/spend_guard.py` only if aggregate columns are needed
- `tests/fixtures/recipe_coverage/`
- evaluation tests adjacent to the reporter

**Acceptance**

- One report attributes 100% of parts, pins, calls, and cost to recipe, lowering, reuse, or LLM.
- Baseline fixtures replay with no live model call.
- Coverage accounting cannot count one part or pin twice.

### Phase 1 — resolver and protected-family gate

**Change**

1. Add typed circuit requirements and legacy normalization.
2. Add resolver aliases/defaults/conflict handling.
3. Resolve recipes during architecture normalization.
4. Persist resolved selections and default assumptions.
5. Add protected-family BOM and wiring gates.
6. Emit stable provenance for resolution, defaulting, blocking, and protected-family rejection.

**Files**

- `kicraft/design/models.py`
- `kicraft/design/recipes/resolver.py` (new)
- `kicraft/design/recipes/registry.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/design/stage_semantics.py`
- architecture/BOM/wiring stage guidance
- `tests/test_design_recipes.py`
- `tests/test_stage_semantics.py`
- `tests/test_stage_driver_retry.py`

**Acceptance**

- An architecture naming ESP32-S3-MINI-1 reaches BOM with its exact recipe selected even if the model omitted `recipe_selections`.
- A conflicting exact variant blocks with one specific diagnostic.
- A BOM candidate containing a protected MCU identity is rejected before parts lookup or commit.
- No protected family can silently use the LLM path.

### Phase 2 — composable recipe engine and pin allocator

**Change**

1. Add recipe ports, port bindings, internal-net scoping, conditional builders, source metadata, maturity, and protected aliases.
2. Add deterministic MCU pin allocation.
3. Upgrade RP2040 to the new contract.
4. Make recipe expansion output one complete BOM/wiring ownership manifest.
5. Keep generic static expansion for recipes that need no parameterized behavior.
6. Before dispatch, omit any sheet work unit whose requirements and pins are completely recipe-owned.

**Files**

- `kicraft/design/recipes/models.py`
- `kicraft/design/recipes/registry.py`
- `kicraft/design/recipes/pin_allocator.py` (new)
- `kicraft/design/recipes/rp2040_minimal.py` or a new `rp2040_minimal_v2.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_work_units.py`
- `tests/test_design_recipes.py`

**Acceptance**

- Two instances have disjoint references and internal nets.
- Required external ports bind exactly once with compatible directions.
- Optional parameters materially change expansion and are not merely validated then discarded.
- MCU application interfaces receive deterministic legal pins.
- An impossible pin request returns a stable capability diagnostic with zero provider calls.

### Phase 3 — Wave-A MCU recipes

**Change**

Implement one exact recipe at a time, beginning with ESP32-S3-MINI-1 because it appears in current recovery fixtures and public examples. For each recipe:

1. Audit the exact vendored symbol/footprint against manufacturer documentation.
2. Add missing production-maturity library bundle assets first.
3. Implement recipe expansion, ports, fixed pins, allocatable pins, exclusions, placement constraints, and assertions.
4. Add alias/protected identities to the resolver.
5. Add frozen architecture/expansion/wiring fixtures.
6. Run a real frozen build tail through schematic, placement, route, promote, and fabrication verification.
7. Only then mark the recipe production-ready and move to the next exact variant.

**Files**

- one module per recipe under `kicraft/design/recipes/`
- `kicraft/design/recipes/__init__.py`
- needed `kicraft/parts_library/<bundle>/` assets
- recipe fixtures/tests
- selected fixed evaluation workspaces

**Acceptance per recipe**

- BOM model calls for the MCU core: zero.
- Wiring model calls for the MCU core: zero.
- All MCU pins are recipe-owned, allocator-owned, or explicit no-connects.
- Programming/recovery path check passes.
- Power polarity, net coverage, dangling-net, family-wiring, and recipe assertions pass.
- Schematic ERC passes.
- Frozen build tail reaches an honest artifact verdict.
- Production provenance names the exact recipe version and zero-call resolution.

### Phase 4 — role-scoped work units

**Change**

1. Plan work units from unresolved requirements rather than one unit per sheet.
2. Subtract recipe/lowerer ownership before creating a model unit.
3. Omit empty units.
4. Preserve accepted residual units independently of recipe-owned content.
5. Route aggregate defects only to the responsible recipe assertion, lowerer, allocator, or model unit.

**Files**

- `kicraft/server/stage_work_units.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_contracts.py`
- `tests/test_stage_work_units.py`
- `tests/test_stage_driver_retry.py`

**Acceptance**

- A mixed ESP32 plus novel-analog sheet calls the model only for the analog requirement.
- The model prompt contains no recipe-owned MCU groups or pins as writable output.
- An aggregate analog defect cannot redraft the MCU recipe.
- A recipe-complete sheet produces no model work unit.

### Phase 5 — Waves B and C recipes

Implement power, USB, bus, and driver recipes in the priority order above. Gate each separately; do not merge an unverified batch of recipe files.

**Acceptance**

- Each recipe meets the per-recipe acceptance bar from Phase 3.
- Parameter boundary tests cover minimum, maximum, and rejected out-of-range electrical conditions.
- USB, charging, converter, motor, and bus recipes carry explicit protection/thermal/layout assertions where applicable.
- No recipe is promoted solely because schema/ERC passes; electrical behavior and real artifact verification are required.

### Phase 6 — lowering registry and Tier-1/Tier-2 lowerers

**Change**

1. Introduce the typed lowerer registry.
2. Migrate existing connector and R-2R behavior without changing observable output.
3. Move the exact MCP6001 follower and MCP1700 application circuits to their Wave-B recipes; delete their prose-match lowerers after cutover.
4. Produce one combined BOM+wiring lowering artifact per matched requirement.
5. Add Tier-1 lowerers, then bounded Tier-2 calculations.
6. Delete migrated regex-chain paths; no compatibility aliases.

**Files**

- `kicraft/design/lowering.py` or `kicraft/design/lowerers/`
- `kicraft/server/stage_work_units.py`
- `kicraft/server/stage_contracts.py`
- typed requirement models
- `tests/test_stage_work_units.py`
- dedicated lowerer behavior tests

**Acceptance**

- Existing R-2R fixtures and the replacement MCP6001/MCP1700 recipe fixtures produce byte-stable canonical BOM/wiring results.
- Ambiguous or incomplete requirements produce no lowering candidate.
- Calculated values are standard-series values within declared tolerance and electrical bounds.
- The same lowering artifact owns both BOM and wiring, preventing topology drift.
- No lowerer depends on ordered first-match behavior.

### Phase 7 — coverage rollout

**Change**

1. Add recipe maturity: `experimental`, `canary`, `production`.
2. Resolve experimental recipes only in self-eval; canary recipes only for selected internal/admin runs; production recipes for user traffic.
3. Surface coverage and calls avoided in the existing provenance UI and admin reports.
4. Promote recipes one at a time after fixed-corpus and live canary evidence.
5. If a protected production recipe is disabled, block that variant rather than allowing model fallback.

**Acceptance**

- All 12 MCU-bearing benchmark briefs use a production or canary MCU recipe in evaluation.
- The four public example briefs never model-author a supported MCU core.
- Recipe-complete MCU sheets have zero BOM and zero wiring calls.
- At least 70% of BOM parts and 70% of wiring pins across the fixed benchmark are recipe/lowering/allocator owned before expanding beyond Wave C.
- Mean BOM+wiring provider calls fall by at least 50% on the fixed corpus without reducing fab-readiness score.
- Protected-family violation count is zero.
- ERC/DRC/fab readiness is no worse than baseline; any regression blocks promotion.

## Verification matrix

### Recipe unit verification

For every recipe version:

- expansion is deterministic across repeated runs;
- reference allocation is collision-free across multiple instances;
- internal nets are instance-scoped;
- exact symbol and footprint exist;
- symbol pin numbers used by the recipe exist;
- every fixed pin is connected exactly once or explicitly no-connect;
- required ports and parameter combinations validate;
- protected aliases cannot be model-authored;
- official source metadata is present;
- electrical assertions pass for accepted parameter boundaries and fail outside them.

### Pin allocator verification

- deterministic output under input reordering;
- atomic I2C/SPI/UART/USB allocation;
- reserved and flash/PSRAM pins never allocated;
- strapping pins avoided unless explicitly allowed;
- impossible parallel-bus request fails with exact missing capability/count;
- rerun preserves a still-valid allocation;
- multiple MCU instances remain isolated.

### Pipeline verification

Use exploding fake clients to prove zero-call paths. A test that merely observes a recipe field is insufficient.

Required scenarios:

1. Recipe-complete ESP32 MCU sheet succeeds when any provider call would raise.
2. Mixed recipe plus novel circuit calls the provider exactly once for the novel role.
3. Model attempt to duplicate ESP32 is rejected before commit.
4. Model attempt to touch recipe/allocator-owned pin is rejected.
5. Unsupported protected variant blocks before BOM.
6. Reopening a failed run retains resolver/recipe/lowerer provenance.
7. Aggregate repair never invalidates a valid recipe expansion.

### Real artifact verification

For every production-ready recipe family:

1. Freeze a complete, non-LLM workspace.
2. Run the real synthesis/build tail.
3. Inspect schematic ERC.
4. Inspect board DRC, unconnected count, shorts, and fabrication package.
5. Verify programming path, power polarity, pin-family constraints, and recipe-specific assertions.
6. Verify placement constraints: antenna keepout, decoupling proximity, connector edge, thermal copper, or current path as applicable.
7. Save the honest verdict; do not promote a recipe on a mocked or narrowed build.

## Source and review policy

- Use manufacturer datasheets, hardware-design guides, and reference schematics as primary sources.
- Record document URL, revision, page/section, reviewer, and review date in recipe metadata or an adjacent machine-readable fixture.
- Existing generated/sample schematics may identify useful candidate topology, but are not authoritative until audited against primary sources.
- Vendored LCSC symbols/footprints must be checked against the manufacturer package drawing and symbol pin table.
- Never bulk-generate recipe pin tables from prose or an LLM response.
- Never generalize one exact package’s pin map to a family.
- Safety-critical power, charging, USB-PD, motor, and high-current recipes require a second human review before production maturity.

## Explicit non-goals

- No generic “any MCU” recipe.
- No generic “any buck converter” recipe.
- No automatic circuit inference from a symbol name alone.
- No relaxation of commit, ERC, DRC, electrical-review, or fabrication gates.
- No model fallback for protected but unsupported variants.
- No attempt to recipe every catalog part.
- No new recipe DSL beyond typed Python builders and Pydantic contracts.
- No production enablement before real artifact verification.

## Recommended first implementation slice

Keep the next session’s first merge small enough to verify honestly:

1. Add typed recipe ports, instance-scoped internal nets, source metadata, maturity, and protected aliases.
2. Add deterministic resolver and protected-family gate.
3. Implement `esp32-s3-mini-1-minimal@1` only.
4. Add the minimal pin allocator capabilities needed by the HUB75 recovery fixture: native USB, BOOT/reset, power, and a bounded parallel-output allocation.
5. Make recipe-complete MCU BOM and wiring work units disappear.
6. Freeze two fixtures:
   - ESP32-S3-MINI-1 MCU-only sheet;
   - ESP32-S3-MINI-1 plus novel analog work on the same sheet.
7. Prove zero MCU calls, one analog-only call, strict ownership rejection, and a real build-tail verdict.
8. Canary against the ESP32 HUB75 and ESP32 robot-controller briefs.

Do not start by adding ten static recipe files to the current contract. Without resolver enforcement, port binding, internal-net scoping, and pin allocation, that would increase apparent coverage while leaving the LLM able to recreate or corrupt the common circuit.

## Completion definition

This program is complete when:

- every supported common family resolves deterministically or blocks before BOM;
- ESP32 and other supported MCU sheets contain no model-authored core parts or pin assignments;
- common recipe/lowerer ownership composes safely inside larger mixed sheets;
- provenance explains every part, pin, call, and failure;
- fixed-corpus provider calls and cost fall materially;
- real ERC/DRC/fabrication results are at least as good as the pre-change baseline.
