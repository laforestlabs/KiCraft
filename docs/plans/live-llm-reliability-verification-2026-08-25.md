# Live LLM reliability verification — 2026-08-25

**Status:** completed — safety controls passed; completion gates failed
**Change under test:** call-specific reasoning guards, streaming collection bounds, bounded serialization escape, and stalled commit-correction reset.
**Source batch:** `/home/kicraft/.kicraft/self_eval/20260825T033602Z`

## Safety and isolation

- Use the production-configured design and judge models through the real capped client and spend ledger.
- Never modify the frozen source batch. `drive_replay` copies each `state.json` into a new `kc-replay-*` workspace.
- Write campaign results under `logs/live_verification/<timestamp>/`.
- Stop on a spend-guard exception, less than $3 daily headroom, or evidence that a source state changed.
- Do not run place/route. These changes affect LLM control paths before synthesis.

## Cohorts and gates

### Judge canary

Rebuild the frozen digest for the four briefs whose Class-J calls previously ended after the design reasoning ceiling: `usb-pd-trigger`, `led-cc-driver`, `esp32-dual-motor`, and `daq-8ch`.

Gate:

- 4/4 valid Class-J verdicts;
- no `reasoning_loop` finish;
- every spend row uses `reasoning_policy_name=eval_judge` rather than `design`;
- total cost is recorded and reported; the plan's $0.45 target is informational because provider output length and pricing are not deterministic correctness properties.

### BOM overflow replay

Replay BOM from the seven frozen failed states: `stm32-min`, `rp2040-min`, `nrf52-beacon`, `esp32-dual-motor`, `can-node`, `daq-8ch`, and `stepper-a4988`.

Gate:

- zero terminal `truncated_json` caused by runaway collection emission;
- zero response accepted beyond the canonical 500-total/450-per-sheet bound;
- at least 5/7 BOM commits, compared with 0/7 in the source batch;
- every non-commit has an honest typed failure and stays within the existing retry/spend budgets.

### Wiring correction replay

Replay the eight source states that terminated at wiring: `r2r-dac`, `rs485-terminal`, `esp32-s3-sensor`, `lora-node`, `dual-rail-supply`, `encoder-oled-panel`, `proto-shield`, and `audio-jack-buffer`.

Gate:

- no rejected topology is committed;
- attempts never exceed the existing five-call wiring budget;
- at least 4/8 commit, compared with 0/8 in the source batch;
- persistent identical rejection signatures terminate as `commit_rejected`, not an unbounded correction loop.

This campaign does not adopt reasoning-disabled wiring globally. The plan's 12/16 adoption decision requires the separate two-arm, repeated experiment and is not necessary to validate the implemented no-progress reset.

### Fresh brief smoke

Run the real design chain, without place/route, in isolated temporary workspaces for:

1. a simple USB-C status LED;
2. an 8x8 addressable LED matrix controller with legitimate repeated parts;
3. an RP2040 CAN sensor node with programming access and connector-heavy wiring.

Gate:

- at least 2/3 complete through wiring;
- no terminal collection overflow or truncated JSON;
- every incomplete run ends with a typed, gate-preserving failure.

## Verdict

The change passes the live canary only if every safety gate holds and the judge, BOM, wiring, and fresh-brief cohort gates all pass. A failed cohort remains evidence: do not weaken deterministic electrical or collection gates, add retries, or raise token caps during this campaign.

## Live result

Campaign: `logs/live_verification/20260825T114515Z/`

| Cohort | Result | Gate |
|---|---:|---|
| prior judge aborts | 4/4 valid; no abort; $0.015227 | pass |
| BOM overflow states | 4/7 committed; 0 truncated; 0 accepted over bounds; $0.390513 | fail (target 5/7) |
| terminal wiring states | 3/8 committed; 5 honest `commit_rejected`; max 5 attempts; $0.157781 | fail (target 4/8) |
| fresh briefs | 1/3 completed; 0 truncated; $0.173901 | fail (target 2/3) |

Total live cost: **$0.737421**.

The judge policy fix is verified: all four spend rows used `eval_judge`, none
aborted, and all four verdicts were valid. The collection guard is also verified
as a safety boundary: no oversized collection was accepted and no overflow
degenerated into `truncated_json`. It is not yet a reliable completion recovery:
`rp2040-min`, `nrf52-beacon`, and `daq-8ch` repeated a 451st same-sheet item and
terminated as `collection_limit`.

Wiring improved three source failures (`r2r-dac`, `lora-node`, and
`proto-shield`) without accepting a rejected topology or exceeding the retry
budget. The remaining five terminated honestly at deterministic gates.

The fresh USB-C status LED completed through wiring. The LED matrix stopped at
BOM with `invalid_json`; the RP2040 CAN sensor stopped at BOM with
`collection_limit`. Because the BOM, wiring, and fresh-brief completion gates
failed, no full 34-brief batch was started.

Frozen source-state SHA-256 values were unchanged. Machine-readable evidence:
`manifest.json`, `judge_results.json`, `bom_results.json`,
`wiring_results.json`, `fresh_results.json`, and `verdict.json` in the campaign
directory.

Recovery plan: `docs/plans/llm-pipeline-reliability-and-model-migration-2026-08-25.md`.
