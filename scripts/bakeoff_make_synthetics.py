#!/usr/bin/env python3
"""Build the 4 calibration-floor synthetics for the electrical-review bakeoff.

Each is a copy of a SOUND corpus board's state.json with exactly ONE injected
defect. Three are objectively confirmable by a deterministic §9 gate; the fourth
is a textbook blocker true by construction (no §9 gate sees it). Run
``bakeoff_label_helper.py`` afterwards to confirm the §9 ones trip their gate and
the decap-drop stays §9-clean.

  syn_vdd_gnd    (§9.16) base run_25  MCP23017 VDD pin onto GND
  syn_self_short (§9.17) base run_03  decoupling C1 shorted across one net
  syn_can_miswire(§9.20) base run_23  SN65HVD230 RS pin onto a +rail (standby)
  syn_decap_drop (none)  base run_16  remove the sole bulk cap on the 10A rail
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CORPUS = Path("logs/bakeoff/20260618T200126Z/corpus")


def load(base: str) -> dict:
    return json.loads((CORPUS / base / "state.json").read_text())


def save(syn_id: str, state: dict) -> None:
    d = CORPUS / syn_id
    d.mkdir(exist_ok=True)
    (d / "state.json").write_text(json.dumps(state, indent=2))
    print(f"wrote {syn_id}/state.json  (stem={state['project_stem']})")


def move_endpoint(state, ref, pin, from_net, to_net):
    conns = state["bom"]["connections"]
    moved = False
    for c in conns:
        if c["net_name"] == from_net:
            keep = [e for e in c["endpoints"]
                    if not (e["ref"] == ref and str(e["pin"]) == str(pin))]
            if len(keep) != len(c["endpoints"]):
                c["endpoints"] = keep
                moved = True
    if not moved:
        sys.exit(f"FAIL: {ref}.{pin} not found on net {from_net}")
    for c in conns:
        if c["net_name"] == to_net:
            c["endpoints"].append({"ref": ref, "pin": pin})
            print(f"  moved {ref}.{pin}: {from_net} -> {to_net}")
            return
    sys.exit(f"FAIL: target net {to_net} not found")


def remove_part(state, ref):
    bom = state["bom"]
    n0 = len(bom["parts"])
    bom["parts"] = [p for p in bom["parts"] if p["ref"] != ref]
    if len(bom["parts"]) == n0:
        sys.exit(f"FAIL: part {ref} not found")
    for c in bom["connections"]:
        c["endpoints"] = [e for e in c["endpoints"] if e["ref"] != ref]
    bom["connections"] = [c for c in bom["connections"] if c["endpoints"]]
    # also drop from ic_groups / placement_hints / zones so the digest is clean
    bom.get("ic_groups", {}).pop(ref, None)
    for k, v in list(bom.get("ic_groups", {}).items()):
        bom["ic_groups"][k] = [r for r in v if r != ref]
    bom["placement_hints"] = [h for h in bom.get("placement_hints", [])
                              if h.get("ref") != ref]
    bom.get("component_zones", {}).pop(ref, None)
    print(f"  removed part {ref} and its connections")


# --- syn_vdd_gnd (§9.16): a positive-supply pin on a ground net ---
s = load("run_25_gpio-expander")
move_endpoint(s, "U1", "9", "VCC", "GND")   # MCP23017 pin 9 = VDD
s["project_stem"] = "SYN_VDD_GND"
save("syn_vdd_gnd", s)

# --- syn_self_short (§9.17): a 2-terminal part with both pins on one net ---
s = load("run_03_thermocouple-amp")
move_endpoint(s, "C1", "2", "GND", "+3V3")  # C1 now +3V3<->+3V3
s["project_stem"] = "SYN_SELF_SHORT"
save("syn_self_short", s)

# --- syn_can_miswire (§9.20): CAN transceiver RS pin tied to a +rail ---
s = load("run_23_can-node")
move_endpoint(s, "U3", "8", "GND", "+3V3")  # SN65HVD230 RS -> standby on rail
s["project_stem"] = "SYN_CAN_MISWIRE"
save("syn_can_miswire", s)

# --- syn_decap_drop (no §9 gate): remove the sole bulk cap on the 10A rail ---
s = load("run_16_highside-switch-10a")
remove_part(s, "C1")                        # 10uF, the only +12V bulk cap
s["project_stem"] = "SYN_DECAP_DROP"
save("syn_decap_drop", s)

print("\ndone: 4 synthetics written under", CORPUS)
