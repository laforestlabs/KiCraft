"""Independent, source-backed qualification of deterministic KiCraft blocks.

Project requests are private, bound to synthesis artifacts. Published contracts contain
only canonical engineering data; project refs, nets, text and paths never travel with
an entry. Exact equality is intentional: no implied operating or physical variants.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kicraft.fsutil import atomic_write_text

REQUEST_FILE = Path('.kicraft/qualified-leaf-requests.json')
SCHEMA = 1
METHOD = 'kicraft-independent-design-qualification@2'
QUALIFICATION_FILE = 'qualification.json'


class QualificationError(ValueError):
    """The block or its entry cannot substantiate the requested qualification."""


def _require(condition: Any, reason: str) -> None:
    if not condition:
        raise QualificationError(reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(payload, dict), f'expected an object in {path.name}')
    return payload


def _write(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _tool_version() -> str:
    result = subprocess.run(['kicad-cli', 'version'], capture_output=True, text=True, check=True, timeout=10)
    version = result.stdout.strip()
    _require(re.fullmatch(r'[0-9][A-Za-z0-9.()+~-]*', version), 'unrecognized KiCad CLI version')
    return version


def _dependencies(parts, project_dir: Path) -> dict:
    from kicraft.design.synthesis.parts_lookup import (
        DEFAULT_KICAD_FOOTPRINT_DIR, DEFAULT_KICAD_SYMBOL_DIR,
        resolve_footprint_library_path, resolve_symbol_library_path,
    )
    import kicraft.parts_library

    trusted = (Path(kicraft.parts_library.__file__).resolve().parent,
               DEFAULT_KICAD_SYMBOL_DIR.resolve(), DEFAULT_KICAD_FOOTPRINT_DIR.resolve())
    result = {}
    for part in parts:
        slib, _sname = part.symbol.split(':', 1)
        flib, fname = part.footprint.split(':', 1)
        assets = (
            (f'symbol:{part.symbol}', resolve_symbol_library_path(slib, project_root=project_dir)),
            (f'footprint:{part.footprint}', resolve_footprint_library_path(flib, project_root=project_dir) / f'{fname}.kicad_mod'),
        )
        for key, asset in assets:
            asset = asset.resolve()
            _require(any(asset.is_relative_to(root) for root in trusted),
                     f'{key} resolves outside reviewed KiCraft/stock assets; sharing would disclose project or user assets')
            if key not in result:
                result[key] = _sha(asset)
    return result


def _fabrication(project: dict) -> dict:
    rules = project['board']['design_settings']
    net = project['net_settings']
    _require(not rules.get('drc_exclusions'), 'board DRC exclusions are not portable evidence')
    _require({r['name'] for r in net['classes']} == {'Default', 'Power'}, 'unsupported fabrication netclasses')
    return {'design_settings': rules, 'classes': net['classes'], 'net_settings_version': net['meta']['version']}


def _require_supported_project_rules(*paths: Path) -> None:
    """Custom native rules are not represented in the portable contract."""
    for path in dict.fromkeys(path.with_suffix('.kicad_dru') for path in paths):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise QualificationError('custom KiCad .kicad_dru rules are not supported for automatic leaf qualification or reuse')


def _recipe_candidate(selection, arch, parts, bom):
    from kicraft.design.recipes import get_recipe, expand_recipe
    from kicraft.design.recipes.registry import complete_unused_published_ports

    definition = get_recipe(selection.recipe)
    if definition.recipe == 'me6211-3v3@2':
        raise QualificationError('ME6211 lacks exact effective bypass capacitance/ESR and board thermal derating evidence')
    _require(definition.family == 'usb-c-power-sink' and definition.maturity == 'production'
             and {a.code for a in definition.electrical_assertions} == {'usb_c_sink_cc_resistors'},
             f'{definition.recipe} has no independent functional qualification evaluator')
    expanded, = complete_unused_published_ports([expand_recipe(selection)], arch)
    by_role = defaultdict(list)
    for part in sorted(parts, key=lambda p: (p.recipe_role or '', _reference_key(p.ref))):
        _require(part.recipe_id == selection.recipe and part.recipe_instance == selection.instance,
                 'physical leaf has mixed or incomplete recipe ownership')
        by_role[part.recipe_role].append(part)
    ref_map = {}
    counts = Counter()
    for template in expanded.parts:
        role = template.recipe_role
        index = counts[role]
        _require(index < len(by_role[role]), 'recipe support network is incomplete')
        part = by_role[role][index]
        _require((part.symbol, part.footprint, part.value, part.mpn, part.assembly) ==
                 (template.symbol, template.footprint, template.value, template.mpn, template.assembly),
                 f'recipe part substitution at role {role}')
        ref_map[template.ref] = part.ref
        counts[role] += 1
    _require(len(ref_map) == len(parts), 'physical leaf has additional recipe parts')
    owner = next((m for m in bom.recipe_ownership if m.recipe == selection.recipe and m.instance == selection.instance), None)
    _require(owner is not None and set(owner.refs) == set(ref_map.values()), 'missing exact recipe ownership manifest')
    expected = {(ref_map[row.ref], row.pin): row.net for row in expanded.ownership.pins}
    _require(expected == {(row.ref, row.pin): row.net for row in owner.pins}, 'owned recipe pin topology differs from resolved recipe')
    bindings = selection.port_bindings
    vbus = bindings.get('vbus')
    _require(vbus is not None and arch.rail_voltages.get(vbus) == 5.0 and bindings.get('gnd') == 'GND',
             'USB fixed default-power qualification requires an explicit 5 V VBUS and canonical GND')
    _require(not selection.pin_allocations, 'unreviewed USB pin allocation')
    ports = [p.model_dump(mode='json') for p in definition.ports]
    _require(definition.source_documents and definition.electrical_assertions, 'missing reviewed USB source evidence')
    evidence = {
        'sources': [d.model_dump(mode='json') for d in definition.source_documents],
        'checks': [a.code for a in definition.electrical_assertions],
        'claim': 'Fixed-5V USB Type-C Rd topology; no data, PD, source-current, or hardware validation claim',
        'operating_point': {'input_nominal_v': 5.0, 'ground_v': 0.0, 'usb_mode': 'default-current-sink'},
    }
    ordered = [(canonical, ref_map[canonical]) for canonical in ref_map]
    return definition.recipe, definition.family, expanded.parameters, ports, bindings, ordered, expected, evidence, list(owner.placement_constraints)


def _reference_key(ref: str) -> tuple[str, int]:
    match = re.fullmatch(r'([A-Z]+)([0-9]+)', ref)
    _require(match is not None, 'noncanonical reference designator')
    return match[1], int(match[2])


def _lowerer_candidate(requirement, arch, parts, bom):
    from kicraft.design.lowering import verified_lowering

    proof = verified_lowering(requirement, bom)
    _require(proof is not None, 'lowerer lacks exact part, role or connectivity evidence')
    artifact, role_refs = proof
    owner_ids = {p.resolution_id for p in parts}
    _require(len(owner_ids) == 1 and None not in owner_ids and set(role_refs.values()) == {p.ref for p in parts},
             'physical leaf has mixed or incomplete lowerer ownership')
    owner_id, = owner_ids
    groups = {group.role: group for group in artifact.groups}
    _require(all(p.value == groups[p.lowering_role].value for p in parts),
             'lowerer part value is not canonical; project-supplied labels cannot be shared')
    bindings = requirement.ports
    if owner_id == 'pin-header@1':
        return _passive_header(owner_id, arch, parts, artifact, role_refs, bindings)
    if owner_id in {'switch-input@1', 'led-current-resistor@1'}:
        return _reviewed_control(owner_id, requirement, arch, artifact, role_refs)
    raise QualificationError(f'{owner_id} has no independent functional qualification evaluator')


def _passive_header(owner_id, arch, parts, artifact, role_refs, bindings):
    # Only passive continuity, not current rating or remote-device behavior.
    _require(len(parts) == 1 and not artifact.no_connects and all(p.net for p in artifact.pins),
             'header must have fully bound contacts')
    _require(len(bindings) == len(artifact.pins) and set(bindings.values()) == {p.net for p in artifact.pins},
             'each qualified header contact needs one uniquely bound published port')
    supplied = {p.net: p.pin for p in artifact.pins}
    ports = [dict(name=p, direction='passive', required=True, contact=supplied[n],
                  voltage_v=arch.rail_voltages.get(n, 0.0 if n == 'GND' else None))
             for p, n in bindings.items()]
    _require(all(p['voltage_v'] is not None and 0 <= p['voltage_v'] <= 5 for p in ports),
             'header operating envelope must be explicit low-voltage DC (0 to 5 V)')
    part, = parts
    expected = {(role_refs[(p.role, p.index)], p.pin): p.net for p in artifact.pins}
    evidence = {'sources': ['pin-header@1 deterministic contact map and native stock connector pin geometry'],
                'checks': ['one_distinct_conductor_per_contact'],
                'claim': '0–5 V DC passive contact presentation, no current rating or hardware validation claim',
                'operating_point': {p['name']: p['voltage_v'] for p in ports}}
    return owner_id, 'passive-contact-header', {}, ports, bindings, [('J1', part.ref)], expected, evidence, []


def _reviewed_control(owner_id, requirement, arch, artifact, role_refs):
    from kicraft.design.part_identity import reviewed_part

    bindings = requirement.ports
    groups = {group.role: group for group in artifact.groups}
    _require(bindings.get('gnd') == 'GND' and not artifact.no_connects and not artifact.local_bypasses,
             'control needs canonical ground and a complete, portable DC topology')
    sources = []
    parameters = {}
    operating: dict[str, Any] = {}
    if owner_id == 'led-current-resistor@1':
        led = groups['led']
        record = reviewed_part(led.mpn) if led.mpn else None
        _require(record is not None and record.manufacturer_sources,
                 'LED qualification requires an exact reviewed diode forward-voltage/current source')
        rating = record.operating_limits.get('forward_current_ma')
        vf = record.operating_limits.get('forward_voltage_v')
        _require(isinstance(rating, (int, float)) and isinstance(vf, (int, float)),
                 'the exact LED lacks reviewed forward-voltage/current evidence')
        _require(len(artifact.calculations) == 1, 'missing independently recomputed LED sizing evidence')
        calc, = artifact.calculations
        drive = arch.rail_voltages.get(bindings.get('drive'))
        _require(drive == calc.inputs['rail_voltage'] and vf == calc.inputs['led_vf']
                 and 0 < calc.inputs['target_current_ma'] <= rating,
                 'LED sizing inputs differ from exact rail or diode evidence')
        resistance = _resistance_ohms(calc.chosen_value)
        worst_a = drive / resistance
        # Zero forward drop bounds the LED current without treating a nominal
        # datasheet Vf as a guaranteed minimum.
        _require(worst_a * 1000 <= rating and worst_a * drive <= .01,
                 'LED worst-case current or series-resistor dissipation exceeds low-power qualification')
        sources.extend(record.manufacturer_sources)
        operating = {'drive_v': drive, 'ground_v': 0.0, 'worst_case_current_a_at_nominal_resistance': worst_a}
        parameters = {'series_resistance_ohm': resistance, **calc.inputs}
        function = 'resistor-limited-indicator'
        claim = 'Reviewed low-power indicator topology and nominal resistor sizing; no brightness or hardware validation claim'
        directions = {'drive': 'input', 'gnd': 'power'}
        checks = [calc.model_dump(mode='json')]
    else:
        _require(set(groups) in ({'switch', 'pullup'}, {'switch', 'pulldown'}),
                 'switch qualification requires an independently owned external bias; jumper or MCU-internal bias is not portable')
        group = groups['switch']
        record = reviewed_part(group.mpn) if group.mpn else None
        _require(record is not None and record.manufacturer_sources
                 and 'momentary-button' in record.physical_features,
                 'switch qualification requires exact SPST contact ratings')
        pull = groups.get('pullup') or groups['pulldown']
        resistance = _resistance_ohms(pull.value)
        supply = arch.rail_voltages.get(bindings.get('vdd'))
        limits = record.operating_limits
        _require(supply is not None and limits.get('ulc_min_current_test_voltage_v', float('inf')) <= supply <= 5.5
                 and supply <= limits.get('voltage_v', 0), 'switch needs an explicit reviewed logic rail')
        current = supply / resistance
        _require(limits.get('ulc_min_current_a', float('inf')) <= current <= limits.get('current_a', 0)
                 and supply * current <= min(limits.get('power_va', 0), .01), 'switch bias current is outside reviewed ratings')
        active = 'low' if pull.role == 'pullup' else 'high'
        sources.extend(record.manufacturer_sources)
        operating = {'vdd_v': supply, 'ground_v': 0.0, 'closed_bias_current_a': current, 'load': 'high-impedance logic input'}
        parameters = {'dc_active_level': active, 'external_bias_ohm': resistance}
        function = 'externally-biased-momentary-switch'
        claim = 'Reviewed SPST contact and external DC bias with a high-impedance load; no debounce or hardware validation claim'
        directions = {'vdd': 'power', 'gnd': 'power', 'signal': 'output'}
        checks = ['exact_normal_open_contact', 'explicit_bias_rail', 'source_backed_contact_current']
    _require(bindings.keys() == directions.keys() and len(set(bindings.values())) == len(bindings),
             'control ports are missing or combined with another boundary role')
    ports = [dict(name=name, direction=direction, required=True) for name, direction in directions.items()]
    expected = {(role_refs[(p.role, p.index)], p.pin): p.net for p in artifact.pins}
    counters = Counter()
    ordered = []
    for group in artifact.groups:
        for index in range(group.quantity):
            counters[group.reference_prefix] += 1
            ordered.append((f'{group.reference_prefix}{counters[group.reference_prefix]}', role_refs[(group.role, index)]))
    evidence = {'sources': sources, 'checks': checks, 'claim': claim, 'operating_point': operating}
    return owner_id, function, parameters, ports, bindings, ordered, expected, evidence, []


def _resistance_ohms(value: str) -> float:
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([kKmM]?)', value)
    _require(match is not None and float(match[1]) > 0, 'unsupported resistor value in DC calculation')
    multiplier = {'': 1, 'k': 1000, 'm': 1000000}[match[2].lower()]
    return float(match[1]) * multiplier


def _obligation_applies(relation: str, observed: float, value: float | None,
                        minimum: float | None = None, maximum: float | None = None) -> bool:
    if relation == 'equal':
        return observed == value
    if relation == 'minimum':
        return value is not None and observed >= value
    if relation == 'maximum':
        return value is not None and observed <= value
    return relation == 'range' and minimum is not None and maximum is not None and minimum <= observed <= maximum


def _require_supported_obligations(owner: str, requirements: list, ports: list[dict]) -> None:
    """Confirm typed consumer obligations against the actual proof, never prose."""
    permitted_parameters = {
        'pin-header@1': {'rows', 'gender', 'positions'},
        'led-current-resistor@1': {'rail_voltage', 'led_vf', 'target_current_ma'},
        'switch-input@1': {'pull_policy', 'active_level', 'resistance', 'voltage'},
    }.get(owner, set())
    for requirement in requirements:
        unsupported = set(requirement.parameters) - permitted_parameters
        _require(not unsupported,
                 f'{owner} does not qualify the requested parameter(s): {", ".join(sorted(unsupported))}')
        for obligation in requirement.obligations:
            if obligation.kind == 'physical':
                # Exact BOM/footprint identity is qualified separately. Final
                # physical-class checks still apply to the consuming board.
                continue
            if obligation.kind == 'quantitative':
                unit = obligation.unit.strip().casefold()
                # The obligation is scoped to this reviewed lowerer instance.
                # Pin/contact units name its topology, regardless of the header's purpose.
                if owner == 'pin-header@1' and unit in {'pin', 'pins', 'contact', 'contacts'}:
                    _require(_obligation_applies(obligation.relation, len(ports), obligation.value,
                                                 obligation.minimum, obligation.maximum),
                             'requested contact count exceeds qualified header topology')
                    continue
                terms = set(re.findall(r'[a-z]+', obligation.quantity.casefold()))
                if (owner.startswith('usb-c-5v-sink@') and unit in {'v', 'volt', 'volts'}
                    and terms <= {'nominal', 'usb', 'c', 'input', 'vbus', 'power', 'supply', 'voltage'}
                    and terms & {'voltage', 'vbus'}):
                    _require((obligation.relation == 'equal' and obligation.value == 5.0)
                             or (obligation.relation == 'range' and obligation.minimum == obligation.maximum == 5.0),
                             'requested input voltage envelope exceeds the reviewed fixed 5 V operating point')
                    continue
                raise QualificationError(f'{owner} lacks evidence for quantitative obligation {obligation.quantity!r} ({obligation.unit})')
            if (obligation.kind == 'quantity' and owner == 'pin-header@1'
                and obligation.subject in {'pin', 'pins', 'contact', 'contacts', 'header-pins', 'header-contacts'}):
                _require(obligation.minimum <= len(ports), 'requested contact count exceeds qualified header topology')
                continue
            raise QualificationError(f'{owner} lacks evidence for {obligation.kind} obligation')


def _request(sheet, state, project_dir: Path, fab: dict, power: set[str], physical: dict, tool: str) -> dict:
    arch, bom = state.architecture, state.bom
    parts = sheet.parts
    _require(parts and not sheet.sheet.from_library, 'not an independently generated physical leaf')
    _require(all(p.assembly for p in parts), 'board-fabricated terminals are not portable rigid geometry')
    recipe_ids = {(p.recipe_id, p.recipe_instance) for p in parts}
    lower_ids = {p.lowering_requirement_id for p in parts}
    if len(recipe_ids) == 1 and None not in next(iter(recipe_ids)):
        recipe, instance = next(iter(recipe_ids))
        selection = next((s for s in arch.recipe_selections if (s.recipe, s.instance) == (recipe, instance)), None)
        _require(selection is not None, 'recipe selection is unavailable')
        candidate = _recipe_candidate(selection, arch, parts, bom)
        bound_requirements = [r for r in arch.requirements if r.id in selection.requirement_ids]
    elif len(lower_ids) == 1 and None not in lower_ids:
        requirement = next((r for r in arch.requirements if r.id in lower_ids), None)
        _require(requirement is not None, 'lowerer requirement is unavailable')
        candidate = _lowerer_candidate(requirement, arch, parts, bom)
        bound_requirements = [requirement]
    else:
        raise QualificationError('no independently owned reviewed circuit spans this physical leaf')
    owner, family, parameters, ports, bindings, ordered, expected, evidence, placement = candidate
    _require_supported_obligations(owner, bound_requirements, ports)
    ref_map = {ref: canonical for canonical, ref in ordered}
    _require(not any(set(a.refs) & ref_map.keys() for a in bom.arrays), 'board-coupled array geometry is not portable')
    _require(not any(set(e.refs) & ref_map.keys() for e in bom.edge_interfaces), 'board-fabricated edge geometry is not portable')
    _require(not any(r.get('owner_ref') in ref_map for r in physical.get('thermal_requirements') or []),
             'board-dependent thermal geometry is not portable')

    actual = {(ep.ref, ep.pin): c.net_name for c in bom.connections for ep in c.endpoints if ep.ref in ref_map}
    actual.update({(ep.ref, ep.pin): None for ep in bom.no_connect_pins if ep.ref in ref_map})
    _require(actual == expected, 'BOM wiring is not exactly the reviewed block graph')
    by_net = defaultdict(list)
    for (ref, pin), net in expected.items():
        if net is not None:
            by_net[net].append([ref_map[ref], pin])
    boundary = set(bindings.values()) & by_net.keys()
    for conn in bom.connections:
        if conn.net_name in by_net and any(ep.ref not in ref_map for ep in conn.endpoints):
            _require(conn.net_name in boundary, 'private block net reaches external circuit')
    for net, _ep in sheet.inter_sheet_endpoints:
        _require(net.name in boundary, 'unpublished hierarchical port crosses the block boundary')

    aliases = defaultdict(list)
    for key, net in bindings.items():
        if net in by_net:
            aliases[net].append(key)
    net_map = {net: 'PORT_' + '_'.join(sorted(keys)).upper() for net, keys in aliases.items()}
    for index, net in enumerate(sorted(by_net.keys() - boundary, key=lambda n: sorted(by_net[n])), 1):
        net_map[net] = f'LOCAL_{index}'
    _require(len(set(net_map.values())) == len(by_net), 'ambiguous normalized port names')
    part_by_ref = {p.ref: p for p in parts}
    safe_zone_values = {
        'edge', 'corner', 'center', 'interior', 'top', 'bottom', 'left', 'right',
        'top-left', 'top-right', 'bottom-left', 'bottom-right',
    }
    zones = {ref_map[ref]: {k: v for k, v in zone.items() if k in ('target', 'value', 'side')}
             for ref, zone in bom.component_zones.items() if ref in ref_map}
    _require(all(value in safe_zone_values for zone in zones.values() for value in zone.values()),
             'unreviewed placement zone must not enter the cross-project catalogue')
    contract = {
        'method': METHOD, 'owner': owner, 'function': family, 'parameters': parameters,
        'parts': [{'ref': canonical, **{key: getattr(part_by_ref[ref], key) for key in ('symbol', 'footprint', 'value', 'mpn')}}
                  for canonical, ref in ordered],
        'nets': {net_map[net]: sorted(pads) for net, pads in by_net.items()},
        'no_connects': sorted([ref_map[ref], pin] for (ref, pin), net in expected.items() if net is None),
        'ports': [dict(p, binding=net_map.get(bindings.get(p['name']))) for p in sorted(ports, key=lambda p: p['name'])],
        'placement_constraints': placement,
        'component_zones': zones,
        'fabrication': {**fab, 'power_nets': sorted(net_map[net] for net in by_net.keys() & power)},
        'dependencies': _dependencies(parts, project_dir), 'electrical_evidence': evidence,
        'tool': tool, 'transforms': ['rigid-translation', 'rigid-in-plane-rotation'],
    }
    return {'identity': _digest(contract), 'contract': contract, 'ref_map': ref_map, 'net_map': net_map,
            'schematic': f'{sheet.sheet.stem}.kicad_sch', 'boundary': sorted(boundary)}


def synthesis_requests(state, project_dir: Path, sheet_instances, pcb: Path, pro: Path, ap: Path, results) -> dict:
    """Derive hash-bound private requests after the actual synthesis checks run."""
    checks = {r.name: r for r in results}
    required = ('9.12 ERC', '9.13 netlist faithfulness')
    shared = ''
    if any(name not in checks or not checks[name].ok or 'skipped' in checks[name].message.lower() for name in required):
        shared = 'native synthesis ERC or schematic-to-BOM netlist evidence missing'
    try:
        if not shared:
            _require_supported_project_rules(pcb, pro)
            from kicraft.pcb_layers import declared_copper_layers, has_two_ounce_outer_stackup
            _require(declared_copper_layers(pcb) == ['F.Cu', 'B.Cu'], 'only exact two-layer stack-up is currently supported')
            _require(not has_two_ounce_outer_stackup(pcb), 'heavy-copper qualification is not supported')
            cfg = _read(ap)
            fab = _fabrication(_read(pro))
            fab['stackup'] = _stackup(pcb)
            tool = _tool_version()
    except (QualificationError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        shared = str(exc)
    requests = {}
    for sheet in sheet_instances:
        path = f'/{sheet.instance_uuid}'
        try:
            _require(not shared, shared)
            request = _request(sheet, state, project_dir, fab, set(cfg['power_nets']), cfg, tool)
            request['schematic_sha256'] = _sha(project_dir / request['schematic'])
            requests[path] = request
        except (QualificationError, OSError, LookupError, TypeError, ValueError) as exc:
            requests[path] = {'reason': str(exc)}
    root_schematic = pro.with_suffix('.kicad_sch')
    return {'schema': SCHEMA, 'pcb_sha256': _sha(pcb), 'project_sha256': _sha(pro),
            'root_schematic_sha256': _sha(root_schematic), 'config_sha256': _sha(ap),
            'synthesis_checks': list(required), 'requests': requests}


def _stackup(path: Path) -> dict:
    from kicraft.pcb_layers import has_two_ounce_outer_stackup
    text = path.read_text(encoding='utf-8')
    thickness = re.search(r'(?m)^\s*\(thickness ([\d.]+)\)', text)
    _require(thickness is not None, 'missing PCB thickness')
    _require('(stackup' not in text or has_two_ounce_outer_stackup(path), 'explicit stack-up has no reviewed qualification method')
    return {'layers': ['F.Cu', 'B.Cu'], 'thickness_mm': float(thickness[1]), 'outer_copper': 'KiCad two-layer default'}


def read_requests(project: Path, pcb: Path, schematic: Path) -> tuple[dict, str]:
    file = project / REQUEST_FILE
    if not file.is_file():
        return {}, 'verified synthesis qualification request unavailable'
    try:
        _require_supported_project_rules(pcb, schematic)
        data = _read(file)
        _require(data['schema'] == SCHEMA, 'unsupported qualification request schema')
        _require(data['pcb_sha256'] == _sha(pcb), 'synthesized PCB changed since qualification input was evaluated')
        _require(data['root_schematic_sha256'] == _sha(schematic), 'root sheet bindings changed since qualification input was evaluated')
        _require(data['project_sha256'] == _sha(schematic.with_suffix('.kicad_pro')), 'fabrication rules changed since qualification input was evaluated')
        ap = schematic.with_name(schematic.stem + '_autoplacer.json')
        _require(data['config_sha256'] == _sha(ap), 'physical constraints changed since qualification input was evaluated')
        _require(data['synthesis_checks'] == ['9.12 ERC', '9.13 netlist faithfulness'], 'required synthesis checks unavailable')
        for path, request in data['requests'].items():
            if 'contract' not in request:
                continue
            _require(request['identity'] == _digest(request['contract']), 'qualification request identity does not match contract')
            name = Path(request['schematic'])
            if name.name != str(name) or request['schematic_sha256'] != _sha(project / name):
                data['requests'][path] = {'reason': 'leaf schematic changed since qualification input was evaluated'}
        return data['requests'], ''
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {}, str(exc)


def _pad_graph(components: dict, ref_map: dict, net_map: dict) -> dict:
    graph = defaultdict(set)
    _require(set(components) == set(ref_map), 'native physical leaf component inventory differs from ownership')
    for ref, component in components.items():
        for pad in component['pads']:
            net = pad.get('net')
            if net:
                _require(net in net_map, 'native PCB has an unowned pad net')
                graph[net_map[net]].add((ref_map[ref], str(pad['pad_id'])))
    return {net: sorted([ref, pin] for ref, pin in pins) for net, pins in graph.items()}


def _require_layout_graph(layout: dict, contract: dict, ref_map: dict, net_map: dict) -> None:
    _require(_pad_graph(layout['components'], ref_map, net_map) == contract['nets'],
             'schematic/BOM and native PCB pad connectivity differ')
    for item in [*layout.get('traces', ()), *layout.get('vias', ())]:
        _require(item['net'] in net_map, 'unowned or boundary-crossing copper cannot be extracted')


def validate_entry(leaf) -> dict:
    """Fail closed for missing evidence, invalid identity, or any changed payload."""
    from .installer import verify_content_hash_at_install

    verify_content_hash_at_install(leaf)
    entry = _read(leaf.dir / QUALIFICATION_FILE)
    contract = entry['contract']
    _require(contract['method'] == METHOD, 'requalification required: obsolete independent DRC policy lacks full dangling-copper evidence')
    _require(entry['schema'] == SCHEMA and entry['identity'] == _digest(contract), 'invalid block qualification identity')
    _require(leaf.manifest.name == _entry_name(contract) and leaf.manifest.version == _entry_version(entry['identity']),
             'manifest and qualified immutable identity differ')
    _require(contract['dependencies'] and contract['electrical_evidence']['sources'] and contract['tool'],
             'qualified entry lacks dependency, functional, or tool evidence')
    evidence = entry['evidence']
    _require(evidence['artifact_source'] == 'independent-local-solve' and
             evidence['method'] == METHOD and evidence['independent_drc']['unconnected'] == 0 and
             evidence['independent_drc']['shorts'] == 0 and evidence['independent_drc']['ran'] is True and
             evidence['independent_drc']['dangling_violations'] == 0 and
             set(evidence['independent_drc']['included_severities']) == {'error', 'warning', 'exclusion'} and
             not evidence['independent_drc']['rejected_rules'], 'local independent DRC evidence missing or failed')
    _require(set(evidence['artifacts']) == {'schematic.kicad_sch', 'leaf_routed.kicad_pcb', 'solved_layout.json', 'metadata.json'},
             'qualified artifact evidence is incomplete')
    for name, digest in evidence['artifacts'].items():
        _require(_sha(leaf.dir / name) == digest, f'qualified artifact evidence does not match {name}')
    layout = _read(leaf.dir / 'solved_layout.json')
    _require_layout_graph(layout, contract, {r: r for r in layout['components']}, {n: n for n in contract['nets']})
    return entry


def _entry_name(contract: dict) -> str:
    return contract['owner'].split('@')[0].replace('_', '-')


def _entry_version(identity: str) -> str:
    return f'1.0.0+{identity}'


def discover(library, requests: dict) -> tuple[dict[str, list], list[dict]]:
    """Select only validated entries; report quarantined catalogue rows separately."""
    desired = {(r['contract']['owner'], r['identity']) for r in requests.values() if 'contract' in r}
    names = {_entry_name(r['contract']) for r in requests.values() if 'contract' in r}
    if not names:
        return {}, []
    loaded, broken = library.load_all()
    entries = defaultdict(list)
    rejected = [{'entry': b.dir.name, 'status': 'quarantined', 'reason': b.reason} for b in broken]
    for leaf in sorted(loaded, key=lambda row: row.slug):
        if leaf.manifest.name not in names:
            continue
        if not (leaf.dir / QUALIFICATION_FILE).is_file():
            rejected.append({'entry': leaf.slug, 'status': 'unqualified', 'reason': 'independent qualification evidence missing'})
            continue
        try:
            entry = validate_entry(leaf)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            rejected.append({'entry': leaf.slug, 'status': 'quarantined', 'reason': str(exc)})
            continue
        key = (entry['contract']['owner'], entry['identity'])
        if key not in desired:
            rejected.append({'entry': leaf.slug, 'status': 'incompatible',
                             'reason': 'exact operating, dependency, interface, or fabrication contract differs'})
            continue
        entries[entry['identity']].append((leaf, entry))
    return dict(entries), rejected


def _template(contract: dict, directory: Path):
    from kicraft.design.models import Architecture, BOM, BomPart, InterSheetNet, NetConnection, PinEndpoint, Sheet
    from kicraft.design.synthesis.emitter import emit_schematic
    from kicraft.design.synthesis.validation import check_netlist_faithfulness

    ports = defaultdict(list)
    for port in contract['ports']:
        if port['binding']:
            ports[port['binding']].append(port['direction'])
    sheet = Sheet(name='BLOCK', stem='BLOCK', function=contract['function'])
    boundary = Sheet(name='BOUNDARY', stem='BOUNDARY', function='Qualification interface')
    power = contract['fabrication']['power_nets']
    inter_sheet_nets = [
        InterSheetNet(name=net, endpoints=[
            {'sheet': sheet.name, 'direction': 'passive' if direction == 'power' else direction},
            {'sheet': boundary.name, 'direction': 'passive'},
        ])
        for net, directions in sorted(ports.items())
        for direction in [directions[0]] if net not in power
    ]
    architecture = Architecture(sheets=[sheet, boundary], power_nets=power, inter_sheet_nets=inter_sheet_nets)
    bom = BOM(
        parts=[BomPart(sheet=sheet.name, **part) for part in contract['parts']],
        connections=[
            NetConnection(sheet=sheet.name, net_name=net,
                          endpoints=[PinEndpoint(ref=ref, pin=pin) for ref, pin in pins])
            for net, pins in contract['nets'].items()
        ],
        no_connect_pins=[PinEndpoint(ref=ref, pin=pin) for ref, pin in contract['no_connects']],
    )
    root, _ = emit_schematic(directory, 'qualified-block', architecture, bom)
    netlist = check_netlist_faithfulness(directory, root.stem, bom)
    _require(netlist.ok and 'skipped' not in netlist.message.lower(),
             'sanitized schematic did not reproduce the reviewed block netlist')
    return bom


def _native_footprints(board) -> dict:
    return {fp.GetReference(): fp for fp in board.GetFootprints()}


def _board_poses(board) -> dict:
    import pcbnew

    return {
        ref: (pcbnew.ToMM(fp.GetPosition().x), pcbnew.ToMM(fp.GetPosition().y),
              float(fp.GetOrientationDegrees()) % 360, board.GetLayerName(fp.GetLayer()))
        for ref, fp in _native_footprints(board).items()
    }


def _native_pad_inventory(board, ref_map: dict, net_map: dict) -> dict:
    result = {}
    for ref, footprint in _native_footprints(board).items():
        _require(ref in ref_map, 'native board contains an unrelated footprint')
        center = footprint.GetPosition()
        pads = Counter()
        for p in footprint.Pads():
            net = p.GetNetname() or None
            _require(net is None or net in net_map, 'native pad carries an unknown network')
            pos = p.GetPosition()
            size, drill = p.GetSize(), p.GetDrillSize()
            pads[(p.GetNumber(), net_map[net] if net else None,
                  pos.x - center.x, pos.y - center.y, size.x, size.y, drill.x, drill.y,
                  p.GetShape(), p.GetAttribute(), tuple(sorted(p.GetLayerSet().Seq())))] += 1
        fpid = footprint.GetFPID()
        result[ref_map[ref]] = ((str(fpid.GetLibNickname()), str(fpid.GetLibItemName())), pads)
    return result


def _pos_key(x: float, y: float) -> tuple[int, int]:
    # Native KiCad serializes integer nanometers; allow 1 µm for the float transform.
    return round(x * 1000), round(y * 1000)


def _copper_keys(layout: dict, *, dx: float = 0, dy: float = 0) -> Counter:
    keys = Counter()
    for row in layout['traces']:
        ends = tuple(sorted(_pos_key(row[p]['x'] + dx, row[p]['y'] + dy) for p in ('start', 'end')))
        keys['segment', row['net'], row['layer'], round(row['width_mm'] * 1000), ends] += 1
    for row in layout['vias']:
        p = row['pos']
        keys['via', row['net'], round(row['size_mm'] * 1000), round(row['drill_mm'] * 1000),
             _pos_key(p['x'] + dx, p['y'] + dy)] += 1
    return keys


def _native_copper(board, nets: dict, *, restricted: bool = True) -> dict:
    import pcbnew

    def point(pos):
        return {'x': pcbnew.ToMM(pos.x), 'y': pcbnew.ToMM(pos.y)}

    result: dict[str, list] = {'traces': [], 'vias': []}
    for item in board.GetTracks():
        net = item.GetNetname()
        if not restricted and net not in nets:
            continue
        _require(net in nets, 'native copper crosses or leaves the owned block')
        if isinstance(item, pcbnew.PCB_VIA):
            through = item.GetViaType() == pcbnew.VIATYPE_THROUGH
            if not restricted and not through:
                continue
            _require(through, 'blind, buried, or microvia geometry is not portable')
            result['vias'].append({'net': nets[net], 'pos': point(item.GetPosition()),
                                   'size_mm': pcbnew.ToMM(item.GetWidth(pcbnew.F_Cu)),
                                   'drill_mm': pcbnew.ToMM(item.GetDrill())})
        elif item.GetClass() == 'PCB_TRACK':
            layer = board.GetLayerName(item.GetLayer())
            if not restricted and layer not in ('F.Cu', 'B.Cu'):
                continue
            _require(layer in ('F.Cu', 'B.Cu'), 'inner-layer trace copper is not portable')
            result['traces'].append({'net': nets[net], 'layer': layer,
                                     'start': point(item.GetStart()), 'end': point(item.GetEnd()),
                                     'width_mm': pcbnew.ToMM(item.GetWidth())})
        elif restricted:
            raise QualificationError('native copper shape is not represented in parent stamping')
    return result


def _geometry_translation(board, layout: dict, ref_map: dict) -> tuple[float, float]:
    poses = _board_poses(board)
    _require(set(poses) == set(ref_map) and set(ref_map.values()) == set(layout['components']),
             'layout and native footprint inventories differ')
    first = next(iter(poses))
    first_pos = layout['components'][ref_map[first]]['pos']
    dx, dy = poses[first][0] - first_pos['x'], poses[first][1] - first_pos['y']
    for ref, (x, y, angle, layer) in poses.items():
        part = layout['components'][ref_map[ref]]
        _require(abs(x - part['pos']['x'] - dx) < .001 and abs(y - part['pos']['y'] - dy) < .001 and
                 abs((angle - part['rotation'] + 180) % 360 - 180) < .001 and layer == part['layer'],
                 'native footprint pose differs from authored layout')
    return dx, dy


def _check_source_features(board, private_nets: set[str]) -> None:
    import pcbnew

    # An ordinary board-wide public power plane is allowed only if removing it
    # leaves a standalone DRC-clean block. No room for dropping rule areas or
    # private polygonal copper (parent stamping cannot represent those yet).
    for zone in board.Zones():
        _require(not zone.GetIsRuleArea() and zone.GetNetname() not in private_nets,
                 'block depends on a native keepout or private copper area')
    for fp in board.GetFootprints():
        _require(not list(fp.Zones()), 'footprint contains nonportable native zones')
    for drawing in board.GetDrawings():
        layer = board.GetLayerName(drawing.GetLayer())
        _require(layer not in ('F.Cu', 'B.Cu'), 'block contains board copper not represented as tracks')
        if layer == 'Edge.Cuts':
            _require(isinstance(drawing, pcbnew.PCB_SHAPE), 'unsupported standalone edge geometry')


def _vector(row: dict, dx: float = 0, dy: float = 0):
    import pcbnew

    return pcbnew.VECTOR2I(pcbnew.FromMM(row['x'] + dx), pcbnew.FromMM(row['y'] + dy))


def _stamp_native(directory: Path, request: dict, layout: dict, source_path: Path, bom) -> Path:
    import pcbnew
    from kicraft.design.synthesis.kicad_pcb_stub import write_empty_pcb

    source = pcbnew.LoadBoard(str(source_path))
    ref_map, net_map = request['ref_map'], request['net_map']
    contract = request['contract']
    _check_source_features(source, set(net_map) - set(request['boundary']))
    dx, dy = _geometry_translation(source, layout, ref_map)
    _require(_copper_keys(_native_copper(source, net_map)) == _copper_keys(layout, dx=dx, dy=dy),
             'native tracks/vias differ from extractable local layout')

    pcb_path = write_empty_pcb(directory, 'BLOCK', bom, project_root=directory)
    board = pcbnew.LoadBoard(str(pcb_path))
    for item in list(board.GetDrawings()):
        board.Remove(item)
    for item in source.GetDrawings():
        if item.GetLayer() == pcbnew.Edge_Cuts:
            # Copy only dimensions of the routed rectangle/segments, not
            # any source-board object identity, text or project properties.
            shape = pcbnew.PCB_SHAPE()
            shape.SetShape(item.GetShape())
            shape.SetStart(item.GetStart())
            shape.SetEnd(item.GetEnd())
            shape.SetWidth(item.GetWidth())
            shape.SetLayer(pcbnew.Edge_Cuts)
            board.Add(shape)
    board.GetDesignSettings().SetBoardThickness(pcbnew.FromMM(contract['fabrication']['stackup']['thickness_mm']))
    for ref, fp in _native_footprints(board).items():
        row = layout['components'][ref]
        _require(row['layer'] == 'F.Cu', 'back-side footprints are not yet supported for qualification')
        fp.SetOrientationDegrees(row['rotation'])
        fp.SetPosition(_vector(row['pos'], dx, dy))
        fp.SetPath(pcbnew.KIID_PATH())
        fp.Models().clear()  # 3D links are not electrical/physical dependencies of this package.
    n_ident = {n: n for n in contract['nets']}
    r_ident = {r: r for r in layout['components']}
    _require(_native_pad_inventory(source, ref_map, net_map) == _native_pad_inventory(board, r_ident, n_ident),
             'sanitized footprint dependency or pad geometry differs from the locally routed board')
    codes = {name: board.FindNet(name).GetNetCode() for name in n_ident}
    for row in layout['traces']:
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(_vector(row['start'], dx, dy))
        track.SetEnd(_vector(row['end'], dx, dy))
        track.SetWidth(pcbnew.FromMM(row['width_mm']))
        track.SetLayer(pcbnew.F_Cu if row['layer'] == 'F.Cu' else pcbnew.B_Cu)
        board.Add(track)
        track.SetNetCode(codes[row['net']])
    for row in layout['vias']:
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(_vector(row['pos'], dx, dy))
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetWidth(pcbnew.FromMM(row['size_mm']))
        via.SetDrill(pcbnew.FromMM(row['drill_mm']))
        board.Add(via)
        via.SetNetCode(codes[row['net']])
    _require(pcbnew.SaveBoard(str(pcb_path), board), 'could not write sanitized native block')
    _require(_copper_keys(_native_copper(pcbnew.LoadBoard(str(pcb_path)), n_ident)) ==
             _copper_keys(layout, dx=dx, dy=dy), 'staged native board lost routed copper')
    return pcb_path


def _strict_drc(pcb: Path, fabrication: dict, *, staged_layout: dict | None = None) -> dict:
    rules = fabrication['design_settings']
    _write(pcb.with_suffix('.kicad_pro'), {
        'board': {'design_settings': rules},
        'meta': {'filename': pcb.with_suffix('.kicad_pro').name, 'version': 1},
        'net_settings': {'classes': fabrication['classes'], 'meta': {'version': fabrication['net_settings_version']},
                         'netclass_patterns': [{'netclass': 'Power', 'pattern': name}
                                               for name in fabrication['power_nets']]},
    })
    report_path = pcb.with_suffix('.qualification-drc.json')

    def native_report() -> dict:
        report_path.unlink(missing_ok=True)
        result = subprocess.run(
            ['kicad-cli', 'pcb', 'drc', '--format', 'json', '--severity-all', '--all-track-errors',
             '--output', str(report_path), str(pcb)],
            capture_output=True, text=True, timeout=60,
        )
        _require(result.returncode == 0 and report_path.is_file(), 'native independent qualification DRC did not run')
        data = _read(report_path)
        _require(set(data['included_severities']) == {'error', 'warning', 'exclusion'},
                 'native DRC did not include all severities')
        return data

    report = native_report()
    normalized = []
    # Only the unpublished, no-zone staging board is eligible for this
    # normalization. The exact result must undergo standalone DRC again.
    if staged_layout is not None and not report['unconnected_items']:
        normalized = _remove_staged_dangling(pcb, staged_layout, report, native_report)
        if normalized:
            report = native_report()
    severities = report['included_severities']
    _require(report['kicad_version'], 'native DRC tool identity is missing')
    dangling = {'track_dangling', 'via_dangling'}
    violations = [row['type'] for row in report['violations']
                  if row['severity'] != 'warning' or row['type'] in dangling]
    unconnected = len(report['unconnected_items'])
    _require(not violations and unconnected == 0,
             f'independent no-plane DRC failed ({unconnected} opens; {dict(Counter(violations))})')
    return {'ran': True, 'unconnected': unconnected, 'shorts': 0, 'dangling_violations': 0,
            'included_severities': severities, 'rejected_rules': violations, 'normalization': normalized}


def _copper_text_items(text: str) -> dict[str, tuple[str, int, int]]:
    from kicraft.pcb_layers import _sexpr_end

    result = {}
    for item in re.finditer(r'(?m)^\t\((via|segment)(?=\s)', text):
        start = item.start() + 1
        end = _sexpr_end(text, start)
        uuid = re.search(r'\(uuid "([0-9a-f-]+)"\)', text[start:end])
        _require(uuid is not None and uuid[1] not in result, 'native copper UUID is missing or ambiguous')
        result[uuid[1]] = (item[1], start, end)
    return result


def _remove_staged_dangling(pcb: Path, layout: dict, report: dict, native_report) -> list[dict]:
    """Trial-remove exactly reported dangling items, preserving all anchors."""
    import pcbnew
    from kicraft.autoplacer.brain.escape_cleanup import _partition
    from kicraft.autoplacer.hardware.local_contract import _copper_graph, _position

    kinds = {'via_dangling': 'via', 'track_dangling': 'segment'}
    def report_bad(data):
        return [row for row in data['violations'] if row['severity'] != 'warning' or row['type'] in kinds]

    bad = report_bad(report)
    if any(row['type'] not in kinds or row['severity'] != 'warning' for row in bad):
        return []
    normalized = []
    for _ in range(16):
        if not bad:
            break
        text = pcb.read_text(encoding='utf-8')
        spans = _copper_text_items(text)
        original = pcbnew.LoadBoard(str(pcb))
        _require(not list(original.Zones()), 'redundant copper proof requires the standalone no-plane board')
        copper = {str(item.m_Uuid.AsString()): item for item in original.GetTracks()}
        for row in bad:
            items = row['items']
            if len(items) != 1:
                continue
            uuid = items[0]['uuid']
            span = spans.get(uuid)
            if not span or span[0] != kinds[row['type']] or uuid not in copper:
                continue
            net = copper[uuid].GetNetname()
            graph, pads, _ = _copper_graph(original, net)
            anchors = [node for candidates in pads.values() for node in candidates]
            for other_id, item in copper.items():
                if uuid == other_id or item.GetNetname() != net:
                    continue
                if isinstance(item, pcbnew.PCB_VIA):
                    anchors.extend((_position(item.GetPosition()), layer) for layer in (pcbnew.F_Cu, pcbnew.B_Cu)
                                   if item.IsOnLayer(layer))
                elif item.GetClass() == 'PCB_TRACK':
                    anchors.extend((_position(point), item.GetLayer()) for point in (item.GetStart(), item.GetEnd()))
                else:
                    raise QualificationError('staged dangling cleanup requires linear two-layer copper')
            before = _partition(graph, anchors)
            atomic_write_text(pcb, text[:span[1]] + text[span[2]:])
            changed = pcbnew.LoadBoard(str(pcb))
            after_graph, after_pads, _ = _copper_graph(changed, net)
            after = _partition(after_graph, anchors)
            if pads == after_pads and before == after:
                updated = native_report()
                next_bad = report_bad(updated)
                if (not updated['unconnected_items'] and
                    all(r['severity'] == 'warning' and r['type'] in kinds for r in next_bad)):
                    normalized.append({'kind': span[0], 'net': net, 'preserved_anchor_count': len(anchors)})
                    bad = next_bad
                    break
            atomic_write_text(pcb, text)
        else:
            break
    if not normalized:
        return []
    board = pcbnew.LoadBoard(str(pcb))
    refs = {ref: ref for ref in layout['components']}
    dx, dy = _geometry_translation(board, layout, refs)
    nets = {row['net']: row['net'] for row in layout['traces'] + layout['vias']}
    remaining = _copper_keys(_native_copper(board, nets))
    _require(not (remaining - _copper_keys(layout, dx=dx, dy=dy)),
             'dangling normalization added or modified qualified copper geometry')
    for kind in ('traces', 'vias'):
        kept = []
        for row in layout[kind]:
            key = next(iter(_copper_keys({'traces': [row] if kind == 'traces' else [],
                                          'vias': [row] if kind == 'vias' else []}, dx=dx, dy=dy)))
            if remaining[key]:
                kept.append(row)
                remaining[key] -= 1
        layout[kind] = kept
    _require(not +remaining, 'normalized copper is not completely represented in the solved layout')
    return normalized


def _canonical_layout(source: dict, request: dict) -> dict:
    from kicraft.cli._replicate_leaves import remap_solved_layout

    identity = dict(sheet_name='BLOCK', sheet_file='schematic.kicad_sch',
                    instance_path='/BLOCK', parent_instance_path='/')
    identity['subcircuit_id'] = dict(identity)
    refs, nets = request['ref_map'], request['net_map']
    remapped = remap_solved_layout(source, refs, nets, identity)
    # Only geometry and typed ports cross the privacy boundary. In particular
    # annotations, route-debug paths, run notes and escape-source IDs do not.
    out = {key: remapped[key] for key in ('schema_version', 'frozen', 'bounding_box', 'components',
                                          'traces', 'vias', 'interface_anchors')}
    out.update(identity)
    out['silkscreen'] = []
    out['score'] = float(remapped.get('score', 0))
    out['ports'] = []
    for port in source['ports']:
        name = nets.get(port['net_name'])
        _require(name is not None, 'layout contains an unbound interface port')
        out['ports'].append({key: value for key, value in port.items() if key not in ('description', 'name', 'net_name')}
                            | {'name': name, 'net_name': name})
    for row, original in zip(out['interface_anchors'], source['interface_anchors']):
        canonical = nets.get(original['port_name'])
        if canonical is None:
            source_ports = {p['name']: p['net_name'] for p in source['ports']}
            canonical = nets.get(source_ports.get(original['port_name']))
        _require(canonical is not None, 'layout contains an unbound interface anchor')
        row['port_name'] = canonical
    for row in out['traces'] + out['vias']:
        row.pop('escape_ownership', None)
    for part in out['components'].values():
        part['locked'] = False
    _require(not source.get('antenna_edge_intents'), 'antenna geometry depends on parent boundary')
    return out


def qualify_and_publish(library, request: dict, metadata: dict, round_index: int) -> dict:
    """Qualify the exact winner of an independent local solve and publish it."""
    from kicraft.autoplacer.brain.pins import _round_snapshot_files
    from .extractor import PromoteRequest, extract_leaf

    source_dir = Path(metadata['artifact_paths']['artifact_dir'])
    winner = _round_snapshot_files(source_dir, round_index)
    _require(winner, 'no complete routed local solve snapshot')
    contract = request['contract']
    _require(_tool_version() == contract['tool'], 'KiCad changed after synthesis')
    source = _read(Path(metadata['artifact_paths']['solved_layout_json']))
    _require(source.get('validation', {}).get('leaf_acceptance_result', {}).get('accepted') is True,
             'the local solver did not accept this round')
    _require_layout_graph(source, contract, request['ref_map'], request['net_map'])
    layout = _canonical_layout(source, request)
    receipt_meta = {
        **{k: layout[k] for k in ('sheet_name', 'sheet_file', 'instance_path', 'subcircuit_id', 'parent_instance_path')},
        'schema_version': metadata['schema_version'], 'solver_version': METHOD, 'component_refs': list(layout['components']),
        'internal_nets': sorted(set(contract['nets']) - {p['binding'] for p in contract['ports']}),
        'external_nets': sorted({p['binding'] for p in contract['ports'] if p['binding']}),
        'interface_ports': layout['ports'], 'local_board_outline': metadata['local_board_outline'],
        'artifact_paths': {}, 'notes': [],
    }
    name, version = _entry_name(contract), _entry_version(request['identity'])
    with tempfile.TemporaryDirectory(prefix='kicraft-block-') as tmp:
        root = Path(tmp)
        bom = _template(contract, root)
        board = _stamp_native(root, request, layout, winner['leaf_routed.kicad_pcb'], bom)
        drc = _strict_drc(board, contract['fabrication'], staged_layout=layout)
        snapshots = root / '.experiments/subcircuits/BLOCK'
        snapshots.mkdir(parents=True)
        prefix = 'round_0000_'
        board_out = snapshots / (prefix + 'leaf_routed.kicad_pcb')
        board.replace(board_out)
        meta_out, layout_out = (snapshots / (prefix + n) for n in ('metadata.json', 'solved_layout.json'))
        _write(meta_out, receipt_meta)
        _write(layout_out, layout)
        evidence = {'method': METHOD, 'independent_drc': drc, 'artifact_source': 'independent-local-solve',
                    'artifacts': {n: _sha(p) for n, p in (
                        ('schematic.kicad_sch', root / 'BLOCK.kicad_sch'), ('leaf_routed.kicad_pcb', board_out),
                        ('solved_layout.json', layout_out), ('metadata.json', meta_out))}}
        qualified = {'schema': SCHEMA, 'identity': request['identity'], 'contract': contract, 'evidence': evidence}
        promote = PromoteRequest(
            source_project_dir=root, source_project_stem='qualified-block', source_sheet_name='BLOCK',
            source_sheet_stem='BLOCK', source_leaf_key='BLOCK', source_round=0,
            name=name, version=version, description=contract['electrical_evidence']['claim'],
            kicad_version=contract['tool'], bom_rows=[dict(p, mpn=p['mpn'] or '') for p in contract['parts']],
            qualification=qualified,
        )
        try:
            extract_leaf(promote, library.base_dir, render=False)
            status = 'published'
        except FileExistsError:
            # A concurrent process can win the immutable variant; do not label
            # its installation ours or accept a conflicting/corrupt winner.
            status = 'already_published'
        loaded = library.find(f'{name}@{version}')
        _require(loaded is not None, 'published variant not discoverable or its target is quarantined')
        found = validate_entry(loaded)
        _require(found['identity'] == request['identity'], 'concurrent variant identity differs')
        return {'status': status, 'identity': found['identity'], 'entry': loaded.slug,
                'content_hash': loaded.manifest.content_hash, 'independent_drc': found['evidence']['independent_drc']}


def audit_composed_board(board_path: str, expectations: list[dict]) -> dict:
    """Read delivery PCB, independently check each qualified rigid instance."""
    import pcbnew

    board = pcbnew.LoadBoard(board_path)
    native_fps = _native_footprints(board)
    all_nets = {row['net'] for expected in expectations for row in
                [*expected['layout']['traces'], *expected['layout']['vias']]}
    all_nets.update(net for expected in expectations for net in expected['internal_nets'])
    native = _copper_keys(_native_copper(board, {net: net for net in all_nets}, restricted=False))
    results = []
    for expected in expectations:
        layout = expected['layout']
        dx, dy = expected['page_offset']
        required = _copper_keys(layout, dx=dx, dy=dy)
        missing = sum((required - native).values())
        private = set(expected['internal_nets'])
        added_private = sum(number for key, number in (native - required).items() if key[1] in private)
        # Nonlinear copper and fills never pass a private boundary even if the
        # straight-track representation could not encode their geometry.
        added_private += sum(
            1 for item in board.GetTracks()
            if item.GetNetname() in private and (
                not isinstance(item, pcbnew.PCB_VIA) and item.GetClass() != 'PCB_TRACK'
                or isinstance(item, pcbnew.PCB_VIA) and item.GetViaType() != pcbnew.VIATYPE_THROUGH
            )
        )
        added_private += sum(1 for item in board.Zones() if item.GetNetname() in private)
        changed_footprints = []
        for ref, wanted in layout['components'].items():
            fp = native_fps.get(ref)
            if fp is None:
                changed_footprints.append(ref)
                continue
            p = fp.GetPosition()
            x, y = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            angle = fp.GetOrientationDegrees()
            pads = Counter((str(pad.GetNumber()), pad.GetNetname() or None) for pad in fp.Pads())
            wanted_pads = Counter((str(pad['pad_id']), pad.get('net') or None) for pad in wanted['pads'])
            if (abs(x - wanted['pos']['x'] - dx) >= .001 or abs(y - wanted['pos']['y'] - dy) >= .001
                or abs((angle - wanted['rotation'] + 180) % 360 - 180) >= .001
                or board.GetLayerName(fp.GetLayer()) != wanted['layer']
                or any(pads[key] < n for key, n in wanted_pads.items())
                or {key for key in pads if key[1]} != {key for key in wanted_pads if key[1]}):
                changed_footprints.append(ref)
        results.append({
            'instance_path': expected['instance_path'], 'entry': expected['entry'],
            'identity': expected['identity'], 'content_hash': expected['content_hash'],
            'expected_tracks': len(layout['traces']), 'expected_vias': len(layout['vias']),
            'missing_copper': missing, 'added_private_copper': added_private,
            'changed_footprints': changed_footprints,
            'status': 'invalidated' if missing or added_private or changed_footprints else 'preserved',
        })
    return {'accepted': all(row['status'] == 'preserved' for row in results),
            'pcb_sha256': _sha(Path(board_path)), 'instances': results}


def check_composed_geometry(board_path: Path, expectations: list[dict]) -> dict:
    """Use the same native pcbnew subprocess environment as the composer."""
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    with tempfile.TemporaryDirectory(prefix='kicraft-qualified-audit-') as tmp:
        request = Path(tmp) / 'expected.json'
        output = Path(tmp) / 'result.json'
        _write(request, {'blocks': expectations})
        run_pcbnew_script(
            'import json\nfrom pathlib import Path\n'
            'from kicraft.leaf_library.qualification import audit_composed_board\n'
            f'blocks = json.loads(Path({str(request)!r}).read_text())[\"blocks\"]\n'
            f'result = audit_composed_board({str(board_path)!r}, blocks)\n'
            f'Path({str(output)!r}).write_text(json.dumps(result))\n'
        )
        return _read(output)


def verify_installed_instance(artifact_dir: Path, decision: dict, *, snapshot_id: str | None = None) -> None:
    """Check the project can replay its import without accessing the library."""
    prefix = f'round_{snapshot_id}_' if snapshot_id is not None else ''
    layout = _read(artifact_dir / (prefix + 'solved_layout.json'))
    evidence = _read(artifact_dir / (prefix + 'qualification.json'))
    _require(evidence['contract']['method'] == METHOD, 'installed block needs requalification under the current independent DRC policy')
    _require(layout.get('qualified_reuse') == decision and evidence['identity'] == decision['identity'],
             'installed leaf identity differs from its qualified receipt')
    _require(_digest({k: layout[k] for k in ('components', 'traces', 'vias')}) == decision['installed_geometry_sha256'],
             'installed layout geometry differs from its qualified receipt')
    _require(_sha(artifact_dir / (prefix + 'leaf_routed.kicad_pcb')) == decision['installed_pcb_sha256'],
             'installed native PCB differs from its qualified receipt')
