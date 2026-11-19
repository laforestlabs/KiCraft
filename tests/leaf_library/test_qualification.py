import shutil
from types import SimpleNamespace

import pytest

from kicraft.cli._replicate_leaves import remap_solved_layout
from kicraft.design.models import Architecture, BOM, CircuitRequirement, RecipeSelection, Sheet
from kicraft.design.recipes import expand_recipe
from kicraft.leaf_library import qualification as q


def _usb_request(tmp_path, *, instance, rail, voltage=5.0, start=1, stackup='two-layer', physical=None):
    sheet = Sheet(name='POWER INPUT', stem='POWER_INPUT', function='private project text')
    selection = RecipeSelection(
        recipe='usb-c-5v-sink@1', instance=instance, sheets={'power': sheet.name},
        port_bindings={'vbus': rail, 'gnd': 'GND'},
    )
    exp = expand_recipe(selection, next_reference={'J': start, 'R': start})
    arch = Architecture(sheets=[sheet], inter_sheet_nets=[], rail_voltages={rail: voltage},
                        power_nets=[rail, 'GND'], recipe_selections=[selection])
    bom = BOM(parts=exp.parts, connections=exp.connections, no_connect_pins=exp.no_connect_pins,
              recipe_ownership=[exp.ownership])
    state = SimpleNamespace(architecture=arch, bom=bom)
    sheet_instance = SimpleNamespace(sheet=sheet, parts=bom.parts, inter_sheet_endpoints=[])
    return q._request(sheet_instance, state, tmp_path, {'stackup': stackup}, {rail, 'GND'}, physical or {}, '9.0.0')


def test_recipe_variant_identity_is_ref_and_net_invariant_not_fabrication_invariant(tmp_path, monkeypatch):
    # The actual reviewed expansion/ownership checks run; stock library installation
    # is deliberately not a prerequisite to the pure identity assertion.
    monkeypatch.setattr(q, '_dependencies', lambda parts, directory: {'stock_asset': 'digest'})
    a = _usb_request(tmp_path, instance='PRIVATE_MOCK_1', rail='PRIVATE_A')
    b = _usb_request(tmp_path, instance='PRIVATE_MOCK_2', rail='PRIVATE_B', start=7)
    assert a['contract'] == b['contract']
    assert a['identity'] == b['identity']
    assert a['ref_map'] != b['ref_map'] and a['net_map'] != b['net_map']
    assert 'PRIVATE' not in str(a['contract'])
    assert 'private project text' not in str(a['contract'])
    other_stackup = _usb_request(tmp_path, instance='other', rail='VBUS', stackup='four-layer')
    assert other_stackup['identity'] != a['identity']
    with pytest.raises(q.QualificationError, match='explicit 5 V'):
        _usb_request(tmp_path, instance='other', rail='VBUS', voltage=9.0)


def test_canonical_thermal_owner_ref_prevents_portable_cutout(tmp_path, monkeypatch):
    monkeypatch.setattr(q, '_dependencies', lambda parts, directory: {'stock_asset': 'digest'})
    request = _usb_request(tmp_path, instance='usb', rail='VBUS')
    owner_ref = next(iter(request['ref_map']))
    with pytest.raises(q.QualificationError, match='board-dependent thermal geometry'):
        _usb_request(tmp_path, instance='usb', rail='VBUS',
                     physical={'thermal_requirements': [{'owner_ref': owner_ref}]})
    assert _usb_request(tmp_path, instance='usb', rail='VBUS', physical={
        'thermal_requirements': [{'owner_ref': 'U100'}],
    })['identity'] == request['identity']


def test_repeat_instances_cannot_swap_pad_roles_or_share_private_nets(tmp_path, monkeypatch):
    monkeypatch.setattr(q, '_dependencies', lambda parts, directory: {'stock_asset': 'digest'})
    a = _usb_request(tmp_path, instance='a', rail='VBUS_A')
    b = _usb_request(tmp_path, instance='b', rail='VBUS_B', start=8)
    contract = a['contract']
    layout = {'components': {p['ref']: {'pads': [], 'ref': p['ref']} for p in contract['parts']},
              'traces': [], 'vias': []}
    for net, pins in contract['nets'].items():
        for ref, pin in pins:
            layout['components'][ref]['pads'].append({'ref': ref, 'pad_id': pin, 'net': net})
    local = next(net for net in contract['nets'] if net.startswith('LOCAL_'))
    layout['traces'].append({'net': local, 'start': {'x': 0., 'y': 0.}, 'end': {'x': 1., 'y': 0.},
                             'width_mm': .2, 'layer': 'F.Cu'})

    def instantiate(request):
        refs = {v: k for k, v in request['ref_map'].items()}
        nets = {v: k for k, v in request['net_map'].items()}
        instance = remap_solved_layout(layout, refs, nets, {'instance_path': '/different'})
        q._require_layout_graph(instance, contract, request['ref_map'], request['net_map'])
        return instance

    one, two = instantiate(a), instantiate(b)
    assert set(one['components']).isdisjoint(two['components'])
    assert one['traces'][0]['net'] != two['traces'][0]['net']
    assert one['traces'][0]['start'] == two['traces'][0]['start']
    two['traces'][0]['net'] = one['traces'][0]['net']
    with pytest.raises(q.QualificationError, match='boundary-crossing'):
        q._require_layout_graph(two, contract, b['ref_map'], b['net_map'])
    two = instantiate(b)
    ref = next(iter(two['components']))
    two['components'][ref]['pads'][0]['net'] = 'GND'
    with pytest.raises(q.QualificationError, match='pad connectivity'):
        q._require_layout_graph(two, contract, b['ref_map'], b['net_map'])


def test_stale_electrical_root_cannot_supply_qualification_request(tmp_path):
    board = tmp_path / 'board.kicad_pcb'
    rules = tmp_path / 'board.kicad_pro'
    config = tmp_path / 'board_autoplacer.json'
    root = tmp_path / 'board.kicad_sch'
    for path in (board, rules, config, root):
        path.write_text(path.name)
    data = dict(schema=q.SCHEMA, pcb_sha256=q._sha(board), project_sha256=q._sha(rules),
                root_schematic_sha256=q._sha(root), config_sha256=q._sha(config),
                synthesis_checks=['9.12 ERC', '9.13 netlist faithfulness'], requests={})
    path = tmp_path / q.REQUEST_FILE
    path.parent.mkdir()
    q._write(path, data)
    assert q.read_requests(tmp_path, board, root) == ({}, '')
    root.write_text('altered cross-instance port binding')
    assert q.read_requests(tmp_path, board, root) == ({}, 'root sheet bindings changed since qualification input was evaluated')


@pytest.mark.parametrize('pcb_stem', ['board', 'layout'])
def test_custom_native_rules_block_catalogue_lookup_and_publication(tmp_path, pcb_stem):
    pcb = tmp_path / f'{pcb_stem}.kicad_pcb'
    pro = tmp_path / 'board.kicad_pro'
    ap = tmp_path / 'board_autoplacer.json'
    root = tmp_path / 'board.kicad_sch'
    for path in (pcb, pro, ap, root):
        path.write_text(path.name)
    contract = {'owner': 'usb-c-5v-sink@1', 'method': q.METHOD}
    checked = ['/instance', '/other']
    request = {'contract': contract, 'identity': q._digest(contract),
               'schematic': root.name, 'schematic_sha256': q._sha(root)}
    checks = ['9.12 ERC', '9.13 netlist faithfulness']
    data = dict(schema=q.SCHEMA, pcb_sha256=q._sha(pcb), project_sha256=q._sha(pro),
                root_schematic_sha256=q._sha(root), config_sha256=q._sha(ap),
                synthesis_checks=checks, requests={path: request for path in checked})
    sidecar = tmp_path / q.REQUEST_FILE
    sidecar.parent.mkdir()
    q._write(sidecar, data)
    assert q.read_requests(tmp_path, pcb, root) == (data['requests'], '')

    # No ordinary input hash changes; a new rule sidecar still invalidates
    # lookup before an immutable package can be selected.
    pcb.with_suffix('.kicad_dru').write_text('(version 1)\n(rule "private" (constraint clearance (min 1mm)))\n')
    requests, reason = q.read_requests(tmp_path, pcb, root)
    assert not requests and 'custom KiCad .kicad_dru' in reason
    result = q.synthesis_requests(
        None, tmp_path, [SimpleNamespace(instance_uuid=path[1:]) for path in checked],
        pcb, pro, ap, [SimpleNamespace(name=name, ok=True, message='verified') for name in checks],
    )
    assert result['requests'] == {path: {'reason': reason} for path in checked}


def test_native_final_geometry_audit_rejects_private_track_changes(tmp_path):
    pcbnew = pytest.importorskip('pcbnew')
    board = pcbnew.BOARD()
    point = lambda x, y: pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
    traces = []
    for y, name in ((10, 'INSTANCE_A'), (20, 'INSTANCE_B')):
        net = pcbnew.NETINFO_ITEM(board, name)
        board.Add(net)
        trace = pcbnew.PCB_TRACK(board)
        trace.SetStart(point(10, y))
        trace.SetEnd(point(12, y))
        trace.SetWidth(pcbnew.FromMM(.2))
        trace.SetLayer(pcbnew.F_Cu)
        board.Add(trace)
        trace.SetNetCode(net.GetNetCode())
        traces.append(trace)
    board_path = tmp_path / 'composed.kicad_pcb'
    expected = [
        {'instance_path': '/' + name, 'entry': 'block@1.0.0', 'identity': 'one', 'content_hash': 'verified',
         'internal_nets': [name], 'page_offset': [10, 10],
         'layout': {'components': {}, 'vias': [], 'traces': [
             {'net': name, 'layer': 'F.Cu', 'width_mm': .2, 'start': {'x': 0, 'y': y}, 'end': {'x': 2, 'y': y}},
         ]}}
        for y, name in ((0, 'INSTANCE_A'), (10, 'INSTANCE_B'))
    ]
    assert pcbnew.SaveBoard(str(board_path), board)
    assert q.audit_composed_board(str(board_path), expected)['accepted']
    traces[1].SetEnd(point(13, 20))  # preserve counts, mutate only one private segment
    assert pcbnew.SaveBoard(str(board_path), board)
    result = q.audit_composed_board(str(board_path), expected)
    assert not result['accepted']
    assert [item['status'] for item in result['instances']] == ['preserved', 'invalidated']
    assert result['instances'][1]['missing_copper'] == result['instances'][1]['added_private_copper'] == 1


def test_quantitative_requirements_cannot_inherit_unrated_layout():
    contacts = [{'voltage_v': 5.0}, {'voltage_v': 0.0}]
    requirement = CircuitRequirement(
        id='header', sheet='OUTPUT', role='connector', family='pin-header',
        ports={'pin1': 'VBUS', 'pin2': 'GND'},
        obligations=[{'kind': 'quantitative', 'original_obligation_id': 'count',
                      'quantity': 'test header pins', 'value': 2, 'relation': 'equal', 'unit': 'pins'}],
    )
    q._require_supported_obligations('pin-header@1', [requirement], contacts)
    four_contacts = CircuitRequirement.model_validate({**requirement.model_dump(), 'obligations': [
        {**requirement.obligations[0].model_dump(), 'value': 4},
    ]})
    with pytest.raises(q.QualificationError, match='contact count'):
        q._require_supported_obligations('pin-header@1', [four_contacts], contacts)
    current = {'kind': 'quantitative', 'original_obligation_id': 'current',
               'quantity': 'output current', 'value': .5, 'relation': 'minimum', 'unit': 'A'}
    required_current = CircuitRequirement.model_validate({**requirement.model_dump(), 'obligations': [current]})
    for owner in ('pin-header@1', 'usb-c-5v-sink@1'):
        with pytest.raises(q.QualificationError, match='lacks evidence.*output current'):
            q._require_supported_obligations(owner, [required_current], contacts)
    required_voltage = CircuitRequirement.model_validate({**requirement.model_dump(), 'obligations': [
        {'kind': 'quantitative', 'original_obligation_id': 'voltage', 'quantity': 'input voltage',
         'minimum': 5.0, 'maximum': 9.0, 'relation': 'range', 'unit': 'V'},
    ]})
    with pytest.raises(q.QualificationError, match='voltage envelope exceeds'):
        q._require_supported_obligations('usb-c-5v-sink@1', [required_voltage], contacts)


def test_independent_drc_rejects_dangling_via_until_proven_private_escape_cleanup(tmp_path):
    pcbnew = pytest.importorskip('pcbnew')
    if not shutil.which('kicad-cli'):
        pytest.skip('native independent DRC requires kicad-cli')
    from kicraft.autoplacer.brain.copper_accounting import attach_signal_escape_ownership
    from kicraft.autoplacer.brain.leaf_routing import _finish_private_signal_escapes
    from kicraft.autoplacer.routing_board import import_routed_copper
    from kicraft.design.synthesis.kicad_pro import DEFAULT_NETCLASS, DEFAULT_RULES, POWER_NETCLASS

    board = pcbnew.BOARD()
    net = pcbnew.NETINFO_ITEM(board, 'PRIVATE')
    board.Add(net)

    def point(x, y):
        return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))

    for ref, x, y in [('U1', 5, 5), ('R1', 7, 8)]:
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(point(x, y))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber('1')
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetSize(point(1, 1))
        layers = pcbnew.LSET()
        layers.AddLayer(pcbnew.F_Cu)
        pad.SetLayerSet(layers)
        pad.SetNet(net)
        fp.Add(pad)
        pad.SetPosition(point(x, y))

    def segment(start, end):
        trace = pcbnew.PCB_TRACK(board)
        trace.SetStart(point(*start))
        trace.SetEnd(point(*end))
        trace.SetWidth(pcbnew.FromMM(.2))
        trace.SetLayer(pcbnew.F_Cu)
        trace.SetNet(net)
        board.Add(trace)
        return trace

    main = segment((5, 5), (7, 5))
    tail = segment((7, 5), (8, 5))
    load = segment((7, 5), (7, 8))
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(point(8, 5))
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetWidth(pcbnew.FromMM(.6))
    via.SetDrill(pcbnew.FromMM(.3))
    via.SetNet(net)
    board.Add(via)
    outline = pcbnew.PCB_SHAPE(board)
    outline.SetShape(pcbnew.SHAPE_T_RECT)
    outline.SetStart(point(1, 1))
    outline.SetEnd(point(14, 13))
    outline.SetLayer(pcbnew.Edge_Cuts)
    outline.SetWidth(pcbnew.FromMM(.05))
    board.Add(outline)
    pcb = tmp_path / 'test.kicad_pcb'
    assert pcbnew.SaveBoard(str(pcb), board)
    fabrication = {'design_settings': {'rules': DEFAULT_RULES, 'meta': {'version': 2}},
                   'classes': [DEFAULT_NETCLASS, POWER_NETCLASS], 'net_settings_version': 3, 'power_nets': []}
    with pytest.raises(q.QualificationError, match='via_dangling'):
        q._strict_drc(pcb, fabrication)
    assert q._read(pcb.with_suffix('.qualification-drc.json'))['unconnected_items'] == []

    staged = tmp_path / 'staged.kicad_pcb'
    shutil.copyfile(pcb, staged)
    staged_layout = q._native_copper(board, {'PRIVATE': 'PRIVATE'})
    staged_layout['components'] = {
        ref: {'pos': {'x': x, 'y': y}, 'rotation': angle, 'layer': layer}
        for ref, (x, y, angle, layer) in q._board_poses(board).items()
    }
    before_copy = q._sha(pcb)
    evidence = q._strict_drc(staged, fabrication, staged_layout=staged_layout)
    assert evidence['dangling_violations'] == 0
    assert {row['kind'] for row in evidence['normalization']} == {'via', 'segment'}
    assert len(staged_layout['traces']) == 2 and not staged_layout['vias']
    assert q._sha(pcb) == before_copy

    copper = import_routed_copper(str(pcb))
    branch = {
        'owner': 'compiler.signal_escape', 'ref': 'U1', 'pad': '1',
        'items': [
            {'kind': 'track', 'uuid': main.m_Uuid.AsString(), 'net': 'PRIVATE', 'start_mm': [5, 5],
             'end_mm': [7, 5], 'layer': 'F.Cu', 'width_mm': .2},
            {'kind': 'track', 'uuid': tail.m_Uuid.AsString(), 'net': 'PRIVATE', 'start_mm': [7, 5],
             'end_mm': [8, 5], 'layer': 'F.Cu', 'width_mm': .2},
            {'kind': 'via', 'uuid': via.m_Uuid.AsString(), 'net': 'PRIVATE', 'position_mm': [8, 5],
             'size_mm': .6, 'drill_mm': .3},
        ],
    }
    assert attach_signal_escape_ownership(copper, {'owned_signal_escapes': [branch]})['bound'] == 3
    before = q._sha(pcb)
    assert _finish_private_signal_escapes(pcb, pcb, copper, set()) == {}
    assert q._sha(pcb) == before
    proof = _finish_private_signal_escapes(pcb, pcb, copper, {'PRIVATE'})
    assert proof['accepted'] and len(proof['removed']) == 2
    assert all(row['proof']['partition_before'] == row['proof']['partition_after'] for row in proof['removed'])
    kept = pcbnew.LoadBoard(str(pcb))
    assert {item.m_Uuid.AsString() for item in kept.GetTracks()} == {main.m_Uuid.AsString(), load.m_Uuid.AsString()}
    assert q._strict_drc(pcb, fabrication)['dangling_violations'] == 0
