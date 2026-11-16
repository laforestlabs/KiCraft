"""The standalone player uses the same replay and transform code in Node."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
from kicraft.autoplacer.brain.subcircuit_instances import _transform_point
from kicraft.autoplacer.brain.types import Point


_NODE = shutil.which("node")
_JS = Path(__file__).resolve().parents[1] / "kicraft/server/static/layout-history.js"
_HARNESS = r"""
global.window = {};
const fs = require('fs');
eval(fs.readFileSync(process.argv[1], 'utf8'));
const h = window.kicraftLayoutHistory;
const events = JSON.parse(fs.readFileSync(0, 'utf8'));
const state = h.makeState(), undos = [];
const snapshot = s => JSON.stringify({objects:Array.from(s.objects).sort(([a],[b]) => a.localeCompare(b)), templates:Array.from(s.templates).sort(([a],[b]) => a.localeCompare(b)),seq:s.seq,sourceScene:s.sourceScene,sourceFilter:s.sourceFilter,geometrySource:s.geometrySource});
const snapshots = [snapshot(state)];
for (const e of events) { undos.push(h.apply(state,e)); snapshots.push(snapshot(state)); }
const final = snapshot(state);
for (let i=undos.length-1;i>=0;i--) {
  h.reverse(state,undos[i]);
  if(snapshot(state)!==snapshots[i]) throw Error('reverse mismatch '+i);
}
for (let i=0;i<events.length;i++) {
  h.apply(state,events[i]);
  if(snapshot(state)!==snapshots[i+1]) throw Error('seek mismatch '+i);
}
process.stdout.write(JSON.stringify({final, child:h.transformChild([4,7],[2,3],{x:10,y:20,rotation:90,side:'back'}), back:h.transformPoint([2,1],{x:10,y:20,rotation:90,side:'back'}), net:h.mapNet('source',[{source:'sibling'},{sibling:'parent'}]), arc:h.arcLength([1,0],[0,1],[-1,0]), order:h.overviewOrder([{header:{scope_id:'selected',started_ns:2}},{header:{scope_id:'later',started_ns:3}},{header:{scope_id:'early',started_ns:1}}],'selected').map(s=>s.header.scope_id)}));
"""


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_forward_reverse_seek_and_child_transform():
    label = '<img src=x onerror="alert(1)">'
    events = [
        {"seq": 1, "kind": "define", "template_id": "t", "geometry": {"pads": []}},
        {"seq": 2, "kind": "scene", "geometry_source": "placement_model", "source_scene": {"scope_id": "board", "seq": 7},
         "source_filter": ["footprint", "outline", "zone", "keepout"], "objects": [
            {"id": "fp", "type": "footprint", "ref": label, "pose": {"x": 1, "y": 2, "rotation": 0}},
            {"id": "a", "type": "via", "position": [1, 2], "layers": ["F.Cu", "In1.Cu", "B.Cu"], "net": label},
            {"id": "b", "type": "via", "position": [1, 2], "layers": ["F.Cu", "B.Cu"]},
        ]},
        {"seq": 3, "kind": "delta", "geometry_source": "native_board", "remove": ["a"], "put": [{"id": "fp", "type": "footprint", "pose": {"x": 4, "y": 5, "rotation": 0}}]},
        {"seq": 4, "kind": "decision", "action": "reject", "reason": label},
        {"seq": 5, "kind": "scene", "geometry_source": "router", "objects": []},
    ]
    result = subprocess.run([_NODE, "-e", _HARNESS, str(_JS)], input=json.dumps(events), text=True, capture_output=True, check=True)
    out = json.loads(result.stdout)
    expected = _transform_point(Point(4 - 2, 7 - 3), Point(10, 20), 90)
    assert out["child"] == pytest.approx([expected.x, expected.y])
    assert out["net"] == "parent"
    assert out["arc"] == pytest.approx(3.141592653589793)
    assert out["order"] == ["early", "later", "selected"]
    assert out["back"] == pytest.approx([9, 18])
    assert json.loads(out["final"])["objects"] == []
    assert json.loads(out["final"])["geometrySource"] == "router"


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_invalid_records_stop_without_mutation():
    script = r"""
global.window={};
eval(require('fs').readFileSync(process.argv[1],'utf8'));
const h=window.kicraftLayoutHistory, s=h.makeState();
h.apply(s,{seq:1,kind:'scene',objects:[{id:'a',type:'via'}]});
for (const e of [
  {seq:3,kind:'decision'},
  {seq:2,kind:'bogus'},
  {seq:2,kind:'delta',remove:['a','missing'],put:[]},
  {seq:2,kind:'delta',remove:[],put:[{id:'x',type:'unknown'}]},
]) {
  let rejected=false; try { h.apply(s,e); } catch (_) { rejected=true; }
  if(!rejected || s.seq!==1 || !s.objects.has('a')) throw Error('invalid event mutated state');
}
h.apply(s,{seq:2,kind:'end'});
let rejected=false;try{h.apply(s,{seq:3,kind:'decision'});}catch(_){rejected=true;}
if(!rejected)throw Error('accepted event after end');
"""
    subprocess.run([_NODE, "-e", script, str(_JS)], check=True)


def test_player_uses_inert_text_and_offline_assets():
    source = _JS.read_text()
    assert ".textContent" in source
    assert ".innerHTML" not in source
    assert "fetch(" not in source
    assert "DecompressionStream" in source
