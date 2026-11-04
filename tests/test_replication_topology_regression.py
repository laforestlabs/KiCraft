import pytest

from kicraft.cli._replicate_leaves import build_replication_maps


def _component(net):
    return {"value": "part", "pads": [{"pad_id": "1", "net": net}]}


@pytest.mark.parametrize("representative,sibling", [(None, "VCC"), ("VCC", None)])
def test_connected_and_unconnected_pads_are_not_replicated(representative, sibling):
    maps = build_replication_maps(
        ["U1"], ["U2"], {"U1": _component(representative)}, {"U2": _component(sibling)}
    )
    assert maps is None
