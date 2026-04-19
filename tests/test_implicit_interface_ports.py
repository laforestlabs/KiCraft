"""Tests for _infer_implicit_interface_ports in subcircuit_extractor.

All tests use synthetic data only; no pcbnew dependency.
"""

from __future__ import annotations


from kicraft.autoplacer.brain.subcircuit_extractor import (
    NetPartition,
    _infer_implicit_interface_ports,
)
from kicraft.autoplacer.brain.types import (
    InterfaceDirection,
    InterfacePort,
    InterfaceRole,
    InterfaceSide,
    Net,
    SubcircuitAccessPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_port(
    name: str,
    net_name: str,
    role: InterfaceRole = InterfaceRole.BIDIR,
) -> InterfacePort:
    """Create a minimal declared interface port."""
    return InterfacePort(
        name=name,
        net_name=net_name,
        role=role,
        direction=InterfaceDirection.UNKNOWN,
        preferred_side=InterfaceSide.ANY,
        access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
        cardinality=1,
        bus_index=None,
        required=True,
        description="declared port",
        raw_direction="",
        source_uuid=None,
        source_kind="schematic_pin",
    )


def _make_net(name: str, is_power: bool = False) -> Net:
    """Create a minimal Net."""
    return Net(
        name=name,
        pad_refs=[("R1", "1"), ("R2", "1")],
        is_power=is_power,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoExternalNets:
    """When there are no external nets, no implicit ports are added."""

    def test_empty_external_returns_original(self):
        ports = [_make_port("VBUS", "VBUS")]
        partition = NetPartition(
            internal={"SIG1": _make_net("SIG1")},
            external={},
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        assert len(result) == 1
        assert result[0].name == "VBUS"

    def test_empty_everything_returns_empty(self):
        result = _infer_implicit_interface_ports([], NetPartition())
        assert result == []


class TestImplicitPowerNets:
    """Power nets without declared ports get POWER_IN role."""

    def test_gnd_inferred_as_power_in(self):
        ports = [_make_port("VBUS", "VBUS")]
        partition = NetPartition(
            internal={},
            external={"GND": _make_net("GND", is_power=True)},
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        assert len(result) == 2
        # Original declared port
        assert result[0].name == "VBUS"
        # Implicit GND port
        gnd_port = result[1]
        assert gnd_port.name == "GND"
        assert gnd_port.net_name == "GND"
        assert gnd_port.role == InterfaceRole.POWER_IN
        assert gnd_port.required is False
        assert gnd_port.source_kind == "implicit_external_net"

    def test_multiple_power_nets(self):
        ports = []
        partition = NetPartition(
            internal={},
            external={
                "GND": _make_net("GND", is_power=True),
                "+3V3": _make_net("+3V3", is_power=True),
                "VBAT": _make_net("VBAT", is_power=True),
            },
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        assert len(result) == 3
        names = sorted(p.name for p in result)
        assert names == ["+3V3", "GND", "VBAT"]
        for p in result:
            assert p.role == InterfaceRole.POWER_IN


class TestImplicitSignalNets:
    """Non-power external nets without declared ports get BIDIR role."""

    def test_signal_net_inferred_as_bidir(self):
        ports = []
        partition = NetPartition(
            internal={},
            external={"SDA": _make_net("SDA", is_power=False)},
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        assert len(result) == 1
        assert result[0].name == "SDA"
        assert result[0].role == InterfaceRole.BIDIR
        assert result[0].required is False


class TestAlreadyDeclaredNetsNotDuplicated:
    """Nets that already have declared ports are not duplicated."""

    def test_declared_net_not_added_again(self):
        ports = [_make_port("GND", "GND", role=InterfaceRole.POWER_IN)]
        partition = NetPartition(
            internal={},
            external={"GND": _make_net("GND", is_power=True)},
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        # Should still be just 1 port (the declared one)
        assert len(result) == 1
        assert result[0].name == "GND"
        assert result[0].source_kind == "schematic_pin"  # original, not implicit

    def test_mix_declared_and_undeclared(self):
        ports = [_make_port("VBUS", "VBUS")]
        partition = NetPartition(
            internal={},
            external={
                "VBUS": _make_net("VBUS", is_power=True),
                "GND": _make_net("GND", is_power=True),
                "SCL": _make_net("SCL", is_power=False),
            },
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        # VBUS was declared, so only GND and SCL are new
        assert len(result) == 3
        names = [p.name for p in result]
        assert "VBUS" in names  # original
        assert "GND" in names   # implicit power
        assert "SCL" in names   # implicit signal


class TestNetNameStripping:
    """Net names with leading '/' have it stripped for port name."""

    def test_leading_slash_stripped(self):
        ports = []
        partition = NetPartition(
            internal={},
            external={"/GND": _make_net("/GND", is_power=True)},
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        assert len(result) == 1
        assert result[0].name == "GND"  # stripped
        assert result[0].net_name == "/GND"  # preserved


class TestImplicitPortAttributes:
    """Verify all attributes of implicit ports are correct."""

    def test_implicit_port_attributes(self):
        ports = []
        partition = NetPartition(
            internal={},
            external={"NET1": _make_net("NET1", is_power=False)},
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        p = result[0]
        assert p.direction == InterfaceDirection.UNKNOWN
        assert p.preferred_side == InterfaceSide.ANY
        assert p.access_policy == SubcircuitAccessPolicy.INTERFACE_ONLY
        assert p.cardinality == 1
        assert p.bus_index is None
        assert p.required is False
        assert p.raw_direction == ""
        assert p.source_uuid is None
        assert p.source_kind == "implicit_external_net"
        assert "implicit" in p.description.lower()


class TestDeterministicOrdering:
    """Implicit ports are added in sorted net name order."""

    def test_sorted_by_net_name(self):
        ports = []
        partition = NetPartition(
            internal={},
            external={
                "ZZZ": _make_net("ZZZ"),
                "AAA": _make_net("AAA"),
                "MMM": _make_net("MMM"),
            },
            ignored={},
        )
        result = _infer_implicit_interface_ports(ports, partition)
        names = [p.net_name for p in result]
        assert names == ["AAA", "MMM", "ZZZ"]
