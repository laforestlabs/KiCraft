"""Design-layer contracts for deterministic versioned circuit recipes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kicraft.design.models import (
    BomPart,
    EdgeInterface,
    JsonScalar,
    NetConnection,
    PinEndpoint,
    RecipeOwnershipManifest,
    RecipeSelection,
)

RecipeMaturity = Literal["experimental", "canary", "production"]
RecipePortDirection = Literal["input", "output", "bidirectional", "passive", "power"]


class RecipeComponentGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    reference_prefix: str = Field(pattern=r"^[A-Z]+$")
    quantity: int = Field(default=1, ge=1, le=500)
    value: str
    symbol: str
    footprint: str
    sheet_role: str
    assembly: bool = True
    mpn: str | None = None
    datasheet: str | None = None
    sourcing_note: str | None = None


class RecipePinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    index: int = Field(default=0, ge=0)
    pin: str
    net: str


class RecipeNoConnectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    index: int = Field(default=0, ge=0)
    pin: str


class RecipeEdgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    side: str
    pitch_mm: float = Field(gt=0)


class RecipePort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    direction: RecipePortDirection
    required: bool = True
    allow_ground: bool = False
    #: The port this one is tied to by the recipe when the design wires neither: an enable pin
    #: sits on the input rail (always-on), an inverting input on the output (unity-gain
    #: follower). Names another port of the same recipe; the derived binding comes from here.
    default_tie: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    # An externally biased control input requires an MCU pad without a reset
    # pull-up. This is a DC compatibility requirement, not a ramp guarantee.
    reset_control: Literal["external_low"] | None = None
    # Inclusive reviewed steady input envelope for this whole circuit, not a
    # silicon absolute maximum or a guarantee about startup or output loading.
    supply_voltage_range_v: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _validate_supply_voltage_range(self):
        if self.supply_voltage_range_v is not None:
            low, high = self.supply_voltage_range_v
            if self.direction != "power":
                raise ValueError("supply voltage range is only valid on power ports")
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("supply voltage range must have finite ordered bounds")
        return self


class RecipeAllocatablePin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    index: int = Field(default=0, ge=0)
    pin: str
    gpio: int | None = Field(default=None, ge=0)
    capabilities: tuple[str, ...]
    strapping: bool = False
    reserved: bool = False
    input_only: bool = False
    # Pin-specific reset documentation; unknown is deliberately not a proof.
    reset_bias: Literal["unknown", "no_pull_up", "pull_up"] = "unknown"


class RecipeSourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    revision: str
    reviewed_date: str
    sections: tuple[str, ...] = ()


class RecipePlacementConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    role: str | None = None
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)


class RecipeElectricalAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str


class RecipeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    recipe: str
    family: str | None = None
    exact_part: str | None = None
    default_for_family: bool = False
    maturity: RecipeMaturity = "experimental"
    protected_aliases: tuple[str, ...] = ()
    identity_aliases: tuple[str, ...] = ()
    # Continuous output current this part is rated for, in amperes, from the
    # datasheet in `source_documents`. Set only for recipes that generate a rail
    # (a converter, LDO or charger); None means "this part is not rated here",
    # which the design checks treat as unproven rather than as a number.
    rated_output_current_a: float | None = Field(default=None, gt=0)
    # A qualified series switch transfers load current without converter losses.
    pass_through: bool = False
    # Conservative design allocations, not measured consumption or silicon ratings.
    # Keys name the actual supply ports so shared-rail budgets can be summed.
    supply_current_reservations_a: dict[str, float] = Field(default_factory=dict)
    required_sheet_roles: tuple[str, ...]
    parameter_defaults: dict[str, JsonScalar] = Field(default_factory=dict)
    allowed_parameters: dict[str, tuple[str | int | float | bool, ...]] = Field(
        default_factory=dict
    )
    ports: tuple[RecipePort, ...] = ()
    # Reviewed protocols served by a physical external interface, not merely
    # by an onboard load or an arbitrary part name.
    external_interface_protocols: tuple[str, ...] = ()
    # Optional MCU-only mapping for ports whose documented peripheral function
    # is selected only when the architecture binds it.  The allocator reserves
    # these pads only for bound ports, so a muxed ADC/GPIO function can remain
    # legal when UART/SPI/I2C is absent.
    fixed_port_pins: dict[str, str] | None = None
    internal_nets: tuple[str, ...] = ()
    parts: tuple[RecipeComponentGroup, ...]
    pins: tuple[RecipePinSpec, ...]
    no_connects: tuple[RecipeNoConnectSpec, ...] = ()
    allocatable_pins: tuple[RecipeAllocatablePin, ...] = ()
    edges: tuple[RecipeEdgeSpec, ...] = ()
    placement_constraints: tuple[RecipePlacementConstraint, ...] = ()
    electrical_assertions: tuple[RecipeElectricalAssertion, ...] = ()
    source_documents: tuple[RecipeSourceDocument, ...] = ()
    @model_validator(mode="after")
    def _default_ties_name_another_port(self):
        names = {port.name for port in self.ports}
        if self.pass_through:
            power_ports = {port.name for port in self.ports if port.direction == "power"}
            if not {"input", "output"} <= power_ports:
                raise ValueError(
                    f"recipe {self.recipe} pass-through requires input and output power ports"
                )
        for port, current in self.supply_current_reservations_a.items():
            if port not in names or not math.isfinite(current) or current <= 0:
                raise ValueError(
                    f"recipe {self.recipe} supply reservation must name a declared port "
                    "and a positive finite current"
                )
        for port in self.ports:
            if port.default_tie is None:
                continue
            if port.default_tie not in names or port.default_tie == port.name:
                raise ValueError(
                    f"recipe {self.recipe} port {port.name!r} default_tie "
                    f"{port.default_tie!r} must name another port of the same recipe"
                )
        if self.fixed_port_pins is not None:
            unknown_ports = set(self.fixed_port_pins) - names
            allocatable = {pin.pin for pin in self.allocatable_pins}
            unknown_pins = set(self.fixed_port_pins.values()) - allocatable
            if unknown_ports or unknown_pins:
                raise ValueError(
                    f"recipe {self.recipe} fixed port mapping must name declared ports and "
                    f"allocatable pins; unknown_ports={sorted(unknown_ports)}, "
                    f"unknown_pins={sorted(unknown_pins)}"
                )
        return self


class ResolvedRecipeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: RecipeSelection
    parameters: dict[str, JsonScalar]


class RecipeExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: RecipeSelection
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)
    parts: list[BomPart]
    connections: list[NetConnection]
    no_connect_pins: list[PinEndpoint] = Field(default_factory=list)
    edge_interfaces: list[EdgeInterface] = Field(default_factory=list)
    ownership: RecipeOwnershipManifest


@dataclass(frozen=True)
class RegisteredRecipe:
    definition: RecipeDefinition
    expand: Callable[[ResolvedRecipeSelection], RecipeExpansion] | None = None
