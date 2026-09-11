"""Design-layer contracts for deterministic versioned circuit recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    required_sheet_roles: tuple[str, ...]
    parameter_defaults: dict[str, JsonScalar] = Field(default_factory=dict)
    allowed_parameters: dict[str, tuple[str | int | float | bool, ...]] = Field(
        default_factory=dict
    )
    ports: tuple[RecipePort, ...] = ()
    internal_nets: tuple[str, ...] = ()
    parts: tuple[RecipeComponentGroup, ...]
    pins: tuple[RecipePinSpec, ...]
    no_connects: tuple[RecipeNoConnectSpec, ...] = ()
    allocatable_pins: tuple[RecipeAllocatablePin, ...] = ()
    edges: tuple[RecipeEdgeSpec, ...] = ()
    placement_constraints: tuple[RecipePlacementConstraint, ...] = ()
    electrical_assertions: tuple[RecipeElectricalAssertion, ...] = ()
    source_documents: tuple[RecipeSourceDocument, ...] = ()


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
