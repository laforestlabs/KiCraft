"""Design-layer contracts for deterministic versioned circuit recipes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kicraft.design.models import (
    BomPart,
    EdgeInterface,
    NetConnection,
    PinEndpoint,
    RecipeSelection,
)


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


class RecipeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    recipe: str
    required_sheet_roles: tuple[str, ...]
    parameter_defaults: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    allowed_parameters: dict[str, tuple[str | int | float | bool, ...]] = Field(
        default_factory=dict
    )
    parts: tuple[RecipeComponentGroup, ...]
    pins: tuple[RecipePinSpec, ...]
    no_connects: tuple[RecipeNoConnectSpec, ...] = ()
    edges: tuple[RecipeEdgeSpec, ...] = ()


class RecipeExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: RecipeSelection
    parts: list[BomPart]
    connections: list[NetConnection]
    no_connect_pins: list[PinEndpoint] = Field(default_factory=list)
    edge_interfaces: list[EdgeInterface] = Field(default_factory=list)
