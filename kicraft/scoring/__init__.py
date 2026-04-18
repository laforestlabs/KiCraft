"""Layout scoring check registry."""
from .trace_check import TraceWidthCheck
from .drc_check import DRCCheck
from .connectivity_check import ConnectivityCheck
from .placement_check import PlacementCheck
from .geometry_check import GeometryCheck
from .compactness_check import CompactnessCheck
from .orientation_check import OrientationCheck
from .visual_check import VisualCheck

ALL_CHECKS = [
    TraceWidthCheck(),
    DRCCheck(),
    ConnectivityCheck(),
    PlacementCheck(),
    GeometryCheck(),
    CompactnessCheck(),
    OrientationCheck(),
    VisualCheck(),
]
