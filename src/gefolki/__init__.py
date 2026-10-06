"""GeFolki: dense optical-flow coregistration of heterogeneous remote sensing images."""

from importlib.metadata import PackageNotFoundError, version

from .backend import Backend, backend_info, get_backend, gpu_available
from .flow import FlowParams, efolki, estimate_flow, folki, gefolki
from .locate import LocateResult, locate, locate_raster
from .pipeline import PRESETS, RegistrationResult, apply_flow, estimate_raster_flow, register
from .warp import warp

try:
    __version__ = version("gefolki")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Backend",
    "FlowParams",
    "LocateResult",
    "PRESETS",
    "RegistrationResult",
    "__version__",
    "apply_flow",
    "backend_info",
    "efolki",
    "estimate_flow",
    "estimate_raster_flow",
    "folki",
    "gefolki",
    "get_backend",
    "gpu_available",
    "locate",
    "locate_raster",
    "register",
    "warp",
]
