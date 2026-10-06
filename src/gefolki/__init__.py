"""GeFolki: dense optical-flow coregistration of heterogeneous remote sensing images."""

from importlib.metadata import PackageNotFoundError, version

from .backend import Backend, get_backend, gpu_available
from .flow import FlowParams, efolki, estimate_flow, folki, gefolki
from .warp import warp

try:
    __version__ = version("gefolki")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Backend",
    "FlowParams",
    "__version__",
    "efolki",
    "estimate_flow",
    "folki",
    "gefolki",
    "get_backend",
    "gpu_available",
    "warp",
]
