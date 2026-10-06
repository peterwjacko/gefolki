"""Array backend selection: NumPy/SciPy on CPU or CuPy/cupyx on a CUDA GPU."""

from __future__ import annotations

import functools
import importlib.util
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np
import scipy.ndimage

Device = Literal["auto", "cpu", "gpu"]

_gpu_error: str | None = None


@dataclass(frozen=True)
class Backend:
    """Array module (``xp``), ndimage module (``ndi``) and CPU thread count."""

    name: Literal["cpu", "gpu"]
    xp: ModuleType
    ndi: ModuleType
    threads: int = 1

    @property
    def is_gpu(self) -> bool:
        return self.name == "gpu"

    def asarray(self, a: Any, dtype: Any = np.float32) -> Any:
        """Convert ``a`` to a backend array (copies host data to the device on GPU)."""
        return self.xp.asarray(a, dtype=dtype)

    to_device = asarray

    def to_host(self, a: Any) -> np.ndarray:
        """Return ``a`` as a NumPy array."""
        return a.get() if self.is_gpu and not isinstance(a, np.ndarray) else np.asarray(a)


@functools.cache
def gpu_available() -> bool:
    """True if CuPy imports, sees a device and can compile and run a kernel (cached)."""
    global _gpu_error
    try:
        import cupy as cp
    except Exception as e:  # ImportError, or broken CUDA libs at import time
        _gpu_error = f"cannot import cupy ({e}); install the 'gpu' extra"
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            _gpu_error = "no CUDA device found"
            return False
    except Exception as e:
        _gpu_error = f"CUDA device query failed ({e})"
        return False
    try:
        # Bare cupy can pass the device query yet fail at first NVRTC compile (e.g. CUDA
        # toolkit too new for the GPU), so compile and run a tiny kernel.
        k = cp.ElementwiseKernel("float32 x", "float32 y", "y = x * 2.0f + 1.0f", "gefolki_probe")
        out = k(cp.arange(4, dtype=cp.float32))
        if not np.allclose(out.get(), [1, 3, 5, 7]):
            _gpu_error = "GPU probe kernel returned wrong values"
            return False
    except Exception as e:
        _gpu_error = f"GPU kernel compile/run failed ({e})"
        return False
    return True


def get_backend(device: Device | Backend = "auto", threads: int | None = None) -> Backend:
    """Return a Backend.

    ``device``: "auto" (GPU if usable, else CPU), "cpu" or "gpu" (raises RuntimeError if no
    usable GPU). A Backend instance is returned unchanged. ``threads``: CPU worker threads
    (default ``os.cpu_count()``); ignored on GPU.
    """
    if isinstance(device, Backend):
        return device
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'gpu', got {device!r}")
    if device == "gpu" and not gpu_available():
        raise RuntimeError(f"GPU backend requested but unavailable: {_gpu_error}")
    if device != "cpu" and gpu_available():
        import cupy as cp
        import cupyx.scipy.ndimage as cndi

        return Backend("gpu", cp, cndi, 1)
    n = threads if threads is not None else (os.cpu_count() or 1)
    return Backend("cpu", np, scipy.ndimage, max(1, int(n)))


def _version(module: str, *dists: str) -> str | None:
    """Version of importable ``module`` (from distribution ``dists`` or ``module``), or None."""
    from importlib.metadata import PackageNotFoundError, version

    if importlib.util.find_spec(module) is None:
        return None
    for d in (*dists, module):
        try:
            return version(d)
        except PackageNotFoundError:
            pass
    return "unknown"


def backend_info() -> dict[str, Any]:
    """Describe the available compute backends (for ``gefolki info`` and bug reports).

    Keys: ``default`` ("gpu" or "cpu", what device="auto" picks), ``cpu_threads``,
    ``numpy``/``scipy``/``scikit-image`` versions, ``numba`` and ``cucim`` versions (None if
    not installed), ``gpu_available``, ``gpu_error`` and, when a GPU is usable, ``cupy``,
    ``cuda_runtime``, ``cuda_driver`` and ``gpu`` (name, compute capability, total and free
    memory in bytes).
    """
    import scipy

    ok = gpu_available()
    info: dict[str, Any] = {
        "default": "gpu" if ok else "cpu",
        "cpu_threads": os.cpu_count() or 1,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-image": _version("skimage", "scikit-image"),
        "numba": _version("numba"),
        "cucim": _version("cucim", "cucim-cu12", "cucim-cu13"),
        "gpu_available": ok,
        "gpu_error": _gpu_error,
    }
    if ok:
        import cupy as cp

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        free, total = cp.cuda.runtime.memGetInfo()
        info |= {
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "gpu": {
                "id": dev.id,
                "name": props["name"].decode(),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "memory_total": int(total),
                "memory_free": int(free),
            },
        }
    return info
