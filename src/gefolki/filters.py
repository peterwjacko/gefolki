"""Backend-generic image filters used by the flow solver.

All functions take a :class:`~gefolki.backend.Backend` and arrays native to it. On CPU the
heavy ndimage calls are split into chunks run on a thread pool (SciPy releases the GIL).
Chunking never changes results: separable 1-D filters are split along the axis they do not
filter, so every 1-D line is processed whole; pointwise and windowed ops use halos.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .backend import Backend

BURT_KERNEL = np.array([0.05, 0.25, 0.4, 0.25, 0.05], dtype=np.float32)  # a = 0.4
_MIN_CHUNK = 64  # rows per chunk; below this threading overhead dominates
_COL_BLOCK = 32  # columns per transposed block for axis-0 filtering

try:  # optional CPU fast path for the rank filters
    import numba

    @numba.njit(parallel=True, cache=True)
    def _rank_numba(a, r, sup):  # pragma: no cover - compiled
        h, w = a.shape
        out = np.zeros((h, w), np.float32)
        for i in numba.prange(h):
            for j in range(w):
                c = a[i, j]
                n = 0
                for dy in range(-r, r + 1):
                    ii = i + dy
                    for dx in range(-r, r + 1):
                        jj = j + dx
                        if 0 <= ii < h and 0 <= jj < w:
                            q = a[ii, jj]
                        else:
                            q = 0.0
                        if sup:
                            n += q > c
                        else:
                            n += q < c
                out[i, j] = n
        return out

except ImportError:  # pragma: no cover - depends on environment
    numba = None

_RANK_CUDA = r"""
extern "C" __global__ void rank_filter(const float* a, float* out, int h, int w, int r, int sup) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;
    float c = a[i * w + j];
    int n = 0;
    for (int dy = -r; dy <= r; dy++) {
        int ii = i + dy;
        for (int dx = -r; dx <= r; dx++) {
            int jj = j + dx;
            float q = (ii >= 0 && ii < h && jj >= 0 && jj < w) ? a[ii * w + jj] : 0.0f;
            n += sup ? (q > c) : (q < c);
        }
    }
    out[i * w + j] = (float)n;
}
"""


@functools.cache
def _executor(threads: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(threads, thread_name_prefix="gefolki")


@functools.cache
def _rank_kernel():
    import cupy as cp

    return cp.RawKernel(_RANK_CUDA, "rank_filter")


def _spans(n: int, bk: Backend) -> list[tuple[int, int]]:
    k = max(1, min(bk.threads, n // _MIN_CHUNK))
    edges = [round(i * n / k) for i in range(k + 1)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _run_chunks(n: int, bk: Backend, fn: Callable[[int, int], None]) -> None:
    """Call ``fn(start, stop)`` over ``range(n)`` split into chunks, in parallel on CPU."""
    spans = _spans(n, bk)
    if len(spans) == 1:
        fn(0, n)
    else:
        list(_executor(bk.threads).map(lambda s: fn(*s), spans))


def _filter1d(bk: Backend, a: Any, axis: int, fn: Callable[..., Any]) -> Any:
    """Apply ``fn(src, axis=, output=)`` along ``axis``, chunked across the other axis."""
    if bk.is_gpu:
        return fn(a, axis=axis)
    out = np.empty_like(a)
    if axis == 0:
        # Strided axis-0 lines are slow in SciPy; filter transposed column blocks instead.
        def block(s: int) -> None:
            t = np.ascontiguousarray(a[:, s : s + _COL_BLOCK].T)
            ft = np.empty_like(t)
            fn(t, axis=1, output=ft)
            out[:, s : s + _COL_BLOCK] = ft.T

        starts = range(0, a.shape[1], _COL_BLOCK)
        if bk.threads == 1 or len(starts) == 1:
            for s in starts:
                block(s)
        else:
            list(_executor(bk.threads).map(block, starts))
    else:
        _run_chunks(a.shape[0], bk, lambda s, e: fn(a[s:e], axis=1, output=out[s:e]))
    return out


def box_filter(a: Any, r: int, bk: Backend) -> Any:
    """Mean over a (2r+1)x(2r+1) window, zero padded, same size."""
    if r == 0:
        return a
    f = functools.partial(bk.ndi.uniform_filter1d, size=2 * r + 1, mode="constant", cval=0.0)
    return _filter1d(bk, _filter1d(bk, a, 0, f), 1, f)


def burt_reduce(a: Any, bk: Backend) -> Any:
    """One Burt pyramid step: separable [.05,.25,.4,.25,.05] zero-padded blur, then [::2, ::2]."""
    k = bk.asarray(BURT_KERNEL)
    f = functools.partial(bk.ndi.correlate1d, weights=k, mode="constant", cval=0.0)
    t = _filter1d(bk, a, 0, f)[::2]
    t = bk.xp.ascontiguousarray(t)
    return bk.xp.ascontiguousarray(_filter1d(bk, t, 1, f)[:, ::2])


def pyramid(a: Any, levels: int, bk: Backend) -> list[Any]:
    """Burt pyramid of ``levels + 1`` images, finest first."""
    p = [a]
    for _ in range(levels):
        p.append(burt_reduce(p[-1], bk))
    return p


def _upsample_axis(a: Any, n: int, axis: int, xp: Any) -> Any:
    """Linear interpolation of ``a`` at coords i/2, i in range(n), along ``axis`` (clamped)."""
    a = xp.moveaxis(a, axis, 0)
    out = xp.empty((n, *a.shape[1:]), dtype=a.dtype)
    out[0::2] = a[: (n + 1) // 2]
    nxt = xp.concatenate([a[1:], a[-1:]])
    out[1::2] = (0.5 * (a + nxt))[: n // 2]
    return xp.moveaxis(out, 0, axis)


def upsample_flow(f: Any, shape: tuple[int, int], bk: Backend) -> Any:
    """Bilinear x2 upsampling of a flow component onto ``shape``; values doubled."""
    xp = bk.xp
    out = _upsample_axis(_upsample_axis(f, shape[0], 0, xp), shape[1], 1, xp)
    return xp.ascontiguousarray(2 * out)


def _rank_numpy(p: np.ndarray, r: int, sup: bool, out: np.ndarray, s: int, e: int) -> None:
    """Rank count for output rows [s, e) by shift-and-compare; ``p`` is zero padded by r."""
    w = out.shape[1]
    c = p[s + r : e + r, r : r + w]
    acc = np.zeros(c.shape, np.uint8)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            q = p[s + dy : e + dy, dx : dx + w]
            acc += (q > c) if sup else (q < c)
    out[s:e] = acc


def _rank(a: Any, r: int, sup: bool, bk: Backend) -> Any:
    if bk.is_gpu:
        xp = bk.xp
        a = xp.ascontiguousarray(a, dtype=xp.float32)
        h, w = a.shape
        out = xp.empty((h, w), xp.float32)
        grid = ((w + 31) // 32, (h + 7) // 8)
        _rank_kernel()(
            grid, (32, 8), (a, out, np.int32(h), np.int32(w), np.int32(r), np.int32(sup))
        )
        return out
    a = np.ascontiguousarray(a, dtype=np.float32)
    if numba is not None:
        prev = numba.get_num_threads()
        numba.set_num_threads(min(bk.threads, numba.config.NUMBA_NUM_THREADS))
        try:
            return _rank_numba(a, r, sup)
        finally:
            numba.set_num_threads(prev)
    p = np.pad(a, r)
    out = np.empty(a.shape, np.float32)
    _run_chunks(a.shape[0], bk, lambda s, e: _rank_numpy(p, r, sup, out, s, e))
    return out


def rank_sup(a: Any, r: int, bk: Backend) -> Any:
    """Count of (2r+1)^2 neighbours strictly greater than the centre (zero padding)."""
    return _rank(a, r, True, bk)


def rank_inf(a: Any, r: int, bk: Backend) -> Any:
    """Count of (2r+1)^2 neighbours strictly less than the centre (zero padding)."""
    return _rank(a, r, False, bk)


def gradients(a: Any, bk: Backend) -> tuple[Any, Any]:
    """Central-difference gradients (np.gradient semantics); returns (Ix, Iy)."""
    gy, gx = bk.xp.gradient(a)
    return gx, gy


def interp2(img: Any, xs: Any, ys: Any, bk: Backend, order: int = 1) -> Any:
    """Sample ``img`` at (xs, ys) (column, row coords; edge values beyond the border)."""
    ndi = bk.ndi
    if bk.is_gpu:
        coords = bk.xp.stack([ys, xs])
        return ndi.map_coordinates(img, coords, order=order, mode="nearest")
    if order > 1:
        img = ndi.spline_filter(img, order=order, mode="nearest", output=np.float32)
    out = np.empty(xs.shape, dtype=img.dtype)

    def f(s: int, e: int) -> None:
        coords = np.stack([ys[s:e], xs[s:e]])
        ndi.map_coordinates(
            img, coords, output=out[s:e], order=order, mode="nearest", prefilter=False
        )

    _run_chunks(xs.shape[0], bk, f)
    return out


def clahe(a: Any, bk: Backend) -> Any:
    """CLAHE matching Matlab adapthisteq defaults: 8x8 tiles, clip limit 0.01, 256 bins.

    Uses cuCIM on GPU when installed, otherwise scikit-image (host round-trip on GPU).
    """
    kernel = tuple(max(1, math.ceil(n / 8)) for n in a.shape)
    if bk.is_gpu:
        try:
            from cucim.skimage.exposure import equalize_adapthist as eq_gpu
        except ImportError:
            pass
        else:
            return eq_gpu(a, kernel_size=kernel, clip_limit=0.01, nbins=256).astype(bk.xp.float32)
    from skimage.exposure import equalize_adapthist

    host = bk.to_host(a)
    if host.max() == host.min():  # skimage cannot rescale a constant image
        out = np.zeros(host.shape, np.float32)
    else:
        out = equalize_adapthist(host, kernel_size=kernel, clip_limit=0.01, nbins=256)
    return bk.asarray(out)
