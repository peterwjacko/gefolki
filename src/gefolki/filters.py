"""Backend-generic image filters used by the flow solver.

All functions take a :class:`~gefolki.backend.Backend` and arrays native to it. On CPU the
heavy ndimage calls are split into chunks run on a thread pool (SciPy releases the GIL).
Chunking never changes results: separable 1-D filters are split along the axis they do not
filter, so every 1-D line is processed whole; pointwise and windowed ops use halos.
"""

from __future__ import annotations

import contextlib
import functools
import math
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .backend import Backend

BURT_KERNEL = np.array([0.05, 0.25, 0.4, 0.25, 0.05], dtype=np.float32)  # a = 0.4
_MIN_CHUNK = 64  # rows per chunk; below this threading overhead dominates
_COL_BLOCK = 32  # columns per transposed block for axis-0 filtering
_BOX_COLS = 64  # columns per numba box-filter work block

try:  # optional CPU fast paths (rank filters, box filter, warp)
    import numba
except ImportError:  # pragma: no cover - depends on environment
    numba = None

if numba is not None:

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

    # The kernels below reproduce SciPy's arithmetic exactly (float64 accumulation, same
    # operation order, float32 stores), so results are bitwise identical to the SciPy path.

    # SciPy's uniform_filter1d keeps a float64 running sum ``t += in[i+r] - in[i-r-1]`` (zero
    # padding) and stores ``t / size`` as float32. Where one operand is padding the update is
    # exactly ``+ in[i+r]`` or ``- in[i-r-1]``, so splitting the line into segments keeps the bits.

    @numba.njit(parallel=True, cache=True)
    def _box_axis0(a, r):  # pragma: no cover - compiled
        h, w = a.shape
        size = np.float64(2 * r + 1)
        out = np.empty((h, w), np.float32)
        lo, hi = min(r + 1, h), max(h - r, 1)  # i >= lo: subtract in[i-r-1]; i < hi: add
        cb = _BOX_COLS
        for blk in numba.prange((w + cb - 1) // cb):  # a block of columns at a time
            j0 = blk * cb
            n = min(w, j0 + cb) - j0
            acc = np.zeros(cb, np.float64)
            for k in range(min(r + 1, h)):
                for j in range(n):
                    acc[j] += a[k, j0 + j]
            for j in range(n):
                out[0, j0 + j] = acc[j] / size
            for i in range(1, min(lo, hi)):
                for j in range(n):
                    acc[j] += a[i + r, j0 + j]
                    out[i, j0 + j] = acc[j] / size
            for i in range(min(lo, hi), max(lo, hi)):
                if lo <= hi:
                    for j in range(n):
                        acc[j] += np.float64(a[i + r, j0 + j]) - np.float64(a[i - r - 1, j0 + j])
                        out[i, j0 + j] = acc[j] / size
                else:
                    for j in range(n):
                        out[i, j0 + j] = acc[j] / size
            for i in range(max(lo, hi), h):
                for j in range(n):
                    acc[j] -= a[i - r - 1, j0 + j]
                    out[i, j0 + j] = acc[j] / size
        return out

    @numba.njit(parallel=True, cache=True)
    def _box_axis1(a, r):  # pragma: no cover - compiled
        h, w = a.shape
        size = np.float64(2 * r + 1)
        out = np.empty((h, w), np.float32)
        lo, hi = min(r + 1, w), max(w - r, 1)
        for i in numba.prange(h):
            s = 0.0
            for k in range(min(r + 1, w)):
                s += a[i, k]
            out[i, 0] = s / size
            for j in range(1, min(lo, hi)):
                s += a[i, j + r]
                out[i, j] = s / size
            if lo <= hi:
                for j in range(lo, hi):
                    s += np.float64(a[i, j + r]) - np.float64(a[i, j - r - 1])
                    out[i, j] = s / size
            else:
                for j in range(hi, lo):
                    out[i, j] = s / size
            for j in range(max(lo, hi), w):
                s -= a[i, j - r - 1]
                out[i, j] = s / size
        return out

    @numba.njit(parallel=True, cache=True)
    def _upsample_numba(f, rows, cols):  # pragma: no cover - compiled
        """2 * linear x2 upsampling (same float32 operations as the NumPy version)."""
        h, w = f.shape
        half = np.float32(0.5)
        tmp = np.empty((rows, w), np.float32)
        for i in numba.prange(rows):
            k = i // 2
            for j in range(w):
                if i % 2 == 0:
                    tmp[i, j] = f[k, j]
                else:
                    tmp[i, j] = half * (f[k, j] + f[min(k + 1, h - 1), j])
        out = np.empty((rows, cols), np.float32)
        two = np.float32(2)
        for i in numba.prange(rows):
            for j in range(cols):
                k = j // 2
                if j % 2 == 0:
                    out[i, j] = two * tmp[i, k]
                else:
                    out[i, j] = two * (half * (tmp[i, k] + tmp[i, min(k + 1, w - 1)]))
        return out

    @numba.njit(parallel=True, cache=True)
    def _clahe_bins_numba(q, lo, hi, binsize):  # pragma: no cover - compiled
        """Histogram bin of each pixel (see _CLAHE_BINS)."""
        rows, cols = q.shape
        out = np.empty((rows, cols), np.int32)
        for i in numba.prange(rows):
            for j in range(cols):
                g = np.float64(q[i, j])
                if lo != hi:
                    g = (min(max(g, lo), hi) - lo) / (hi - lo) * 16383.0
                else:
                    g = min(max(g, 0.0), 16383.0)
                out[i, j] = np.int32(np.rint(g)) // binsize
        return out

    @numba.njit(parallel=True, cache=True)
    def _clahe_clip_map_numba(hist, clim, scale, maxval):  # pragma: no cover - compiled
        """skimage clip_histogram + map_histogram per tile (see _CLAHE_CUDA)."""
        n, nbins = hist.shape
        lut = np.empty((n, nbins), np.int32)
        for t in numba.prange(n):
            hh = hist[t].copy()
            n_excess = 0
            for b in range(nbins):
                if hh[b] > clim:
                    n_excess += hh[b] - clim
                    hh[b] = clim
            bin_incr = n_excess // nbins
            upper = clim - bin_incr
            for b in range(nbins):
                if hh[b] < upper:
                    hh[b] += bin_incr
                    n_excess -= bin_incr
            for b in range(nbins):
                if hh[b] >= upper and hh[b] < clim:
                    n_excess += hh[b] - clim
                    hh[b] = clim
            while n_excess > 0:
                prev = n_excess
                for index in range(nbins):
                    under = 0
                    for b in range(nbins):
                        under += hh[b] < clim
                    step = max(1, under // n_excess)
                    cnt = 0
                    for b in range(index, nbins, step):
                        if hh[b] < clim:
                            hh[b] += 1
                            cnt += 1
                    n_excess -= cnt
                    if n_excess <= 0:
                        break
                if prev == n_excess:
                    break
            cum = 0
            for b in range(nbins):
                cum += hh[b]
                lut[t, b] = int(min(np.float64(cum) * scale, np.float64(maxval)))
        return lut

    @numba.njit(parallel=True, cache=True)
    def _clahe_interp_numba(bins, lut, kr, kc, nr, nc):  # pragma: no cover - compiled
        """Bilinear blend of the 4 neighbouring tile mappings (see _CLAHE_INTERP)."""
        rows, cols = bins.shape
        out = np.empty((rows, cols), np.uint16)
        for row in numba.prange(rows):
            pr = row + kr // 2
            br = pr // kr
            fr = np.float64(pr - br * kr) / kr
            for col in range(cols):
                pc = col + kc // 2
                bc = pc // kc
                fc = np.float64(pc - bc * kc) / kc
                b = bins[row, col]
                res = np.float32(0)
                for er in range(2):
                    hr = min(max(br + er - 1, 0), nr - 1)
                    wr = fr if er else 1.0 - fr
                    for ec in range(2):
                        hc = min(max(bc + ec - 1, 0), nc - 1)
                        wc = fc if ec else 1.0 - fc
                        res += np.float32(np.float64(lut[hr * nc + hc, b]) * (wc * wr))
                out[row, col] = np.uint16(res)
        return out

    @numba.njit(parallel=True, cache=True)
    def _warp_numba(imgs, u, v):  # pragma: no cover - compiled
        """Bilinear sample of each imgs[k] at clip(x+u), clip(y+v) (map_coordinates order 1)."""
        n, h, w = imgs.shape
        out = np.empty((n, h, w), np.float32)
        xmax = np.float32(w - 1)
        ymax = np.float32(h - 1)
        for i in numba.prange(h):
            for j in range(w):
                xs = np.float32(j) + u[i, j]
                xs = min(max(xs, np.float32(0)), xmax)
                ys = np.float32(i) + v[i, j]
                ys = min(max(ys, np.float32(0)), ymax)
                cy = np.float64(ys)
                cx = np.float64(xs)
                y0 = int(np.floor(cy))
                x0 = int(np.floor(cx))
                wy0 = 1.0 - (cy - np.floor(cy))
                wy1 = 1.0 - wy0
                wx0 = 1.0 - (cx - np.floor(cx))
                wx1 = 1.0 - wx0
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w - 1)
                for k in range(n):
                    t = 0.0
                    t += np.float64(imgs[k, y0, x0]) * wy0 * wx0
                    t += np.float64(imgs[k, y0, x1]) * wy0 * wx1
                    t += np.float64(imgs[k, y1, x0]) * wy1 * wx0
                    t += np.float64(imgs[k, y1, x1]) * wy1 * wx1
                    out[k, i, j] = t
        return out


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


_numba_lock = threading.Lock()  # numba's workqueue layer forbids concurrent parallel calls
_numba_concurrent = False  # True once a thread-safe layer (tbb, omp) is known to be active


@contextlib.contextmanager
def _numba_threads(bk: Backend) -> Iterator[None]:
    """Run numba parallel kernels with ``bk.threads`` threads (``set_num_threads`` is
    thread-local). Calls are serialised unless numba runs a thread-safe threading layer."""
    global _numba_concurrent
    with contextlib.nullcontext() if _numba_concurrent else _numba_lock:
        prev = numba.get_num_threads()
        numba.set_num_threads(min(bk.threads, numba.config.NUMBA_NUM_THREADS))
        try:
            yield
        finally:
            numba.set_num_threads(prev)
    if not _numba_concurrent:
        try:
            _numba_concurrent = numba.threading_layer() != "workqueue"
        except ValueError:  # no parallel kernel has run yet
            pass


def _use_numba(bk: Backend, *arrays: Any) -> bool:
    return (
        numba is not None
        and not bk.is_gpu
        and all(a.dtype == np.float32 and a.flags.c_contiguous for a in arrays)
    )


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


_BOX_CUDA = r"""
extern "C" __global__ void box1d(const float* in, float* out, int nlines, int len, int r,
                                 int seg) {
    // Running-sum mean along axis 0 of a (len, nlines) array, zero padding, float64
    // accumulator as in SciPy (but multiplying by 1/size: double division is slow on GPUs).
    // Each thread handles a segment of ``seg`` rows of one column (coalesced loads).
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int s0 = blockIdx.y * seg;
    if (line >= nlines || s0 >= len) return;
    int s1 = min(len, s0 + seg);
    const float* p = in + line;
    float* q = out + line;
    long long st = nlines;
    double inv = 1.0 / (2 * r + 1);
    double acc = 0.0;
    for (int k = max(0, s0 - r); k <= min(len - 1, s0 + r); k++) acc += p[k * st];
    q[s0 * st] = (float)(acc * inv);
    for (int i = s0 + 1; i < s1; i++) {
        int ia = i + r, isb = i - r - 1;
        if (ia < len && isb >= 0) acc += (double)p[ia * st] - (double)p[isb * st];
        else if (ia < len) acc += p[ia * st];
        else if (isb >= 0) acc -= p[isb * st];
        q[i * st] = (float)(acc * inv);
    }
}
"""
_BOX_SEG = 256  # rows per GPU thread


@functools.cache
def _box_kernel():
    import cupy as cp

    return cp.RawKernel(_BOX_CUDA, "box1d", options=("--fmad=false",))


def _box_axis0_gpu(a: Any, r: int, xp: Any) -> Any:
    h, w = a.shape
    out = xp.empty_like(a)
    grid = ((w + 127) // 128, -(-h // _BOX_SEG))
    _box_kernel()(grid, (128,), (a, out, np.int32(w), np.int32(h), np.int32(r), np.int32(_BOX_SEG)))
    return out


def _box_gpu(a: Any, r: int, xp: Any) -> Any:
    # Axis 1 runs on the transpose: strided running sums along rows are far slower.
    t = _box_axis0_gpu(xp.ascontiguousarray(a, dtype=xp.float32), r, xp)
    t = xp.ascontiguousarray(t.T)
    t = _box_axis0_gpu(t, r, xp)
    return xp.ascontiguousarray(t.T)


def box_filter(a: Any, r: int, bk: Backend) -> Any:
    """Mean over a (2r+1)x(2r+1) window, zero padded, same size."""
    if r == 0:
        return a
    if bk.is_gpu:
        return _box_gpu(a, r, bk.xp)
    if _use_numba(bk, a):
        with _numba_threads(bk):
            return _box_axis1(_box_axis0(a, r), r)
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
    if _use_numba(bk, f):
        with _numba_threads(bk):
            return _upsample_numba(f, *shape)
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
        with _numba_threads(bk):
            return _rank_numba(a, r, sup)
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


_WARP_CUDA = r"""
int row = i / w, col = i - row * w;
float xs = fminf(fmaxf((float)col + u, 0.0f), (float)(w - 1));
float ys = fminf(fmaxf((float)row + v, 0.0f), (float)(h - 1));
float fy = floorf(ys), fx = floorf(xs);
int y0 = (int)fy, x0 = (int)fx;
int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
float wy0 = 1.0f - (ys - fy), wx0 = 1.0f - (xs - fx);
float wy1 = 1.0f - wy0, wx1 = 1.0f - wx0;
for (int k = 0; k < n; k++) {
    const float* p = &imgs[(long long)k * h * w];
    out[(long long)k * h * w + i] = p[y0 * w + x0] * wy0 * wx0 + p[y0 * w + x1] * wy0 * wx1
                                    + p[y1 * w + x0] * wy1 * wx0 + p[y1 * w + x1] * wy1 * wx1;
}
"""


@functools.cache
def _warp_kernel():
    import cupy as cp

    return cp.ElementwiseKernel(
        "float32 u, float32 v, raw float32 imgs, int32 h, int32 w, int32 n",
        "raw float32 out",
        _WARP_CUDA,
        "gefolki_warp_flow",
    )


def warp_flow(imgs: Any, u: Any, v: Any, bk: Backend) -> Any:
    """Bilinear sample of each image of ``imgs`` (n, H, W) at clip(x + u), clip(y + v).

    Equivalent to ``interp2(img, clip(x + u), clip(y + v))`` per image, without building
    coordinate arrays; returns (n, H, W) float32.
    """
    n, rows, cols = imgs.shape
    if bk.is_gpu:
        xp = bk.xp
        out = xp.empty((n, rows, cols), xp.float32)
        _warp_kernel()(u, v, imgs, np.int32(rows), np.int32(cols), np.int32(n), out)
        return out
    if _use_numba(bk, imgs, u, v):
        with _numba_threads(bk):
            return _warp_numba(imgs, u, v)
    out = np.empty((n, rows, cols), np.float32)
    x = np.arange(cols, dtype=np.float32)[None, :]
    y = np.arange(rows, dtype=np.float32)[:, None]

    def f(s: int, e: int) -> None:
        xs = np.clip(x + u[s:e], 0, cols - 1)
        ys = np.clip(y[s:e] + v[s:e], 0, rows - 1)
        coords = np.stack([ys, xs])
        for k in range(n):
            bk.ndi.map_coordinates(
                imgs[k], coords, output=out[k, s:e], order=1, mode="nearest", prefilter=False
            )

    _run_chunks(rows, bk, f)
    return out


_CLAHE_GRAY = 2**14  # skimage NR_OF_GRAY
_CLAHE_CUDA = r"""
extern "C" __global__ void clahe_clip_map(const long long* hist, int* lut, int ntiles, int nbins,
                                          long long clim, double scale, int maxval) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntiles) return;
    long long hh[256];
    for (int b = 0; b < nbins; b++) hh[b] = hist[(long long)t * nbins + b];
    // skimage clip_histogram, step by step
    long long n_excess = 0;
    for (int b = 0; b < nbins; b++)
        if (hh[b] > clim) { n_excess += hh[b] - clim; hh[b] = clim; }
    long long bin_incr = n_excess / nbins;
    long long upper = clim - bin_incr;
    for (int b = 0; b < nbins; b++)
        if (hh[b] < upper) { hh[b] += bin_incr; n_excess -= bin_incr; }
    for (int b = 0; b < nbins; b++)
        if (hh[b] >= upper && hh[b] < clim) { n_excess += hh[b] - clim; hh[b] = clim; }
    while (n_excess > 0) {
        long long prev = n_excess;
        for (int index = 0; index < nbins; index++) {
            long long under = 0;
            for (int b = 0; b < nbins; b++) under += hh[b] < clim;
            long long step = under / n_excess;
            if (step < 1) step = 1;
            long long cnt = 0;
            for (long long b = index; b < nbins; b += step)
                if (hh[b] < clim) { hh[b] += 1; cnt++; }
            n_excess -= cnt;
            if (n_excess <= 0) break;
        }
        if (prev == n_excess) break;
    }
    // skimage map_histogram
    long long cum = 0;
    for (int b = 0; b < nbins; b++) {
        cum += hh[b];
        double val = (double)cum * scale;
        if (val > maxval) val = maxval;
        lut[t * nbins + b] = (int)val;
    }
}
"""
_CLAHE_BINS = r"""
// skimage: rescale_intensity(q, out_range=(0, 2**14 - 1)) in float64, round, then // binsize
double g = (double)q;
if (lo != hi) g = (fmin(fmax(g, lo), hi) - lo) / (hi - lo) * 16383.0;
else g = fmin(fmax(g, 0.0), 16383.0);
bin = (int)rint(g) / binsize;
"""
_CLAHE_INTERP = r"""
int row = i / w, col = i - row * w;
int pr = row + kr / 2, pc = col + kc / 2;
int br = pr / kr, bc = pc / kc;
double fr = (double)(pr - br * kr) / kr, fc = (double)(pc - bc * kc) / kc;
float res = 0.0f;
for (int er = 0; er < 2; er++) {
    int hr = min(max(br + er - 1, 0), nr - 1);
    for (int ec = 0; ec < 2; ec++) {
        int hc = min(max(bc + ec - 1, 0), nc - 1);
        int m = lut[(hr * nc + hc) * nbins + bin];
        double wgt = (ec ? fc : 1.0 - fc) * (er ? fr : 1.0 - fr);
        res += (float)((double)m * wgt);
    }
}
out = (unsigned short)res;
"""


@functools.cache
def _clahe_kernels() -> tuple[Any, Any, Any]:
    import cupy as cp

    clip_map = cp.RawKernel(_CLAHE_CUDA, "clahe_clip_map", options=("--fmad=false",))
    interp = cp.ElementwiseKernel(
        "int32 bin, raw int32 lut, int32 w, int32 kr, int32 kc, int32 nr, int32 nc, int32 nbins",
        "uint16 out",
        _CLAHE_INTERP,
        "gefolki_clahe_interp",
        options=("--fmad=false",),
    )
    bins = cp.ElementwiseKernel(
        "uint16 q, float64 lo, float64 hi, int32 binsize",
        "int32 bin",
        _CLAHE_BINS,
        "gefolki_clahe_bins",
        options=("--fmad=false",),
    )
    return clip_map, interp, bins


def _rescale(a: Any, xp: Any) -> Any:
    """skimage rescale_intensity(a) for a float32 image with min >= 0 (out range [0, 1])."""
    lo, hi = float(a.min()), float(a.max())
    a = xp.clip(a, xp.float32(lo), xp.float32(hi))
    if lo == hi:
        return xp.clip(a, xp.float32(0), xp.float32(1))
    return (a - xp.float32(lo)) / xp.float32(hi - lo)


def _clahe_port(a: Any, kernel: tuple[int, int], clip_limit: float, nbins: int, bk: Backend) -> Any:
    """Port of skimage ``equalize_adapthist`` for a 2-D float image in [0, 1] (NumPy + numba,
    or CuPy).

    Follows skimage step by step (quantisation, tile histograms, clipping, mapping, bilinear
    interpolation) so the output matches it (bitwise in tests).
    """
    xp = bk.xp
    rows, cols = a.shape
    kr, kc = kernel
    # img_as_uint (float32 maths), then rescale_intensity to 14 bits (float64) and round.
    q = xp.clip(xp.rint(a.astype(xp.float32) * xp.float32(65535)), 0, 65535).astype(xp.uint16)
    lo, hi = float(q.min()), float(q.max())
    binsize = 1 + _CLAHE_GRAY // nbins
    if bk.is_gpu:
        bins = _clahe_kernels()[2](q, lo, hi, np.int32(binsize))
    else:
        with _numba_threads(bk):
            bins = _clahe_bins_numba(q, lo, hi, binsize)
    del q
    # Tile histograms over the reflect-padded image (as skimage pads).
    pad = [
        (k // 2, (k - s % k) % k + math.ceil(k / 2)) for k, s in zip(kernel, a.shape, strict=True)
    ]
    nr = (rows + sum(pad[0])) // kr - 1
    nc = (cols + sum(pad[1])) // kc - 1
    region = xp.pad(bins, pad, mode="reflect")[
        kr // 2 : kr // 2 + nr * kr, kc // 2 : kc // 2 + nc * kc
    ]
    tile = (xp.arange(nr)[:, None] * nc + xp.arange(nc)[None, :]).astype(xp.int32)
    keys = tile[:, None, :, None] * nbins + region.reshape(nr, kr, nc, kc)
    hist = xp.bincount(keys.ravel(), minlength=nr * nc * nbins).astype(xp.int64)
    del region, keys
    clim = max(int(clip_limit * kr * kc), 1) if clip_limit > 0 else kr * kc
    scale = (_CLAHE_GRAY - 1) / (kr * kc)
    n = nr * nc
    if not bk.is_gpu:
        with _numba_threads(bk):
            lut = _clahe_clip_map_numba(hist.reshape(n, nbins), clim, scale, _CLAHE_GRAY - 1)
            out = _clahe_interp_numba(bins, lut, kr, kc, nr, nc)
    else:
        lut = xp.empty(n * nbins, xp.int32)
        clip_map, interp, _ = _clahe_kernels()
        args = (hist, lut, np.int32(n), np.int32(nbins), np.int64(clim))
        clip_map(((n + 63) // 64,), (64,), (*args, np.float64(scale), np.int32(_CLAHE_GRAY - 1)))
        ints = (cols, kr, kc, nr, nc, nbins)
        out = interp(bins, lut, *(np.int32(i) for i in ints))
    return _rescale(out.astype(xp.float32), xp)


def clahe(a: Any, bk: Backend) -> Any:
    """CLAHE matching Matlab adapthisteq defaults: 8x8 tiles, clip limit 0.01, 256 bins.

    Uses cuCIM on GPU when installed; otherwise a port of scikit-image's algorithm (CuPy on
    GPU, numba on CPU) that reproduces its output; scikit-image itself on CPU without numba.
    """
    kernel = tuple(max(1, math.ceil(n / 8)) for n in a.shape)
    if bk.is_gpu:
        try:
            from cucim.skimage.exposure import equalize_adapthist as eq_gpu
        except ImportError:
            pass
        else:
            return eq_gpu(a, kernel_size=kernel, clip_limit=0.01, nbins=256).astype(bk.xp.float32)
    if float(a.max()) == float(a.min()):  # skimage cannot rescale a constant image
        return bk.xp.zeros(a.shape, bk.xp.float32)
    if bk.is_gpu or numba is not None:
        return _clahe_port(a, kernel, 0.01, 256, bk)
    from skimage.exposure import equalize_adapthist

    host = bk.to_host(a)
    return bk.asarray(equalize_adapthist(host, kernel_size=kernel, clip_limit=0.01, nbins=256))
