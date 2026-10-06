"""Dense flow estimation: Folki, EFolki and GeFolki share one pyramidal solver."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import filters as F
from .backend import Backend, Device, get_backend

DEFAULT_RADIUS = tuple(range(32, 4, -4))  # 32, 28, ..., 8
Radius = int | Sequence[int]


@dataclass(frozen=True)
class FlowParams:
    """Solver parameters (defaults follow the GeFolki manual).

    levels: pyramid levels (levels + 1 images). radius: window radii, applied coarse to fine
    at every level. iterations: solver iterations per radius. rank: rank-filter radius
    (0 = use intensities). contrast_adapt: GeFolki CLAHE-based contrast-inversion test.
    """

    levels: int = 6
    radius: tuple[int, ...] = DEFAULT_RADIUS
    iterations: int = 2
    rank: int = 4
    contrast_adapt: bool = False

    def __post_init__(self) -> None:
        r = (self.radius,) if isinstance(self.radius, int) else tuple(self.radius)
        object.__setattr__(self, "radius", r)
        if self.levels < 0 or self.iterations < 1 or self.rank < 0 or not r or min(r) < 1:
            raise ValueError(f"invalid FlowParams: {self}")


def _normalise(a: Any, valid: Any, xp: Any) -> Any:
    """Scale to [0, 1] by min/max over valid pixels; invalid pixels -> 0."""
    if not bool(valid.any()):
        return xp.zeros_like(a)
    v = a[valid]
    lo, hi = v.min(), v.max()
    ptp = hi - lo
    if float(ptp) == 0.0:
        return xp.zeros_like(a)
    return xp.where(valid, (a - lo) / ptp, 0).astype(xp.float32)


def estimate_gpu_bytes_per_pixel(params: FlowParams | None = None) -> int:
    """Approximate peak GPU memory of :func:`estimate_flow` per image pixel, in bytes.

    Measured peaks of the CuPy memory pool (including its fragmentation) for 2048^2 to
    6000^2 images: GeFolki ~110 B/px (92 B/px live arrays), EFolki/Folki ~90 B/px; the
    values returned add ~10% headroom. Use it to choose tile sizes:
    ``max_pixels = free_bytes // estimate_gpu_bytes_per_pixel(params)``. The input arrays,
    if already on the device, are not included.
    """
    p = params or FlowParams()
    return 120 if p.contrast_adapt else 100


# Fused element-wise steps of the solver iteration. CPU: numba kernels when installed, else
# NumPy on row chunks run on the thread pool (both bitwise identical to whole-array NumPy);
# GPU: one CuPy kernel per step.
_EW_OPTS = ("--fmad=false",)  # no FMA contraction: keep GPU arithmetic close to CPU


@functools.cache
def _ew_kernels() -> dict[str, Any]:
    import cupy as cp

    k = functools.partial(cp.ElementwiseKernel, options=_EW_OPTS)
    return {
        "absdiff": k(
            "float32 h0, float32 h1w",
            "float32 e1, float32 e2",
            "e1 = fabsf(h0 - h1w); e2 = fabsf(1.0f - h0 - h1w);",
            "gefolki_absdiff",
        ),
        "residual": k(
            "float32 r0, float32 r1w, float32 ix, float32 iy, float32 u, float32 v",
            "float32 p, float32 q",
            "float it = r0 - r1w + u * ix + v * iy; p = it * ix; q = it * iy;",
            "gefolki_residual",
        ),
        "residual_sel": k(
            "float32 r0, float32 r1w, float32 r1iw, float32 c1, float32 c2, "
            "float32 ix, float32 iy, float32 u, float32 v",
            "float32 p, float32 q",
            "float it = r0 - (c1 > c2 ? r1iw : r1w) + u * ix + v * iy; p = it * ix; q = it * iy;",
            "gefolki_residual_sel",
        ),
        "update": k(
            "float32 g, float32 h, float32 a, float32 b, float32 c, float32 d",
            "float32 u, float32 v",
            "float uu = (g * b - c * h) / d; float vv = (a * h - c * g) / d;"
            "bool ok = isfinite(uu) && isfinite(vv); u = ok ? uu : 0.0f; v = ok ? vv : 0.0f;",
            "gefolki_update",
        ),
    }


def _chunked(bk: Backend, n_out: int, args: tuple[Any, ...], fn: Any) -> list[np.ndarray]:
    """Run ``fn(*arg_rows, *out_rows)`` on row chunks; returns ``n_out`` new float32 arrays."""
    outs = [np.empty(args[0].shape, np.float32) for _ in range(n_out)]
    F._run_chunks(
        args[0].shape[0], bk, lambda s, e: fn(*(a[s:e] for a in args), *(o[s:e] for o in outs))
    )
    return outs


def _absdiff_np(h0, h1w, e1, e2):
    np.abs(h0 - h1w, out=e1)
    np.abs(1 - h0 - h1w, out=e2)


def _residual_np(r0, r1w, ix, iy, u, v, p, q, sel=None):
    if sel is not None:
        r1w = np.where(sel[0] > sel[1], sel[2], r1w)
    it = r0 - r1w + u * ix + v * iy
    np.multiply(it, ix, out=p)
    np.multiply(it, iy, out=q)


def _residual_sel_np(r0, r1w, r1iw, c1, c2, ix, iy, u, v, p, q):
    _residual_np(r0, r1w, ix, iy, u, v, p, q, (c1, c2, r1iw))


def _update_np(g, h, a, b, c, d, u, v):
    with np.errstate(divide="ignore", invalid="ignore"):
        uu = (g * b - c * h) / d
        vv = (a * h - c * g) / d
    bad = ~(np.isfinite(uu) & np.isfinite(vv))
    u[...] = np.where(bad, 0, uu)
    v[...] = np.where(bad, 0, vv)


def _fused(name: str, n_out: int, bk: Backend, *args: Any) -> list[Any]:
    if bk.is_gpu:
        return list(_ew_kernels()[name](*args))
    if F._use_numba(bk, *args):
        with F._numba_threads(bk):
            outs = _NUMBA_EW[name](*(a.ravel() for a in args))
        return [o.reshape(args[0].shape) for o in outs]
    return _chunked(bk, n_out, args, _CPU_EW[name])


_CPU_EW = {
    "absdiff": _absdiff_np,
    "residual": _residual_np,
    "residual_sel": _residual_sel_np,
    "update": _update_np,
}

if F.numba is not None:  # one parallel pass per step; same float32 operations as NumPy
    _njit = F.numba.njit(parallel=True, cache=True, error_model="numpy")
    _prange = F.numba.prange

    @_njit
    def _absdiff_nb(h0, h1w):  # pragma: no cover - compiled
        e1 = np.empty(h0.size, np.float32)
        e2 = np.empty(h0.size, np.float32)
        one = np.float32(1)
        for i in _prange(h0.size):
            e1[i] = abs(h0[i] - h1w[i])
            e2[i] = abs(one - h0[i] - h1w[i])
        return e1, e2

    @_njit
    def _residual_nb(r0, r1w, ix, iy, u, v):  # pragma: no cover - compiled
        p = np.empty(r0.size, np.float32)
        q = np.empty(r0.size, np.float32)
        for i in _prange(r0.size):
            it = r0[i] - r1w[i] + u[i] * ix[i] + v[i] * iy[i]
            p[i] = it * ix[i]
            q[i] = it * iy[i]
        return p, q

    @_njit
    def _residual_sel_nb(r0, r1w, r1iw, c1, c2, ix, iy, u, v):  # pragma: no cover - compiled
        p = np.empty(r0.size, np.float32)
        q = np.empty(r0.size, np.float32)
        for i in _prange(r0.size):
            r = r1iw[i] if c1[i] > c2[i] else r1w[i]
            it = r0[i] - r + u[i] * ix[i] + v[i] * iy[i]
            p[i] = it * ix[i]
            q[i] = it * iy[i]
        return p, q

    @_njit
    def _update_nb(g, h, a, b, c, d):  # pragma: no cover - compiled
        u = np.empty(g.size, np.float32)
        v = np.empty(g.size, np.float32)
        zero = np.float32(0)
        for i in _prange(g.size):
            uu = (g[i] * b[i] - c[i] * h[i]) / d[i]
            vv = (a[i] * h[i] - c[i] * g[i]) / d[i]
            ok = np.isfinite(uu) and np.isfinite(vv)
            u[i] = uu if ok else zero
            v[i] = vv if ok else zero
        return u, v

    _NUMBA_EW = {
        "absdiff": _absdiff_nb,
        "residual": _residual_nb,
        "residual_sel": _residual_sel_nb,
        "update": _update_nb,
    }


def _solve_level(j0: Any, j1: Any, u: Any, v: Any, p: FlowParams, bk: Backend) -> tuple[Any, Any]:
    xp = bk.xp
    if p.rank:
        r0 = F.rank_sup(j0, p.rank, bk)
        r1_sup = F.rank_sup(j1, p.rank, bk)
        r1_inf = F.rank_inf(j1, p.rank, bk) if p.contrast_adapt else None
    else:
        r0, r1_sup = j0, j1
        r1_inf = 1 - j1 if p.contrast_adapt else None
    if p.contrast_adapt:
        if bk.is_gpu or bk.threads == 1:
            h0, h1 = F.clahe(j0, bk), F.clahe(j1, bk)
        else:  # skimage CLAHE is single-threaded: equalise both images at once
            h0, h1 = F._executor(bk.threads).map(lambda a: F.clahe(a, bk), (j0, j1))
        imgs = xp.stack([r1_sup, h1, r1_inf])  # warped together, sharing coordinates
        del h1, r1_inf
    else:
        imgs = r1_sup[None]
    del r1_sup, j0, j1  # (the caller holds no references: frees device memory early)
    ix, iy = F.gradients(r0, bk)

    for rad in p.radius:
        a = F.box_filter(ix * ix, rad, bk)
        b = F.box_filter(iy * iy, rad, bk)
        c = F.box_filter(ix * iy, rad, bk)
        d = a * b - c * c
        for _ in range(p.iterations):
            w = F.warp_flow(imgs, u, v, bk)
            if p.contrast_adapt:
                e1, e2 = _fused("absdiff", 2, bk, h0, w[1])
                crit1 = F.box_filter(e1, p.rank, bk)
                del e1
                crit2 = F.box_filter(e2, p.rank, bk)
                del e2
                pq = _fused("residual_sel", 2, bk, r0, w[0], w[2], crit1, crit2, ix, iy, u, v)
                del crit1, crit2
            else:
                pq = _fused("residual", 2, bk, r0, w[0], ix, iy, u, v)
            del w
            g = F.box_filter(pq[0], rad, bk)
            h = F.box_filter(pq[1], rad, bk)
            del pq
            u, v = _fused("update", 2, bk, g, h, a, b, c, d)
    return u, v


def estimate_flow(
    master: Any,
    slave: Any,
    params: FlowParams | None = None,
    *,
    mask: Any = None,
    device: Device | Backend = "auto",
    threads: int | None = None,
    return_device: bool = False,
) -> tuple[Any, Any]:
    """Estimate dense flow (u, v) such that slave(x + u, y + v) ~ master(x, y).

    master, slave: 2-D arrays of equal shape (NumPy or CuPy). mask: optional boolean
    valid-pixel mask (True = valid) applied to both images; non-finite pixels are also
    invalid. Returns float32 NumPy arrays (u = column shift, v = row shift in pixels), or
    backend arrays if ``return_device``.
    """
    p = params or FlowParams()
    bk = get_backend(device, threads)
    u, v = _pyramid_flow(*_prepare(master, slave, mask, bk), p, bk)
    if return_device:
        return u, v
    u, v = bk.to_host(u), bk.to_host(v)
    if bk.is_gpu:  # hand cached device memory back (e.g. to other processes or cuCIM)
        bk.xp.get_default_memory_pool().free_all_blocks()
    return u, v


def _prepare(master: Any, slave: Any, mask: Any, bk: Backend) -> tuple[Any, Any]:
    """Backend float32 copies normalised to [0, 1] with invalid pixels set to 0."""
    xp = bk.xp
    m, s = bk.asarray(master), bk.asarray(slave)
    if m.ndim != 2 or m.shape != s.shape:
        raise ValueError(f"master/slave must be 2-D with equal shape, got {m.shape}, {s.shape}")
    valid = xp.isfinite(m) & xp.isfinite(s)
    if mask is not None:
        valid &= bk.asarray(mask, dtype=bool)
    return _normalise(m, valid, xp), _normalise(s, valid, xp)


def _pyramid_flow(m: Any, s: Any, p: FlowParams, bk: Backend) -> tuple[Any, Any]:
    xp = bk.xp
    # Keep the coarsest level at least 2 px per side so gradients are defined.
    levels = p.levels
    while levels and -(-min(m.shape) // 2**levels) < 2:
        levels -= 1
    pyr0, pyr1 = F.pyramid(m, levels, bk), F.pyramid(s, levels, bk)
    del m, s
    u = v = None
    while pyr0:  # coarse to fine, dropping each level once solved
        shape = pyr0[-1].shape
        if u is None:
            u = xp.zeros(shape, xp.float32)
            v = xp.zeros(shape, xp.float32)
        else:
            u, v = F.upsample_flow(u, shape, bk), F.upsample_flow(v, shape, bk)
        u, v = _solve_level(pyr0.pop(), pyr1.pop(), u, v, p, bk)
    return u, v


def folki(
    master: Any,
    slave: Any,
    *,
    levels: int = 6,
    radius: Radius = (8,),
    iterations: int = 2,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Folki: intensity-based flow (no rank transform). kwargs go to estimate_flow."""
    p = FlowParams(levels, radius, iterations, rank=0, contrast_adapt=False)
    return estimate_flow(master, slave, p, **kwargs)


def efolki(
    master: Any,
    slave: Any,
    *,
    levels: int = 6,
    radius: Radius = DEFAULT_RADIUS,
    iterations: int = 2,
    rank: int = 4,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """EFolki: rank-transformed flow (robust to monotonic radiometric differences)."""
    p = FlowParams(levels, radius, iterations, rank=rank, contrast_adapt=False)
    return estimate_flow(master, slave, p, **kwargs)


def gefolki(
    master: Any,
    slave: Any,
    *,
    levels: int = 6,
    radius: Radius = DEFAULT_RADIUS,
    iterations: int = 2,
    rank: int = 4,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """GeFolki: EFolki plus local contrast-inversion handling (heterogeneous sensors)."""
    p = FlowParams(levels, radius, iterations, rank=rank, contrast_adapt=True)
    return estimate_flow(master, slave, p, **kwargs)
