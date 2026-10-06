"""Dense flow estimation: Folki, EFolki and GeFolki share one pyramidal solver."""

from __future__ import annotations

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


def _solve_level(j0: Any, j1: Any, u: Any, v: Any, p: FlowParams, bk: Backend) -> tuple[Any, Any]:
    xp = bk.xp
    if p.rank:
        r0 = F.rank_sup(j0, p.rank, bk)
        r1_sup = F.rank_sup(j1, p.rank, bk)
        r1_inf = F.rank_inf(j1, p.rank, bk) if p.contrast_adapt else None
    else:
        r0, r1_sup, r1_inf = j0, j1, 1 - j1
    if p.contrast_adapt:
        h0, h1 = F.clahe(j0, bk), F.clahe(j1, bk)
    ix, iy = F.gradients(r0, bk)
    rows, cols = j0.shape
    x = xp.arange(cols, dtype=xp.float32)[None, :]
    y = xp.arange(rows, dtype=xp.float32)[:, None]

    for rad in p.radius:
        a = F.box_filter(ix * ix, rad, bk)
        b = F.box_filter(iy * iy, rad, bk)
        c = F.box_filter(ix * iy, rad, bk)
        d = a * b - c * c
        for _ in range(p.iterations):
            xs = xp.clip(x + u, 0, cols - 1)
            ys = xp.clip(y + v, 0, rows - 1)
            r1w = F.interp2(r1_sup, xs, ys, bk)
            if p.contrast_adapt:
                h1w = F.interp2(h1, xs, ys, bk)
                crit1 = F.box_filter(xp.abs(h0 - h1w), p.rank, bk)
                crit2 = F.box_filter(xp.abs(1 - h0 - h1w), p.rank, bk)
                r1w = xp.where(crit1 > crit2, F.interp2(r1_inf, xs, ys, bk), r1w)
            it = r0 - r1w + u * ix + v * iy
            g = F.box_filter(it * ix, rad, bk)
            h = F.box_filter(it * iy, rad, bk)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (g * b - c * h) / d
                v = (a * h - c * g) / d
            bad = ~(xp.isfinite(u) & xp.isfinite(v))
            u = xp.where(bad, 0, u).astype(xp.float32)
            v = xp.where(bad, 0, v).astype(xp.float32)
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
    xp = bk.xp
    m, s = bk.asarray(master), bk.asarray(slave)
    if m.ndim != 2 or m.shape != s.shape:
        raise ValueError(f"master/slave must be 2-D with equal shape, got {m.shape}, {s.shape}")
    valid = xp.isfinite(m) & xp.isfinite(s)
    if mask is not None:
        valid &= bk.asarray(mask, dtype=bool)
    m, s = _normalise(m, valid, xp), _normalise(s, valid, xp)

    # Keep the coarsest level at least 2 px per side so gradients are defined.
    levels = p.levels
    while levels and -(-min(m.shape) // 2**levels) < 2:
        levels -= 1
    pyr0, pyr1 = F.pyramid(m, levels, bk), F.pyramid(s, levels, bk)

    u = v = None
    for j0, j1 in zip(reversed(pyr0), reversed(pyr1), strict=True):
        if u is None:
            u = xp.zeros(j0.shape, xp.float32)
            v = xp.zeros(j0.shape, xp.float32)
        else:
            u, v = F.upsample_flow(u, j0.shape, bk), F.upsample_flow(v, j0.shape, bk)
        u, v = _solve_level(j0, j1, u, v, p, bk)
    if return_device:
        return u, v
    return bk.to_host(u), bk.to_host(v)


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
