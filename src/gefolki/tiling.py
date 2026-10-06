"""Flow estimation for images too large for memory/GPU: overlapping tiles, feathered blend.

Tiles of ``tile_size`` px overlap by ``overlap`` px. Each tile's flow gets a weight that
ramps linearly from ~0 at an interior tile edge to 1 at ``overlap`` px inside (image borders
keep weight 1), and the blended flow is sum(w * flow) / sum(w). Tiles run in a thread pool
on CPU (each with ``threads // workers`` threads) or one after another on GPU.

Default overlap: ``min(tile_size // 4, max(128, 8 * max(radius)))`` (256 px for radius 32).
The solver's coarse levels see a window of ``(2 r + 1) * 2**levels`` px, larger than any
practical tile, so tile borders always lose some context; an overlap of several fine-level
windows lets the feather hide the border errors (see tests: tiled vs whole-image flow).
GeFolki's CLAHE uses 8x8 tiles of the image it is given, so per-tile CLAHE differs from
whole-image CLAHE; expect slightly larger tiled/whole differences with ``contrast_adapt``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .backend import Backend, Device, get_backend
from .flow import FlowParams, estimate_flow

BYTES_PER_PIXEL = 160  # peak solver memory per pixel (float32 pyramids, rank images, temps)
GPU_TILE = 4096
CPU_TILE = 2048

Progress = Callable[[str, int, int], None]


def default_overlap(tile_size: int, params: FlowParams) -> int:
    return min(tile_size // 4, max(128, 8 * max(params.radius)))


def _available_bytes(bk: Backend) -> int:
    if bk.is_gpu:
        free, _ = bk.xp.cuda.runtime.memGetInfo()
        return int(free)
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
        return 8 << 30


def fits_in_memory(shape: tuple[int, int], bk: Backend, fraction: float = 0.5) -> bool:
    """True if a whole-image flow on ``shape`` should fit in ``fraction`` of free memory."""
    return shape[0] * shape[1] * BYTES_PER_PIXEL <= fraction * _available_bytes(bk)


def tile_spans(n: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Start/stop pairs covering ``range(n)`` with tiles of ``tile`` overlapping >= ``overlap``."""
    if n <= tile:
        return [(0, n)]
    step = tile - overlap
    k = -(-(n - tile) // step) + 1
    starts = np.linspace(0, n - tile, k).round().astype(int)
    return [(int(s), int(s) + tile) for s in starts]


def _ramp(start: int, stop: int, n: int, overlap: int) -> np.ndarray:
    """1-D feather weights for a tile [start, stop) of an axis of length n."""
    i = np.arange(stop - start, dtype=np.float32)
    w = np.ones(stop - start, np.float32)
    if start > 0:
        w = np.minimum(w, (i + 0.5) / overlap)
    if stop < n:
        w = np.minimum(w, (stop - start - i - 0.5) / overlap)
    return w


def estimate_flow_tiled(
    master: np.ndarray,
    slave: np.ndarray,
    params: FlowParams | None = None,
    *,
    mask: np.ndarray | None = None,
    device: Device | Backend = "auto",
    threads: int | None = None,
    tile_size: int | None = None,
    overlap: int | None = None,
    progress: Progress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flow (u, v) like :func:`gefolki.estimate_flow`, tiled when needed.

    tile_size: None = whole image if it fits in memory (GPU: free device memory), else
    4096 px tiles on GPU / 2048 on CPU; 0 = never tile; >0 = tile with that size.
    """
    p = params or FlowParams()
    bk = get_backend(device, threads)
    shape = master.shape
    if tile_size is None:
        tile_size = 0 if fits_in_memory(shape, bk) else (GPU_TILE if bk.is_gpu else CPU_TILE)
    if not tile_size or (shape[0] <= tile_size and shape[1] <= tile_size):
        if progress:
            progress("flow", 0, 1)
        u, v = estimate_flow(master, slave, p, mask=mask, device=bk)
        if progress:
            progress("flow", 1, 1)
        return u, v

    ov = default_overlap(tile_size, p) if overlap is None else overlap
    ov = max(1, min(ov, tile_size // 2))
    rows, cols = tile_spans(shape[0], tile_size, ov), tile_spans(shape[1], tile_size, ov)
    tiles = [(r, c) for r in rows for c in cols]
    acc_u = np.zeros(shape, np.float32)
    acc_v = np.zeros(shape, np.float32)
    acc_w = np.zeros(shape, np.float32)

    if bk.is_gpu:
        workers, tile_bk = 1, bk
    else:
        by_mem = int(0.5 * _available_bytes(bk) // (tile_size**2 * BYTES_PER_PIXEL))
        workers = max(1, min(bk.threads, len(tiles), by_mem))
        tile_bk = get_backend("cpu", max(1, bk.threads // workers))

    def run(rc: tuple[tuple[int, int], tuple[int, int]]) -> Any:
        (r0, r1), (c0, c1) = rc
        sl = np.s_[r0:r1, c0:c1]
        m = None if mask is None else mask[sl]
        if m is not None and not m.any():
            return sl, None, None
        u, v = estimate_flow(master[sl], slave[sl], p, mask=m, device=tile_bk)
        return sl, u, v

    def add(res: Any) -> None:
        sl, u, v = res
        if u is None:
            return
        (rs, cs) = sl
        w = np.outer(_ramp(rs.start, rs.stop, shape[0], ov), _ramp(cs.start, cs.stop, shape[1], ov))
        acc_u[sl] += w * u
        acc_v[sl] += w * v
        acc_w[sl] += w

    done = 0
    if progress:
        progress("flow", 0, len(tiles))
    if workers == 1:
        results = map(run, tiles)
    else:
        pool = ThreadPoolExecutor(workers, thread_name_prefix="gefolki-tile")
        results = pool.map(run, tiles)
    try:
        for res in results:
            add(res)
            done += 1
            if progress:
                progress("flow", done, len(tiles))
    finally:
        if workers > 1:
            pool.shutdown()
    np.maximum(acc_w, 1e-12, out=acc_w)
    return acc_u / acc_w, acc_v / acc_w
