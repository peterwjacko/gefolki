"""Find where a small slave image (e.g. an airborne SAR chip) lies inside a large master image.

Port of the legacy ``mining.py``: exhaustive search of the offset minimising the mean squared
difference of rank_inf maps, first on area-averaged decimated images, then at full resolution
within a margin around the coarse hit. All offsets are scored at once with FFT correlations.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from os import PathLike
from typing import Any

import scipy.fft

from . import filters as F
from .backend import Backend, Device, get_backend

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class LocateResult:
    """Position of the slave inside the master.

    row, col: master pixel of the slave's top-left corner; height, width: slave shape.
    score: mean squared rank difference at the best offset (lower is better).
    bounds: (xmin, xmax, ymin, ymax) in master pixels, x = column, y = row, max exclusive.
    map_bounds: the same extent (xmin, xmax, ymin, ymax) in master map coordinates when the
    master is georeferenced (``transform`` given), else None.
    """

    row: int
    col: int
    height: int
    width: int
    score: float
    bounds: tuple[int, int, int, int]
    map_bounds: Bounds | None = None


def _block_mean(a: Any, m: Any, d: int, xp: Any) -> tuple[Any, Any]:
    """Area average over d x d blocks of valid pixels; block valid if >= half its pixels are."""
    h, w = a.shape[0] // d, a.shape[1] // d
    a, m = a[: h * d, : w * d], m[: h * d, : w * d].astype(xp.float32)
    s = (a * m).reshape(h, d, w, d).sum(axis=(1, 3))
    n = m.reshape(h, d, w, d).sum(axis=(1, 3))
    return s / xp.maximum(n, 1), n >= d * d / 2


@functools.cache
def _gpu_fft_ok() -> bool:
    """cuFFT loads (the CUDA 12 cuFFT library may be missing: pip install nvidia-cufft-cu12)."""
    import cupy as cp

    try:
        cp.fft.rfft2(cp.ones((4, 4), cp.float32))
    except Exception:  # ImportError for libcufft, or CUDA errors
        return False
    return True


def _score_map(img: Any, tpl: Any, valid: Any, bk: Backend) -> Any:
    """Mean of (img[o + p] - tpl[p])^2 over valid p, for every offset o fully inside img.

    sum_p m(I - T)^2 = sum m T^2 - 2 corr(I, m T) + corr(I^2, m), the correlations done by FFT.
    A circular FFT of the image size is exact for the fully-inside ("valid") offsets.
    """
    if bk.is_gpu and not _gpu_fft_ok():  # host FFT fallback
        cpu = get_backend("cpu")
        args = (bk.to_host(img), bk.to_host(tpl), bk.to_host(valid))
        return bk.asarray(_score_map(*args, cpu), bk.xp.float64)
    xp = bk.xp
    if bk.is_gpu:
        fft, workers = xp.fft, contextlib.nullcontext()
    else:
        fft, workers = scipy.fft, scipy.fft.set_workers(bk.threads)
    shape = tuple(scipy.fft.next_fast_len(int(n), real=True) for n in img.shape)
    img = img.astype(xp.float64)
    m = valid.astype(xp.float64)
    mt = m * tpl.astype(xp.float64)
    with workers:
        f = fft.rfft2(img * img, shape) * fft.rfft2(m[::-1, ::-1], shape)
        f -= 2 * fft.rfft2(img, shape) * fft.rfft2(mt[::-1, ::-1], shape)
        full = fft.irfft2(f, shape)
    h, w = tpl.shape
    valid_part = full[h - 1 : img.shape[0], w - 1 : img.shape[1]]
    return ((mt * mt).sum() + valid_part) / m.sum()


def _rank_crop(a: Any, r0: int, r1: int, c0: int, c1: int, rank: int, bk: Backend) -> Any:
    """rank_inf of a[r0:r1, c0:c1] as computed on the whole image (halo, no crop edge effect)."""
    h0, w0 = max(0, r0 - rank), max(0, c0 - rank)
    sub = a[h0 : min(a.shape[0], r1 + rank), w0 : min(a.shape[1], c1 + rank)]
    rk = F.rank_inf(bk.xp.ascontiguousarray(sub), rank, bk)
    return rk[r0 - h0 : r0 - h0 + r1 - r0, c0 - w0 : c0 - w0 + c1 - c0]


def _argmin(score: Any, bk: Backend) -> tuple[int, int, float]:
    i = int(bk.xp.argmin(score))
    r, c = divmod(i, score.shape[1])
    return r, c, float(score[r, c])


def locate(
    master: Any,
    slave: Any,
    *,
    mask: Any = None,
    decimation: int = 8,
    rank: int = 3,
    margin: int = 100,
    transform: Any = None,
    device: Device | Backend = "auto",
    threads: int | None = None,
) -> LocateResult:
    """Find the offset of 2-D ``slave`` inside 2-D ``master``.

    mask: optional slave valid mask (e.g. alpha > 0); invalid pixels are ignored.
    decimation: coarse-pass factor (area averaging); 1 searches all offsets at full resolution.
    rank: rank_inf radius. margin: full-resolution search radius (px) around the coarse hit.
    transform: master affine geotransform (``affine.Affine``); fills ``map_bounds``.
    """
    bk = get_backend(device, threads)
    xp = bk.xp
    if master.ndim != 2 or slave.ndim != 2:
        raise ValueError(f"master {master.shape} and slave {slave.shape} must be 2-D")
    H, W = master.shape
    h, w = slave.shape
    if h > H or w > W:
        raise ValueError(f"slave {slave.shape} larger than master {master.shape}")
    if decimation < 1:
        raise ValueError("decimation must be >= 1")
    M = bk.asarray(master)
    S = bk.asarray(slave)
    valid = xp.ones((h, w), bool) if mask is None else bk.asarray(mask, bool)
    if mask is not None and tuple(valid.shape) != (h, w):
        raise ValueError(f"mask {valid.shape} does not match slave {slave.shape}")

    # Coarse pass: all offsets on decimated rank maps.
    d = decimation
    if d > 1 and h // d > 2 * rank and w // d > 2 * rank:
        Mg, _ = _block_mean(M, xp.ones((H, W), bool), d, xp)
        Sg, vg = _block_mean(S, valid, d, xp)
        Sg_rank = F.rank_inf(Sg, rank, bk)
        r, c, _ = _argmin(_score_map(F.rank_inf(Mg, rank, bk), Sg_rank, vg & (Sg_rank > 0), bk), bk)
        r0, r1 = max(0, r * d - margin), min(H, r * d + h + margin)
        c0, c1 = max(0, c * d - margin), min(W, c * d + w + margin)
    else:
        r0, r1, c0, c1 = 0, H, 0, W

    # Fine pass: full resolution inside the search window.
    S_rank = F.rank_inf(S, rank, bk)
    score = _score_map(_rank_crop(M, r0, r1, c0, c1, rank, bk), S_rank, valid & (S_rank > 0), bk)
    r, c, s = _argmin(score, bk)
    row, col = r0 + r, c0 + c
    bounds = (col, col + w, row, row + h)
    map_bounds = None
    if transform is not None:
        a, b, c0, d, e, f = tuple(transform)[:6]
        corners = [(x, y) for x in bounds[:2] for y in bounds[2:]]
        xs = [a * x + b * y + c0 for x, y in corners]
        ys = [d * x + e * y + f for x, y in corners]
        map_bounds = (min(xs), max(xs), min(ys), max(ys))
    return LocateResult(row, col, h, w, s, bounds, map_bounds)


def locate_raster(
    master_path: str | PathLike,
    slave_path: str | PathLike,
    *,
    master_band: int = 1,
    slave_band: int = 1,
    chip_output: str | PathLike | None = None,
    **kwargs: Any,
) -> LocateResult:
    """:func:`locate` on raster files; ``map_bounds`` is in the master CRS.

    The slave's dataset mask (alpha band or nodata) is used as valid mask. ``chip_output``:
    write the matching master window (all bands, georeferenced) as GeoTIFF.
    Other keyword arguments go to :func:`locate`.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="FALSE"):
        with rasterio.open(master_path) as ms:
            master = ms.read(master_band)
            transform = None if ms.transform.is_identity else ms.transform
        with rasterio.open(slave_path) as ss:
            slave = ss.read(slave_band)
            m = ss.read_masks(slave_band) > 0
        res = locate(master, slave, mask=None if m.all() else m, transform=transform, **kwargs)
        if chip_output is not None:
            with rasterio.open(master_path) as ms:
                win = Window(res.col, res.row, res.width, res.height)
                profile = ms.profile
                profile.update(
                    driver="GTiff",
                    width=res.width,
                    height=res.height,
                    transform=ms.window_transform(win),
                )
                for k in ("blockxsize", "blockysize", "tiled"):
                    profile.pop(k, None)
                with rasterio.open(chip_output, "w", **profile) as out:
                    out.write(ms.read(window=win))
    return res
