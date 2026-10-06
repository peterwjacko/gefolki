"""Georeferenced registration: master resampled onto the slave grid, flow, warp all bands.

``register(master, slave, output)`` keeps the slave's grid, CRS, dtype, nodata and band
metadata; only pixel content moves. Flow convention: registered(x, y) = slave(x + u, y + v).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from . import io
from .backend import Backend, Device, get_backend
from .flow import FlowParams
from .tiling import estimate_flow_tiled
from .warp import warp

Method = Literal["folki", "efolki", "gefolki"]
Progress = Callable[[str, int, int], None]

PRESETS: dict[str, FlowParams] = {
    # Airborne hyperspectral (green band average) onto an RGB orthomosaic (luminance).
    # Liffey FX10 lines: shifts up to ~17 px globally / ~28 px locally at 0.4 m need >= 5
    # levels; rank-based EFolki matches better than GeFolki here (no contrast inversion
    # between green radiance and RGB luminance; CLAHE adds noise). See tests/test_pipeline.
    "hyperspectral-rgb": FlowParams(levels=5, radius=(32, 24, 16, 8), iterations=2, rank=4),
    # From the GeFolki manual / Brigot et al. 2016.
    "sar-sar": FlowParams(levels=3, radius=(32,), iterations=2, rank=4, contrast_adapt=False),
    "lidar-sar": FlowParams(levels=6, iterations=2, rank=4, contrast_adapt=False),
    "optical-sar": FlowParams(levels=6, iterations=2, rank=4, contrast_adapt=True),
    "optical-optical": FlowParams(levels=5, radius=(16, 8), iterations=4, rank=4),
}
_ORDER = {"nearest": 0, "bilinear": 1, "cubic": 3}
_CHUNK_BYTES = 1 << 30  # target raw bytes per band chunk when warping
_GDAL_CACHE_MB = 256


@dataclass
class RasterFlow:
    """Flow on the slave grid. ``grid`` describes the slave (CRS, transform, bands)."""

    u: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    grid: io.RasterInfo
    params: FlowParams
    timings: dict[str, float] = field(default_factory=dict)

    def stats(self) -> dict[str, float]:
        """Median and 95th percentile flow magnitude (px) over valid pixels."""
        mag = np.hypot(self.u, self.v)[self.valid]
        if not mag.size:
            return {"median": float("nan"), "p95": float("nan")}
        return {"median": float(np.median(mag)), "p95": float(np.percentile(mag, 95))}


@dataclass
class RegistrationResult:
    output: Path
    flow_path: Path | None
    flow_stats: dict[str, float]
    timings: dict[str, float]
    params: FlowParams


def resolve_params(
    method: Method | None = None, params: FlowParams | None = None, preset: str | None = None
) -> FlowParams:
    """Flow parameters from ``params`` or ``preset`` (else manual defaults), then ``method``
    overrides the variant flags: folki -> rank 0; efolki -> no contrast adaptation;
    gefolki -> contrast adaptation (rank 4 if it was 0)."""
    if params is None:
        params = PRESETS[preset] if preset else FlowParams(contrast_adapt=True)
    if method is None:
        return params
    if method == "folki":
        return replace(params, rank=0, contrast_adapt=False)
    rank = params.rank or 4
    if method == "efolki":
        return replace(params, rank=rank, contrast_adapt=False)
    if method == "gefolki":
        return replace(params, rank=rank, contrast_adapt=True)
    raise ValueError(f"method must be folki, efolki or gefolki, got {method!r}")


def _auto_bands(master: io.RasterInfo, slave: io.RasterInfo) -> tuple[io.BandSpec, io.BandSpec]:
    m = "rgb-gray" if io.is_rgb(master) else 1
    s: io.BandSpec = 1
    if io.is_rgb(master) and slave.wavelengths is not None:
        w = slave.wavelengths
        s = "500-600" if ((w >= 500) & (w <= 600)).any() else 1
    return m, s


def estimate_raster_flow(
    master: str | os.PathLike,
    slave: str | os.PathLike,
    *,
    master_bands: io.BandSpec = None,
    slave_bands: io.BandSpec = None,
    method: Method | None = None,
    params: FlowParams | None = None,
    preset: str | None = None,
    device: Device | Backend = "auto",
    threads: int | None = None,
    tile_size: int | None = None,
    progress: Progress | None = None,
) -> RasterFlow:
    """Flow between two rasters on the slave grid (no warping).

    master_bands / slave_bands: see :func:`gefolki.io.select_bands`; None = auto (master
    'rgb-gray' if RGB else band 1; slave mean of 500-600 nm if it has wavelengths and the
    master is RGB, else band 1). Params: ``params``, else ``preset``, else
    PRESETS['hyperspectral-rgb'] for that RGB/hyperspectral case, else GeFolki defaults;
    ``method`` overrides the variant (see :func:`resolve_params`).
    """
    t0 = time.perf_counter()
    mi, si = io.read_info(master), io.read_info(slave)
    auto_m, auto_s = _auto_bands(mi, si)
    msel = io.select_bands(mi, auto_m if master_bands is None else master_bands)
    ssel = io.select_bands(si, auto_s if slave_bands is None else slave_bands)
    if params is None and preset is None and io.is_rgb(mi) and si.wavelengths is not None:
        preset = "hyperspectral-rgb"
    p = resolve_params(method, params, preset)
    bk = get_backend(device, threads)
    timings: dict[str, float] = {}

    if progress:
        progress("read_master", 0, 1)
    t = time.perf_counter()
    mimg, mvalid = io.read_on_grid(
        master, msel, si.crs, si.transform, si.shape, threads=threads or os.cpu_count()
    )
    timings["read_master"] = time.perf_counter() - t
    if progress:
        progress("read_slave", 0, 1)
    t = time.perf_counter()
    simg, svalid = io.read_selection(slave, ssel)
    timings["read_slave"] = time.perf_counter() - t

    t = time.perf_counter()
    valid = mvalid & svalid
    u, v = estimate_flow_tiled(
        mimg, simg, p, mask=valid, device=bk, tile_size=tile_size, progress=progress
    )
    timings["flow"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    return RasterFlow(u, v, valid, si, p, timings)


def _chunk_bands(info: io.RasterInfo) -> int:
    per_band = info.width * info.height * np.dtype(info.dtype).itemsize
    return int(np.clip(_CHUNK_BYTES // max(per_band, 1), 1, 16))


def _warp_to(
    slave: str | os.PathLike,
    u: np.ndarray,
    v: np.ndarray,
    output: str | os.PathLike,
    info: io.RasterInfo,
    *,
    order: int,
    bk: Backend,
    output_format: str | None,
    progress: Progress | None,
) -> None:
    """Warp all slave bands in chunks; read-ahead and writing overlap with warping."""
    chunk = _chunk_bands(info)
    total = info.count
    if bk.is_gpu:  # keep the flow on the device across chunks
        u, v = bk.asarray(u), bk.asarray(v)
    with (
        # GDAL's default block cache (5% of RAM) fills with dirty output blocks; a small
        # cache keeps memory at ~3 chunks without slowing the band-interleaved writes.
        io.gdal_env(GDAL_CACHEMAX=_GDAL_CACHE_MB),
        io.create_output(output, info, fmt=output_format) as dst,
        ThreadPoolExecutor(1) as reader,
        ThreadPoolExecutor(1) as writer,
    ):
        chunks = io.iter_band_chunks(slave, chunk)  # only ever advanced in the reader thread
        try:
            nxt = reader.submit(next, chunks, None)
            pending = None
            done = 0
            while (item := nxt.result()) is not None:
                nxt = reader.submit(next, chunks, None)
                idx, data = item
                out = warp(data, u, v, order=order, nodata=info.nodata, device=bk)
                if pending is not None:
                    pending.result()
                pending = writer.submit(dst.write, out, idx)
                done += len(idx)
                if progress:
                    progress("warp", done, total)
            if pending is not None:
                pending.result()
        finally:
            reader.submit(chunks.close).result()


def apply_flow(
    slave: str | os.PathLike,
    flow: str | os.PathLike | tuple[np.ndarray, np.ndarray],
    output: str | os.PathLike,
    *,
    resampling: str = "bilinear",
    device: Device | Backend = "auto",
    threads: int | None = None,
    output_format: str | None = None,
    progress: Progress | None = None,
) -> Path:
    """Warp every band of ``slave`` by a flow (GeoTIFF from :func:`gefolki.io.write_flow`,
    or (u, v) arrays on the slave grid) and write ``output`` on the slave grid."""
    info = io.read_info(slave)
    u, v = io.read_flow(flow) if isinstance(flow, str | os.PathLike) else flow
    if u.shape != info.shape:
        raise ValueError(f"flow shape {u.shape} does not match slave {info.shape}")
    try:
        order = _ORDER[resampling]
    except KeyError:
        raise ValueError(f"resampling must be one of {list(_ORDER)}") from None
    bk = get_backend(device, threads)
    _warp_to(slave, u, v, output, info, order=order, bk=bk, output_format=output_format,
             progress=progress)  # fmt: skip
    return Path(output)


def register(
    master: str | os.PathLike,
    slave: str | os.PathLike,
    output: str | os.PathLike,
    *,
    master_bands: io.BandSpec = None,
    slave_bands: io.BandSpec = None,
    method: Method | None = None,
    params: FlowParams | None = None,
    preset: str | None = None,
    device: Device | Backend = "auto",
    threads: int | None = None,
    tile_size: int | None = None,
    flow_output: str | os.PathLike | None = None,
    output_format: str | None = None,
    resampling: str = "bilinear",
    progress: Progress | None = None,
) -> RegistrationResult:
    """Register ``slave`` to ``master`` and write all warped slave bands to ``output``.

    The master is resampled onto the slave grid (overviews, average when downsampling).
    Output keeps the slave grid/CRS/dtype/nodata/band metadata; format from
    ``output_format`` ('GTiff', 'COG', 'ENVI') or the path suffix (.tif -> GTiff; none,
    .bsq/.bil/.img/.dat -> ENVI). ``flow_output``: optional 2-band float32 GeoTIFF (u, v px).
    ``resampling``: slave band interpolation, nearest/bilinear/cubic. tile_size: see
    :func:`gefolki.tiling.estimate_flow_tiled`. Other arguments: :func:`estimate_raster_flow`.
    """
    t0 = time.perf_counter()
    bk = get_backend(device, threads)
    rf = estimate_raster_flow(
        master, slave, master_bands=master_bands, slave_bands=slave_bands, method=method,
        params=params, preset=preset, device=bk, threads=threads, tile_size=tile_size,
        progress=progress,
    )  # fmt: skip
    timings = dict(rf.timings)
    flow_path = None
    if flow_output is not None:
        flow_path = io.write_flow(flow_output, rf.u, rf.v, rf.grid)
    t = time.perf_counter()
    apply_flow(slave, (rf.u, rf.v), output, resampling=resampling, device=bk,
               output_format=output_format, progress=progress)  # fmt: skip
    timings["warp_write"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    return RegistrationResult(Path(output), flow_path, rf.stats(), timings, rf.params)
