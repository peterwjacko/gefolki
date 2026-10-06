"""Raster I/O for georeferenced registration (rasterio/GDAL).

Opens rasters with ENVI sidecars found regardless of ``GDAL_DISABLE_READDIR_ON_OPEN``,
reads band metadata (wavelengths, fwhm), builds single-band registration images from band
selections, reads a master resampled onto a slave grid (using overviews), and writes
outputs on the slave grid with metadata preserved (GTiff, COG or ENVI).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "FALSE"}
ENVI_SUFFIXES = {"", ".bsq", ".bil", ".bip", ".img", ".dat", ".envi"}
# ENVI header keys GDAL derives itself; never copied between files.
_ENVI_STRUCTURAL = {
    "samples", "lines", "bands", "header_offset", "file_type", "data_type", "interleave",
    "byte_order", "map_info", "projection_info", "coordinate_system_string", "band_names",
    "data_ignore_value", "wavelength", "fwhm", "wavelength_units", "description",
}  # fmt: skip
_LUMA = (0.299, 0.587, 0.114)

BandSpec = int | str | Sequence[int] | None


@contextmanager
def gdal_env(**options: Any) -> Iterator[None]:
    """rasterio.Env that lets GDAL list directories (ENVI .hdr sidecars) plus ``options``."""
    with rasterio.Env(**GDAL_ENV, **options):
        yield


@contextmanager
def open_raster(path: str | os.PathLike, mode: str = "r", **kwargs: Any) -> Iterator[Any]:
    """``rasterio.open`` inside :func:`gdal_env`."""
    with gdal_env(), rasterio.open(path, mode, **kwargs) as ds:
        yield ds


# ---------------------------------------------------------------------------------- metadata


def _to_nm(values: list[float], units: str | None) -> np.ndarray:
    a = np.asarray(values, np.float64)
    u = (units or "").lower()
    if "micro" in u or u in ("um", "µm"):
        return a * 1000.0
    if "nano" in u or u == "nm":
        return a
    return a * 1000.0 if a.size and np.nanmax(a) < 30 else a  # unknown: guess from range


def _envi_list(s: str | None) -> list[float] | None:
    if not s:
        return None
    try:
        return [float(x) for x in s.strip().strip("{}").split(",") if x.strip()]
    except ValueError:
        return None


def _band_values(ds: Any, key: str) -> tuple[list[float] | None, str | None]:
    """Per-band numeric metadata ``key`` (band tags, then ENVI list); returns (values, units)."""
    units = ds.tags().get("wavelength_units") or ds.tags(ns="ENVI").get("wavelength_units")
    vals = []
    for b in range(1, ds.count + 1):
        t = ds.tags(b)
        try:
            vals.append(float(t[key]))
            units = t.get("wavelength_units", units)
        except (KeyError, ValueError):
            break
    if len(vals) == ds.count:
        return vals, units
    envi = _envi_list(ds.tags(ns="ENVI").get(key))
    if envi is not None and len(envi) == ds.count:
        return envi, units
    return None, units


def band_wavelengths(ds: Any) -> np.ndarray | None:
    """Band centre wavelengths in nm, or None.

    Sources in order: band tags ``wavelength`` (GDAL ENVI/GTiff convention), IMAGERY domain
    ``CENTRAL_WAVELENGTH_UM``, ENVI header ``wavelength`` list, numeric band descriptions.
    Dataset-level ``Band_N`` tags are ignored (often stale after band subsetting).
    """
    vals, units = _band_values(ds, "wavelength")
    if vals is not None:
        return _to_nm(vals, units)
    try:
        um = [float(ds.tags(b, ns="IMAGERY")["CENTRAL_WAVELENGTH_UM"]) for b in ds.indexes]
        return np.asarray(um) * 1000.0
    except (KeyError, ValueError):
        pass
    num = re.compile(r"^\s*(\d+(?:\.\d*)?)\s*([a-zA-Zµ]*)")
    parsed = [num.match(d or "") for d in ds.descriptions]
    if ds.count > 1 and all(parsed):
        return _to_nm([float(m.group(1)) for m in parsed], parsed[0].group(2) or units)
    return None


def band_fwhm(ds: Any) -> np.ndarray | None:
    """Band full width at half maximum in nm, or None."""
    vals, units = _band_values(ds, "fwhm")
    return None if vals is None else _to_nm(vals, units)


@dataclass
class RasterInfo:
    """Grid and band metadata of a raster (everything needed to write a look-alike)."""

    path: Path
    driver: str
    width: int
    height: int
    count: int
    dtype: str
    crs: Any
    transform: Affine
    nodata: float | None
    descriptions: tuple[str | None, ...]
    colorinterp: tuple[ColorInterp, ...]
    wavelengths: np.ndarray | None = None  # nm
    fwhm: np.ndarray | None = None  # nm
    wavelength_units: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    band_tags: list[dict[str, str]] = field(default_factory=list)
    envi: dict[str, str] = field(default_factory=dict)
    interleave: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def res(self) -> tuple[float, float]:
        return math.hypot(self.transform.a, self.transform.d), math.hypot(
            self.transform.b, self.transform.e
        )


def read_info(path: str | os.PathLike) -> RasterInfo:
    """Read grid and band metadata of ``path``."""
    with open_raster(path) as ds:
        envi = ds.tags(ns="ENVI") if "ENVI" in (ds.tag_namespaces() or []) else {}
        units = ds.tags().get("wavelength_units") or envi.get("wavelength_units")
        if units is None and ds.count:
            units = ds.tags(1).get("wavelength_units")
        return RasterInfo(
            path=Path(path),
            driver=ds.driver,
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtype=ds.dtypes[0],
            crs=ds.crs,
            transform=ds.transform,
            nodata=ds.nodata,
            descriptions=ds.descriptions,
            colorinterp=ds.colorinterp,
            wavelengths=band_wavelengths(ds),
            fwhm=band_fwhm(ds),
            wavelength_units=units,
            tags=ds.tags(),
            band_tags=[ds.tags(b) for b in ds.indexes],
            envi=envi,
            interleave=(ds.interleaving.name if ds.interleaving else None),
        )


# ------------------------------------------------------------------------------ band choice


@dataclass(frozen=True)
class BandSelection:
    """Bands (1-based) and weights combined into one registration image."""

    indexes: tuple[int, ...]
    weights: tuple[float, ...]
    label: str = ""

    def combine(self, data: np.ndarray) -> np.ndarray:
        """Weighted sum of ``data`` (len(indexes), H, W) -> float32 (H, W)."""
        out = np.zeros(data.shape[1:], np.float32)
        for w, b in zip(self.weights, data, strict=True):
            out += np.float32(w) * b.astype(np.float32, copy=False)
        return out


def is_rgb(info: RasterInfo) -> bool:
    """True for 3/4-band rasters whose first bands are RGB (by colour interp or layout)."""
    ci = info.colorinterp
    if {ColorInterp.red, ColorInterp.green, ColorInterp.blue} <= set(ci):
        return True
    return info.count in (3, 4) and info.wavelengths is None and info.dtype == "uint8"


def _rgb_indexes(info: RasterInfo) -> tuple[int, int, int]:
    ci = list(info.colorinterp)
    try:
        rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        return tuple(ci.index(c) + 1 for c in rgb)
    except ValueError:
        if info.count in (3, 4):
            return 1, 2, 3
        raise ValueError(f"{info.path}: 'rgb-gray' needs a 3/4-band RGB(A) raster") from None


def select_bands(info: RasterInfo, spec: BandSpec) -> BandSelection:
    """Resolve a band spec.

    - int or sequence of ints (1-based, like GDAL), or "3" / "1,2,3": mean of those bands;
    - "500-600" (nm, optional "nm" suffix): mean of bands with wavelength in the range;
    - "rgb-gray" / "luminance" / "gray": 0.299 R + 0.587 G + 0.114 B of an RGB(A) raster.
    """
    if spec is None:
        spec = 1
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("rgb-gray", "rgb-grey", "luminance", "gray", "grey"):
            return BandSelection(_rgb_indexes(info), _LUMA, "rgb-gray")
        m = re.fullmatch(r"(\d+(?:\.\d*)?)\s*-\s*(\d+(?:\.\d*)?)\s*(nm)?", s)
        if m:
            lo, hi = sorted((float(m.group(1)), float(m.group(2))))
            if info.wavelengths is None:
                raise ValueError(f"{info.path}: no wavelength metadata for band range {spec!r}")
            inside = (info.wavelengths >= lo) & (info.wavelengths <= hi)
            idx = tuple(int(i) + 1 for i in np.flatnonzero(inside))
            if not idx:
                raise ValueError(f"{info.path}: no band within {lo:g}-{hi:g} nm")
            return BandSelection(idx, (1 / len(idx),) * len(idx), f"{lo:g}-{hi:g} nm")
        try:
            spec = [int(x) for x in s.split(",")]
        except ValueError:
            raise ValueError(f"invalid band spec {spec!r}") from None
    idx = (int(spec),) if isinstance(spec, int | np.integer) else tuple(int(i) for i in spec)
    if not idx or min(idx) < 1 or max(idx) > info.count:
        raise ValueError(f"{info.path}: band indexes {idx} outside 1..{info.count}")
    return BandSelection(idx, (1 / len(idx),) * len(idx), ",".join(map(str, idx)))


# ------------------------------------------------------------------------------------ reading


def _valid(ds: Any, data: np.ndarray, idx: Sequence[int], window: Window | None) -> np.ndarray:
    """Valid-pixel mask of the selected bands: nodata, alpha or internal mask (AND)."""
    flags = ds.mask_flag_enums[idx[0] - 1]
    if MaskFlags.all_valid in flags:
        return np.ones(data.shape[1:], bool)
    if MaskFlags.nodata in flags and ds.nodata is not None:
        nd = ds.nodata
        bad = np.isnan(data) if np.isnan(nd) else data == np.asarray(nd).astype(data.dtype)
        return ~bad.any(axis=0)
    return ds.read_masks(idx[0], window=window) > 0  # alpha / per-dataset mask


def read_selection(
    path: str | os.PathLike,
    selection: BandSelection,
    *,
    window: Window | None = None,
    block_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Registration image (float32) and valid mask of ``selection`` on the raster's own grid.

    Reads row blocks so only ``len(indexes) * block_rows`` rows of raw data are in memory.
    """
    with open_raster(path) as ds:
        win = window or Window(0, 0, ds.width, ds.height)
        h, w = int(win.height), int(win.width)
        img = np.empty((h, w), np.float32)
        valid = np.empty((h, w), bool)
        idx = list(selection.indexes)
        for r in range(0, h, block_rows):
            n = min(block_rows, h - r)
            sub = Window(win.col_off, win.row_off + r, w, n)
            data = ds.read(idx, window=sub)
            img[r : r + n] = selection.combine(data)
            valid[r : r + n] = _valid(ds, data, idx, sub)
    return img, valid


def _overview_level(ds: Any, ratio: float) -> int | None:
    """Index of the coarsest overview whose decimation factor is <= ``ratio`` (None = full)."""
    best, best_f = None, 1.0
    for i, f in enumerate(ds.overviews(1)):
        if best_f < f <= ratio * (1 + 1e-6):
            best, best_f = i, f
    return best


def read_on_grid(
    path: str | os.PathLike,
    selection: BandSelection,
    crs: Any,
    transform: Affine,
    shape: tuple[int, int],
    *,
    resampling: Resampling | None = None,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Registration image and valid mask of ``path`` resampled onto a target grid.

    Picks the coarsest overview that is still at least as fine as the target, reads only the
    window covering the target extent, combines the selected bands, then reprojects.
    Resampling defaults to average when downsampling, bilinear otherwise. Pixels outside the
    source or touching invalid source pixels are invalid.
    """
    h, w = shape
    threads = threads or os.cpu_count() or 1
    dst_bounds = rasterio.transform.array_bounds(h, w, transform)
    with open_raster(path) as full:
        src_crs = full.crs or crs
        b = transform_bounds(crs, src_crs, *dst_bounds) if full.crs else dst_bounds
        src_res = min(abs(full.transform.a), abs(full.transform.e))
        ratio = min((b[2] - b[0]) / w, (b[3] - b[1]) / h) / src_res
        level = _overview_level(full, ratio)
    kwargs = {} if level is None else {"overview_level": level}
    with open_raster(path, **kwargs) as ds:
        res = min(abs(ds.transform.a), abs(ds.transform.e))
        down = min((b[2] - b[0]) / w, (b[3] - b[1]) / h) / res > 1.01
        rs = resampling or (Resampling.average if down else Resampling.bilinear)
        win = from_bounds(*b, transform=ds.transform)
        win = Window(
            math.floor(win.col_off) - 2, math.floor(win.row_off) - 2,
            math.ceil(win.width) + 5, math.ceil(win.height) + 5,
        )  # fmt: skip
        try:
            win = win.intersection(Window(0, 0, ds.width, ds.height))
        except rasterio.errors.WindowError:  # no overlap
            return np.zeros(shape, np.float32), np.zeros(shape, bool)
        win = win.round_offsets().round_lengths()
        data = ds.read(list(selection.indexes), window=win)
        src = selection.combine(data)
        valid = _valid(ds, data, selection.indexes, win)
        src_tr = ds.window_transform(win)
    del data
    img = np.zeros(shape, np.float32)
    vmask = np.zeros(shape, np.float32)
    common = dict(src_transform=src_tr, src_crs=src_crs, dst_transform=transform, dst_crs=crs)
    with gdal_env():
        src[~valid] = np.nan
        reproject(src, img, src_nodata=np.nan, dst_nodata=np.nan, resampling=rs,
                  num_threads=threads, **common)  # fmt: skip
        # 1 = invalid, 2 = valid, 0 = outside the source (dst nodata)
        reproject(valid.astype(np.float32) + 1, vmask, dst_nodata=0, resampling=rs,
                  num_threads=threads, **common)  # fmt: skip
    valid = (vmask > 1.999) & np.isfinite(img)
    img[~valid] = 0
    return img, valid


def iter_band_chunks(
    path: str | os.PathLike, chunk: int, window: Window | None = None
) -> Iterator[tuple[list[int], np.ndarray]]:
    """Yield (1-based band indexes, (k, H, W) data) for consecutive chunks of ``chunk`` bands."""
    with open_raster(path) as ds:
        for s in range(1, ds.count + 1, chunk):
            idx = list(range(s, min(s + chunk, ds.count + 1)))
            yield idx, ds.read(idx, window=window)


# ------------------------------------------------------------------------------------ writing


def output_format(path: str | os.PathLike, fmt: str | None = None) -> str:
    """'GTiff', 'COG' or 'ENVI': explicit ``fmt`` or guessed from the path suffix."""
    if fmt:
        f = {"gtiff": "GTiff", "tif": "GTiff", "geotiff": "GTiff", "cog": "COG", "envi": "ENVI"}
        try:
            return f[fmt.lower()]
        except KeyError:
            raise ValueError(f"unsupported output format {fmt!r}") from None
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        return "GTiff"
    if suffix in ENVI_SUFFIXES:
        return "ENVI"
    raise ValueError(f"cannot infer output format from {path!r}; pass a format")


def _gtiff_options(dtype: str, count: int) -> dict[str, Any]:
    kind = np.dtype(dtype).kind
    return dict(
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
        predictor=3 if kind == "f" else 2, bigtiff="IF_SAFER", num_threads="ALL_CPUS",
        interleave="band" if count > 1 else "pixel",
    )  # fmt: skip


def _fmt_list(vals: np.ndarray) -> str:
    return "{" + ", ".join(f"{v:.6f}" for v in vals) + "}"


def write_metadata(ds: Any, info: RasterInfo, driver: str) -> None:
    """Copy descriptions, tags and wavelength/fwhm metadata of ``info`` onto ``ds``."""
    if info.tags:
        ds.update_tags(**info.tags)
    units = info.wavelength_units or ("Nanometers" if info.wavelengths is not None else None)
    for b in range(1, ds.count + 1):
        if info.descriptions[b - 1]:
            ds.set_band_description(b, info.descriptions[b - 1])
        tags = dict(info.band_tags[b - 1]) if info.band_tags else {}
        if info.wavelengths is not None:
            tags.setdefault("wavelength", f"{info.wavelengths[b - 1]:.6f}")
        if info.fwhm is not None:
            tags.setdefault("fwhm", f"{info.fwhm[b - 1]:.6f}")
        if units and "wavelength" in tags:
            tags.setdefault("wavelength_units", units)
        if tags:
            ds.update_tags(b, **tags)
    if driver == "ENVI":
        envi = {k: v for k, v in info.envi.items() if k not in _ENVI_STRUCTURAL}
        if "description" in info.envi:
            envi["description"] = info.envi["description"]
        if info.wavelengths is not None:
            # Keep the source text where it describes the same bands (exact values, units).
            envi["wavelength"] = info.envi.get("wavelength") or _fmt_list(info.wavelengths)
            envi["wavelength_units"] = units
        if info.fwhm is not None:
            envi["fwhm"] = info.envi.get("fwhm") or _fmt_list(info.fwhm)
        if envi:
            ds.update_tags(ns="ENVI", **envi)


@contextmanager
def create_output(
    path: str | os.PathLike,
    info: RasterInfo,
    *,
    fmt: str | None = None,
    count: int | None = None,
    dtype: str | None = None,
    nodata: float | None = ...,  # type: ignore[assignment]
    metadata: bool = True,
) -> Iterator[Any]:
    """Open ``path`` for writing on ``info``'s grid (CRS, transform, size, dtype, nodata).

    GTiff: tiled, DEFLATE, BIGTIFF=IF_SAFER, band interleaved. COG: written as a temporary
    GTiff then converted. ENVI: interleave from suffix (.bil/.bip) or BSQ; wavelength, fwhm,
    wavelength units and data ignore value end up in the .hdr.
    """
    driver = output_format(path, fmt)
    path = Path(path)
    count = count or info.count
    dtype = dtype or info.dtype
    profile: dict[str, Any] = dict(
        width=info.width, height=info.height, count=count, dtype=dtype, crs=info.crs,
        transform=info.transform, nodata=info.nodata if nodata is ... else nodata,
    )  # fmt: skip
    target = path
    if driver == "ENVI":
        il = {".bil": "bil", ".bip": "bip"}.get(path.suffix.lower(), "bsq")
        profile.update(driver="ENVI", interleave=il)
    else:
        profile.update(driver="GTiff", **_gtiff_options(dtype, count))
        if driver == "COG":
            target = path.with_name(path.name + ".tmp.tif")
    with open_raster(target, "w", **profile) as ds:
        if metadata:
            write_metadata(ds, info, driver)
        yield ds
    if driver == "COG":
        from rasterio.shutil import copy as rio_copy

        with gdal_env():
            rio_copy(target, path, driver="COG", compress="deflate", bigtiff="IF_SAFER",
                     num_threads="ALL_CPUS", overview_resampling="average")  # fmt: skip
        target.unlink()


def write_flow(path: str | os.PathLike, u: np.ndarray, v: np.ndarray, info: RasterInfo) -> Path:
    """Write flow as a 2-band float32 GeoTIFF on ``info``'s grid (band 1 u, band 2 v, px)."""
    with create_output(
        path, info, fmt="GTiff", count=2, dtype="float32", nodata=None, metadata=False
    ) as ds:
        ds.write(u.astype(np.float32, copy=False), 1)
        ds.write(v.astype(np.float32, copy=False), 2)
        ds.set_band_description(1, "u (column shift, px)")
        ds.set_band_description(2, "v (row shift, px)")
        ds.update_tags(convention="registered(x, y) = slave(x + u, y + v)")
    return Path(path)


def read_flow(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read a flow GeoTIFF written by :func:`write_flow`; returns (u, v) float32."""
    with open_raster(path) as ds:
        return ds.read(1).astype(np.float32), ds.read(2).astype(np.float32)
