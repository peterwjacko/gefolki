"""Command-line interface: ``gefolki register|flow|warp|locate|info|presets|inspect``."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import warnings
from contextlib import contextmanager
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rasterio.errors import NotGeoreferencedWarning
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

import gefolki

from . import io
from .flow import FlowParams
from .pipeline import PRESETS, resolve_params

app = typer.Typer(
    help="GeFolki dense optical-flow coregistration of remote sensing images.",
    no_args_is_help=True,
    add_completion=False,
)
out = Console(soft_wrap=True)
err = Console(stderr=True)

Preset = Enum("Preset", {k: k for k in PRESETS}, type=str)


class Method(StrEnum):
    folki = "folki"
    efolki = "efolki"
    gefolki = "gefolki"


class Device(StrEnum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


class Format(StrEnum):
    GTiff = "GTiff"
    COG = "COG"
    ENVI = "ENVI"


class Resampling(StrEnum):
    nearest = "nearest"
    bilinear = "bilinear"
    cubic = "cubic"


# ------------------------------------------------------------------------------ helpers


def parse_radius(text: str) -> tuple[int, ...]:
    """'32,24,16,8' -> (32, 24, 16, 8); 'start:stop:step' (stop inclusive) -> range,
    e.g. '32:8:4' -> (32, 28, ..., 8)."""
    try:
        if ":" in text:
            start, stop, *rest = (int(x) for x in text.split(":"))
            step = abs(rest[0]) if rest else 1
            if len(rest) > 1 or step == 0:
                raise ValueError
            sign = -1 if stop < start else 1
            return tuple(range(start, stop + sign, sign * step))
        return tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError:
        raise typer.BadParameter(f"expected '32,24,16,8' or '32:8:4', got {text!r}") from None


def _backend_info() -> dict[str, Any]:
    return dict(gefolki.backend_info())


def _check_device(device: Device) -> None:
    if device is Device.gpu and not gefolki.gpu_available():
        reason = _backend_info().get("gpu_error") or "unknown reason"
        err.print(f"[red]Error:[/] GPU requested but unavailable: {reason}")
        raise typer.Exit(2)


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _print_json(obj: Any) -> None:
    print(json.dumps(_jsonable(obj), indent=2))


@contextmanager
def _progress(enabled: bool):
    """Yield a pipeline progress callback ``(stage, done, total)`` drawing rich bars."""
    if not enabled:
        yield None
        return
    tasks: dict[str, Any] = {}
    columns = (
        TextColumn("{task.description:<12}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=err, transient=False) as bar:

        def update(stage: str, done: int, total: int) -> None:
            for name, tid in tasks.items():  # earlier stages are finished
                if name != stage:
                    bar.update(tid, completed=bar.tasks[tid].total)
            if stage not in tasks:
                tasks[stage] = bar.add_task(stage, total=max(total, 1))
            bar.update(tasks[stage], completed=done, total=max(total, 1))

        yield update
        for tid in tasks.values():
            bar.update(tid, completed=bar.tasks[tid].total)


@contextmanager
def _errors():
    """Turn library errors into a red message and exit code 1."""
    try:
        yield
    except (typer.Exit, typer.Abort, typer.BadParameter):
        raise
    except Exception as e:  # rasterio, ValueError, RuntimeError, ...
        err.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1) from None


def _flow_params(
    master: Path,
    slave: Path,
    preset: Preset | None,
    method: Method | None,
    overrides: dict[str, Any],
) -> FlowParams | None:
    """Explicit params when any solver option is given (on top of preset and method)."""
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides:
        return None
    if preset is not None:
        base = PRESETS[preset.value]
    elif io.is_rgb(io.read_info(master)) and io.read_info(slave).wavelengths is not None:
        base = PRESETS["hyperspectral-rgb"]  # same auto choice as estimate_raster_flow
    else:
        base = FlowParams(contrast_adapt=True)
    base = resolve_params(method.value if method else None, base)
    return dataclasses.replace(base, **overrides)


def _fmt_params(p: FlowParams) -> str:
    r = ",".join(map(str, p.radius))
    return (
        f"levels={p.levels} radius={r} iterations={p.iterations} rank={p.rank} "
        f"contrast_adapt={p.contrast_adapt}"
    )


def _fmt_timings(t: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.2f}s" for k, v in t.items())


def _version(value: bool) -> None:
    if value:
        print(f"gefolki {gefolki.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """GeFolki dense optical-flow coregistration of remote sensing images."""
    # Unreferenced inputs (e.g. PNG chips for locate) are expected; the warning is noise.
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# ------------------------------------------------------------------------------ options

ExistingFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, show_default=False)]
OutFile = Annotated[Path, typer.Argument(dir_okay=False, show_default=False)]
PresetOpt = Annotated[Preset | None, typer.Option(help="Parameter preset (see `gefolki presets`).")]
MethodOpt = Annotated[
    Method | None, typer.Option(help="Variant: folki (no rank), efolki, gefolki (CLAHE test).")
]
LevelsOpt = Annotated[int | None, typer.Option(min=0, help="Pyramid levels.")]
RadiusOpt = Annotated[
    str | None, typer.Option(help="Window radii coarse to fine: '32,24,16,8' or '32:8:4'.")
]
ItersOpt = Annotated[int | None, typer.Option(min=1, help="Solver iterations per radius.")]
RankOpt = Annotated[int | None, typer.Option(min=0, help="Rank filter radius (0 = none).")]
ContrastOpt = Annotated[
    bool | None,
    typer.Option("--contrast-adapt/--no-contrast-adapt", help="GeFolki contrast adaptation."),
]
MasterBandsOpt = Annotated[
    str | None,
    typer.Option(help="Master bands: '1', '1,2,3', '500-600' (nm) or 'rgb-gray'. Default auto."),
]
SlaveBandsOpt = Annotated[
    str | None,
    typer.Option(help="Slave bands: '1', '1,2,3', '500-600' (nm) or 'rgb-gray'. Default auto."),
]
DeviceOpt = Annotated[Device, typer.Option(help="Compute device.")]
ThreadsOpt = Annotated[int | None, typer.Option(min=1, help="CPU threads (default: all).")]
TileOpt = Annotated[int | None, typer.Option(min=64, help="Flow tile size px (default auto).")]
FormatOpt = Annotated[
    Format | None, typer.Option("--format", help="Output format (default from suffix).")
]
ResamplingOpt = Annotated[Resampling, typer.Option(help="Slave band interpolation.")]
QuietOpt = Annotated[
    bool | None, typer.Option("--quiet/--verbose", "-q/-v", help="Less / more output.")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print the result as JSON.")]


def _solver(
    master: Path,
    slave: Path,
    preset: Preset | None,
    method: Method | None,
    levels: int | None,
    radius: str | None,
    iterations: int | None,
    rank: int | None,
    contrast_adapt: bool | None,
) -> dict[str, Any]:
    """Keyword arguments for estimate_raster_flow / register from solver options."""
    overrides = dict(
        levels=levels,
        radius=parse_radius(radius) if radius else None,
        iterations=iterations,
        rank=rank,
        contrast_adapt=contrast_adapt,
    )
    params = _flow_params(master, slave, preset, method, overrides)
    if params is not None:
        return {"params": params}
    return {"preset": preset.value if preset else None, "method": method.value if method else None}


# ------------------------------------------------------------------------------ commands


@app.command()
def register(
    master: ExistingFile,
    slave: ExistingFile,
    output: OutFile,
    preset: PresetOpt = None,
    method: MethodOpt = None,
    levels: LevelsOpt = None,
    radius: RadiusOpt = None,
    iterations: ItersOpt = None,
    rank: RankOpt = None,
    contrast_adapt: ContrastOpt = None,
    master_bands: MasterBandsOpt = None,
    slave_bands: SlaveBandsOpt = None,
    device: DeviceOpt = Device.auto,
    threads: ThreadsOpt = None,
    tile_size: TileOpt = None,
    flow_output: Annotated[
        Path | None, typer.Option(dir_okay=False, help="Also write flow (2-band GeoTIFF).")
    ] = None,
    fmt: FormatOpt = None,
    resampling: ResamplingOpt = Resampling.bilinear,
    quiet: QuietOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Register SLAVE to MASTER; write all warped slave bands to OUTPUT on the slave grid."""
    _check_device(device)
    with _errors():
        kw = _solver(master, slave, preset, method, levels, radius, iterations, rank,
                     contrast_adapt)  # fmt: skip
        with _progress(not quiet and not as_json) as progress:
            res = gefolki.register(
                master, slave, output, master_bands=master_bands, slave_bands=slave_bands,
                device=device.value, threads=threads, tile_size=tile_size,
                flow_output=flow_output, output_format=fmt.value if fmt else None,
                resampling=resampling.value, progress=progress, **kw,
            )  # fmt: skip
    if as_json:
        _print_json(res)
        return
    if quiet:
        return
    s = res.flow_stats
    out.print(f"wrote {res.output}" + (f" (flow: {res.flow_path})" if res.flow_path else ""))
    out.print(f"flow magnitude px: median={s['median']:.2f} p95={s['p95']:.2f}")
    out.print(f"total {res.timings['total']:.2f}s")
    if quiet is False:
        out.print(f"params: {_fmt_params(res.params)}")
        out.print(f"timings: {_fmt_timings(res.timings)}")


@app.command()
def flow(
    master: ExistingFile,
    slave: ExistingFile,
    flow_output: OutFile,
    preset: PresetOpt = None,
    method: MethodOpt = None,
    levels: LevelsOpt = None,
    radius: RadiusOpt = None,
    iterations: ItersOpt = None,
    rank: RankOpt = None,
    contrast_adapt: ContrastOpt = None,
    master_bands: MasterBandsOpt = None,
    slave_bands: SlaveBandsOpt = None,
    device: DeviceOpt = Device.auto,
    threads: ThreadsOpt = None,
    tile_size: TileOpt = None,
    quiet: QuietOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Estimate the flow only and write it to FLOW_OUTPUT (band 1 u, band 2 v, px)."""
    _check_device(device)
    with _errors():
        kw = _solver(master, slave, preset, method, levels, radius, iterations, rank,
                     contrast_adapt)  # fmt: skip
        with _progress(not quiet and not as_json) as progress:
            rf = gefolki.estimate_raster_flow(
                master, slave, master_bands=master_bands, slave_bands=slave_bands,
                device=device.value, threads=threads, tile_size=tile_size, progress=progress,
                **kw,
            )  # fmt: skip
        path = io.write_flow(flow_output, rf.u, rf.v, rf.grid)
    stats = rf.stats()
    if rf.valid.any():
        stats["mean_u"] = float(rf.u[rf.valid].mean())
        stats["mean_v"] = float(rf.v[rf.valid].mean())
    stats["valid_fraction"] = float(rf.valid.mean())
    if as_json:
        _print_json({"flow_path": path, "flow_stats": stats, "timings": rf.timings,
                     "params": rf.params})  # fmt: skip
        return
    if quiet:
        return
    out.print(f"wrote {path}")
    out.print("  ".join(f"{k}={v:.3f}" for k, v in stats.items()))
    if quiet is False:
        out.print(f"params: {_fmt_params(rf.params)}")
        out.print(f"timings: {_fmt_timings(rf.timings)}")


@app.command()
def warp(
    slave: ExistingFile,
    flow: ExistingFile,
    output: OutFile,
    resampling: ResamplingOpt = Resampling.bilinear,
    device: DeviceOpt = Device.auto,
    threads: ThreadsOpt = None,
    fmt: FormatOpt = None,
    quiet: QuietOpt = None,
) -> None:
    """Warp every band of SLAVE by FLOW (from `gefolki flow`) and write OUTPUT."""
    _check_device(device)
    with _errors(), _progress(not quiet) as progress:
        if (n := io.read_info(flow).count) != 2:
            raise ValueError(f"flow must be a 2-band (u, v) raster, {flow} has {n} bands")
        path = gefolki.apply_flow(
            slave, flow, output, resampling=resampling.value, device=device.value,
            threads=threads, output_format=fmt.value if fmt else None, progress=progress,
        )  # fmt: skip
    if not quiet:
        out.print(f"wrote {path}")


@app.command()
def locate(
    master: ExistingFile,
    slave: ExistingFile,
    master_band: Annotated[int, typer.Option(min=1, help="Master band (1-based).")] = 1,
    slave_band: Annotated[int, typer.Option(min=1, help="Slave band (1-based).")] = 1,
    chip_output: Annotated[
        Path | None, typer.Option(dir_okay=False, help="Write the matching master window.")
    ] = None,
    decimation: Annotated[int, typer.Option(min=1, help="Coarse-pass decimation.")] = 8,
    rank: Annotated[int, typer.Option(min=0, help="rank_inf radius.")] = 3,
    margin: Annotated[int, typer.Option(min=0, help="Fine search radius px.")] = 100,
    device: DeviceOpt = Device.auto,
    threads: ThreadsOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Find where the small SLAVE image lies inside MASTER."""
    _check_device(device)
    with _errors():
        res = gefolki.locate_raster(
            master, slave, master_band=master_band, slave_band=slave_band,
            chip_output=chip_output, decimation=decimation, rank=rank, margin=margin,
            device=device.value, threads=threads,
        )  # fmt: skip
    if as_json:
        _print_json(res)
        return
    xmin, xmax, ymin, ymax = res.bounds
    out.print(f"row={res.row} col={res.col} size={res.height}x{res.width} score={res.score:.4g}")
    out.print(f"pixel bounds: x {xmin}:{xmax}  y {ymin}:{ymax}")
    if res.map_bounds is not None:
        x0, x1, y0, y1 = res.map_bounds
        out.print(f"map bounds:   x {x0:.6f}..{x1:.6f}  y {y0:.6f}..{y1:.6f}")
    if chip_output is not None:
        out.print(f"wrote {chip_output}")


@app.command()
def info(as_json: JsonOpt = False) -> None:
    """Show backend (GPU, threads) and library versions."""
    import rasterio

    backend = _backend_info()  # includes numpy, scipy, scikit-image, numba, cucim, cupy
    versions = {
        "gefolki": gefolki.__version__,
        "python": ".".join(map(str, sys.version_info[:3])),
        "rasterio": rasterio.__version__,
        "GDAL": rasterio.__gdal_version__,
        "typer": typer.__version__,
    }
    data = {"backend": backend, "versions": versions}
    if as_json:
        _print_json(data)
        return
    table = Table(show_header=False, box=None)
    for k, v in (versions | backend).items():
        table.add_row(k, "[dim]-[/]" if v is None else str(v))
    out.print(table)


@app.command()
def presets(as_json: JsonOpt = False) -> None:
    """List parameter presets."""
    if as_json:
        _print_json(PRESETS)
        return
    table = Table("preset", "levels", "radius", "iterations", "rank", "contrast_adapt")
    for col in table.columns:
        col.no_wrap = col.header != "radius"
    table.columns[2].overflow = "fold"
    for name, p in PRESETS.items():
        table.add_row(name, str(p.levels), ", ".join(map(str, p.radius)), str(p.iterations),
                      str(p.rank), str(p.contrast_adapt))  # fmt: skip
    out.print(table)


@app.command()
def inspect(
    raster: ExistingFile,
    bands: Annotated[bool, typer.Option("--bands", help="List every band.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Show size, bands, dtype, CRS, resolution, nodata and wavelengths of RASTER."""
    with _errors():
        ri = io.read_info(raster)
    w = ri.wavelengths
    summary: dict[str, Any] = {
        "path": ri.path,
        "driver": ri.driver,
        "width": ri.width,
        "height": ri.height,
        "count": ri.count,
        "dtype": ri.dtype,
        "crs": ri.crs.to_string() if ri.crs else None,
        "resolution": list(ri.res),
        "bounds": list(_bounds(ri)),
        "nodata": ri.nodata,
        "interleave": ri.interleave,
        "rgb": io.is_rgb(ri),
        "wavelengths_nm": None,
    }
    if w is not None:
        summary["wavelengths_nm"] = {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean_step": float(np.diff(w).mean()) if len(w) > 1 else None,
        }
    band_rows = [
        {
            "band": i + 1,
            "description": ri.descriptions[i],
            "colorinterp": ri.colorinterp[i].name,
            "wavelength_nm": None if w is None else float(w[i]),
            "fwhm_nm": None if ri.fwhm is None else float(ri.fwhm[i]),
        }
        for i in range(ri.count)
    ]
    if as_json:
        _print_json({**summary, "bands": band_rows})
        return
    table = Table(show_header=False, box=None)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for k, v in summary.items():
        if k == "wavelengths_nm" and v is not None:
            step = f", step ~{v['mean_step']:.2f}" if v["mean_step"] is not None else ""
            v = f"{v['min']:.1f}-{v['max']:.1f} nm{step}"
        elif k in ("resolution", "bounds"):
            v = ", ".join(f"{x:.6g}" for x in v)
        table.add_row(k, str(v))
    out.print(table)
    if bands:
        bt = Table("band", "description", "colorinterp", "wavelength_nm", "fwhm_nm")
        for r in band_rows:
            bt.add_row(*("" if v is None else str(v) for v in r.values()))
        out.print(bt)


def _bounds(ri: io.RasterInfo) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) in map units."""
    from rasterio.transform import array_bounds

    b = array_bounds(ri.height, ri.width, ri.transform)  # (west, south, east, north)
    return b[0], b[1], b[2], b[3]
