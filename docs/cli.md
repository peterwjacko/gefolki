# Command line reference

`gefolki` is installed with the package (`python -m gefolki` works too).
`gefolki --help` and `gefolki COMMAND --help` list every option.

| command | does |
|---|---|
| [`register MASTER SLAVE OUTPUT`](#register) | flow + warp of all slave bands; output on the slave grid |
| [`flow MASTER SLAVE FLOW_OUTPUT`](#flow) | flow only: 2-band float32 GeoTIFF (u, v in px) |
| [`warp SLAVE FLOW OUTPUT`](#warp) | warp every slave band by a flow from `gefolki flow` |
| [`locate MASTER SLAVE`](#locate) | find where a small SLAVE image lies inside MASTER |
| [`inspect RASTER`](#inspect) | size, bands, dtype, CRS, resolution, nodata, wavelengths |
| [`presets`](#presets) | list parameter presets |
| [`info`](#info) | backends (GPU, threads) and library versions |

Inputs are any raster GDAL reads (GeoTIFF, COG, ENVI, PNG, ...). The master is resampled
onto the slave grid, so the two can differ in CRS, resolution and extent; the output
keeps the slave's grid. See the [user guide](user-guide.md) for the method.

## register

```bash
gefolki register MASTER SLAVE OUTPUT [options]
```

Estimates the flow between MASTER and SLAVE on the slave grid, warps every SLAVE band and
writes OUTPUT with the slave's grid, CRS, dtype, nodata, band names and wavelengths.

```bash
# hyperspectral line onto an RGB ortho (auto: hyperspectral-rgb preset, 500-600 nm vs luminance)
gefolki register rgb_ortho.tif line.bsq line_reg.bsq --flow-output line_flow.tif

# preset and device
gefolki register optical.tif radar.tif radar_reg.tif --preset optical-sar --device gpu

# explicit solver options, on top of --preset / --method
gefolki register a.tif b.tif out.tif --method efolki --levels 5 --radius 32:8:8 \
    --iterations 2 --rank 4 --no-contrast-adapt --slave-bands 2 --format COG -v
```

### Flow options (`register` and `flow`)

| option | values | meaning |
|---|---|---|
| `--preset` | `hyperspectral-rgb`, `sar-sar`, `lidar-sar`, `optical-sar`, `optical-optical` | parameter preset (see [`presets`](#presets)) |
| `--method` | `folki`, `efolki`, `gefolki` | variant: no rank filter / rank filter / rank filter and contrast adaptation |
| `--levels` | int ≥ 0 | pyramid levels |
| `--radius` | `32,24,16,8` or `start:stop:step` | window radii, coarse to fine; `32:8:4` = 32, 28, ..., 8 (stop included) |
| `--iterations` | int ≥ 1 | solver iterations per radius |
| `--rank` | int ≥ 0 | rank filter radius (0 = none) |
| `--contrast-adapt` / `--no-contrast-adapt` | | GeFolki contrast adaptation |
| `--master-bands`, `--slave-bands` | `1`, `1,2,3`, `500-600`, `rgb-gray` | bands used for the flow; default auto (see [choosing bands](user-guide.md#choosing-bands)) |
| `--device` | `auto` (default), `cpu`, `gpu` | compute device |
| `--threads` | int ≥ 1 | CPU threads (default: all) |
| `--tile-size` | int ≥ 64 | flow tile size in px (default: tile only when the image does not fit in memory) |
| `-q` / `-v` | | less / more output |
| `--json` | | print the result as JSON |

Parameters resolve in this order: the preset (or the defaults: levels 6, radius 32..8
step 4, iterations 2, rank 4, contrast adaptation on), then `--method`, then single
options (`--levels`, `--radius`, ...).

### Output options (`register` only)

| option | values | meaning |
|---|---|---|
| `--flow-output` | file | also write the flow (2-band float32 GeoTIFF: u, v in px) |
| `--format` | `GTiff`, `COG`, `ENVI` | output format; default from the suffix: `.tif` GTiff; `.bsq`, `.bil`, `.img`, `.dat` or none ENVI |
| `--resampling` | `nearest`, `bilinear` (default), `cubic` | slave band interpolation |

ENVI output interleave follows the suffix: `.bil`, `.bip`, else BSQ.

### Output

Default: the output path, flow statistics (median and 95th percentile of the flow
magnitude, px) and timings, with progress bars on a terminal. `--json` prints:

```json
{
  "output": "radar_reg.tif",
  "flow_path": "flow.tif",
  "flow_stats": {"median": 8.18, "p95": 10.07},
  "timings": {"read_master": 0.03, "read_slave": 0.005, "flow": 0.23, "warp_write": 0.03, "total": 0.51},
  "params": {"levels": 6, "radius": [32, 28, 24, 20, 16, 12, 8], "iterations": 2, "rank": 4, "contrast_adapt": true}
}
```

## flow

```bash
gefolki flow MASTER SLAVE FLOW_OUTPUT [flow options]
```

Like `register` without the warp. FLOW_OUTPUT is a 2-band float32 GeoTIFF on the slave
grid: band 1 `u` (column shift), band 2 `v` (row shift), in pixels, with
`slave(x + u, y + v) ≈ master(x, y)`.

```bash
gefolki flow master.tif slave.tif flow.tif --preset optical-sar --json
```

## warp

```bash
gefolki warp SLAVE FLOW OUTPUT [--resampling nearest|bilinear|cubic] [--format GTiff|COG|ENVI]
             [--device auto|cpu|gpu] [--threads N] [-q|-v]
```

Warps every SLAVE band by FLOW (from `gefolki flow` or `--flow-output`) and writes OUTPUT
on the slave grid. FLOW must have the slave's shape. Use it to inspect a flow before
warping, or to apply one flow to several rasters on the same grid.

```bash
gefolki warp slave.tif flow.tif slave_reg.tif --resampling cubic
gefolki warp slave_dem.tif flow.tif slave_dem_reg.tif --resampling nearest
```

## locate

```bash
gefolki locate MASTER SLAVE [options]
```

Finds where the small SLAVE image lies inside the large MASTER (e.g. an airborne SAR chip
in a satellite scene), by an exhaustive search on rank-filtered images: first on
decimated images, then at full resolution around the coarse hit.

| option | default | meaning |
|---|---|---|
| `--master-band`, `--slave-band` | 1 | band to use (1-based) |
| `--chip-output` | | write the matching master window (all bands, georeferenced GeoTIFF) |
| `--decimation` | 8 | coarse-pass decimation factor (1 = full-resolution search everywhere) |
| `--rank` | 3 | rank filter radius |
| `--margin` | 100 | full-resolution search radius around the coarse hit, px |
| `--device`, `--threads`, `--json` | | as above |

The slave's alpha band or nodata marks its invalid pixels.

```bash
$ gefolki locate S1_Jacksonville_GEE.tif JacksonvilleNavalAirStation_sandiaKu.png
row=852 col=1112 size=360x806 score=296.9
pixel bounds: x 1112:1918  y 852:1212
map bounds:   x -81.730162..-81.657758  y 30.211241..30.243581
```

`row`, `col`: master pixel of the slave's top-left corner; `score`: mean squared rank
difference (lower is better); map bounds in the master CRS when it is georeferenced.

## inspect

```bash
gefolki inspect RASTER [--bands] [--json]
```

Prints path, driver, size, band count, dtype, CRS, resolution, bounds, nodata,
interleave, whether it is RGB, and wavelengths (nm, from ENVI headers or GDAL metadata).
`--bands` lists every band: description, colour interpretation, wavelength and FWHM.

## presets

```bash
gefolki presets [--json]
```

| preset | levels | radius | iterations | rank | contrast_adapt |
|---|---|---|---|---|---|
| `hyperspectral-rgb` | 5 | 32, 24, 16, 8 | 2 | 4 | False |
| `sar-sar` | 3 | 32 | 2 | 4 | False |
| `lidar-sar` | 6 | 32, 28, 24, 20, 16, 12, 8 | 2 | 4 | False |
| `optical-sar` | 6 | 32, 28, 24, 20, 16, 12, 8 | 2 | 4 | True |
| `optical-optical` | 5 | 16, 8 | 4 | 4 | False |

## info

```bash
gefolki info [--json]
```

Shows the package and library versions (Python, NumPy, SciPy, scikit-image, rasterio,
GDAL, numba, CuPy, cuCIM), the default device, CPU threads and, if CuPy is installed, the
GPU, CUDA runtime and driver versions, and why a GPU was rejected (`gpu_error`). Run it
first when `--device gpu` fails.

## Exit codes

0 on success; 2 for invalid command-line arguments (missing file, bad option value) or
`--device gpu` without a usable GPU; 1 for errors while processing (bad band spec,
unreadable raster, ...), printed as `Error: ...`.
