# gefolki

Dense optical-flow coregistration of heterogeneous remote sensing images: SAR/SAR,
optical/SAR, LIDAR/SAR, hyperspectral/optical and so on. A Python library and a `gefolki`
command line tool, with optional GPU (CuPy) and multithreaded CPU backends.

GeFolki was developed by ONERA in the framework of the MEDUSA project, first for SAR/SAR
coregistration, then for other cases of remote sensing image coregistration. This
repository is a rewrite of the original Matlab/Python code as an installable package
(numpy, scipy, scikit-image, rasterio/GDAL, typer). It adds georeferenced, multi-band
registration of rasters (GeoTIFF, COG, ENVI), tiling for very large images and pattern
localisation.

## Citations for {ge ,e, _}Folki algorithms :

If you are using the result of GeFolki in your project, we kindly ask you to cite :

- for SAR/SAR coregistration (interferometry, change detection, and so on)

Aurélien Plyer, Elise Colin-Koeniguer, Flora Weissgerber, "A New Coregistration Algorithm for Recent Applications on Urban SAR Images", Geoscience and Remote Sensing Letters, IEEE , vol.12, no.11, pp. 2198 – 2202, nov 2015

- for other geosensing cases (optics/SAR, optics/hyperspectral, LIDAR/SAR, etc.)

Guillaume Brigot, Elise Colin-Koeniguer, Aurélien Plyer, Fabrice Janez, "Adaptation and Evaluation of an Optical Flow Method Applied to Coregistration of Forest Remote Sensing Images", IEEE Journal of Selected Topics in Applied Earth Observations ans Remote Sensing, Volume 9, Issue 7, July 2016

- for measurement applications (ex : PIV, material deformation, ...) : 

Champagnat, F., Plyer, A., Le Besnerais, G., Leclaire, B., & Le Sant, Y. (2009, August). How to calculate dense piv vector fields at video rate. In Proceedings of 8th International Symposium on Particle Image Velocimetry-PIV09 (Vol. 11, pp. 15-20).

- for computer vision (ex : robotics) :

Plyer, A., Le Besnerais, G., & Champagnat, F. (2016). Massively parallel Lucas Kanade optical flow for real-time video processing applications. Journal of Real-Time Image Processing, 11(4), 713-730.

The original user manual (`manual_gefolki_english.pdf`) and the coregistration course
(`COREGISTRATION.pdf`) are kept in this repository.

## Install

Python 3.12 or newer.

```bash
# uv
uv add "gefolki[gpu,numba] @ git+https://github.com/peterwjacko/gefolki"
# pip
pip install "gefolki[gpu,numba] @ git+https://github.com/peterwjacko/gefolki"
# from a checkout
uv sync --extra gpu --extra numba      # or: pip install -e ".[gpu,numba]"
```

| extra | installs | for |
|---|---|---|
| (none) | numpy, scipy, scikit-image, rasterio, typer | CPU, multithreaded |
| `numba` | numba | faster CPU rank filter and solver steps |
| `gpu` | `cupy-cuda12x`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cufft-cu12` | NVIDIA GPU |
| `gpu-clahe` | `cucim-cu12` | GeFolki's CLAHE on the GPU (else it runs on the CPU) |
| `dev` | pytest, ruff, tifffile | tests and linting |

The `gpu` extra ships the CUDA 12 runtime and compiler as wheels, so no system CUDA toolkit
is needed, only an NVIDIA driver. Pascal cards (sm_61, e.g. Quadro P4000, GTX 10xx) need
these CUDA 12 wheels: CUDA 13 dropped Pascal, so `cupy-cuda13x` (or a system CUDA 13 NVRTC)
fails at the first kernel compile. `device="auto"` uses the GPU only if a test kernel
compiles and runs, else the CPU; `gefolki info` shows what was found and why a GPU was
rejected.

## Python quickstart

```python
import numpy as np
import gefolki as g

# Arrays: 2-D master and slave of equal shape (any dtype; normalised internally)
u, v = g.gefolki(master, slave)  # radar/optical etc. (contrast adaptation)
u, v = g.efolki(master, slave, levels=5, radius=(16, 8), iterations=4, mask=master > 0)
registered = g.warp(slave, u, v)  # slave resampled onto the master grid
stack_reg = g.warp(cube, u, v, nodata=0)  # (bands, H, W) stacks too

p = g.FlowParams(levels=6, radius=(32, 24, 16, 8), iterations=2, rank=4, contrast_adapt=True)
u, v = g.estimate_flow(master, slave, p, device="gpu")  # "auto" | "cpu" | "gpu", threads=N

# Files: master is resampled onto the slave grid; all slave bands are warped and written
# with the slave's grid, CRS, dtype, nodata, band names and wavelengths.
res = g.register(
    "rgb_ortho.tif",
    "line.bsq",
    "line_reg.bsq",
    preset="hyperspectral-rgb",
    slave_bands="500-600",
    flow_output="flow.tif",
)
print(res.flow_stats, res.timings)

rf = g.estimate_raster_flow("master.tif", "slave.tif", method="efolki")  # flow only
g.apply_flow("slave.tif", "flow.tif", "slave_reg.tif")  # warp only

hit = g.locate(big_image, chip)  # where a small chip lies in a big image
hit = g.locate_raster("S1.tif", "chip.png")  # same, with map bounds
```

Band specs (`master_bands`, `slave_bands`): `1`, `"1,2,3"` (mean), `"500-600"` (mean of
bands within that wavelength range, nm) or `"rgb-gray"` (luminance). Default: RGB master
-> `rgb-gray`; slave with wavelengths and RGB master -> `"500-600"` and the
`hyperspectral-rgb` preset; otherwise band 1 and GeFolki defaults.

See `examples/quickstart.ipynb` (EFolki, GeFolki, accuracy, `register`, CLI) and
`examples/register_hyperspectral.py` (batch registration of hyperspectral flight lines).

## Command line

`gefolki --help` and `gefolki COMMAND --help` list every option.

| command | does |
|---|---|
| `register MASTER SLAVE OUTPUT` | flow + warp all slave bands; output on the slave grid |
| `flow MASTER SLAVE FLOW_OUTPUT` | flow only: 2-band float32 GeoTIFF (u, v in px) |
| `warp SLAVE FLOW OUTPUT` | warp every slave band by a flow from `gefolki flow` |
| `locate MASTER SLAVE` | find where a small SLAVE image lies inside MASTER |
| `inspect RASTER` | size, bands, dtype, CRS, resolution, nodata, wavelengths |
| `presets` | list parameter presets |
| `info` | backends (GPU, threads) and library versions |

```bash
# Hyperspectral line onto an RGB ortho (auto: hyperspectral-rgb preset, 500-600 nm vs luminance)
gefolki register rgb_ortho.tif line.bsq line_reg.bsq --flow-output line_flow.tif

# Explicit solver options (on top of --preset / --method)
gefolki register optical.tif radar.tif radar_reg.tif --preset optical-sar --device gpu
gefolki register a.tif b.tif out.tif --method efolki --levels 5 --radius 32:8:8 \
    --iterations 2 --rank 4 --no-contrast-adapt --slave-bands 2 --format COG -v

# Two steps: inspect the flow, then warp (also other rasters on the same grid)
gefolki flow master.tif slave.tif flow.tif --json
gefolki warp slave.tif flow.tif slave_reg.tif --resampling cubic

gefolki locate S1_Jacksonville_GEE.tif sandia_chip.png --chip-output match.tif --json
gefolki inspect line.bsq --bands
gefolki presets
gefolki info --json
```

Shared options of `register` / `flow`: `--preset`, `--method folki|efolki|gefolki`,
`--levels`, `--radius` (`32,24,16,8` or `start:stop:step`, stop inclusive),
`--iterations`, `--rank`, `--contrast-adapt/--no-contrast-adapt`, `--master-bands`,
`--slave-bands`, `--device auto|cpu|gpu`, `--threads`, `--tile-size` (px, default auto),
`-q/-v`, `--json`. `register` adds `--flow-output`, `--format GTiff|COG|ENVI` (default
from suffix: `.tif` GTiff; none, `.bsq`, `.bil`, `.img`, `.dat` ENVI) and
`--resampling nearest|bilinear|cubic`. `locate` takes `--master-band`, `--slave-band`,
`--chip-output`, `--decimation` (8), `--rank` (3), `--margin` (100 px).

## Presets

| preset | levels | radius | iterations | rank | contrast_adapt |
|---|---|---|---|---|---|
| `hyperspectral-rgb` | 5 | 32, 24, 16, 8 | 2 | 4 | no |
| `sar-sar` | 3 | 32 | 2 | 4 | no |
| `lidar-sar` | 6 | 32, 28, ..., 8 | 2 | 4 | no |
| `optical-sar` | 6 | 32, 28, ..., 8 | 2 | 4 | yes |
| `optical-optical` | 5 | 16, 8 | 4 | 4 | no |

Without a preset: manual defaults (levels 6, radius 32..8 step 4, 2 iterations, rank 4)
with contrast adaptation (GeFolki). `--method` / `method=` then switches only the variant:
`folki` (rank 0, no contrast adaptation), `efolki` (rank, no contrast adaptation),
`gefolki` (rank and contrast adaptation).

## Algorithm

One pyramidal Lucas-Kanade solver with three variants:

- **Pyramid**: both images normalised to [0, 1] over valid pixels (invalid pixels set to 0
  in both), Burt pyramid (kernel a = 0.4, `levels + 1` images). Flow starts at zero on the
  coarsest level and is upsampled bilinearly (values x2) to each finer level.
- **Rank filter** (EFolki, GeFolki): each pixel is replaced by the count of neighbours in a
  (2 rank + 1)² window that are greater (master, slave) or smaller (inverted slave) than it,
  which removes monotonic radiometric differences between sensors. `rank=0` (Folki) uses
  intensities.
- **Solver**: for each window radius (coarse to fine) and iteration, warp the slave rank
  image by the current flow, then solve the local least-squares system over a
  (2 radius + 1)² box window.
- **Contrast adaptation** (GeFolki): CLAHE of master and slave (8x8 tiles, clip 0.01);
  where the local mean of |H0 - H1| exceeds that of |1 - H0 - H1|, the contrast is
  inverted, and the solver uses the inverted (rank_inf) slave there.
- **Sign convention**: the flow satisfies `slave(x + u, y + v) ≈ master(x, y)`, with `u`
  the column (x) shift and `v` the row (y) shift in pixels. The registered slave is
  `warp(slave, u, v)`, i.e. the slave sampled at `(x + u, y + v)`. If
  `slave(x) = master(x + d)` then `u = -d`. Flow GeoTIFFs store u in band 1, v in band 2.
- **Large images**: `register` tiles the flow when the image does not fit in memory
  (4096 px tiles on GPU, 2048 on CPU, feathered overlaps); warping runs in band chunks.

## Performance

Intel i7-7820X (8 cores / 16 threads), Quadro P4000 (8 GB), numba installed. 2048² images,
default parameters (`benchmarks/bench_flow.py`):

| | CPU, 1 thread | CPU, 16 threads | GPU (P4000) |
|---|---|---|---|
| `gefolki` | 6.2 s | 1.54 s | 0.32 s |
| `efolki` | 3.95 s | 0.96 s | 0.20 s |

Full airborne FX10 flight line (15574 x 1091 px, 224 bands, ENVI) registered to a 107 GB
RGB orthomosaic GeoTIFF (2.36 cm, read through its overviews onto the 0.4 m line grid):
**22 s total on GPU, 54 s on CPU**. Residual shift vs the RGB after registration:
0.05 px, against 1.1 px for the reference product registered with ENVI.

## ENVI files

ENVI rasters are read and written through GDAL; ENVI software is not needed.

- Wavelengths, fwhm and wavelength units come from the `.hdr` (or GDAL band metadata) and
  are written back to output `.hdr` files, along with band names, the data ignore value and
  other non-structural header keys. Band selection by wavelength (`"500-600"`) uses them.
- Output interleave follows the suffix: `.bil`, `.bip`, else BSQ.
- With `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` set (common on clusters), GDAL cannot find
  `.hdr` sidecars. gefolki opens all rasters with that option set to `FALSE`, so ENVI
  inputs work regardless.

## Migrating from the legacy code

| legacy (`python/`, `matlab/`) | gefolki |
|---|---|
| `EFolki(I0, I1, iteration=2, radius=[32, 24, 16, 8], rank=4, levels=5)` | `gefolki.efolki(I0, I1, iterations=2, radius=(32, 24, 16, 8), rank=4, levels=5)` |
| `GEFolki(I0, I1, ...)` | `gefolki.gefolki(I0, I1, ...)` |
| `Folki(I0, I1, ...)` | `gefolki.folki(I0, I1, ...)` |
| `wrapData(I, u, v)` | `gefolki.warp(I, u, v)` (also band stacks, nodata) |
| `mining.py --input_master M --input_slave S` | `gefolki.locate` / `gefolki.locate_raster` / `gefolki locate M S` |
| Matlab `GeFolki(I0, I1, para)`, `para.contrast_adapt` true / false | `gefolki.gefolki` / `gefolki.efolki` (`gefolki.FlowParams` for `para`) |
| `main.py`, `main.m`, TP notebooks | `examples/quickstart.ipynb` |

Notes: `iteration` is now `iterations`; images need not be pre-scaled (they are normalised
over valid pixels) and can take a `mask`. Where the legacy Python and Matlab code differ,
gefolki follows Matlab: bilinear flow upsampling between levels (the legacy Python repeated
pixels, which moves results by ~0.15 px median on heterogeneous pairs).

## Development

```bash
uv venv -p 3.13 .venv && uv sync --extra gpu --extra numba --extra dev
uv run pytest                      # all tests; GPU tests skip without a usable GPU
uv run pytest -m "not gpu and not slow and not data"
uv run ruff check . && uv run ruff format --check .
uv run python benchmarks/bench_flow.py --size 2048 --variant gefolki efolki
```

Test markers: `gpu` (needs CuPy and a GPU), `slow`, `data` (private Liffey airborne data,
found through `GEFOLKI_LIFFEY_DATA`, full RGB ortho via `GEFOLKI_LIFFEY_RGB`; skipped when absent). CI runs ruff and
`pytest -m "not gpu"` on Python 3.12 and 3.13. Sample data used by tests and examples is in
`datasets/` (sources in `datasets/readme.txt`).

## License

GPL-3.0-or-later (`copying.txt`).
