# Development guide for `gefolki`

Python package (src layout, hatchling, uv) for GeFolki dense optical-flow coregistration
of remote sensing images, with a typer CLI `gefolki`. GPL-3.0-or-later.

## Commands

```bash
uv venv -p 3.13 .venv && uv sync --extra gpu --extra numba --extra dev   # dev env
uv run pytest                                 # full suite (GPU tests skip without a GPU)
uv run pytest tests/test_flow.py -k evalgefolki  # one test
uv run pytest -m "not gpu and not slow and not data"
uv run ruff check . && uv run ruff format --check .
uv run python benchmarks/bench_flow.py --variant gefolki efolki
uv build                                      # wheel + sdist in dist/
```

CI (`.github/workflows/ci.yml`): `uv sync --locked --extra dev`, ruff check, ruff format
check, `pytest -m "not gpu"` on Python 3.12 and 3.13. Update `uv.lock` (`uv lock`) when
dependencies change.

## Architecture (`src/gefolki/`)

- `backend.py`: `Backend` (name, `xp` = numpy/cupy, `ndi` = scipy/cupyx ndimage, threads);
  `get_backend("auto"|"cpu"|"gpu", threads)`; `gpu_available()` imports CuPy, counts
  devices and compiles/runs a tiny kernel (cached); `backend_info()`.
- `filters.py`: backend-generic primitives: box filter, Burt pyramid, flow upsampling,
  rank filters (CuPy RawKernel / numba / NumPy paths), gradients, bilinear interpolation,
  CLAHE (cuCIM if installed, else scikit-image). CPU paths split rows over a thread pool.
- `flow.py`: `FlowParams`, `estimate_flow` (pyramid driver + per-level solver with fused
  element-wise steps), wrappers `folki` / `efolki` / `gefolki`,
  `estimate_gpu_bytes_per_pixel`.
- `warp.py`: `warp(image or (bands, H, W), u, v, order, nodata)`.
- `tiling.py`: `estimate_flow_tiled`: overlapping feathered tiles when an image does not
  fit in memory (thread pool over tiles on CPU, sequential on GPU).
- `io.py`: rasterio/GDAL helpers: `open_raster` (forces
  `GDAL_DISABLE_READDIR_ON_OPEN=FALSE` so ENVI `.hdr` sidecars are found), `read_info`
  (wavelengths, fwhm, ENVI header), band selection (`select_bands`), `read_on_grid`
  (master reprojected onto the slave grid via overviews), `create_output` (GTiff/COG/ENVI
  with metadata), `write_flow` / `read_flow`.
- `pipeline.py`: `PRESETS`, `resolve_params`, `estimate_raster_flow`, `apply_flow`,
  `register` (files in, registered file out on the slave grid).
- `locate.py`: `locate` / `locate_raster`: find a small chip in a big image (FFT scoring of
  rank_inf maps, coarse then fine).
- `cli.py`: typer app: `register`, `flow`, `warp`, `locate`, `info`, `presets`, `inspect`.
- `__init__.py`: public API (`__all__`).

`examples/`: `quickstart.ipynb` (commit without outputs), `register_hyperspectral.py`.
`datasets/`: sample images used by tests and examples (not shipped in wheel/sdist).

## Conventions

- Flow sign: `slave(x + u, y + v) ≈ master(x, y)`; `u` = column shift, `v` = row shift
  (px); registered slave = `warp(slave, u, v)`. Flow GeoTIFF: band 1 u, band 2 v. Keep it.
- Where the legacy Python and Matlab GeFolki disagree, follow Matlab `GeFolki.m`
  (manual defaults: levels 6, radius 32..8 step 4, iterations 2, rank 4).
- Core functions take NumPy or CuPy 2-D arrays and return NumPy float32 unless
  `return_device=True`; keep data on the device across pyramid levels and iterations.
- Georeferenced outputs keep the slave's grid, CRS, dtype, nodata and band metadata;
  ENVI wavelengths/fwhm are written to the `.hdr`.
- Style: ruff (line length 100), type hints, concise docstrings, no speculative
  abstractions. Tests: pytest; mark `gpu` (skipped without a usable GPU), `slow`, `data`
  (private Liffey data via `GEFOLKI_LIFFEY_DATA`, skipped when absent). Never commit
  private data.
