# Changelog

## 0.1.0 (unreleased)

Rewrite of the original GeFolki Matlab/Python code as the `gefolki` package.

### Added

- `gefolki.folki`, `efolki`, `gefolki` and `estimate_flow` (`FlowParams`): one pyramidal
  solver following the Matlab reference (bilinear flow upsampling, CLAHE contrast
  adaptation with Matlab `adapthisteq` defaults), valid-pixel masks.
- Backends: NumPy/SciPy with multithreading, optional numba kernels, optional CuPy GPU
  (CUDA 12 wheels, Pascal sm_61 supported) and cuCIM CLAHE; `device="auto"` checks that a
  GPU kernel compiles and runs.
- `gefolki.warp` for images and band stacks, with nodata handling.
- Georeferenced registration: `register`, `estimate_raster_flow`, `apply_flow`: master
  resampled onto the slave grid (overviews), band selection by index, wavelength range or
  RGB luminance, all slave bands warped and written with grid, CRS, nodata and band
  metadata (GTiff, COG, ENVI with wavelengths/fwhm); flow as a 2-band GeoTIFF.
- Tiled flow estimation for images larger than memory.
- `gefolki.locate` / `locate_raster`: vectorised port of `mining.py`.
- Parameter presets (`hyperspectral-rgb`, `sar-sar`, `lidar-sar`, `optical-sar`,
  `optical-optical`).
- `gefolki` CLI: `register`, `flow`, `warp`, `locate`, `info`, `presets`, `inspect`.
- Examples: `examples/quickstart.ipynb`, `examples/register_hyperspectral.py`.
- Tests (pytest, `gpu`/`slow`/`data` markers), benchmarks, GitHub Actions CI.

- Documentation in `docs/`: user guide and coregistration course (rewritten from the
  original PDF manual and slides, with figures regenerated from the sample data), CLI and
  API references, implementation notes, migration guide, development guide.
- `datasets/fetch.py`: downloads the sample data from the original ONERA repository
  (pinned commit, SHA-256 checked); tests needing a missing file are skipped.

### Removed

- Legacy `python/` and `matlab/` code and the `GEFOLKI_TP1/TP2` notebooks (see
  `docs/migration.md`).
- `manual_gefolki_english.pdf` and `COREGISTRATION.pdf` (replaced by `docs/`).
- Sample data files from git (`datasets/fetch.py` downloads them).
