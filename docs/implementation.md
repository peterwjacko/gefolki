# Implementation notes

How the package computes the flow, which hardware it uses and how it scales. For the
method itself see [Coregistration with GeFolki](coregistration.md#2-how-gefolki-works).

## Solver

One pyramidal Lucas-Kanade solver (`gefolki/flow.py`) serves all three variants.

1. **Normalisation**: each image is scaled to [0, 1] over the pixels valid in *both*
   (mask, finite values); invalid pixels are set to 0 in both.
2. **Pyramid**: Burt pyramid (5-tap kernel, a = 0.4), `levels + 1` images. `levels` is
   lowered if the coarsest image would be smaller than 2 px.
3. **Coarse to fine**: the flow starts at zero on the coarsest level and is upsampled
   bilinearly (values doubled) to each finer level.
4. **Per level**: rank-filter the master (`rank_sup`: count of the `(2·rank+1)²`
   neighbours strictly greater than the centre, zero padding) and the slave (`rank_sup`,
   and `rank_inf` for the inverted slave when contrast adaptation is on); `rank=0` uses
   the normalised intensities (and `1 − slave` as the inverted slave). Gradients of the
   master rank image by central differences.
5. **Per radius** (large to small) the structure tensor is box-filtered over the
   `(2r+1)²` window; **per iteration** the slave images are warped by the current flow,
   the residual `r0 − r1(x + u) + u·Ix + v·Iy` is formed, box-filtered, and the 2x2
   system is solved at every pixel. Pixels where the system is singular get zero flow.
6. **Contrast adaptation** (GeFolki): CLAHE of both images at the start of each level
   (Matlab `adapthisteq` defaults: 8x8 tiles, clip limit 0.01, 256 bins). At each
   iteration, with `H1` warped by the current flow, the box means over radius `rank` of
   `|H0 − H1|` and `|1 − H0 − H1|` are compared; where the first is larger the inverted
   slave rank image is used in the residual.

The flow sign: `slave(x + u, y + v) ≈ master(x, y)`. `warp(slave, u, v)` samples the
slave at `(x + u, y + v)`. Flow GeoTIFFs store `u` in band 1 and `v` in band 2.

Where the legacy Python and Matlab code differ, `gefolki` follows Matlab `GeFolki.m`.
Differences from the legacy Python: bilinear flow upsampling between levels (it repeated
pixels), no 1e-8 term on the structure tensor diagonal, float32 arithmetic. On a heterogeneous
radar/optical pair this moves results by about 0.15 px median from the legacy Python;
with the legacy upsampling patched in, they agree to about 0.001 px
(`tests/test_flow.py::test_parity_with_legacy_efolki`).

## Backends

| backend | when | notes |
|---|---|---|
| NumPy/SciPy, multithreaded | always | row chunks over a thread pool; `threads` sets the pool size |
| numba | `numba` extra installed | compiled rank filter, box filter, warp, CLAHE and fused element-wise steps; same results as the NumPy path |
| CuPy (GPU) | `gpu` extra and a usable NVIDIA GPU | custom kernels for rank filter, box filter and fused steps; FMA contraction off to stay close to CPU results |
| cuCIM CLAHE | `gpu-clahe` extra | otherwise a CuPy port of scikit-image's CLAHE |

`device="auto"` picks the GPU only if CuPy imports, sees a device and compiles and runs
a test kernel; else the CPU. `gefolki info` shows what was found and why a GPU was
rejected.

The `gpu` extra installs `cupy-cuda12x` with the CUDA 12 runtime, NVRTC and cuFFT as
wheels, so only an NVIDIA driver is needed, no CUDA toolkit. Pascal cards (compute
capability 6.1, e.g. Quadro P4000, GTX 10xx) need these CUDA 12 wheels: CUDA 13 dropped
Pascal, so `cupy-cuda13x` or a system CUDA 13 NVRTC fails at the first kernel compile.

On the GPU, data stay on the device across levels and iterations; arrays are freed as
soon as they are no longer needed, and the memory pool is released after each call.
Peak device memory is about 120 B/px for GeFolki and 100 B/px for EFolki/Folki
(`gefolki.flow.estimate_gpu_bytes_per_pixel`).

## Large images

`register`, `flow` and `estimate_raster_flow` tile the flow estimation
(`gefolki/tiling.py`) when a whole-image run would not fit in half of the free memory
(GPU memory, or RAM at about 160 B/px on CPU):

- tiles of 4096 px on GPU, 2048 px on CPU (`tile_size` / `--tile-size` overrides; 0
  disables tiling);
- overlap `min(tile_size / 4, max(128, 8 · max(radius)))`, 256 px for radius 32;
- each tile's flow is weighted by a linear ramp from the inner tile edges and the
  weighted flows are averaged, which hides tile-border errors;
- tiles run in a thread pool on CPU and one after another on GPU; tiles with no valid
  pixels are skipped.

Coarse pyramid levels see windows larger than any tile, so tiled and whole-image flows
differ slightly near tile borders. CLAHE works on 8x8 tiles of the image it is given,
so per-tile CLAHE differs from whole-image CLAHE: expect slightly larger differences
with contrast adaptation.

Other memory savers:

- the master is read through its overviews, at the coarsest level still at least as fine
  as the slave grid, and only the window covering the slave;
- band selection reads only the bands needed, in row blocks;
- warping and writing run in band chunks of about 1 GB, with reading, warping and
  writing overlapped, and a 256 MB GDAL block cache.

## Formats

All raster I/O goes through rasterio/GDAL.

- **GeoTIFF** output: tiled 256 px, deflate with a predictor, BigTIFF when needed. **COG**
  through GDAL's COG driver.
- **ENVI** files are read and written through GDAL; ENVI software is not needed.
  Wavelengths, FWHM and wavelength units come from the `.hdr` (or GDAL band metadata),
  are converted to nm for band selection and are written back to output `.hdr` files,
  along with band names, the data ignore value and other non-structural header keys.
  Output interleave follows the suffix: `.bil`, `.bip`, else BSQ.
- With `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` set (common on clusters), GDAL cannot
  find `.hdr` sidecars. `gefolki` opens all rasters with that option set to `FALSE`, so
  ENVI inputs work regardless.

## Performance

Intel i7-7820X (8 cores, 16 threads), Quadro P4000 (8 GB), numba installed, 2048 x 2048
images, default parameters (`benchmarks/bench_flow.py`):

| | CPU, 1 thread | CPU, 16 threads | GPU (P4000) |
|---|---|---|---|
| `gefolki` | 6.2 s | 1.54 s | 0.32 s |
| `efolki` | 3.95 s | 0.96 s | 0.20 s |

A full airborne FX10 flight line (15574 x 1091 px, 224 bands, ENVI) registered to a 107 GB
RGB orthomosaic GeoTIFF (2.36 cm, read through its overviews onto the 0.4 m line grid)
takes **22 s on GPU, 54 s on CPU**, end to end. The residual shift against the RGB after
registration is 0.05 px, against 1.1 px for the reference product registered with ENVI.

```bash
uv run python benchmarks/bench_flow.py --size 2048 --variant gefolki efolki
```

## Package layout

| module | |
|---|---|
| `backend.py` | `Backend`, `get_backend`, `gpu_available`, `backend_info` |
| `filters.py` | box filter, Burt pyramid, flow upsampling, rank filters, gradients, interpolation, CLAHE (NumPy, numba and CuPy paths) |
| `flow.py` | `FlowParams`, `estimate_flow`, `folki` / `efolki` / `gefolki` |
| `warp.py` | `warp` |
| `tiling.py` | `estimate_flow_tiled` |
| `io.py` | raster metadata, band selection, reading on a grid, output creation, flow files |
| `pipeline.py` | `PRESETS`, `estimate_raster_flow`, `apply_flow`, `register` |
| `locate.py` | `locate`, `locate_raster` |
| `cli.py` | the `gefolki` command |
