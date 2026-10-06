# Python API reference

```python
import gefolki as g
```

Everything below is exported from `gefolki` (`gefolki.__all__`), except the `gefolki.io`
helpers at the end. Docstrings hold the same information: `help(g.register)`.

Conventions:

- Images are 2-D arrays indexed `[row, col]`; band stacks are `(bands, rows, cols)`.
- Flow: `slave(x + u, y + v) ≈ master(x, y)`, `u` = column shift, `v` = row shift, in
  pixels, float32 arrays of the master's shape.
- `device`: `"auto"` (GPU if usable, else CPU), `"cpu"` or `"gpu"` (raises
  `RuntimeError` if no usable GPU), or a `Backend`. `threads`: CPU threads (default all;
  ignored on GPU).
- Inputs may be NumPy or CuPy arrays. Outputs are NumPy unless `return_device=True`.

| | arrays | files |
|---|---|---|
| flow | [`estimate_flow`](#estimate_flow), [`folki` / `efolki` / `gefolki`](#folki-efolki-gefolki) | [`estimate_raster_flow`](#estimate_raster_flow) |
| warp | [`warp`](#warp) | [`apply_flow`](#apply_flow) |
| both | | [`register`](#register) |
| find a chip | [`locate`](#locate) | [`locate_raster`](#locate_raster) |

## Flow on arrays

### `FlowParams`

```python
g.FlowParams(
    levels=6, radius=(32, 28, 24, 20, 16, 12, 8), iterations=2, rank=4, contrast_adapt=False
)
```

Frozen dataclass of solver parameters (defaults follow the GeFolki manual).

| field | meaning |
|---|---|
| `levels` | pyramid levels (`levels + 1` images); lowered automatically if the coarsest image would be under 2 px |
| `radius` | window radius or tuple of radii, applied large to small at every level |
| `iterations` | solver iterations per radius (≥ 1) |
| `rank` | rank filter radius; 0 = raw intensities |
| `contrast_adapt` | GeFolki contrast-inversion test (CLAHE-based) |

`ValueError` on invalid values. Use `dataclasses.replace(p, levels=4)` to change a field.

### `PRESETS`

`dict[str, FlowParams]`: `hyperspectral-rgb`, `sar-sar`, `lidar-sar`, `optical-sar`,
`optical-optical`. Values in the [user guide](user-guide.md#presets).

### `estimate_flow`

```python
u, v = g.estimate_flow(master, slave, params=None, *, mask=None, device="auto",
                       threads=None, return_device=False)
```

Dense flow between two 2-D arrays of equal shape, any numeric dtype. `params=None` uses
`FlowParams()` (EFolki with the manual defaults). `mask`: boolean, True = valid; NaN and
infinite pixels are invalid too. Invalid pixels are set to 0 in both images; each image
is scaled to [0, 1] over the valid pixels.

### `folki`, `efolki`, `gefolki`

```python
u, v = g.folki(master, slave, *, levels=6, radius=(8,), iterations=2, **kwargs)
u, v = g.efolki(master, slave, *, levels=6, radius=(32, ..., 8), iterations=2, rank=4, **kwargs)
u, v = g.gefolki(master, slave, *, levels=6, radius=(32, ..., 8), iterations=2, rank=4, **kwargs)
```

Wrappers of `estimate_flow` for each variant: Folki (intensities), EFolki (rank filter),
GeFolki (rank filter and contrast adaptation). `kwargs` go to `estimate_flow`
(`mask`, `device`, `threads`, `return_device`).

## Warping arrays

### `warp`

```python
out = g.warp(image, u, v, *, order=1, nodata=None, device="auto", threads=None,
             return_device=False)
```

Samples `image` at `(x + u, y + v)`: the slave on the master grid. `image`: `(rows, cols)`
or `(bands, rows, cols)`. `order`: 0 nearest, 1 bilinear, 3 cubic. Output pixels whose
sample point falls outside the image, or touches an input nodata pixel (equal to
`nodata` in all bands; NaN allowed), are set to `nodata` (0 if `None`). The output keeps
the input dtype; integers are rounded and clipped.

## Files

Raster inputs are paths to anything GDAL reads. The master is read resampled onto the
slave grid, so flow and outputs are on the slave grid.

### `register`

```python
res = g.register(master, slave, output, *, master_bands=None, slave_bands=None,
                 method=None, params=None, preset=None, device="auto", threads=None,
                 tile_size=None, flow_output=None, output_format=None,
                 resampling="bilinear", progress=None)
```

Estimates the flow, warps all slave bands and writes `output` with the slave's grid, CRS,
dtype, nodata, band names, wavelengths and FWHM.

| argument | meaning |
|---|---|
| `master_bands`, `slave_bands` | band spec for the flow image; `None` = auto (see [band specs](#band-specs)) |
| `params` / `preset` / `method` | parameters: `params`, else `preset`, else defaults (contrast adaptation on, or `hyperspectral-rgb` for an RGB master with a hyperspectral slave); `method` (`"folki"`, `"efolki"`, `"gefolki"`) then overrides only the variant |
| `tile_size` | `None`: whole image if it fits in memory, else 4096 px tiles on GPU / 2048 on CPU; `0`: never tile; `> 0`: that tile size |
| `flow_output` | also write the flow: 2-band float32 GeoTIFF, band 1 u, band 2 v |
| `output_format` | `"GTiff"`, `"COG"`, `"ENVI"`; default from the suffix (`.tif`, `.tiff` GTiff; none, `.bsq`, `.bil`, `.bip`, `.img`, `.dat`, `.envi` ENVI) |
| `resampling` | `"nearest"`, `"bilinear"`, `"cubic"` |
| `progress` | callback `progress(stage, done, total)`; stages `read_master`, `read_slave`, `flow`, `warp` |

Returns `RegistrationResult`:

| field | |
|---|---|
| `output` | `Path` of the registered raster |
| `flow_path` | `Path` of the flow GeoTIFF, or `None` |
| `flow_stats` | `{"median": ..., "p95": ...}` flow magnitude over valid pixels, px |
| `timings` | seconds per stage: `read_master`, `read_slave`, `flow`, `warp_write`, `total` |
| `params` | the `FlowParams` used |

GeoTIFF output is tiled (256 px), deflate-compressed, BigTIFF when needed.

### `estimate_raster_flow`

```python
rf = g.estimate_raster_flow(master, slave, *, master_bands=None, slave_bands=None,
                            method=None, params=None, preset=None, device="auto",
                            threads=None, tile_size=None, progress=None)
```

Flow only. Returns a `RasterFlow` with `u`, `v` (float32, slave grid), `valid` (bool),
`grid` (`gefolki.io.RasterInfo` of the slave), `params`, `timings` and `stats()`. Write
the flow with `gefolki.io.write_flow(path, rf.u, rf.v, rf.grid)`.

### `apply_flow`

```python
path = g.apply_flow(slave, flow, output, *, resampling="bilinear", device="auto",
                    threads=None, output_format=None, progress=None)
```

Warps every band of `slave` by `flow` (a flow GeoTIFF path, or a `(u, v)` tuple on the
slave grid) and writes `output` on the slave grid. Bands are processed in chunks, with
reading and writing overlapped with warping.

### Band specs

`master_bands` / `slave_bands` accept:

| spec | meaning |
|---|---|
| `1`, `"1"` | that band (1-based) |
| `[1, 2, 3]`, `"1,2,3"` | mean of those bands |
| `"500-600"`, `"500-600nm"` | mean of the bands whose wavelength (nm) is in the range |
| `"rgb-gray"` (also `"luminance"`, `"gray"`) | 0.299 R + 0.587 G + 0.114 B of an RGB(A) raster |

`None` (auto): master `"rgb-gray"` if RGB, else band 1; slave `"500-600"` if the master
is RGB and the slave has wavelengths in that range, else band 1.

## Finding a chip

### `locate`

```python
hit = g.locate(master, slave, *, mask=None, decimation=8, rank=3, margin=100,
               transform=None, device="auto", threads=None)
```

Finds the offset of a small 2-D `slave` inside a large 2-D `master` by minimising the mean
squared difference of their rank-filtered images over all offsets (FFT correlations):
first on images averaged over `decimation` x `decimation` blocks, then at full resolution
within `margin` px of the coarse hit. `mask`: slave valid pixels. `transform`: the
master's affine geotransform, to fill `map_bounds`.

Returns `LocateResult`: `row`, `col` (master pixel of the slave's top-left corner),
`height`, `width`, `score` (lower is better), `bounds` (`xmin, xmax, ymin, ymax` in master
pixels, max exclusive) and `map_bounds` (same in map units, or `None`).

### `locate_raster`

```python
hit = g.locate_raster(master_path, slave_path, *, master_band=1, slave_band=1,
                      chip_output=None, **kwargs)
```

`locate` on files. The slave's alpha band or nodata is its mask; `map_bounds` is in the
master CRS. `chip_output`: write the matching master window (all bands, georeferenced) as
GeoTIFF. `kwargs` go to `locate`.

## Backends

### `get_backend`, `Backend`

```python
bk = g.get_backend("auto", threads=None)  # Backend(name, xp, ndi, threads)
bk.is_gpu, bk.xp, bk.to_host(a), bk.asarray(a)
```

Pass a `Backend` as `device=` to reuse it across calls.

### `gpu_available`, `backend_info`

`gpu_available()`: True if CuPy imports, sees a device and compiles and runs a test
kernel (cached). `backend_info()`: dict used by `gefolki info` (versions, default device,
CPU threads, GPU name, compute capability, memory, and `gpu_error` explaining why a GPU
was rejected).

## `gefolki.io` helpers

Lower-level raster helpers used by the pipeline, for scripts that need them:

| function | |
|---|---|
| `read_info(path)` | `RasterInfo`: driver, size, count, dtype, CRS, transform, nodata, band descriptions, colour interpretation, wavelengths, FWHM, ENVI header |
| `select_bands(info, spec)` | resolve a band spec to indexes and weights |
| `read_selection(path, selection)` | flow image and valid mask on the raster's own grid |
| `read_on_grid(path, selection, crs, transform, shape)` | flow image and valid mask resampled onto another grid (uses overviews) |
| `write_flow(path, u, v, info)` / `read_flow(path)` | flow GeoTIFF |
| `open_raster(path)` | `rasterio.open` with `GDAL_DISABLE_READDIR_ON_OPEN=FALSE`, so ENVI `.hdr` files are found |
| `is_rgb(info)` | 3- or 4-band RGB(A) raster |
| `output_format(path, fmt=None)` | `"GTiff"`, `"COG"` or `"ENVI"` |

`gefolki.tiling.estimate_flow_tiled(master, slave, params, *, mask, device, tile_size,
overlap)` estimates flow on overlapping, feathered tiles; see
[large images](implementation.md#large-images).
