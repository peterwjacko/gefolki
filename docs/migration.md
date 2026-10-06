# Migrating from the legacy code

The original ONERA repository ([aplyer/gefolki](https://github.com/aplyer/gefolki)) had
`python/` and `matlab/` folders and two notebooks (`GEFOLKI_TP1/TP2`). They are replaced
by the `gefolki` package.

| legacy | `gefolki` |
|---|---|
| `EFolki(I0, I1, iteration=2, radius=[32, 24, 16, 8], rank=4, levels=5)` | `gefolki.efolki(I0, I1, iterations=2, radius=(32, 24, 16, 8), rank=4, levels=5)` |
| `GEFolki(I0, I1, ...)` | `gefolki.gefolki(I0, I1, ...)` |
| `Folki(I0, I1, ...)` | `gefolki.folki(I0, I1, ...)` |
| `wrapData(I, u, v)` | `gefolki.warp(I, u, v)` (also band stacks and nodata) |
| `mining.py --input_master M --input_slave S` | `gefolki.locate` / `gefolki.locate_raster` / `gefolki locate M S` |
| Matlab `W = GeFolki(I1, I2, para)`, `para.contrast_adapt` true / false | `u, v = gefolki.gefolki(...)` / `gefolki.efolki(...)`; `gefolki.FlowParams` for `para` |
| Matlab `W(:,:,1)`, `W(:,:,2)` | `u`, `v` (`np.dstack([u, v])` for the same layout) |
| Matlab `interp2(..., x' + W(:,:,2)', y' + W(:,:,1)', 'nearest')` | `gefolki.warp(I2, u, v, order=0)` |
| `main.py`, `main.m`, TP notebooks | [exercises](coregistration.md#5-exercises), `examples/quickstart.ipynb` |
| `manual_gefolki_english.pdf` | [user guide](user-guide.md) |
| `COREGISTRATION.pdf` | [coregistration with GeFolki](coregistration.md) |

Notes:

- `iteration` is now `iterations`; Matlab's `para.iter` likewise.
- Images need not be pre-scaled: each is normalised over its valid pixels. Pass `mask=`
  instead of zero-filling by hand; NaN pixels are treated as invalid.
- Where the legacy Python and Matlab code differ, `gefolki` follows Matlab: bilinear flow
  upsampling between levels. The legacy Python repeated pixels, which moves results by
  about 0.15 px median on heterogeneous pairs.
- Georeferenced files no longer need manual initialisation: `gefolki.register()` /
  `gefolki register` resample the master onto the slave grid, estimate the flow and write
  every warped band with its metadata.
