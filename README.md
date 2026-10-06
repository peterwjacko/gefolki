# gefolki

Dense optical-flow coregistration of heterogeneous remote sensing images: SAR/SAR,
optical/SAR, LIDAR/SAR, hyperspectral/optical and more. A Python library and a `gefolki`
command line tool, with multithreaded CPU and optional GPU (CuPy) backends.

GeFolki was developed by ONERA in the MEDUSA project, first for SAR/SAR coregistration,
then for other remote sensing pairs. This repository rewrites the original Matlab/Python
code ([aplyer/gefolki](https://github.com/aplyer/gefolki)) as an installable package. It
adds georeferenced, multi-band registration of GeoTIFF, COG and ENVI rasters, tiling for
very large images and chip localisation.

![Radar and optical images before and after GeFolki](docs/figures/optical_sar.jpg)

## Install

Python 3.12 or newer.

```bash
pip install "gefolki[numba] @ git+https://github.com/peterwjacko/gefolki"        # CPU
pip install "gefolki[gpu,numba] @ git+https://github.com/peterwjacko/gefolki"    # + NVIDIA GPU
uv add "gefolki[gpu,numba] @ git+https://github.com/peterwjacko/gefolki"         # with uv
```

| extra | adds |
|---|---|
| (none) | NumPy/SciPy/scikit-image/rasterio backend, multithreaded |
| `numba` | faster CPU kernels (recommended) |
| `gpu` | CuPy with CUDA 12 runtime wheels; needs only an NVIDIA driver |
| `gpu-clahe` | cuCIM, for GeFolki's CLAHE on the GPU |
| `dev` | pytest, ruff, tifffile |

`gefolki info` shows which backends work and, if the GPU is not used, why. See
[implementation notes](docs/implementation.md#backends) for GPU details.

## Usage

Georeferenced files: the master is resampled onto the slave grid, then every slave band
is warped and written with the slave's grid, CRS, dtype, nodata and band metadata.

```bash
gefolki register optical.tif radar.tif radar_registered.tif --preset optical-sar
gefolki register rgb_ortho.tif line.bsq line_registered.bsq --flow-output flow.tif
gefolki locate S1_scene.tif airborne_chip.png --chip-output match.tif
gefolki --help
```

Python, on files or on arrays of equal shape:

```python
import gefolki as g

res = g.register("optical.tif", "radar.tif", "radar_registered.tif", preset="optical-sar")
print(res.flow_stats)

u, v = g.gefolki(master, slave)  # flow: slave(x + u, y + v) ≈ master(x, y)
registered = g.warp(slave, u, v)  # slave resampled onto the master grid
```

`gefolki` (with contrast adaptation) suits heterogeneous pairs such as optical/SAR;
`efolki` suits same-sensor or same-band pairs. Presets cover common cases:
`gefolki presets`.

## Documentation

- [User guide](docs/user-guide.md): the three steps, parameters, presets, masks, files
- [Coregistration with GeFolki](docs/coregistration.md): method, parameter tuning,
  worked examples, exercises
- [Command line](docs/cli.md) and [Python API](docs/api.md) references
- [Implementation notes](docs/implementation.md): solver, backends, large images,
  performance
- [Migrating from the legacy code](docs/migration.md)
- [Sample datasets](datasets/README.md) (`python datasets/fetch.py`) and
  [examples](examples/)
- [Development](docs/development.md), [changelog](CHANGELOG.md)

## Citation

If you use GeFolki in published work, please cite:

- SAR/SAR coregistration (interferometry, change detection, ...): A. Plyer,
  E. Colin-Koeniguer, F. Weissgerber, "A new coregistration algorithm for recent
  applications on urban SAR images", *IEEE Geoscience and Remote Sensing Letters*,
  12(11):2198–2202, 2015.
- Other remote sensing pairs (optical/SAR, optical/hyperspectral, LIDAR/SAR, ...):
  G. Brigot, E. Colin-Koeniguer, A. Plyer, F. Janez, "Adaptation and evaluation of an
  optical flow method applied to coregistration of forest remote sensing images",
  *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*,
  9(7), 2016.
- Measurement (PIV, material deformation, ...): F. Champagnat, A. Plyer, G. Le Besnerais,
  B. Leclaire, Y. Le Sant, "How to calculate dense PIV vector fields at video rate",
  *8th International Symposium on Particle Image Velocimetry*, 2009.
- Computer vision (robotics, ...): A. Plyer, G. Le Besnerais, F. Champagnat, "Massively
  parallel Lucas Kanade optical flow for real-time video processing applications",
  *Journal of Real-Time Image Processing*, 11(4):713–730, 2016.

## License

GPL-3.0-or-later ([copying.txt](copying.txt)).
