"""Synthetic georeferenced master/slave rasters with a known shift."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from scipy import ndimage as ndi

CRS = "EPSG:32755"
X0, Y0 = 487000.0, 5384000.0
WAVELENGTHS = [450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
FWHM = [2.6, 2.6, 2.7, 2.7, 2.8, 2.8]


def texture(shape, seed=0, sigma=3.0):
    rng = np.random.default_rng(seed)
    t = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return ((t - t.min()) / np.ptp(t)).astype(np.float32)


def make_pair(d: Path, n=256, dx=4, dy=-3, margin=32, master_res=0.2, slave_res=0.4):
    """Write master (RGBA uint8, finer res, larger extent) and slave (6-band uint16, nodata 0,
    wavelength tags) GeoTIFFs. slave(x, y) = scene(x + dx, y + dy) in slave px, so the
    expected flow is u = -dx, v = -dy. Returns (master_path, slave_path, true_slave_band1).
    """
    f = round(slave_res / master_res)
    nm = (n + 2 * margin) * f
    base = texture((nm, nm))
    # Master: covers the slave extent plus ``margin`` slave px on each side.
    mtr = from_origin(X0 - margin * slave_res, Y0 + margin * slave_res, master_res, master_res)
    rgba = np.stack([180 * base + 20, 150 * base + 40, 120 * base + 60, np.full_like(base, 255)])
    rgba = rgba.astype(np.uint8)
    rgba[3, :, : f * 8] = 0  # transparent strip on the far left (outside the slave)
    mpath = d / "master.tif"
    with rasterio.open(mpath, "w", driver="GTiff", width=nm, height=nm, count=4, dtype="uint8",
                       crs=CRS, transform=mtr, tiled=True) as ds:  # fmt: skip
        ds.write(rgba)
        ds.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        ds.build_overviews([2, 4], rasterio.enums.Resampling.average)
    # Scene on the slave grid (block average of the master texture), then shifted.
    scene = base.reshape(nm // f, f, nm // f, f).mean(axis=(1, 3))
    true = scene[margin : margin + n, margin : margin + n]
    shifted = scene[margin + dy : margin + dy + n, margin + dx : margin + dx + n]
    bands = np.stack([(1000 + 3000 * k / 6) * shifted + 100 for k in range(6)]).astype(np.uint16)
    bands[:, :10, :] = 0  # nodata rows
    spath = d / "slave.tif"
    with rasterio.open(spath, "w", driver="GTiff", width=n, height=n, count=6, dtype="uint16",
                       crs=CRS, transform=from_origin(X0, Y0, slave_res, slave_res),
                       nodata=0) as ds:  # fmt: skip
        ds.write(bands)
        for b, (w, fw) in enumerate(zip(WAVELENGTHS, FWHM, strict=True), 1):
            ds.set_band_description(b, f"{w:g} nm")
            ds.update_tags(b, wavelength=f"{w:g}", fwhm=f"{fw:g}", wavelength_units="Nanometers")
    return mpath, spath, (1000 * true + 100).astype(np.float32)
