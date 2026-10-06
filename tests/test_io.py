import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gefolki import io

from .geodata import CRS, FWHM, WAVELENGTHS, make_pair


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return make_pair(tmp_path_factory.mktemp("pair"))


def test_read_info_wavelengths(pair):
    _, slave, _ = pair
    info = io.read_info(slave)
    np.testing.assert_allclose(info.wavelengths, WAVELENGTHS)
    np.testing.assert_allclose(info.fwhm, FWHM)
    assert info.nodata == 0 and info.count == 6 and not io.is_rgb(info)
    assert io.is_rgb(io.read_info(pair[0]))


def test_select_bands(pair):
    m, s = io.read_info(pair[0]), io.read_info(pair[1])
    assert io.select_bands(s, 2).indexes == (2,)
    assert io.select_bands(s, [1, 3]).weights == (0.5, 0.5)
    assert io.select_bands(s, "4,5").indexes == (4, 5)
    assert io.select_bands(s, "500-600").indexes == (2, 3, 4)
    assert io.select_bands(s, "640 - 710nm").indexes == (5, 6)
    rgb = io.select_bands(m, "rgb-gray")
    assert rgb.indexes == (1, 2, 3) and sum(rgb.weights) == pytest.approx(1)
    for bad in ("rgb-gray", "800-900", 7, "x"):
        with pytest.raises(ValueError):
            io.select_bands(s, bad)
    with pytest.raises(ValueError):
        io.select_bands(m, "500-600")  # no wavelengths


def test_read_selection_mask_and_blocks(pair):
    s = io.read_info(pair[1])
    sel = io.select_bands(s, [1, 2])
    img, valid = io.read_selection(pair[1], sel, block_rows=7)
    with rasterio.open(pair[1]) as ds:
        ref = ds.read([1, 2]).astype(np.float32).mean(0)
    np.testing.assert_allclose(img, ref, rtol=1e-6)
    assert not valid[:10].any() and valid[10:].all()


def test_read_on_grid_downsamples_with_alpha(pair):
    master, slave, _ = pair
    s, m = io.read_info(slave), io.read_info(master)
    img, valid = io.read_on_grid(master, io.select_bands(m, 2), s.crs, s.transform, s.shape)
    with rasterio.open(master) as ds:
        g = ds.read(2).astype(np.float32)
    f, off = 2, 32 * 2  # master is 2x finer and starts 32 slave px earlier
    blocks = g[off : off + 2 * s.height, off : off + 2 * s.width]
    blocks = blocks.reshape(s.height, f, s.width, f).mean(axis=(1, 3))
    assert valid.all()
    np.testing.assert_allclose(img, blocks, atol=0.51)


def test_read_on_grid_outside_and_overview(pair, tmp_path):
    master, _, _ = pair
    m = io.read_info(master)
    # Target grid at 0.8 m (uses the 2x overview), half of it left of the master extent.
    tr = from_origin(m.transform.c - 50 * 0.8, m.transform.f, 0.8, 0.8)
    img, valid = io.read_on_grid(master, io.select_bands(m, "rgb-gray"), m.crs, tr, (64, 100))
    assert not valid[:, :52].any() and valid[:, 54:].all()  # outside + transparent strip
    assert (img[~valid] == 0).all() and img[valid].min() > 0
    with rasterio.open(master) as ds:
        assert io._overview_level(ds, 4.0) == 1 and io._overview_level(ds, 1.5) is None


def test_envi_round_trip(pair, tmp_path):
    _, slave, _ = pair
    info = io.read_info(slave)
    out = tmp_path / "out.bsq"
    with io.create_output(out, info) as ds:
        ds.write(np.ones((info.count, *info.shape), np.uint16))
    hdr = (tmp_path / "out.hdr").read_text()
    for key in ("wavelength = {", "fwhm = {", "wavelength units = Nanometers",
                "data ignore value = 0", "band names = {"):  # fmt: skip
        assert key in hdr
    back = io.read_info(out)
    assert back.driver == "ENVI" and back.crs == info.crs and back.transform == info.transform
    np.testing.assert_allclose(back.wavelengths, WAVELENGTHS)
    np.testing.assert_allclose(back.fwhm, FWHM)
    assert back.nodata == 0 and back.descriptions[0].startswith("450")
    # ENVI -> GTiff keeps per-band wavelength/fwhm tags
    with io.create_output(tmp_path / "back.tif", back) as ds:
        ds.write(np.ones((info.count, *info.shape), np.uint16))
    again = io.read_info(tmp_path / "back.tif")
    np.testing.assert_allclose(again.wavelengths, WAVELENGTHS)
    np.testing.assert_allclose(again.fwhm, FWHM)


def test_envi_header_found_despite_env(pair, tmp_path, monkeypatch):
    _, slave, _ = pair
    out = tmp_path / "x"  # no suffix -> ENVI
    with io.create_output(out, io.read_info(slave)) as ds:
        ds.write(np.zeros((6, *ds.shape), np.uint16))
    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    assert io.read_info(out).wavelengths is not None


def test_output_formats(pair, tmp_path):
    info = io.read_info(pair[1])
    assert io.output_format("a.tif") == "GTiff" and io.output_format("a.dat") == "ENVI"
    assert io.output_format("a.tif", "cog") == "COG"
    with pytest.raises(ValueError):
        io.output_format("a.png")
    with io.create_output(tmp_path / "c.tif", info, fmt="COG") as ds:
        ds.write(np.full((6, *info.shape), 7, np.uint16))
    with rasterio.open(tmp_path / "c.tif") as ds:
        assert ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert ds.read(1)[0, 0] == 7 and ds.crs == CRS
    assert not (tmp_path / "c.tif.tmp.tif").exists()


def test_flow_file(pair, tmp_path):
    info = io.read_info(pair[1])
    u = np.full(info.shape, 1.5, np.float32)
    p = io.write_flow(tmp_path / "f.tif", u, -u, info)
    with rasterio.open(p) as ds:
        assert ds.count == 2 and ds.dtypes == ("float32", "float32")
        assert ds.descriptions == ("u (column shift, px)", "v (row shift, px)")
        assert ds.transform == info.transform
    uu, vv = io.read_flow(p)
    np.testing.assert_array_equal(uu, u)
    np.testing.assert_array_equal(vv, -u)
