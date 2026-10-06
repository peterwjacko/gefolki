import numpy as np
import pytest
import rasterio

import gefolki as g
from gefolki import io
from gefolki.flow import FlowParams
from gefolki.pipeline import PRESETS, resolve_params

from .geodata import FWHM, WAVELENGTHS, make_pair

DX, DY = 4, -3
FAST = FlowParams(levels=3, radius=(16, 8))
INNER = np.s_[24:-24, 24:-24]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return make_pair(tmp_path_factory.mktemp("pair"), dx=DX, dy=DY)


@pytest.fixture(scope="module")
def registered(pair, tmp_path_factory):
    master, slave, _ = pair
    d = tmp_path_factory.mktemp("out")
    stages = []
    res = g.register(
        master, slave, d / "reg.tif", params=FAST, device="cpu", threads=4,
        flow_output=d / "flow.tif", progress=lambda *a: stages.append(a[0]),
    )  # fmt: skip
    return res, stages


def test_register_recovers_shift(registered):
    res, stages = registered
    u, v = io.read_flow(res.flow_path)
    assert np.median(u[INNER]) == pytest.approx(-DX, abs=0.1)
    assert np.median(v[INNER]) == pytest.approx(-DY, abs=0.1)
    assert res.flow_stats["median"] == pytest.approx(5.0, abs=0.15)
    assert {"read_master", "read_slave", "flow", "warp_write", "total"} <= set(res.timings)
    assert stages[0] == "read_master" and stages[-1] == "warp"


def test_register_output_matches_scene_and_metadata(pair, registered):
    _, slave, true1 = pair
    res, _ = registered
    with rasterio.open(slave) as s, rasterio.open(res.output) as o:
        assert (o.crs, o.transform, o.shape, o.count) == (s.crs, s.transform, s.shape, s.count)
        assert o.dtypes == s.dtypes and o.nodata == 0
        assert o.descriptions == s.descriptions
        out1 = o.read(1).astype(np.float32)
        assert o.tags(3)["wavelength"] == "550" and o.tags(3)["fwhm"] == "2.7"
    valid = out1 > 0
    assert not valid[:7].any()  # warped from nodata rows (shifted by -DY)
    err = np.abs(out1 - true1)[INNER][valid[INNER]]
    assert np.median(err) < 0.02 * true1.mean()
    info = io.read_info(res.output)
    np.testing.assert_allclose(info.wavelengths, WAVELENGTHS)
    np.testing.assert_allclose(info.fwhm, FWHM)


def test_register_envi_output_and_apply_flow(pair, registered, tmp_path):
    master, slave, _ = pair
    res, _ = registered
    out = g.apply_flow(slave, res.flow_path, tmp_path / "reg", device="cpu")  # ENVI
    hdr = (tmp_path / "reg.hdr").read_text()
    assert "wavelength = {" in hdr and "fwhm = {" in hdr and "data ignore value = 0" in hdr
    with rasterio.open(res.output) as a, io.open_raster(out) as b:
        assert b.driver == "ENVI"
        np.testing.assert_array_equal(a.read(), b.read())
    np.testing.assert_allclose(io.read_info(out).wavelengths, WAVELENGTHS)


def test_estimate_raster_flow_bands_and_presets(pair):
    master, slave, _ = pair
    rf = g.estimate_raster_flow(master, slave, device="cpu", tile_size=0)
    assert rf.params == PRESETS["hyperspectral-rgb"]  # auto: RGB master + wavelengths
    assert rf.valid[:10].sum() == 0 and rf.valid[20:].all()
    assert np.median(rf.u[INNER]) == pytest.approx(-DX, abs=0.1)
    rf2 = g.estimate_raster_flow(
        master, slave, master_bands=2, slave_bands="1,2", method="gefolki", params=FAST,
        device="cpu", tile_size=128,
    )  # fmt: skip
    assert rf2.params.contrast_adapt
    assert np.median(rf2.v[INNER]) == pytest.approx(-DY, abs=0.15)


def test_resolve_params():
    assert resolve_params(None, None, "sar-sar") == PRESETS["sar-sar"]
    assert resolve_params("folki", None, "sar-sar").rank == 0
    e = resolve_params("efolki", FlowParams(rank=0, contrast_adapt=True))
    assert e.rank == 4 and not e.contrast_adapt
    assert resolve_params().contrast_adapt
    with pytest.raises(ValueError):
        resolve_params("foo")
    with pytest.raises(KeyError):
        resolve_params(None, None, "nope")
