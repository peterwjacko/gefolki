import json

import numpy as np
import pytest
import rasterio
from typer.testing import CliRunner

import gefolki as g
from gefolki import cli, io

from .conftest import DATA
from .geodata import WAVELENGTHS, make_pair

DX, DY = 4, -3
INNER = np.s_[24:-24, 24:-24]
FAST = ["--levels", "3", "--radius", "16,8", "--device", "cpu", "--threads", "4"]
runner = CliRunner()


def run(*args: object, code: int = 0):
    res = runner.invoke(cli.app, [str(a) for a in args])
    assert res.exit_code == code, res.output
    return res


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return make_pair(tmp_path_factory.mktemp("pair"), dx=DX, dy=DY)


def test_version():
    assert g.__version__ in run("--version").output


def test_parse_radius():
    assert cli.parse_radius("32,24,16,8") == (32, 24, 16, 8)
    assert cli.parse_radius("32:8:4") == (32, 28, 24, 20, 16, 12, 8)
    assert cli.parse_radius("8:16:8") == (8, 16)
    with pytest.raises(Exception, match="expected"):
        cli.parse_radius("32:x")


def test_register_json(pair, tmp_path):
    master, slave, _ = pair
    out, fl = tmp_path / "reg.tif", tmp_path / "flow.tif"
    res = run("register", master, slave, out, *FAST, "--flow-output", fl, "--json")
    data = json.loads(res.stdout)
    assert data["output"] == str(out) and data["flow_path"] == str(fl)
    assert data["params"]["levels"] == 3 and data["params"]["radius"] == [16, 8]
    assert data["flow_stats"]["median"] == pytest.approx(5.0, abs=0.15)
    u, v = io.read_flow(fl)
    assert np.median(u[INNER]) == pytest.approx(-DX, abs=0.1)
    assert np.median(v[INNER]) == pytest.approx(-DY, abs=0.1)
    with rasterio.open(out) as ds, rasterio.open(slave) as s:
        assert (ds.count, ds.dtypes, ds.transform) == (s.count, s.dtypes, s.transform)


def test_register_text_and_options(pair, tmp_path):
    master, slave, _ = pair
    out = tmp_path / "reg.bsq"
    res = run("register", master, slave, out, *FAST, "--method", "efolki", "--slave-bands",
              "500-600", "--resampling", "cubic", "--format", "ENVI", "-v")  # fmt: skip
    assert "flow magnitude" in res.stdout and "params: levels=3" in res.stdout
    assert "contrast_adapt=False" in res.stdout
    np.testing.assert_allclose(io.read_info(out).wavelengths, WAVELENGTHS)
    assert run("register", master, slave, out, *FAST, "-q").stdout == ""


def test_flow_then_warp(pair, tmp_path):
    master, slave, _ = pair
    fl, out = tmp_path / "flow.tif", tmp_path / "warped.tif"
    data = json.loads(run("flow", master, slave, fl, *FAST, "--json").stdout)
    assert data["flow_stats"]["mean_u"] == pytest.approx(-DX, abs=0.3)
    assert data["flow_stats"]["mean_v"] == pytest.approx(-DY, abs=0.3)
    assert "median=" in run("flow", master, slave, fl, *FAST).stdout
    run("warp", slave, fl, out, "--device", "cpu", "-q")
    ref = tmp_path / "ref.tif"
    g.apply_flow(slave, fl, ref, device="cpu")
    with rasterio.open(out) as a, rasterio.open(ref) as b:
        np.testing.assert_array_equal(a.read(), b.read())


def test_warp_rejects_non_flow(pair, tmp_path):
    _, slave, _ = pair
    res = run("warp", slave, slave, tmp_path / "x.tif", "--device", "cpu", code=1)
    assert "2-band" in res.output


def test_missing_file(pair, tmp_path):
    master, _, _ = pair
    res = run("register", master, tmp_path / "nope.tif", tmp_path / "o.tif", code=2)
    assert "does not exist" in res.output


def test_bad_radius(pair, tmp_path):
    master, slave, _ = pair
    res = run("flow", master, slave, tmp_path / "f.tif", "--radius", "a,b", code=2)
    assert "expected" in res.output


def test_gpu_unavailable(pair, tmp_path, monkeypatch):
    master, slave, _ = pair
    monkeypatch.setattr(g, "gpu_available", lambda: False)
    monkeypatch.setattr(cli, "_backend_info", lambda: {"gpu_error": "no CUDA device found"})
    res = run("register", master, slave, tmp_path / "o.tif", "--device", "gpu", code=2)
    assert "GPU requested but unavailable: no CUDA device found" in res.output


def test_locate_jacksonville(tmp_path):
    chip = tmp_path / "chip.tif"
    args = (DATA / "S1_Jacksonville_GEE.tif", DATA / "JacksonvilleNavalAirStation_sandiaKu.png")
    data = json.loads(run("locate", *args, "--device", "cpu", "--json").stdout)
    assert abs(data["row"] - 852) <= 2 and abs(data["col"] - 1112) <= 2
    assert len(data["map_bounds"]) == 4
    res = run("locate", *args, "--device", "cpu", "--chip-output", chip, "--rank", "4")
    assert "map bounds" in res.stdout and "pixel bounds" in res.stdout
    with rasterio.open(chip) as c:
        assert (c.height, c.width) == (360, 806)


def test_info():
    data = json.loads(run("info", "--json").stdout)
    assert "gpu_available" in data["backend"]
    assert data["versions"]["gefolki"] == g.__version__ and data["versions"]["GDAL"]
    assert "numpy" in run("info").stdout


def test_presets():
    data = json.loads(run("presets", "--json").stdout)
    assert set(data) == set(g.PRESETS)
    assert data["hyperspectral-rgb"]["radius"] == [32, 24, 16, 8]
    assert "sar-sar" in run("presets").stdout


def test_inspect(pair):
    master, slave, _ = pair
    data = json.loads(run("inspect", slave, "--json").stdout)
    assert (data["width"], data["count"], data["dtype"], data["nodata"]) == (256, 6, "uint16", 0)
    assert data["crs"] == "EPSG:32755" and data["resolution"] == [0.4, 0.4]
    assert data["wavelengths_nm"]["min"] == 450 and data["bands"][2]["wavelength_nm"] == 550
    assert json.loads(run("inspect", master, "--json").stdout)["rgb"] is True
    assert "550" in run("inspect", slave, "--bands").stdout
