import numpy as np
import pytest
import rasterio
from scipy.ndimage import gaussian_filter

import gefolki as g
from gefolki.backend import get_backend
from gefolki.locate import _score_map

from .conftest import DATA

CPU = get_backend("cpu", threads=4)
ROW, COL, H, W = 237, 411, 120, 200
# Expected chip position: legacy mining.py (rank 3 and 4, decimation 8) gives the same.
JAX = (852, 1112)


@pytest.fixture(scope="module")
def texture() -> np.ndarray:
    rng = np.random.default_rng(1)
    return gaussian_filter(rng.random((600, 800)), 2).astype(np.float32)


def chip(master: np.ndarray) -> np.ndarray:
    """Window of master with a monotonic radiometric change and mild noise."""
    s = master[ROW : ROW + H, COL : COL + W]
    s = 3 * (s - s.min()) ** 0.8 + 10
    return s + np.random.default_rng(2).normal(0, 0.002, s.shape).astype(np.float32)


def test_score_map_matches_brute_force():
    rng = np.random.default_rng(0)
    img, tpl = rng.random((23, 31)), rng.random((7, 9))
    valid = rng.random(tpl.shape) > 0.3
    got = _score_map(img, tpl, valid, CPU)
    want = np.array(
        [
            [((img[r : r + 7, c : c + 9] - tpl)[valid] ** 2).mean() for c in range(31 - 9 + 1)]
            for r in range(23 - 7 + 1)
        ]
    )
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("decimation", [1, 4, 8])
def test_synthetic_exact(texture, decimation):
    res = g.locate(texture, chip(texture), decimation=decimation, device="cpu")
    assert (res.row, res.col, res.height, res.width) == (ROW, COL, H, W)
    assert res.bounds == (COL, COL + W, ROW, ROW + H)
    assert res.map_bounds is None


def test_mask_and_transform(texture):
    s = chip(texture)
    mask = np.ones(s.shape, bool)
    mask[:, :60] = False
    s[~mask] = np.random.default_rng(3).random((~mask).sum()) * 100  # garbage under the mask
    tr = rasterio.Affine(2.0, 0, 1000.0, 0, -2.0, 5000.0)
    res = g.locate(texture, s, mask=mask, transform=tr, device="cpu")
    assert (res.row, res.col) == (ROW, COL)
    assert res.map_bounds == (
        1000 + 2 * COL,
        1000 + 2 * (COL + W),
        5000 - 2 * (ROW + H),
        5000 - 2 * ROW,
    )


def test_edge_offsets(texture):
    """The last valid offsets (bottom-right corner) are searched."""
    s = texture[-H:, -W:]
    res = g.locate(texture, s, device="cpu")
    assert (res.row, res.col) == (600 - H, 800 - W)


def test_bad_shapes(texture):
    with pytest.raises(ValueError):
        g.locate(texture[:50, :50], texture, device="cpu")
    with pytest.raises(ValueError):
        g.locate(texture, texture[:10, :10, None], device="cpu")


def test_jacksonville(tmp_path):
    out = tmp_path / "chip.tif"
    res = g.locate_raster(
        DATA / "S1_Jacksonville_GEE.tif",
        DATA / "JacksonvilleNavalAirStation_sandiaKu.png",
        chip_output=out,
        device="cpu",
    )
    assert abs(res.row - JAX[0]) <= 2 and abs(res.col - JAX[1]) <= 2
    with rasterio.open(DATA / "S1_Jacksonville_GEE.tif") as ms, rasterio.open(out) as c:
        assert (c.height, c.width, c.count) == (360, 806, ms.count)
        assert c.crs == ms.crs
        np.testing.assert_allclose(c.bounds[0], res.map_bounds[0])
        np.testing.assert_allclose(c.bounds[3], res.map_bounds[3])
        win = rasterio.windows.Window(res.col, res.row, res.width, res.height)
        np.testing.assert_array_equal(c.read(), ms.read(window=win))


@pytest.mark.gpu
def test_gpu_matches_cpu(texture):
    s = chip(texture)
    a = g.locate(texture, s, device="cpu")
    b = g.locate(texture, s, device="gpu")
    assert (a.row, a.col) == (b.row, b.col)
    assert b.score == pytest.approx(a.score, rel=1e-6, abs=1e-6)
