import numpy as np
import pytest
from scipy import ndimage as ndi

import gefolki as g
from gefolki.flow import FlowParams
from gefolki.tiling import _ramp, estimate_flow_tiled, tile_spans

from .geodata import texture


def test_tile_spans_cover_with_overlap():
    for n, t, o in [(2048, 1024, 256), (1000, 300, 64), (300, 300, 10), (100, 300, 10)]:
        spans = tile_spans(n, t, o)
        assert spans[0][0] == 0 and spans[-1][1] == n
        for (a0, a1), (b0, _b1) in zip(spans, spans[1:], strict=False):
            assert a1 - b0 >= o and b0 > a0
        assert all(e - s == min(t, n) for s, e in spans)


def test_feather_weights_sum_to_one():
    n, t, o = 1000, 300, 64
    acc = np.zeros(n)
    for s, e in tile_spans(n, t, o):
        acc[s:e] += _ramp(s, e, n, o)
    assert acc.min() > 0.99  # overlaps >= o cross-fade to >= 1
    assert _ramp(0, 300, 300, o).min() == 1  # image borders are not feathered


@pytest.fixture(scope="module")
def big_pair():
    n = 2048
    base = texture((n + 64, n + 64), sigma=2)
    y, x = np.mgrid[:n, :n].astype(np.float32)
    du = 6 + 3 * np.sin(x / 400)  # smooth, spatially varying shift
    dv = -4 + 2 * np.cos(y / 300)
    master = base[32 : 32 + n, 32 : 32 + n]
    slave = ndi.map_coordinates(base, [y + 32 - dv, x + 32 - du], order=1)
    return master, slave, du, dv


def test_tiled_matches_whole(big_pair):
    master, slave, du, dv = big_pair
    p = FlowParams(levels=5, radius=(16, 8))
    uw, vw = g.estimate_flow(master, slave, p, device="cpu")
    calls = []
    ut, vt = estimate_flow_tiled(
        master, slave, p, device="cpu", tile_size=1024, progress=lambda *a: calls.append(a)
    )
    assert calls[-1] == ("flow", 9, 9)
    d = np.hypot(ut - uw, vt - vw)[64:-64, 64:-64]
    assert np.median(d) < 0.01 and np.percentile(d, 95) < 0.05
    assert np.median(np.abs(ut - du)[64:-64, 64:-64]) < 0.1  # slave(x) = master(x - d)


def test_auto_whole_and_mask():
    a = texture((300, 300))
    b = np.roll(a, (2, -3), (0, 1))
    mask = np.ones(a.shape, bool)
    mask[:150] = False  # first tile row fully masked
    p = FlowParams(levels=2, radius=(8,))
    uw, vw = estimate_flow_tiled(a, b, p, device="cpu")  # fits -> whole image
    u0, v0 = g.estimate_flow(a, b, p, device="cpu")
    np.testing.assert_array_equal(uw, u0)
    ut, vt = estimate_flow_tiled(a, b, p, mask=mask, device="cpu", tile_size=128, overlap=32)
    assert np.isfinite(ut).all()
    assert np.median(ut[180:-20, 20:-20]) == pytest.approx(-3, abs=0.1)
    assert np.median(vt[180:-20, 20:-20]) == pytest.approx(2, abs=0.1)
