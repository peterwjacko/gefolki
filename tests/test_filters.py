import numpy as np
import pytest
from scipy import ndimage, signal

from gefolki import filters as F
from gefolki.backend import get_backend

CPU1 = get_backend("cpu", threads=1)
CPU8 = get_backend("cpu", threads=8)


def rng_image(shape=(300, 257), seed=0):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def naive_rank(a, r, sup):
    h, w = a.shape
    p = np.pad(a, r)
    out = np.zeros(a.shape, np.float32)
    for i in range(h):
        for j in range(w):
            win = p[i : i + 2 * r + 1, j : j + 2 * r + 1]
            out[i, j] = np.sum(win > a[i, j]) if sup else np.sum(win < a[i, j])
    return out


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("r", [1, 4])
def test_rank_matches_naive(monkeypatch, use_numba, r):
    if not use_numba:
        monkeypatch.setattr(F, "numba", None)
    elif F.numba is None:
        pytest.skip("numba not installed")
    a = rng_image((37, 51))
    a[5:9, 5:9] = 0.5  # ties are not counted
    for bk in (CPU1, CPU8):
        np.testing.assert_array_equal(F.rank_sup(a, r, bk), naive_rank(a, r, True))
        np.testing.assert_array_equal(F.rank_inf(a, r, bk), naive_rank(a, r, False))


def test_rank_numpy_threaded_identical(monkeypatch):
    monkeypatch.setattr(F, "numba", None)
    a = rng_image()
    np.testing.assert_array_equal(F.rank_sup(a, 4, CPU1), F.rank_sup(a, 4, CPU8))


def test_burt_reduce_matches_conv2():
    a = rng_image((101, 64))
    k = F.BURT_KERNEL.astype(np.float64)
    ref = signal.convolve2d(a.astype(np.float64), np.outer(k, k), mode="same")[::2, ::2]
    for bk in (CPU1, CPU8):
        out = F.burt_reduce(a, bk)
        assert out.shape == (51, 32)
        np.testing.assert_allclose(out, ref, atol=1e-6)


def test_pyramid_shapes():
    p = F.pyramid(rng_image((100, 33)), 3, CPU1)
    assert [x.shape for x in p] == [(100, 33), (50, 17), (25, 9), (13, 5)]


@pytest.mark.parametrize("r", [0, 3, 32])
def test_box_filter(r):
    a = rng_image()
    ref = ndimage.uniform_filter(a.astype(np.float64), 2 * r + 1, mode="constant")
    single = F.box_filter(a, r, CPU1)
    threaded = F.box_filter(a, r, CPU8)
    np.testing.assert_array_equal(single, threaded)  # bitwise identical
    np.testing.assert_allclose(single, ref, atol=1e-6)


def test_interp2_threaded_identical_and_clamped():
    a = rng_image()
    rng = np.random.default_rng(1)
    xs = rng.uniform(-5, a.shape[1] + 5, a.shape).astype(np.float32)
    ys = rng.uniform(-5, a.shape[0] + 5, a.shape).astype(np.float32)
    single = F.interp2(a, xs, ys, CPU1)
    np.testing.assert_array_equal(single, F.interp2(a, xs, ys, CPU8))
    xc, yc = np.clip(xs, 0, a.shape[1] - 1), np.clip(ys, 0, a.shape[0] - 1)
    ref = ndimage.map_coordinates(a, [yc, xc], order=1)
    np.testing.assert_allclose(single, ref, atol=1e-6)


def test_upsample_flow_bilinear():
    coarse = np.arange(12, dtype=np.float32).reshape(3, 4)  # f = 4*row + col
    for shape in [(6, 8), (5, 7)]:
        up = F.upsample_flow(coarse, shape, CPU1)
        assert up.shape == shape
        i, j = np.mgrid[: shape[0], : shape[1]]
        ri, cj = np.minimum(i / 2, 2), np.minimum(j / 2, 3)  # clamped coarse coords
        np.testing.assert_allclose(up, 2 * (4 * ri + cj))


def test_gradients_match_numpy():
    a = rng_image((20, 30))
    gx, gy = F.gradients(a, CPU1)
    ry, rx = np.gradient(a)
    np.testing.assert_array_equal(gx, rx)
    np.testing.assert_array_equal(gy, ry)


def test_clahe_matlab_like():
    a = rng_image((64, 80)) * 0.3 + 0.2
    out = F.clahe(a, CPU1)
    assert out.shape == a.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1 and out.std() > a.std()
    np.testing.assert_array_equal(F.clahe(np.zeros((16, 16), np.float32), CPU1), 0)
