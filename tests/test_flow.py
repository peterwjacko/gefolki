import numpy as np
import pytest
import tifffile

import gefolki as g
from gefolki import filters as F
from gefolki.flow import FlowParams, estimate_gpu_bytes_per_pixel

from .conftest import DATA, read_band

DX, DY = 3, -2  # slave(x, y) = master(x + DX, y + DY)  =>  u = -DX, v = -DY
FAST = dict(levels=3, radius=(16, 8), iterations=2)


def shifted_pair(img, n=256, r0=600, c0=700):
    master = img[r0 : r0 + n, c0 : c0 + n]
    slave = img[r0 + DY : r0 + DY + n, c0 + DX : c0 + DX + n]
    return master, slave


def interior_median(a, m=24):
    return float(np.median(a[m:-m, m:-m]))


@pytest.mark.parametrize("fn", [g.folki, g.efolki, g.gefolki])
def test_constant_shift_recovered(radar, fn):
    master, slave = shifted_pair(radar)
    u, v = fn(master, slave, device="cpu", **FAST)
    assert u.dtype == np.float32 and u.shape == master.shape
    assert interior_median(u) == pytest.approx(-DX, abs=0.05)
    assert interior_median(v) == pytest.approx(-DY, abs=0.05)


def test_gefolki_handles_contrast_inversion(radar):
    master, slave = shifted_pair(radar)
    master = master.max() - master  # inverted radiometry
    u, v = g.gefolki(master, slave, device="cpu", **FAST)
    assert interior_median(u) == pytest.approx(-DX, abs=0.1)
    assert interior_median(v) == pytest.approx(-DY, abs=0.1)
    ue, _ = g.efolki(master, slave, device="cpu", **FAST)
    assert abs(interior_median(ue) + DX) > 0.5  # EFolki alone cannot cope


def test_threads_do_not_change_result(radar):
    master, slave = shifted_pair(radar)
    p = FlowParams(contrast_adapt=True, **FAST)
    u1, v1 = g.estimate_flow(master, slave, p, device="cpu", threads=1)
    u8, v8 = g.estimate_flow(master, slave, p, device="cpu", threads=8)
    np.testing.assert_array_equal(u1, u8)
    np.testing.assert_array_equal(v1, v8)


@pytest.mark.skipif(F.numba is None, reason="numba not installed")
@pytest.mark.parametrize("contrast_adapt", [False, True])
def test_numba_and_numpy_paths_identical(radar, optical, monkeypatch, contrast_adapt):
    master, slave = radar[800:1056, 800:1056], optical[800:1056, 800:1056]
    p = FlowParams(contrast_adapt=contrast_adapt, **FAST)
    fast = g.estimate_flow(master, slave, p, device="cpu", threads=4)
    monkeypatch.setattr(F, "numba", None)
    slow = g.estimate_flow(master, slave, p, device="cpu", threads=4)
    np.testing.assert_array_equal(fast[0], slow[0])
    np.testing.assert_array_equal(fast[1], slow[1])


def test_gpu_bytes_per_pixel_estimate():
    assert estimate_gpu_bytes_per_pixel() == estimate_gpu_bytes_per_pixel(FlowParams())
    assert estimate_gpu_bytes_per_pixel(FlowParams(contrast_adapt=True)) > 100


def test_mask_and_nan_are_ignored(radar):
    master, slave = shifted_pair(radar)
    slave = slave.copy()
    slave[:, :40] = np.nan  # e.g. nodata strip
    mask = np.ones(master.shape, bool)
    mask[:, :40] = False
    u, v = g.efolki(master, slave, mask=mask, device="cpu", **FAST)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert float(np.median(u[24:-24, 80:-24])) == pytest.approx(-DX, abs=0.05)


def test_small_image_and_bad_params():
    a = np.random.default_rng(0).random((12, 9), dtype=np.float32)
    u, v = g.efolki(a, a, device="cpu")  # levels clipped, no crash
    assert u.shape == a.shape and np.isfinite(u).all()
    with pytest.raises(ValueError):
        FlowParams(radius=())
    with pytest.raises(ValueError):
        g.estimate_flow(a, a[:5], device="cpu")


EVAL = {
    # case: (master path, gt flow path, legacy EFolki mean EPE, threshold)
    "S1S2": ("S1S2/S1_patch11.tif", "S1S2/Flow_patch11.tif", 0.10, 0.15),
    "HR": (
        "HR/lngley_05521_09059_000_090813_L090_CX_01_pauli_x016_y020_radar.tif",
        "HR/lngley_05521_09059_000_090813_L090_CX_01_pauli_x016_y020_flow.tif",
        0.37,
        0.45,
    ),
}


@pytest.mark.parametrize("fn", [g.efolki, g.gefolki])
@pytest.mark.parametrize("case", EVAL)
def test_evalgefolki_ground_truth_recovery(case, fn):
    """Warp the master by the EvalGeFolki GT flow, then recover it (legacy settings)."""
    mpath, fpath, legacy, limit = EVAL[case]
    master = read_band(DATA / "EvalGeFolki" / mpath)
    gt = tifffile.imread(DATA / "EvalGeFolki" / fpath)  # (H, W, 2); GDAL sees one page only
    slave = g.warp(master, gt[..., 0], gt[..., 1], device="cpu")  # slave(x) = master(x + gt)
    u, v = fn(master, slave, device="cpu", levels=3, radius=(16, 8), iterations=4)
    epe = np.hypot(u + gt[..., 0], v + gt[..., 1])[30:-30, 30:-30].mean()
    print(f"{case} {fn.__name__}: mean EPE {epe:.4f} px (legacy EFolki {legacy})")
    assert epe <= limit


def legacy_efolki(i0, i1, levels, radius, iterations, rank):
    """Compact port of the legacy python/ EFolki (BurtOF + EFolkiIter) for parity checks.

    Differs from gefolki by: nearest-neighbour (repeat) flow upsampling, a 1e-8 'talon'
    added to the structure tensor diagonal, float64 maths.
    """
    from scipy import ndimage, signal

    def conv_sep(a, k):
        r = k.size // 2
        a = np.pad(a, r)
        return signal.convolve2d(signal.convolve2d(a, k[:, None], "valid"), k[None, :], "valid")

    def rank_sup(a, r):
        p, out = np.pad(a, r), np.zeros(a.shape)
        h, w = a.shape
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                out += p[dy : dy + h, dx : dx + w] > a
        return out

    burt = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
    norm = [(x - x.min()) / (x.max() - x.min()) for x in (i0.astype(float), i1.astype(float))]
    pyr = [norm]
    for _ in range(levels):
        pyr.append([conv_sep(x, burt)[::2, ::2] for x in pyr[-1]])
    u = v = np.zeros(pyr[-1][0].shape)
    for lvl in range(levels, -1, -1):
        j0, j1 = pyr[lvl]
        r0, r1 = rank_sup(j0, rank), rank_sup(j1, rank)
        iy, ix = np.gradient(r0)
        y, x = np.mgrid[: j0.shape[0], : j0.shape[1]]
        for rad in radius:
            k = np.ones(2 * rad + 1) / (2 * rad + 1)
            a, b, c = conv_sep(ix * ix, k) + 1e-8, conv_sep(iy * iy, k) + 1e-8, conv_sep(ix * iy, k)
            d = a * b - c * c
            for _ in range(iterations):
                r1w = ndimage.map_coordinates(r1, [y + v, x + u], order=1, mode="nearest")
                it = r0 - r1w + u * ix + v * iy
                gx, gy = conv_sep(ix * it, k), conv_sep(iy * it, k)
                u, v = (b * gx - c * gy) / d, (a * gy - c * gx) / d
                bad = ~np.isfinite(u) | ~np.isfinite(v)
                u[bad] = v[bad] = 0
        if lvl:
            shape = pyr[lvl - 1][0].shape
            u, v = (2 * np.repeat(np.repeat(f, 2, 0), 2, 1)[: shape[0], : shape[1]] for f in (u, v))
    return u, v


def test_parity_with_legacy_efolki(radar, optical, monkeypatch):
    """Radar/optical flow vs the legacy python EFolki on a crop.

    With the legacy repeat upsampling patched in, results agree to ~1e-3 px (the talon and
    float32 maths are negligible). The default bilinear upsampling (Matlab-faithful) moves
    the result by ~0.15 px median on this heterogeneous pair (measured: median 0.14,
    p95 0.76 px); bounds below are loose on purpose.
    """
    master, slave = radar[800:1056, 800:1056], optical[800:1056, 800:1056]
    kw = dict(levels=3, radius=(16, 8), iterations=2, rank=4)
    lu, lv = legacy_efolki(master, slave, **kw)

    def diff():
        u, v = g.efolki(master, slave, device="cpu", **kw)
        return np.hypot(u - lu, v - lv)[16:-16, 16:-16]

    d = diff()
    print(f"legacy parity: median |d| {np.median(d):.3f} px, p95 {np.percentile(d, 95):.3f}")
    assert np.median(d) < 0.3 and np.percentile(d, 95) < 1.5

    def repeat_upsample(f, shape, bk):
        return 2 * np.repeat(np.repeat(f, 2, 0), 2, 1)[: shape[0], : shape[1]]

    monkeypatch.setattr(g.filters, "upsample_flow", repeat_upsample)
    d = diff()
    assert np.median(d) < 0.005 and np.percentile(d, 99) < 0.05
