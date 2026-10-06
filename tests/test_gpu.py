"""CPU vs GPU equivalence (tolerance based). Skipped without a usable CUDA GPU."""

import sys

import numpy as np
import pytest

import gefolki as g
from gefolki import filters as F
from gefolki.backend import get_backend
from gefolki.flow import FlowParams

from .test_flow import DX, DY, FAST, interior_median, shifted_pair

pytestmark = pytest.mark.gpu
CPU = get_backend("cpu", threads=4)


@pytest.fixture(scope="module")
def gpu():
    return get_backend("gpu")


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).random((300, 257), dtype=np.float32)


def test_auto_picks_gpu(gpu):
    assert gpu.name == "gpu" and get_backend("auto").name == "gpu"
    assert type(gpu.asarray(np.zeros(3))).__module__.startswith("cupy")


def test_rank_kernel_exact(gpu, img):
    for r in (1, 4):
        for fn in (F.rank_sup, F.rank_inf):
            np.testing.assert_array_equal(
                gpu.to_host(fn(gpu.asarray(img), r, gpu)), fn(img, r, CPU)
            )


@pytest.mark.parametrize(
    "op",
    [
        lambda a, bk: F.box_filter(a, 16, bk),
        lambda a, bk: F.burt_reduce(a, bk),
        lambda a, bk: F.upsample_flow(a, (599, 513), bk),
        lambda a, bk: F.gradients(a, bk)[0],
        lambda a, bk: F.interp2(a, a * 300 - 20, (1 - a) * 310 - 5, bk),
    ],
    ids=["box", "burt", "upsample", "gradient", "interp2"],
)
def test_filters_match(gpu, img, op):
    np.testing.assert_allclose(gpu.to_host(op(gpu.asarray(img), gpu)), op(img, CPU), atol=2e-5)


def test_clahe_cpu_fallback(gpu, img, monkeypatch):
    monkeypatch.setitem(sys.modules, "cucim", None)  # force the host round-trip path
    out = F.clahe(gpu.asarray(img), gpu)
    assert type(out).__module__.startswith("cupy")
    np.testing.assert_allclose(gpu.to_host(out), F.clahe(img, CPU), atol=1e-6)


def test_clahe_cucim(gpu, img):
    pytest.importorskip("cucim")
    out = gpu.to_host(F.clahe(gpu.asarray(img), gpu))
    assert np.abs(out - F.clahe(img, CPU)).mean() < 0.01


@pytest.mark.parametrize("contrast_adapt", [False, True])
def test_flow_matches_cpu(gpu, radar, optical, contrast_adapt):
    master, slave = radar[800:1056, 800:1056], optical[800:1056, 800:1056]
    p = FlowParams(contrast_adapt=contrast_adapt, **FAST)
    uc, vc = g.estimate_flow(master, slave, p, device="cpu")
    ug, vg = g.estimate_flow(master, slave, p, device="gpu")
    assert isinstance(ug, np.ndarray) and ug.dtype == np.float32
    d = np.hypot(uc - ug, vc - vg)
    assert np.median(d) < 1e-3 and np.percentile(d, 99) < 0.05


def test_flow_shift_and_return_device(gpu, radar):
    master, slave = shifted_pair(radar)
    u, v = g.gefolki(master, slave, device="gpu", return_device=True, **FAST)
    assert type(u).__module__.startswith("cupy")
    assert interior_median(u.get()) == pytest.approx(-DX, abs=0.05)
    assert interior_median(v.get()) == pytest.approx(-DY, abs=0.05)


def test_warp_matches_cpu(gpu):
    rng = np.random.default_rng(2)
    stack = rng.integers(0, 4000, (3, 64, 80)).astype(np.uint16)
    stack[:, 5:8, 5:8] = 0
    u = rng.uniform(-2, 2, (64, 80)).astype(np.float32)
    v = rng.uniform(-2, 2, (64, 80)).astype(np.float32)
    c = g.warp(stack, u, v, nodata=0, device="cpu")
    gg = g.warp(stack, u, v, nodata=0, device="gpu")
    assert gg.dtype == np.uint16
    assert (np.abs(c.astype(int) - gg) <= 1).all()
