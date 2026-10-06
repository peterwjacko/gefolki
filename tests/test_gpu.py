"""CPU vs GPU equivalence (tolerance based). Skipped without a usable CUDA GPU."""

import sys

import numpy as np
import pytest

import gefolki as g
from gefolki import filters as F
from gefolki.backend import get_backend
from gefolki.flow import FlowParams, estimate_gpu_bytes_per_pixel

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


@pytest.mark.parametrize("crop", [(slice(None), slice(None)), (slice(0, 31), slice(0, 40))])
def test_clahe_cupy_port_matches_skimage(gpu, radar, monkeypatch, crop):
    monkeypatch.setitem(sys.modules, "cucim", None)  # force the CuPy port
    monkeypatch.setattr(F, "numba", None)  # CPU reference: scikit-image itself
    for a in (radar[crop] / 255, np.random.default_rng(0).random((300, 257), np.float32)):
        out = F.clahe(gpu.asarray(a), gpu)
        assert type(out).__module__.startswith("cupy") and out.dtype == np.float32
        np.testing.assert_array_equal(gpu.to_host(out), F.clahe(a, CPU))


def test_warp_flow_matches_cpu(gpu, img):
    rng = np.random.default_rng(4)
    u = rng.uniform(-20, 20, img.shape).astype(np.float32)
    v = rng.uniform(-20, 20, img.shape).astype(np.float32)
    stack = np.stack([img, img * 100])
    out = F.warp_flow(gpu.asarray(stack), gpu.asarray(u), gpu.asarray(v), gpu)
    np.testing.assert_allclose(
        gpu.to_host(out), F.warp_flow(stack, u, v, CPU), rtol=1e-5, atol=1e-5
    )


def test_gpu_memory_within_estimate(gpu, radar):
    import cupy as cp

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    peak = [0]

    class Peak(cp.cuda.MemoryHook):
        name = "peak"

        def alloc_postprocess(self, **kw):
            peak[0] = max(peak[0], pool.total_bytes())

    master = radar[:1024, :1024]
    with Peak():
        g.gefolki(master, np.roll(master, 2, axis=1), device="gpu")
    assert peak[0] <= estimate_gpu_bytes_per_pixel(FlowParams(contrast_adapt=True)) * master.size
    assert pool.total_bytes() == 0  # cached blocks handed back after the run


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
