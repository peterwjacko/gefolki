import numpy as np
import pytest

import gefolki as g


def ramp(shape=(40, 50), dtype=np.float32):
    y, x = np.mgrid[: shape[0], : shape[1]]
    return (100 * y + x + 1).astype(dtype)


def const_flow(shape, u, v):
    return np.full(shape, u, np.float32), np.full(shape, v, np.float32)


def test_integer_shift_and_outside_fill():
    img = ramp()
    u, v = const_flow(img.shape, 2, -1)
    out = g.warp(img, u, v, device="cpu")
    np.testing.assert_array_equal(out[1:, :-2], img[:-1, 2:])
    assert (out[0] == 0).all() and (out[:, -2:] == 0).all()


def test_fractional_shift_bilinear():
    img = ramp()
    out = g.warp(img, *const_flow(img.shape, 0.5, 0.25), device="cpu")
    np.testing.assert_allclose(out[:-1, :-1], img[:-1, :-1] + 0.5 + 25, rtol=1e-6)


def test_stack_dtype_and_nodata():
    img = ramp(dtype=np.uint16)
    stack = np.stack([img, img * 2])
    stack[:, 10, 10] = 0  # nodata pixel in all bands
    stack[1, 20, 20] = 0  # zero in one band only: still valid
    u, v = const_flow(img.shape, 0.5, 0.0)
    out = g.warp(stack, u, v, nodata=0, device="cpu", threads=4)
    assert out.shape == stack.shape and out.dtype == np.uint16
    assert (out[:, 10, 9:11] == 0).all()  # samples touching the nodata pixel
    assert out[0, 10, 8] == np.rint((float(img[10, 8]) + img[10, 9]) / 2)  # rounded
    assert out[1, 20, 20] == round(0.5 * (0 + 2 * img[20, 21]))
    assert (out[:, :, -1] == 0).all()  # sampled beyond right edge


def test_nan_nodata():
    img = ramp()
    img[5, 5] = np.nan
    out = g.warp(img, *const_flow(img.shape, 0.3, 0.3), nodata=np.nan, device="cpu")
    assert np.isnan(out[4:6, 4:6]).all()
    assert np.isfinite(out[:-1, :-1]).sum() == (img.shape[0] - 1) * (img.shape[1] - 1) - 4


def test_cubic_and_bad_shape():
    img = ramp()
    out = g.warp(img, *const_flow(img.shape, 1.0, 0.0), order=3, device="cpu")
    np.testing.assert_allclose(out[:, :-1], img[:, 1:], atol=1e-3)
    with pytest.raises(ValueError):
        g.warp(img, np.zeros((3, 3)), np.zeros((3, 3)), device="cpu")
