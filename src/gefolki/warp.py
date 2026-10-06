"""Apply a dense flow to an image or a band stack."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import filters as F
from .backend import Backend, Device, get_backend

_EPS = 1e-3  # px tolerance for "inside the image"


def warp(
    image: Any,
    u: Any,
    v: Any,
    *,
    order: int = 1,
    nodata: float | None = None,
    device: Device | Backend = "auto",
    threads: int | None = None,
    return_device: bool = False,
) -> Any:
    """Resample ``image`` at (x + u, y + v): the registered slave for flow from estimate_flow.

    image: 2-D (H, W) or stack (bands, H, W); u, v: (H, W) column/row shifts in pixels.
    order: spline order (0 nearest, 1 bilinear, 3 cubic). nodata: pixels whose sample point
    falls outside the image, or touches an input nodata pixel (equal to ``nodata`` in all
    bands; NaN allowed), are set to ``nodata``. With nodata=None, outside pixels become 0.
    Output keeps the input dtype (integers are rounded and clipped).
    """
    bk = get_backend(device, threads)
    xp = bk.xp
    stack = image.ndim == 3
    bands = image if stack else image[None]
    if bands.ndim != 3 or tuple(u.shape) != bands.shape[1:] or tuple(v.shape) != tuple(u.shape):
        raise ValueError(f"image {image.shape} and flow {u.shape}/{v.shape} do not match")
    rows, cols = bands.shape[1:]
    dtype = np.dtype(image.dtype)
    fill = 0 if nodata is None else nodata

    xs = xp.arange(cols, dtype=xp.float32)[None, :] + bk.asarray(u)
    ys = xp.arange(rows, dtype=xp.float32)[:, None] + bk.asarray(v)
    bad = (xs < -_EPS) | (xs > cols - 1 + _EPS) | (ys < -_EPS) | (ys > rows - 1 + _EPS)
    xs, ys = xp.clip(xs, 0, cols - 1), xp.clip(ys, 0, rows - 1)

    if nodata is not None:
        ixp = np if isinstance(bands, np.ndarray) else xp
        invalid = ixp.ones((rows, cols), bool)
        for b in bands:
            invalid &= ixp.isnan(b) if np.isnan(nodata) else (b == nodata)
        if bool(invalid.any()):
            valid_w = F.interp2(bk.asarray(~invalid), xs, ys, bk, order=min(order, 1))
            bad |= valid_w < 1 - 1e-4

    out = (xp if return_device else np).empty(bands.shape, dtype)
    for i, b in enumerate(bands):
        b = bk.asarray(b)
        if nodata is not None and np.isnan(nodata):
            b = xp.nan_to_num(b, nan=0.0)  # NaN would leak through zero interpolation weights
        w = F.interp2(b, xs, ys, bk, order=order)
        w = xp.where(bad, xp.float32(fill), w)
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            w = xp.clip(xp.rint(w), info.min, info.max)
        w = w.astype(dtype)
        out[i] = w if return_device else bk.to_host(w)
    return out if stack else out[0]
