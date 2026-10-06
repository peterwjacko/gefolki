"""Registration on private Liffey airborne data (skipped when absent).

GEFOLKI_LIFFEY_DATA points at the subset directory (line_P3/, tiny/); GEFOLKI_LIFFEY_RGB at the
full RGB orthomosaic (slow test). Numbers are printed
(run with ``-s``) and recorded as junit properties.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from skimage.registration import phase_cross_correlation

import gefolki as g
from gefolki import io

DATA = Path(os.environ.get("GEFOLKI_LIFFEY_DATA", "/nonexistent"))
FULL_RGB = Path(os.environ.get("GEFOLKI_LIFFEY_RGB", "/nonexistent"))
P3 = DATA / "line_P3"
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not P3.is_dir(), reason="Liffey subset not available"),
]


def _norm(a, m):
    a = np.where(m, a, a[m].mean())
    return (a - a.mean()) / a.std()


def residual(a, b, m, margin=64):
    """Phase-correlation shift magnitude (px) between a and b on the central region."""
    c = np.s_[margin:-margin, margin:-margin]
    s, _, _ = phase_cross_correlation(_norm(a, m)[c], _norm(b, m)[c], upsample_factor=20)
    return float(np.hypot(*s))


def _check_line(master, tmp_path, record_property, tag):
    slave = P3 / "P3_rad_geo_unregistered_16b.tif"
    ref = P3 / "P3_rad_registered_16b.tif"
    res = g.register(master, slave, tmp_path / "reg.tif", flow_output=tmp_path / "flow.tif")
    si = io.read_info(slave)
    green = io.select_bands(si, "500-600")
    out, ovalid = io.read_selection(res.output, green)
    before, _ = io.read_selection(slave, green)
    refimg, rvalid = io.read_selection(ref, green)
    rgb, mvalid = io.read_on_grid(master, io.select_bands(io.read_info(master), 2), si.crs,
                                  si.transform, si.shape)  # fmt: skip
    m = ovalid & rvalid & mvalid
    r_before, r_out = residual(rgb, before, m), residual(rgb, out, m)
    r_ref = residual(rgb, refimg, m)
    # Flow implied by the reference product: unregistered -> reference (same sensor).
    u_ref, v_ref = g.efolki(refimg, before, levels=5, radius=(16, 8), mask=m)
    u, v = io.read_flow(res.flow_path)
    du, dv = np.median(np.abs(u - u_ref)[m]), np.median(np.abs(v - v_ref)[m])
    nums = dict(before=r_before, ours=r_out, reference=r_ref, flow_vs_ref_du=du,
                flow_vs_ref_dv=dv, **res.flow_stats, **res.timings)  # fmt: skip
    for k, val in nums.items():
        record_property(f"{tag}_{k}", round(float(val), 3))
    print(f"\n[{tag}]", {k: round(float(val), 3) for k, val in nums.items()})
    assert r_before > 10
    assert r_out <= 1.5
    assert du < 2 and dv < 2


def test_p3_against_rgb_on_grid(tmp_path, record_property):
    _check_line(P3 / "rgb_on_P3grid_40cm.tif", tmp_path, record_property, "p3_grid")


@pytest.mark.slow
@pytest.mark.skipif(not FULL_RGB.exists(), reason="full RGB orthomosaic not available")
def test_p3_against_full_orthomosaic(tmp_path, record_property):
    _check_line(FULL_RGB, tmp_path, record_property, "p3_full")


@pytest.mark.skipif(not (DATA / "tiny").is_dir(), reason="tiny subset not available")
def test_tiny_envi_slave(tmp_path, record_property):
    tiny = DATA / "tiny"
    slave = tiny / "liffey_fx10_refl_tiny512_16b.bsq"
    master = tiny / "liffey_rgb_tiny_10cm.tif"
    res = g.register(master, slave, tmp_path / "reg.bsq")
    hdr = (tmp_path / "reg.hdr").read_text()
    for key in ("wavelength = {", "fwhm = {", "wavelength units = Nanometers",
                "data ignore value = 0"):  # fmt: skip
        assert key in hdr
    si, oi = io.read_info(slave), io.read_info(res.output)
    np.testing.assert_allclose(oi.wavelengths, si.wavelengths)
    assert (oi.crs, oi.transform, oi.dtype, oi.nodata) == (si.crs, si.transform, si.dtype, 0)
    green = io.select_bands(si, "500-600")
    out, ov = io.read_selection(res.output, green)
    before, bv = io.read_selection(slave, green)
    rgb, mv = io.read_on_grid(master, io.select_bands(io.read_info(master), 2), si.crs,
                              si.transform, si.shape)  # fmt: skip
    m = ov & bv & mv
    r0, r1 = residual(rgb, before, m), residual(rgb, out, m)
    record_property("tiny_before", round(r0, 3))
    record_property("tiny_after", round(r1, 3))
    print(f"\n[tiny] residual before {r0:.2f} px, after {r1:.2f} px, {res.flow_stats}")
    assert r1 <= 1.5
