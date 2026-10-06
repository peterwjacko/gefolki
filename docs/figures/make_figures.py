"""Regenerate the documentation figures in docs/figures/ from the sample datasets.

    python datasets/fetch.py
    uv run --with matplotlib --with tifffile python docs/figures/make_figures.py [names...]

Runs on the CPU so the figures do not depend on the machine. Names: see FIGURES below.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tifffile
from matplotlib.patches import FancyArrowPatch, Rectangle
from rasterio.errors import NotGeoreferencedWarning

import gefolki as g
from gefolki import filters as F

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
OUT = Path(__file__).resolve().parent
CPU = g.get_backend("cpu")
plt.rcParams.update(
    {"font.size": 9, "axes.titlesize": 9, "figure.dpi": 100, "savefig.bbox": "tight"}
)


def read(name: str, band: int = 1) -> np.ndarray:
    with rasterio.open(DATA / name) as ds:
        return ds.read(band).astype(np.float32)


def stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def gray(a: np.ndarray) -> np.ndarray:
    """Stretched grey RGB, so it stays grey in panels with a colour map."""
    return np.dstack([stretch(a)] * 3)


def fuse(master: np.ndarray, slave: np.ndarray) -> np.ndarray:
    """Magenta (slave) / green (master) overlay: aligned structures look grey."""
    m, s = stretch(master), stretch(slave)
    return np.dstack([s, m, s])


def checker(master: np.ndarray, slave: np.ndarray, n: int = 8) -> np.ndarray:
    h, w = master.shape
    yy, xx = np.mgrid[:h, :w]
    tiles = ((yy * n // h) + (xx * n // w)) % 2 == 0
    return np.where(tiles, stretch(master), stretch(slave))


def panels(images, titles, path, cols=None, size=3.0, cmap="gray", colorbar=None):
    cols = cols or len(images)
    rows = -(-len(images) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, im, t) in enumerate(zip(axes.flat, images, titles, strict=False)):
        h = ax.imshow(im, cmap=cmap if im.ndim == 2 else None, interpolation="nearest")
        ax.set_title(t)
        if colorbar and i in colorbar:
            fig.colorbar(h, ax=ax, shrink=0.75, label=colorbar[i])
    fig.tight_layout()
    save(fig, path)


def save(fig, name: str) -> None:
    kw = {"pil_kwargs": {"quality": 85}} if name.endswith(".jpg") else {}
    fig.savefig(OUT / name, dpi=100, **kw)
    plt.close(fig)
    print("wrote", OUT / name)


def radar_optical() -> tuple[np.ndarray, np.ndarray]:
    return read("radar_bandep.png", 1), read("optiquehr_georef.png", 2)


# ----------------------------------------------------------------------------- figures


def flow_convention() -> None:
    """Grid sketch of slave(x + u, y + v) = master(x, y)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    x, y, u, v = 2, 2, 3, 1
    for ax, title, (px, py) in zip(
        axes, ("master", "slave"), ((x, y), (x + u, y + v)), strict=True
    ):
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(5.5, -0.5)
        ax.set_xticks(range(8))
        ax.set_yticks(range(6))
        ax.grid(True, color="0.8")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("column x")
        ax.set_ylabel("row y")
        ax.add_patch(Rectangle((px - 0.5, py - 0.5), 1, 1, color="tab:orange"))
    axes[0].annotate("(x, y)", (x, y), (x - 1.6, y + 1.6), arrowprops={"arrowstyle": "->"})
    axes[1].add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ls="--", ec="tab:orange"))
    axes[1].add_patch(
        FancyArrowPatch((x, y), (x + u, y + v), arrowstyle="-|>", mutation_scale=12, lw=1.5)
    )
    axes[1].text(x + u / 2, y + v / 2 - 0.45, "(u, v)", ha="center")
    axes[1].annotate("(x + u, y + v)", (x + u, y + v), (x + u - 1, y + v + 2.2),
                     arrowprops={"arrowstyle": "->"})  # fmt: skip
    fig.suptitle("flow (u, v) at master pixel (x, y):  slave(x + u, y + v) ≈ master(x, y)")
    fig.tight_layout()
    save(fig, "flow_convention.png")


def pyramid() -> None:
    radar, _ = radar_optical()
    pyr = F.pyramid(radar[500:1524, 500:1524] / 255, 5, CPU)
    titles = [f"level {i}: {p.shape[1]} x {p.shape[0]}" for i, p in enumerate(pyr)]
    panels(pyr, titles, "pyramid.jpg", cols=6, size=1.9)


def rank_filter() -> None:
    radar, optical = radar_optical()
    sl = np.s_[1000:1300, 700:1000]
    r, o = radar[sl] / 255, optical[sl] / 255
    panels(
        [r, o, F.rank_sup(r, 4, CPU), F.rank_sup(o, 4, CPU)],
        ["radar (HH-VV)", "optical (green)", "radar, rank filter (r = 4)",
         "optical, rank filter (r = 4)"],
        "rank_filter.jpg", cols=4, size=2.6,
    )  # fmt: skip


def contrast_inversion() -> None:
    radar, optical = radar_optical()
    sl = np.s_[600:1400, 600:1400]
    r, o = radar[sl] / 255, optical[sl] / 255
    u, v = g.gefolki(r, o, device="cpu")
    h0, h1 = F.clahe(r, CPU), F.clahe(g.warp(o, u, v, device="cpu"), CPU)
    c1 = F.box_filter(np.abs(h0 - h1), 4, CPU)
    c2 = F.box_filter(np.abs(1 - h0 - h1), 4, CPU)
    panels(
        [h0, h1, (c1 > c2).astype(np.float32)],
        ["radar, CLAHE", "optical (registered), CLAHE", "contrast inverted (white)"],
        "contrast_inversion.jpg", size=3.0,
    )  # fmt: skip


def optical_sar() -> None:
    radar, optical = radar_optical()
    u, v = g.gefolki(radar, optical, device="cpu")
    reg = g.warp(optical, u, v, device="cpu")
    sl = np.s_[600:1400, 600:1400]
    panels(
        [fuse(radar[sl], optical[sl]), fuse(radar[sl], reg[sl]), np.hypot(u, v)],
        ["before (radar green, optical magenta)", "after GeFolki", "|flow| (px)"],
        "optical_sar.jpg", size=3.2, cmap="viridis", colorbar={2: "px"},
    )  # fmt: skip


def lidar_sar() -> None:
    radar, lidar = read("radar_bandep.png", 1), read("lidar_georef.png")
    u, v = g.estimate_flow(radar, lidar, g.PRESETS["lidar-sar"], device="cpu")
    reg = g.warp(lidar, u, v, device="cpu")
    sl = np.s_[1100:1500, 500:900]
    panels(
        [fuse(radar[sl], lidar[sl]), fuse(radar[sl], reg[sl]), np.hypot(u, v)],
        ["before (radar green, LIDAR magenta)", "after EFolki (lidar-sar)", "|flow| (px)"],
        "lidar_sar.jpg", size=3.2, cmap="viridis", colorbar={2: "px"},
    )  # fmt: skip


def optical_optical() -> None:
    wv, qb = read("WV.tif"), read("QB.tif")
    mask = wv > 0
    u, v = g.efolki(wv, qb, mask=mask, levels=5, radius=(16, 8), iterations=4, device="cpu")
    reg = g.warp(qb, u, v, device="cpu")
    sl = np.s_[600:900, 1000:1300]
    panels(
        [fuse(wv[sl], qb[sl]), fuse(wv[sl], reg[sl]), np.where(mask, np.hypot(u, v), np.nan)],
        ["before (WV green, QB magenta)", "after EFolki", "|flow| (px)"],
        "optical_optical.jpg", size=3.2, cmap="viridis", colorbar={2: "px"},
    )  # fmt: skip


def radius() -> None:
    """Column shift u minus its median: large radii smooth, small radii follow detail."""
    radar, optical = radar_optical()
    images, titles = [], []
    for rad in ((32,), (32, 24, 16, 8), (8,)):
        u, _ = g.gefolki(radar, optical, radius=rad, device="cpu")
        images.append(u - np.median(u))
        titles.append(f"radius {', '.join(map(str, rad))}")
    lim = np.percentile(np.abs(images[1]), 99)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, im, t in zip(axes, images, titles, strict=True):
        h = ax.imshow(im, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(t)
        ax.axis("off")
    fig.colorbar(h, ax=axes, shrink=0.8, label="u - median(u) (px)")
    save(fig, "radius.png")


def levels() -> None:
    """A 40 px shift: recovered only with enough pyramid levels."""
    wv = read("WV.tif")[300:1068, 700:1468]
    d = 40
    slave = g.warp(wv, np.full(wv.shape, -d, np.float32), np.zeros(wv.shape, np.float32),
                   device="cpu")  # fmt: skip
    inner = np.s_[80:-80, 80:-80]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, lv in zip(axes, (2, 3, 4, 5), strict=True):
        u, _ = g.efolki(wv, slave, levels=lv, radius=(16, 8), iterations=4, device="cpu")
        err = np.abs(u - d)[inner]
        h = ax.imshow(err, cmap="magma", vmin=0, vmax=d)
        ax.set_title(f"levels = {lv}\nmean error {err.mean():.1f} px")
        ax.axis("off")
    fig.colorbar(h, ax=axes, shrink=0.8, label=f"|u - {d}| (px)")
    save(fig, "levels.png")


def evalgefolki() -> None:
    s1 = read("EvalGeFolki/S1S2/S1_patch11.tif")
    s2 = read("EvalGeFolki/S1S2/S2_patch11.tif")
    gt = tifffile.imread(DATA / "EvalGeFolki/S1S2/Flow_patch11.tif")
    slave = g.warp(s1, gt[..., 0], gt[..., 1], device="cpu")
    u, v = g.efolki(s1, slave, levels=3, radius=(16, 8), iterations=4, device="cpu")
    err = np.hypot(u + gt[..., 0], v + gt[..., 1])
    epe = err[30:-30, 30:-30].mean()
    panels(
        [gray(s1), gray(s2), np.hypot(gt[..., 0], gt[..., 1]), err],
        ["Sentinel-1 (band 1)", "Sentinel-2 (band 1)", "ground-truth |flow| (px)",
         f"EFolki error (mean {epe:.2f} px)"],
        "evalgefolki.jpg", cols=4, size=2.6, cmap="viridis",
        colorbar={2: "px", 3: "px"},
    )  # fmt: skip


def insar() -> None:
    """L-band SLC pair: interferogram and coherence before / after registration."""
    from matplotlib.colors import hsv_to_rgb
    from scipy.io import loadmat
    from scipy.ndimage import uniform_filter

    s1 = loadmat(DATA / "radar_bandel_hh1.mat")["Radar_bandeL_HH1"]
    s2 = loadmat(DATA / "radar_bandel_hh2.mat")["Radar_bandeL_HH2"]
    u, v = g.estimate_flow(np.abs(s1), np.abs(s2), g.PRESETS["sar-sar"], device="cpu")
    s2r = g.warp(s2.real.astype(np.float32), u, v, device="cpu") + 1j * g.warp(
        s2.imag.astype(np.float32), u, v, device="cpu"
    )

    def interferogram(a, b, n=7):
        i = a * np.conj(b)
        m = uniform_filter(i.real, n) + 1j * uniform_filter(i.imag, n)
        p = uniform_filter(np.abs(a) ** 2, n) * uniform_filter(np.abs(b) ** 2, n)
        coh = np.abs(m) / np.sqrt(np.maximum(p, 1e-12))
        hsv = np.dstack([(np.angle(m) + np.pi) / (2 * np.pi), coh, stretch(np.abs(a))])
        return hsv_to_rgb(hsv), coh

    sl = np.s_[600:1200, 600:1200]
    (rgb0, c0), (rgb1, c1) = interferogram(s1[sl], s2[sl]), interferogram(s1[sl], s2r[sl])
    panels(
        [rgb0, rgb1, c0, c1],
        ["interferogram, before", "interferogram, after EFolki (sar-sar)",
         f"coherence, before (mean {c0.mean():.2f})", f"coherence, after (mean {c1.mean():.2f})"],
        "insar.jpg", cols=4, size=2.7,
    )  # fmt: skip


def locate() -> None:
    res = g.locate_raster(
        DATA / "S1_Jacksonville_GEE.tif",
        DATA / "JacksonvilleNavalAirStation_sandiaKu.png",
        device="cpu",
    )
    s1 = read("S1_Jacksonville_GEE.tif")
    chip = read("JacksonvilleNavalAirStation_sandiaKu.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), width_ratios=(1.6, 1))
    axes[0].imshow(stretch(s1), cmap="gray")
    axes[0].add_patch(
        Rectangle((res.col, res.row), res.width, res.height, fill=False, ec="tab:orange", lw=2)
    )
    axes[0].set_title(f"Sentinel-1 mean, match at row {res.row}, col {res.col}")
    axes[1].imshow(stretch(chip), cmap="gray")
    axes[1].set_title("Ku-band airborne SAR chip (Sandia)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "locate.jpg")


def validation() -> None:
    radar, optical = radar_optical()
    u, v = g.gefolki(radar, optical, device="cpu")
    reg = g.warp(optical, u, v, device="cpu")
    sl = np.s_[700:1300, 700:1300]
    r, o = radar[sl], reg[sl]
    swipe = np.where(np.arange(o.shape[1])[None, :] < o.shape[1] // 2, stretch(r), stretch(o))
    panels(
        [fuse(r, o), checker(r, o), swipe],
        ["colour composite", "checkerboard", "swipe (left radar, right optical)"],
        "validation.jpg", size=3.0,
    )  # fmt: skip


FIGURES = {
    f.__name__: f
    for f in (flow_convention, pyramid, rank_filter, contrast_inversion, optical_sar,
              lidar_sar, optical_optical, radius, levels, evalgefolki, insar, locate, validation)
}  # fmt: skip

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
