"""Time GeFolki flow on a 2048x2048 pair: CPU 1 thread vs N threads vs GPU.

Usage: python benchmarks/bench_flow.py [--size 2048] [--threads N] [--variant gefolki]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import rasterio

import gefolki as g

DATA = Path(__file__).resolve().parents[1] / "datasets"


def make_pair(n: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(DATA / "radar_bandep.png") as ds:
        img = ds.read(1).astype(np.float32)
    reps = -(-n // min(img.shape))
    master = np.pad(img, [(0, reps * s - s) for s in img.shape], mode="reflect")[:n, :n]
    y, x = np.mgrid[:n, :n] / n
    u = (3 * np.sin(2 * np.pi * y) + 2).astype(np.float32)  # smooth non-rigid flow, px
    v = (2 * np.cos(2 * np.pi * x) - 1).astype(np.float32)
    return master, g.warp(master, u, v, device="cpu")


def timeit(fn, repeat: int = 2) -> float:
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--variant", choices=["folki", "efolki", "gefolki"], default="gefolki")
    a = ap.parse_args()
    master, slave = make_pair(a.size)
    fn = getattr(g, a.variant)
    print(f"{a.variant} defaults, {a.size}x{a.size} float32")
    runs = [("cpu", 1), ("cpu", a.threads)] + ([("gpu", 1)] if g.gpu_available() else [])
    for dev, th in runs:
        fn(master[:256, :256], slave[:256, :256], device=dev, threads=th)  # warm-up / JIT
        if dev == "gpu":
            fn(master, slave, device=dev)  # compile kernels for this size
        t = timeit(lambda d=dev, n=th: fn(master, slave, device=d, threads=n))
        label = f"{dev} {th} thread(s)" if dev == "cpu" else "gpu"
        print(f"  {label:<18} {t:7.2f} s")


if __name__ == "__main__":
    main()
