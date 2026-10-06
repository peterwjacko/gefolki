"""Register georectified airborne hyperspectral flight lines to an RGB orthomosaic.

Run this on every line *before* atmospheric correction: each line keeps its own grid,
CRS, dtype, nodata, band names and wavelengths; only pixel content moves (dense,
non-rigid flow from GeFolki). The RGB ortho is resampled onto each line's grid on the fly
(overviews are used), so it can be far larger than memory.

    python examples/register_hyperspectral.py RGB_ORTHO.tif LINES_DIR OUT_DIR [options]

LINES_DIR holds ENVI files (data + .hdr; .bsq/.bil/.img/.dat/no suffix) and/or GeoTIFFs.
Output: OUT_DIR/<line>_reg.<ext> (same format as the input unless --format) and, with
--flow, OUT_DIR/<line>_flow.tif (band 1 u = column shift, band 2 v = row shift, px).

CLI equivalent for one line:
    gefolki register RGB_ORTHO.tif line.bsq line_reg.bsq --preset hyperspectral-rgb \
        --slave-bands 500-600 --flow-output line_flow.tif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gefolki
from gefolki import io

TIFF_SUFFIXES = {".tif", ".tiff"}


def find_lines(directory: Path) -> list[Path]:
    """ENVI data files (found through their .hdr) and GeoTIFFs in ``directory``."""
    lines = []
    for hdr in sorted(directory.glob("*.hdr")):
        stem = str(hdr.with_suffix(""))
        data = [Path(stem + s) for s in sorted(io.ENVI_SUFFIXES)]
        lines += [p for p in data if p.is_file()][:1]
    lines += sorted(p for p in directory.iterdir() if p.suffix.lower() in TIFF_SUFFIXES)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rgb", type=Path, help="RGB(A) orthomosaic (master)")
    ap.add_argument("lines", type=Path, help="directory of georectified flight lines")
    ap.add_argument("out", type=Path, help="output directory")
    ap.add_argument("--preset", default="hyperspectral-rgb", choices=list(gefolki.PRESETS))
    ap.add_argument("--slave-bands", default="500-600", help="nm range or band list")
    ap.add_argument("--master-bands", default="rgb-gray")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--format", choices=["GTiff", "COG", "ENVI"], help="default: as input")
    ap.add_argument("--flow", action="store_true", help="also write each line's flow")
    ap.add_argument("--overwrite", action="store_true", help="redo lines already written")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    lines = find_lines(args.lines)
    if not lines:
        raise SystemExit(f"no ENVI or GeoTIFF rasters in {args.lines}")
    print(f"{len(lines)} lines, preset {args.preset}: {gefolki.PRESETS[args.preset]}")
    t0 = time.perf_counter()
    for i, line in enumerate(lines, 1):
        is_tif = line.suffix.lower() in TIFF_SUFFIXES
        fmt = args.format or ("GTiff" if is_tif else "ENVI")
        ext = {"GTiff": ".tif", "COG": ".tif", "ENVI": ".bsq"}[fmt]
        out = args.out / f"{line.stem}_reg{ext}"
        flow = args.out / f"{line.stem}_flow.tif" if args.flow else None
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(lines)}] {line.name}: exists, skipped")
            continue
        res = gefolki.register(
            args.rgb,
            line,
            out,
            preset=args.preset,
            master_bands=args.master_bands,
            slave_bands=args.slave_bands,
            device=args.device,
            output_format=fmt,
            flow_output=flow,
        )
        s = res.flow_stats
        print(
            f"[{i}/{len(lines)}] {line.name} -> {out.name}: flow median {s['median']:.2f} px, "
            f"p95 {s['p95']:.2f} px, {res.timings['total']:.1f} s"
        )
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
