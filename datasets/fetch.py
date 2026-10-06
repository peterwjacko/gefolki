"""Download the GeFolki sample datasets from the original ONERA repository.

The files are not stored in this repository. They come from
https://github.com/aplyer/gefolki/tree/<COMMIT>/datasets, pinned to one commit and checked
by SHA-256. Files already present with the right hash are skipped.

    python datasets/fetch.py                 # everything (~160 MB)
    python datasets/fetch.py WV.tif QB.tif   # some files
    python datasets/fetch.py EvalGeFolki     # a folder (any path prefix works)
    python datasets/fetch.py --list          # names and sizes

Standard library only. See datasets/README.md for sources and licences.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "3a77736109ef66470349fb82f2d04ae161a64850"
BASE = f"https://raw.githubusercontent.com/aplyer/gefolki/{COMMIT}/datasets/"
HERE = Path(__file__).resolve().parent
HR = "EvalGeFolki/HR/lngley_05521_09059_000_090813_L090_CX_01_pauli_x016_y020"

# path: (size in bytes, sha256)
FILES = {
    "0414.png": (11701, "113dd32fe0e2392c8fec5fbb7cab693ff1d8e488322e4c3fc2d737fd51e164d0"),
    "4645.png": (16557, "1b761407173a08c59d97d9058c42cff29dc78f33c39b689bffe3b622d550a205"),
    f"{HR}_flow.tif": (8583346, "eb925e385b27db2aa4f06c8e7e5b40950e3e0237e4df22c4dab57f25b2458361"),
    f"{HR}_optique.tif": (
        3150444,
        "42f30b63a9e5709b762901ce0a68126f87b64bb5ede0909783a99fed6b8c10f8",
    ),
    f"{HR}_radar.tif": (
        3150254,
        "40c8eadab4ccb601b863aa21a83cd88d7df22f875b4a64a3cad84904a867ee9a",
    ),
    "EvalGeFolki/S1S2/Flow_patch11.tif": (
        1690930,
        "fefa7980f7f97cb421db7bab75c9f8160517a07d04c9b9e318f7e131091f6162",
    ),
    "EvalGeFolki/S1S2/S1_patch11.tif": (
        1607796,
        "bbd89b1d42e2cd2b90a54f9b262a7758b1eef75c6e9d9baa51e295ec75165ce0",
    ),
    "EvalGeFolki/S1S2/S2_patch11.tif": (
        2412416,
        "619fc68a5ffb3f17defa72edfc20df3c8a4a1656edcd8fee1943097ac260be7c",
    ),
    "JacksonvilleNavalAirStation_sandiaKu.png": (
        202939,
        "c16fc968092c05f8c35e46dbb804b90e61b31eff0d1b4e11164f72f78ed7b54e",
    ),
    "QB.tif": (11305510, "ae2701a5d1d3acb97520ba3a56f62662cd171f2023ff9363310a5970557bf0a0"),
    "S1_Jacksonville_GEE.tif": (
        9079247,
        "b8e7a88e0dd2c6927d88530668d249ad86d0e12e55ffd9282236644b7bb35f07",
    ),
    "S1_Washington_GEE.tif": (
        8591090,
        "53511179003793623ccc84f92a0afae82029543fa67ce4ec091e23b901e1b3d2",
    ),
    "WV.tif": (8076400, "5bca96283ede500fa2b6f9acf5f6cb05317df8883682e246e2bc7b15399654dc"),
    "WhashingtonDC_sandiaKu_project.png": (
        1360659,
        "9304cf3fb9c30adc4533c4cb108bca9f3f8d6740271a7fbeec8b2d4707d8e9ad",
    ),
    "lidar_georef.png": (
        2284432,
        "a89fc46a1b0c565db7918a56a55c6cd3fad837702b72118f314ed198d125b67a",
    ),
    "optiquehr_georef.png": (
        6760910,
        "221caf71f2c879cd464d70f2772fe7986b3b58321991a2f4afa6cbc54b7e8e6c",
    ),
    "radar_bandel_hh1.mat": (
        37017094,
        "6a0f879bfdb7b8e34738bd1765ba4b4ca05023631e329239d1c49ad36d006d6d",
    ),
    "radar_bandel_hh2.mat": (
        45739455,
        "6cbc7b83cb1586992ffa401e1c1262b93dc3f979b276420fb256e2a5664feeb1",
    ),
    "radar_bandep.png": (
        7657532,
        "4463e9a70695ff18bba436257173da11ecd9f35df5d29e285bf9a1e2a2a44f56",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> str:
    """Download ``name`` into ``dest`` unless present and valid; returns a status word."""
    size, digest = FILES[name]
    path = dest / name
    if path.is_file() and path.stat().st_size == size and sha256(path) == digest:
        return "ok"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(BASE + name, timeout=60) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if sha256(tmp) != digest:
        tmp.unlink()
        raise RuntimeError(f"{name}: SHA-256 mismatch")
    tmp.replace(path)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="files or path prefixes (default: all)")
    ap.add_argument("--dest", type=Path, default=HERE, help="target folder (default: here)")
    ap.add_argument("--list", action="store_true", help="list files and exit")
    a = ap.parse_args(argv)
    if a.list:
        for name, (size, _) in FILES.items():
            print(f"{size / 1e6:8.2f} MB  {name}")
        return 0
    names = [n for n in FILES if not a.names or any(n.startswith(p) for p in a.names)]
    if not names:
        ap.error(f"no dataset matches {a.names}")
    for name in names:
        print(f"{fetch(name, a.dest):10s} {name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
