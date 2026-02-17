# Copilot Instructions for `gefolki`

## Build, test, and lint commands

This repository does not define a formal build system, test suite, or lint configuration (no `pyproject.toml`, `requirements*.txt`, CI workflows, or test files were found).

Use the project demos as executable checks:

- Python coregistration demo:
  - `cd python && python3 main.py`
- Python mining demo (from `README.md`):
  - `cd python && python3 mining.py --input_master '../datasets/S1_Jacksonville_GEE.tif' --input_slave '../datasets/JacksonvilleNavalAirStation_sandiaKu.png'`
- MATLAB demo:
  - `cd matlab` then run `main.m` in MATLAB/Octave.

Single-test equivalent: run one demo path (for example, the `python/mining.py` command above with fixed sample inputs) to validate a targeted workflow.

## High-level architecture

- `python/algorithm.py` exposes the three public registration entry points (`Folki`, `EFolki`, `GEFolki`) by wrapping iterative solvers from `python/folki.py` in a multiscale pyramid driver (`BurtOF` in `python/pyramid.py`).
- `python/pyramid.py` handles coarse-to-fine optical-flow estimation:
  - normalizes both images,
  - builds pyramids with `pyrUp`,
  - runs the selected solver at each level,
  - upsamples flow (`u`, `v`) between levels.
- `python/folki.py` contains the per-level solver logic:
  - `FolkiIter`: base Lucas-Kanade-style flow,
  - `EFolkiIter`: rank-filter-enhanced variant,
  - `GEFolkiIter`: heterogeneous-image variant combining rank filters and local CLAHE preprocessing.
- `python/primitive.py` provides low-level numerical primitives (interpolation and 2D convolution), while `python/tools.py` provides `wrapData` to warp an image using estimated flow.
- `python/main.py` is the end-to-end demo script for LIDAR/RADAR and optical/RADAR registration.
- `python/mining.py` is a separate pattern-localization workflow (coarse decimated search + fine local refinement) and is not part of the optical-flow registration pipeline.

## Key conventions in this repository

- Run Python scripts from inside the `python/` directory; dataset paths in demos are written relative to that location (`../datasets/...`).
- Registration functions generally expect 2D arrays (single-channel images). Demo scripts explicitly select channels (for example `radar[:, :, 0]`) and cast to `float32` before processing.
- The optical-flow displacement convention is `(u, v)` and warping uses `interp2(I, X+u, Y+v)` (`python/tools.py`), so keep this sign/order convention consistent when adding new code.
- `algorithm.py` is the stable API surface for selecting algorithm variants; prefer adding new top-level variants there rather than importing iterative implementations directly in application code.
