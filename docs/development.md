# Development

```bash
git clone https://github.com/peterwjacko/gefolki && cd gefolki
uv venv -p 3.13 .venv && uv sync --extra gpu --extra numba --extra dev
python datasets/fetch.py               # sample data for tests and examples (~160 MB)
```

## Tests and linting

```bash
uv run pytest                                   # all tests; GPU tests skip without a usable GPU
uv run pytest -m "not gpu and not slow and not data"
uv run pytest tests/test_flow.py -k evalgefolki # one test
uv run ruff check . && uv run ruff format --check .
```

Markers:

| marker | needs | without it |
|---|---|---|
| `gpu` | CuPy and a usable CUDA GPU | skipped |
| `slow` | time | runs (deselect with `-m "not slow"`) |
| `data` | private Liffey airborne data: `GEFOLKI_LIFFEY_DATA` (subsets), `GEFOLKI_LIFFEY_RGB` (full RGB ortho) | skipped |

Tests that need a sample file (`datasets/`) skip with a hint to run `datasets/fetch.py`.
Never commit data: `datasets/` ignores everything except its README and fetch script.

CI (`.github/workflows/ci.yml`) runs on Python 3.12 and 3.13: `uv sync --locked --extra
dev`, fetch of the sample files the tests use (cached), `ruff check`, `ruff format
--check` and `pytest -m "not gpu"`. Run `uv lock` after changing dependencies.

## Benchmarks

```bash
uv run python benchmarks/bench_flow.py --size 2048 --variant gefolki efolki
```

Best-of-2 wall times for CPU (1 thread and all threads) and GPU, transfers included, and
the GPU memory peak per pixel.

## Documentation

Markdown in `docs/`, rendered by GitHub (diagrams in Mermaid). Figures are generated from
the sample data:

```bash
python datasets/fetch.py
uv run --with matplotlib --with tifffile python docs/figures/make_figures.py          # all
uv run --with matplotlib --with tifffile python docs/figures/make_figures.py radius   # one
```

Photographic figures are JPEG (quality 85), line art PNG; keep each under ~150 kB.

## Style

- ruff, line length 100; type hints; short docstrings; no speculative abstractions.
- Core functions take NumPy or CuPy 2-D arrays and return NumPy float32 unless
  `return_device=True`; keep data on the device across levels and iterations.
- Keep the flow sign convention: `slave(x + u, y + v) ≈ master(x, y)`.
- `examples/quickstart.ipynb` is committed without outputs.

## Release

```bash
uv build      # wheel and sdist in dist/ (datasets and doc figures excluded from the sdist)
```

Update `CHANGELOG.md` and the version in `pyproject.toml`.
