from pathlib import Path

import numpy as np
import pytest
import rasterio

from gefolki import gpu_available

DATA = Path(__file__).resolve().parents[1] / "datasets"


def pytest_collection_modifyitems(config, items):
    if gpu_available():
        return
    skip = pytest.mark.skip(reason="no usable CUDA GPU / CuPy")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def read_band(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(band).astype(np.float32)


@pytest.fixture(scope="session")
def radar() -> np.ndarray:
    """Radar P-band image (first channel), 2000x2000 float32."""
    return read_band(DATA / "radar_bandep.png")


@pytest.fixture(scope="session")
def optical() -> np.ndarray:
    """Optical image co-located with ``radar`` (green channel)."""
    return read_band(DATA / "optiquehr_georef.png", 2)
