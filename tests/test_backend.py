import numpy as np
import pytest

import gefolki as g
from gefolki import backend
from gefolki.backend import get_backend


def test_cpu_backend():
    bk = get_backend("cpu", threads=3)
    assert bk.name == "cpu" and bk.xp is np and bk.threads == 3
    a = bk.asarray([1, 2])
    assert a.dtype == np.float32
    assert isinstance(bk.to_host(a), np.ndarray)


def test_backend_passthrough_and_bad_name():
    bk = get_backend("cpu")
    assert get_backend(bk) is bk
    with pytest.raises(ValueError):
        get_backend("tpu")


def test_gpu_unavailable_raises(monkeypatch):
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    monkeypatch.setattr(backend, "_gpu_error", "test reason")
    with pytest.raises(RuntimeError, match="test reason"):
        get_backend("gpu")
    assert get_backend("auto").name == "cpu"


def test_backend_info():
    info = g.backend_info()
    assert info["default"] in ("cpu", "gpu") and info["cpu_threads"] >= 1
    assert info["numpy"] == np.__version__
    assert info["gpu_available"] == (info["default"] == "gpu")
    if info["gpu_available"]:
        gpu = info["gpu"]
        assert gpu["name"] and gpu["memory_total"] >= gpu["memory_free"] > 0
        assert "." in gpu["compute_capability"]
    else:
        assert info["gpu_error"]
