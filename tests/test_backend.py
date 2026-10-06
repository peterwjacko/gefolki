import numpy as np
import pytest

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
