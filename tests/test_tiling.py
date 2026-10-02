"""Solver-neutral row tiling: window sizing and host/device row windows."""

from types import SimpleNamespace

import numpy as np
import pytest

from utils import kernel, require_nmf_runtime


def _budget_never_used():
    raise AssertionError("only automatic sizing on CUDA may evaluate the budget")


@pytest.mark.parametrize("ratio,expected", [(None, 13), (1, 13), (0.5, 6), (1 / 3, 4), (1e-9, 1)])
def test_explicit_ratios_never_query_cuda(kernel, ratio, expected):
    # No CUDA API exists on this fake backend: explicit sizing must not use it.
    assert kernel.tiling._resolve_window_rows(
        SimpleNamespace(), ratio, "cuda:7", 13, _budget_never_used, "test solver"
    ) == expected


def test_auto_ratio_off_cuda_keeps_all_rows(kernel):
    assert kernel.tiling._resolve_window_rows(
        SimpleNamespace(), 0, "cpu", 13, _budget_never_used, "test solver"
    ) == 13


def test_auto_ratio_fills_free_cuda_memory_after_the_reserve(kernel, monkeypatch):
    tiling = kernel.tiling
    fixed, per_row = 1000, 100
    reserve = tiling._VRAM_RESERVE_BYTES
    free_bytes = reserve + fixed + 8 * per_row - 1
    devices = []

    def mem_get_info(device):
        devices.append(device)
        return free_bytes, 16 * (1 << 30)

    backend = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=mem_get_info))

    def rows():
        return tiling._resolve_window_rows(
            backend, 0, "cuda:7", 2065, lambda: (fixed, per_row), "test solver"
        )

    assert rows() == 7
    free_bytes = reserve + fixed + per_row
    assert rows() == 1
    free_bytes -= 1
    with pytest.raises(MemoryError, match="GPU test solver .* enough VRAM for one row"):
        rows()
    free_bytes = 8 * (1 << 30)
    assert rows() == 2065  # The full matrix stays resident when it fits.
    assert devices == ["cuda:7"] * 4

    # The fractional reserve applies once it exceeds the absolute minimum.
    monkeypatch.setattr(tiling, "_VRAM_RESERVE_BYTES", 0)
    monkeypatch.setattr(tiling, "_VRAM_RESERVE_FRACTION", 0.25)
    free_bytes = (4 * (fixed + 7 * per_row) + 2) // 3
    assert rows() == 7


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf, "nan", "inf", "invalid"])
def test_invalid_row_tiling_ratios_are_rejected(kernel, value):
    with pytest.raises(ValueError, match="row tiling ratio"):
        kernel.utils._resolve_gpu_opts({"row_tiling_ratio": value})


@pytest.mark.parametrize("value,expected", [(None, None), (0, 0), ("0.5", 0.5), (1, 1)])
def test_valid_row_tiling_ratios_are_parsed(kernel, value, expected):
    assert kernel.utils._resolve_gpu_opts({"row_tiling_ratio": value})[
        "row_tiling_ratio"
    ] == expected


def _host_arrays(n_rows):
    X = np.arange(n_rows * 5, dtype=np.float64).reshape(n_rows, 5)
    Wt = np.arange(2 * 3 * n_rows, dtype=np.float64).reshape(2, 3, n_rows)
    return X, Wt


def _copying_backend(torch, uploads):
    # CPU tensors would alias NumPy; copy so device and host stay distinct.
    def as_tensor(value, **kwargs):
        uploads.append(np.shape(value))
        return torch.as_tensor(value, **kwargs).clone()

    return SimpleNamespace(as_tensor=as_tensor)


@pytest.mark.parametrize("window_rows", [13, 20])
def test_row_windows_keep_a_single_window_resident(kernel, window_rows):
    torch = require_nmf_runtime()
    X, Wt = _host_arrays(13)
    original_Wt = Wt.copy()
    uploads = []
    windows = kernel.tiling._RowWindows(
        _copying_backend(torch, uploads), torch.empty(0, dtype=torch.float64),
        13, window_rows, {"X": (X, 0), "Wt": (Wt, 2)}, writable=("Wt",),
    )

    assert windows.resident
    for _ in range(3):
        bounds = []
        for start, stop, tensors in windows:
            bounds.append((start, stop))
            tensors["Wt"].add_(1)
        assert bounds == [(0, 13)]
    assert uploads == [X.shape, Wt.shape]  # Uploaded once, not per pass.
    np.testing.assert_array_equal(windows.host("Wt"), original_Wt + 3)
    np.testing.assert_array_equal(Wt, original_Wt)  # Resident: host untouched.


@pytest.mark.parametrize("window_rows", [1, 4, 5, 12])
def test_row_windows_stream_rows_and_write_back_writable_arrays(kernel, window_rows):
    torch = require_nmf_runtime()
    n_rows = 13
    X, Wt = _host_arrays(n_rows)
    original_X, expected_Wt = X.copy(), Wt.copy()
    uploads = []
    windows = kernel.tiling._RowWindows(
        _copying_backend(torch, uploads), torch.empty(0, dtype=torch.float64),
        n_rows, window_rows, {"X": (X, 0), "Wt": (Wt, 2)}, writable=("Wt",),
    )

    assert not windows.resident
    for _ in range(2):
        bounds = []
        for start, stop, tensors in windows:
            bounds.append((start, stop))
            np.testing.assert_array_equal(tensors["X"].numpy(), X[start:stop])
            np.testing.assert_array_equal(tensors["Wt"].numpy(), expected_Wt[:, :, start:stop])
            tensors["X"].add_(100)  # Not writable: must never reach the host.
            tensors["Wt"].add_(1)
            expected_Wt[:, :, start:stop] += 1
            held = tensors
        assert held == {}  # Released once the caller moved on.
        assert bounds == [
            (start, min(start + window_rows, n_rows))
            for start in range(0, n_rows, window_rows)
        ]
    assert len(uploads) == 2 * 2 * len(bounds)  # X and Wt, per window, per pass.
    np.testing.assert_array_equal(windows.host("Wt"), expected_Wt)
    np.testing.assert_array_equal(X, original_X)


def test_row_windows_write_back_even_when_the_caller_breaks(kernel):
    torch = require_nmf_runtime()
    X, Wt = _host_arrays(13)
    expected = Wt.copy()
    expected[:, :, :4] += 1
    windows = kernel.tiling._RowWindows(
        _copying_backend(torch, []), torch.empty(0, dtype=torch.float64),
        13, 4, {"X": (X, 0), "Wt": (Wt, 2)}, writable=("Wt",),
    )

    for _, _, tensors in windows:
        tensors["Wt"].add_(1)
        break
    np.testing.assert_array_equal(Wt, expected)
