"""CPU-only checks for CD's tiling memory model, window wiring and stop rule.

Solver-neutral window sizing and row windows are covered in test_tiling.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from utils import kernel, require_nmf_runtime


# Hand-computed budgets for G=13, K=3, B=2; H is already allocated.
# Per row: W and cross windows (2BK) plus one H·Xᵀ output (K); factorization
# adds the X window (G); fixed-H takes the larger of that and its one-off
# product pass (G + BK + K). The torch sweep adds 10B values + 8B index bytes.
# Fixed: orders and state B(8K + 9) bytes + 3B values, plus (B + 1)K² Gram
# values; factorization adds BK² + (B + 1)KG values and, with the torch
# sweep, G times its per-row sweep bytes.
@pytest.mark.parametrize("update_h,dtype,torch_sweep,fixed,per_row", [
    (False, np.float32, False, 198, 88),
    (False, np.float64, False, 330, 176),
    (True, np.float32, False, 738, 112),
    (True, np.float64, False, 1410, 224),
    (False, np.float32, True, 198, 156),
    (True, np.float32, True, 1986, 208),
])
def test_cd_window_budget(kernel, update_h, dtype, torch_sweep, fixed, per_row):
    itemsize = np.dtype(dtype).itemsize
    assert kernel.solver_cd._window_budget(
        update_h, 2, 3, 13, itemsize, torch_sweep
    ) == (fixed, per_row)


@pytest.mark.parametrize("update_h,torch_sweep", [(False, False), (True, False), (True, True)])
def test_fit_cd_sizes_auto_windows_from_the_cd_budget(kernel, monkeypatch, update_h, torch_sweep):
    cd = kernel.solver_cd
    monkeypatch.setattr(cd, "_HALS_CUDA_BACKEND", None if torch_sweep else object())
    window_rows = []
    loop = "_fit_cd_factorize" if update_h else "_fit_cd_fixed_h"
    monkeypatch.setattr(cd, loop, lambda *args: window_rows.append(args[5]))
    fixed, per_row = cd._window_budget(update_h, 2, 3, 13, 4, torch_sweep)
    reserve = kernel.tiling._VRAM_RESERVE_BYTES
    free_bytes = reserve + fixed + 8 * per_row - 1
    backend = SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda device: (free_bytes, 16 * (1 << 30))
    ))
    rc = SimpleNamespace(k=3, device="cuda:7", opt={"row_tiling_ratio": 0})
    X = np.empty((2065, 13), dtype=np.float32)

    def fit():
        cd._fit_cd(backend, rc, X, None, None, [0, 1], {}, update_h)

    fit()
    assert window_rows == [7]
    free_bytes = reserve + fixed + per_row - 1
    mode = "factorization" if update_h else "fixed-H"
    with pytest.raises(MemoryError, match=f"GPU {mode} CD .* one row"):
        fit()


@pytest.mark.parametrize("tol", [0, 1e-4])
def test_stop_rule_zero_initial_violation_stops_immediately(kernel, tol):
    torch = require_nmf_runtime()
    stop = kernel.solver_cd._StopRule(torch, 2, tol, "cpu")
    assert stop.update(1, torch.zeros(2))
    assert stop.n_iter.tolist() == [1, 1]
    assert stop.active.tolist() == [False, False]


def test_stop_rule_tracks_and_freezes_each_replicate(kernel):
    torch = require_nmf_runtime()
    stop = kernel.solver_cd._StopRule(torch, 3, 0.1, "cpu")
    initial = torch.tensor([10.0, 20.0, 0.0], dtype=torch.float64)
    assert not stop.update(1, initial)
    initial.zero_()  # The stored baseline must own its data.
    assert stop.violation_init.tolist() == [10.0, 20.0, 0.0]
    assert not stop.update(2, torch.tensor([1.0, 4.0, 0.0]))
    assert stop.active.tolist() == [False, True, False]
    assert stop.n_iter.tolist() == [2, 2, 1]
    assert stop.update(3, torch.tensor([999.0, 1.0, 999.0]))
    assert stop.active.tolist() == [False, False, False]
    assert stop.n_iter.tolist() == [2, 3, 1]


def test_stop_rule_tol_zero_still_stops_at_exact_zero(kernel):
    torch = require_nmf_runtime()
    stop = kernel.solver_cd._StopRule(torch, 1, 0, "cpu")
    assert not stop.update(1, torch.tensor([1.0]))
    assert not stop.update(2, torch.tensor([1e-20]))
    assert stop.update(3, torch.tensor([0.0]))
    assert stop.n_iter.item() == 3


@pytest.mark.parametrize("update_h", [False, True], ids=["fixed-h", "factorize"])
def test_row_windows_never_upload_the_full_matrix(kernel, monkeypatch, update_h):
    """Below ratio 1, X, W and cross-products move one window at a time."""
    torch = require_nmf_runtime()
    rng = np.random.default_rng(5)
    n_rows = 101  # Distinct from every other dimension below.
    X = rng.random((n_rows, 7)) + 0.1
    nmf_kwargs = dict(
        solver="cd", n_components=3, init="random", max_iter=4, tol=0.0,
        update_H=update_h,
    )
    if not update_h:
        nmf_kwargs["H"] = rng.random((3, 7)) + 0.1
    shapes = []
    as_tensor = torch.as_tensor

    def recording_as_tensor(value, *args, **kwargs):
        shapes.append(np.shape(value))
        return as_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", recording_as_tensor)
    outputs = kernel.solver_cd._nmf_gpu_cd(
        X, [0, 1], nmf_kwargs,
        dict(device="cpu", dtype="fp64", row_tiling_ratio=0.25),
    )

    window = int(n_rows * 0.25)
    row_counts = [shape[0] if len(shape) == 2 else shape[-1] for shape in shapes if len(shape) >= 2]
    assert n_rows not in row_counts
    assert max(row_counts) == max(window, 7)  # 7 is H's feature axis.
    assert all(np.isfinite(W).all() and (W >= 0).all() for _, W in outputs)
