"""Batched sklearn-compatible Fast-HALS coordinate-descent NMF solver."""

import numpy as np

from . import tiling, utils

# ---------------------------------------------------------------------
# CD solver (sklearn-compatible Fast-HALS; batch-aware)
# ---------------------------------------------------------------------


def _numpy_staging_dtype(torch, dtype):
    """Return the host dtype used for sklearn initialization before transfer."""
    if dtype is torch.float64:
        return np.float64
    if dtype is torch.float32:
        return np.float32
    raise TypeError("solver='cd' supports gpu dtype fp32 or fp64")


def _to_checked_custom_factor(value, shape, name, dtype):
    """Validate and cast a custom CD factor."""
    if value is None:
        raise ValueError(f"init='custom' requires {name}")
    array = value.toarray() if hasattr(value, "toarray") else np.asarray(value)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"custom {name} shape must be {shape}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"custom {name} contains NaN/inf")
    if array.size and array.min() < 0:
        raise ValueError(f"custom {name} must be non-negative")
    return np.ascontiguousarray(array, dtype=dtype)


def _cd_regularization(nmf_kwargs, n_samples, n_features):
    """Return sklearn's sample/feature-scaled CD regularization terms."""
    alpha_w = float(nmf_kwargs.get("alpha_W", 0.0))
    alpha_h_raw = nmf_kwargs.get("alpha_H", "same")
    alpha_h = alpha_w if alpha_h_raw == "same" else float(alpha_h_raw)
    l1_ratio = float(nmf_kwargs.get("l1_ratio", 0.0))
    if alpha_w < 0 or alpha_h < 0:
        raise ValueError("alpha_W and alpha_H must be non-negative")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be in the range [0, 1]")
    return (
        n_features * alpha_w * l1_ratio,
        n_features * alpha_w * (1.0 - l1_ratio),
        n_samples * alpha_h * l1_ratio,
        n_samples * alpha_h * (1.0 - l1_ratio),
    )


def _validate_cd_runtime(torch, rc, nmf_kwargs):
    """Validate the sklearn CD contract before allocating factor tensors."""
    legacy_regularization = {"alpha", "regularization"}.intersection(nmf_kwargs)
    if legacy_regularization:
        raise ValueError(
            "solver='cd' does not accept deprecated alpha/regularization; "
            "use alpha_W, alpha_H, and l1_ratio"
        )
    beta_loss = nmf_kwargs.get("beta_loss", "frobenius")
    if not (beta_loss == 2 or str(beta_loss).lower() == "frobenius"):
        raise ValueError("solver='cd' supports only beta_loss='frobenius'")
    if rc.dtype is torch.bfloat16:
        raise ValueError("solver='cd' supports gpu dtype fp32 or fp64, not bf16")
    if rc.max_iter < 1:
        raise ValueError("max_iter must be at least 1 for solver='cd'")
    if rc.tol < 0:
        raise ValueError("tol must be non-negative for solver='cd'")
    if not isinstance(nmf_kwargs.get("shuffle", False), (bool, np.bool_)):
        raise ValueError("shuffle must be a boolean for solver='cd'")


def _hals_sweep_torch(factor, gram, cross, permutation, active):
    """Apply one literal torch port of sklearn's serial CD coordinate sweep."""
    replicates, components, rows = factor.shape
    violation = factor.new_zeros(replicates)
    zero = factor.new_zeros(())
    cyclic = permutation is None

    for coordinate in range(components):
        if cyclic:
            component = coordinate
            gram_row = gram[:, component, :]
            cross_row = cross[:, component, :]
            old_value = factor[:, component, :].clone()
        else:
            component = permutation[:, coordinate]
            gram_row = gram.gather(
                1, component[:, None, None].expand(-1, 1, components)
            ).squeeze(1)
            factor_index = component[:, None, None].expand(-1, 1, rows)
            cross_row = cross.gather(1, factor_index).squeeze(1)
            old_value = factor.gather(1, factor_index).squeeze(1)

        # Match sklearn's _cdnmf_fast.pyx summation and Gauss-Seidel order.
        gradient = -cross_row.clone()
        for other_component in range(components):
            gradient = (
                gradient
                + gram_row[:, other_component, None]
                * factor[:, other_component, :]
            )

        projected_gradient = gradient.where(
            old_value != 0, zero.minimum(gradient)
        )
        violation = violation + projected_gradient.abs().sum(dim=1) * active

        if cyclic:
            hessian = gram[:, component, component, None]
        else:
            hessian = gram_row.gather(1, component[:, None])
        nonzero_hessian = hessian != 0
        safe_hessian = hessian.where(nonzero_hessian, hessian.new_ones(()))
        candidate = (old_value - gradient / safe_hessian).clamp_min(0)
        new_value = candidate.where(nonzero_hessian, old_value)
        new_value = new_value.where(active[:, None], old_value)

        if cyclic:
            factor[:, component, :] = new_value
        else:
            factor.scatter_(1, factor_index, new_value[:, None, :])

    return violation


_HALS_CUDA_BACKEND_UNSET = object()
_HALS_CUDA_BACKEND = _HALS_CUDA_BACKEND_UNSET


def _get_hals_cuda_backend():
    """Resolve and cache the optional fused CUDA sweep."""
    global _HALS_CUDA_BACKEND
    if _HALS_CUDA_BACKEND is _HALS_CUDA_BACKEND_UNSET:
        try:
            from .solver_cd_triton import hals_sweep_cuda
        except (ImportError, ModuleNotFoundError):
            hals_sweep_cuda = None
        _HALS_CUDA_BACKEND = hals_sweep_cuda
    return _HALS_CUDA_BACKEND


def _hals_sweep(factor, gram, cross, permutation, active):
    """Use the fused CUDA sweep when available, otherwise use torch."""
    if factor.is_cuda:
        hals_sweep_cuda = _get_hals_cuda_backend()
        if hals_sweep_cuda is not None:
            if permutation is None:
                permutation = factor.new_tensor(
                    np.tile(
                        np.arange(factor.shape[1], dtype=np.int64),
                        (factor.shape[0], 1),
                    )
                ).long()
            return hals_sweep_cuda(factor, gram, cross, permutation, active)
    return _hals_sweep_torch(factor, gram, cross, permutation, active)


def _regularize_cd_products(gram, cross, l1_reg, l2_reg):
    """Apply sklearn's L2 diagonal addition and L1 cross-product shift."""
    if l2_reg != 0.0:
        gram.diagonal(dim1=-2, dim2=-1).add_(l2_reg)
    if l1_reg != 0.0:
        cross.sub_(l1_reg)
    return gram, cross


def _coordinate_orders(torch, seeds, components, shuffle, device):
    """Return next_order(), giving each sweep's [replicate, component] order.

    Shuffled orders come from one sklearn RandomState per seed, one draw per
    sweep. The cyclic order is None for the torch sweep's fast path and a
    fixed index tensor for the CUDA sweep.
    """
    if shuffle:
        from sklearn.utils import check_random_state

        rngs = [check_random_state(seed) for seed in seeds]

        def next_order():
            orders = np.stack([rng.permutation(components) for rng in rngs])
            return torch.as_tensor(orders, dtype=torch.int64, device=device)

        return next_order

    cyclic = None
    if device.startswith("cuda") and _get_hals_cuda_backend() is not None:
        cyclic = (
            torch.arange(components, dtype=torch.int64, device=device)
            .expand(len(seeds), -1)
            .contiguous()
        )
    return lambda: cyclic


class _StopRule:
    """sklearn's per-replicate CD stop: violation / first violation <= tol."""

    def __init__(self, torch, replicates, tol, device):
        self.tol = tol
        self.active = torch.ones(replicates, dtype=torch.bool, device=device)
        self.n_iter = torch.zeros(replicates, dtype=torch.int64, device=device)
        self.violation_init = None

    def update(self, iteration, violation):
        """Record one iteration; return True once every replicate has stopped."""
        self.n_iter = self.n_iter.new_full((), iteration).where(
            self.active, self.n_iter
        )
        if self.violation_init is None:
            self.violation_init = violation.clone()

        zero_init = self.violation_init == 0
        denominator = self.violation_init.where(
            ~zero_init, self.violation_init.new_ones(())
        )
        converged = self.active & (
            zero_init | ((violation / denominator) <= self.tol)
        )
        self.active = self.active & ~converged
        return not bool(self.active.any())


def _window_budget(
    update_h, replicates, components, n_features, itemsize, torch_sweep
):
    """Return (fixed, per-row) device bytes for one fit; H is already allocated.

    A window of w rows needs fixed + w * per_row bytes. The resident full
    matrix is the single window w = n_rows.
    """
    R, k, m, s = replicates, components, n_features, itemsize
    # Coordinate orders (int64), n_iter, active flags and violation totals.
    fixed = R * (8 * k + 8 + 1 + 3 * s)
    # The W-sweep Gram, plus one replicate's GEMM output before it is copied.
    fixed += (R + 1) * k * k * s
    # W and cross-product windows, plus one replicate's H·Xᵀ output.
    per_row = (2 * R * k + k) * s
    # Without Triton, the torch sweep holds about ten [replicate, row]
    # temporaries plus an int64 gather index for shuffled orders.
    sweep = 10 * R * s + 8 * R if torch_sweep else 0
    per_row += sweep
    if update_h:
        # The X window, the H-update Gram and cross accumulators, one
        # replicate's W·X output, and the H sweep over n_features columns.
        per_row += m * s
        fixed += R * k * k * s + (R + 1) * k * m * s + sweep * m
    else:
        # The one-off cross-product pass holds an X window, every replicate's
        # cross window and one replicate's GEMM output.
        per_row = max(per_row, (m + R * k + k) * s)
    return fixed, per_row


def _fixed_h_cross(torch, H, Xcompute, window_rows, l1_w):
    """Return H·Xᵀ minus the L1 shift for every row, one window at a time."""
    replicates, components, _ = H.shape
    n_rows = Xcompute.shape[0]
    cross_host = np.empty(
        (replicates, components, n_rows), dtype=Xcompute.dtype
    )
    windows = tiling._RowWindows(
        torch, H, n_rows, window_rows, {"X": (Xcompute, 0)}
    )
    for start, stop, window in windows:
        cross = H.new_empty((replicates, components, stop - start))
        for replicate in range(replicates):
            cross[replicate].copy_(H[replicate] @ window["X"].T)
        if l1_w != 0.0:
            cross.sub_(l1_w)
        cross_host[:, :, start:stop] = cross.cpu().numpy()
        del cross
    return cross_host


def _fit_cd_fixed_h(
    torch, rc, Xcompute, Wt_host, H, window_rows, seeds, nmf_kwargs
):
    """Update W against a fixed H, one row window at a time.

    H·Hᵀ and H·Xᵀ never change, so they are computed once, on the device, and
    then travel with W's rows. One window keeps everything resident, which is
    the pre-tiling arithmetic.
    """
    replicates, components, _ = H.shape
    n_rows = Xcompute.shape[0]
    l1_w, l2_w, _, _ = _cd_regularization(
        nmf_kwargs, n_rows, Xcompute.shape[1]
    )
    next_order = _coordinate_orders(
        torch, seeds, components, nmf_kwargs.get("shuffle", False), rc.device
    )
    stop_rule = _StopRule(torch, replicates, rc.tol, rc.device)
    gram = H.new_empty((replicates, components, components))
    tf32 = utils._want_tf32(torch, rc)

    with torch.no_grad(), utils._cuda_tf32(torch, tf32, rc.device):
        for replicate in range(replicates):
            gram[replicate].copy_(H[replicate] @ H[replicate].T)
        if l2_w != 0.0:
            gram.diagonal(dim1=-2, dim2=-1).add_(l2_w)
        cross_host = _fixed_h_cross(torch, H, Xcompute, window_rows, l1_w)
        windows = tiling._RowWindows(
            torch, H, n_rows, window_rows,
            {"Wt": (Wt_host, 2), "cross": (cross_host, 2)}, writable=("Wt",),
        )

        for iteration in range(1, rc.max_iter + 1):
            order = next_order()
            violation = None
            for start, stop, window in windows:
                window_violation = _hals_sweep(
                    window["Wt"], gram, window["cross"], order, stop_rule.active
                )
                violation = (
                    window_violation if violation is None
                    else violation + window_violation
                )
            if stop_rule.update(iteration, violation):
                break

    return windows.host("Wt"), H, stop_rule.n_iter


def _fit_cd_factorize(
    torch, rc, Xcompute, Wt_host, H, window_rows, seeds, nmf_kwargs
):
    """Update W and H, streaming X and W in row windows when they do not fit.

    Products are computed at window size and summed over windows for the H
    update. One window keeps X and W resident and is exactly the pre-tiling
    full-matrix arithmetic. Smaller windows split the sums over cells, so
    their results agree numerically rather than bit for bit.
    """
    replicates, components, n_features = H.shape
    n_rows = Xcompute.shape[0]
    l1_w, l2_w, l1_h, l2_h = _cd_regularization(
        nmf_kwargs, n_rows, n_features
    )
    next_order = _coordinate_orders(
        torch, seeds, components, nmf_kwargs.get("shuffle", False), rc.device
    )
    stop_rule = _StopRule(torch, replicates, rc.tol, rc.device)
    gram_w = H.new_empty((replicates, components, components))
    gram_h = H.new_empty((replicates, components, components))
    cross_h = H.new_empty(H.shape)
    tf32 = utils._want_tf32(torch, rc)
    windows = tiling._RowWindows(
        torch, H, n_rows, window_rows,
        {"X": (Xcompute, 0), "Wt": (Wt_host, 2)}, writable=("Wt",),
    )

    with torch.no_grad(), utils._cuda_tf32(torch, tf32, rc.device):
        for iteration in range(1, rc.max_iter + 1):
            # One 2-D GEMM per replicate: a batched matmul may pick another
            # reduction path as gpu-batch changes, which Fast-HALS amplifies.
            for replicate in range(replicates):
                gram_w[replicate].copy_(H[replicate] @ H[replicate].T)
            if l2_w != 0.0:
                gram_w.diagonal(dim1=-2, dim2=-1).add_(l2_w)
            # One order for the whole W sweep, not a fresh one per window.
            order_w = next_order()
            violation = None

            for start, stop, window in windows:
                Xw, Ww = window["X"], window["Wt"]
                cross_w = H.new_empty((replicates, components, stop - start))
                for replicate in range(replicates):
                    cross_w[replicate].copy_(H[replicate] @ Xw.T)
                if l1_w != 0.0:
                    cross_w.sub_(l1_w)
                window_violation = _hals_sweep(
                    Ww, gram_w, cross_w, order_w, stop_rule.active
                )
                violation = (
                    window_violation if violation is None
                    else violation + window_violation
                )
                # Use UPDATED W. Raw products accumulate across all cells;
                # apply H regularization only once, with full sample scaling.
                for replicate in range(replicates):
                    gram = Ww[replicate] @ Ww[replicate].T
                    cross = Ww[replicate] @ Xw
                    if start == 0:
                        gram_h[replicate].copy_(gram)
                        cross_h[replicate].copy_(cross)
                    else:
                        gram_h[replicate].add_(gram)
                        cross_h[replicate].add_(cross)
                # Drop this window before the next one is uploaded.
                del Xw, Ww, cross_w

            _regularize_cd_products(gram_h, cross_h, l1_h, l2_h)
            violation = violation + _hals_sweep(
                H, gram_h, cross_h, next_order(), stop_rule.active
            )
            if stop_rule.update(iteration, violation):
                break

    return windows.host("Wt"), H, stop_rule.n_iter


def _fit_cd(torch, rc, Xcompute, Wt_host, H, seeds, nmf_kwargs, update_h):
    """Run batched sklearn-compatible Fast-HALS to projected-gradient convergence.

    Wt_host is in host memory and H is already on the device. Without row
    tiling this is the pre-tiling full-matrix computation; with tiling, the
    products are computed one row window at a time. Returns the updated
    (Wt_host, H, n_iter).
    """
    n_rows, n_features = Xcompute.shape
    window_rows = tiling._resolve_window_rows(
        torch, rc.opt["row_tiling_ratio"], rc.device, n_rows,
        budget=lambda: _window_budget(
            update_h, len(seeds), rc.k, n_features, Xcompute.dtype.itemsize,
            torch_sweep=_get_hals_cuda_backend() is None,
        ),
        solver="factorization CD" if update_h else "fixed-H CD",
    )
    fit = _fit_cd_factorize if update_h else _fit_cd_fixed_h
    return fit(torch, rc, Xcompute, Wt_host, H, window_rows, seeds, nmf_kwargs)


def _nmf_gpu_cd(X, seeds, nmf_kwargs, gpu_kwargs=None):
    """Run full or fixed-H sklearn-compatible CD replicates."""
    torch = utils._loud_import_torch()
    seeds = utils._normalize_seeds(seeds)
    rc = utils._gpu_setup(torch, X, nmf_kwargs, gpu_kwargs)
    _validate_cd_runtime(torch, rc, nmf_kwargs)

    host_dtype = _numpy_staging_dtype(torch, rc.dtype)
    Xcompute = np.ascontiguousarray(rc.Xnp, dtype=host_dtype)
    replicates = len(seeds)
    update_h = nmf_kwargs.get("update_H", True) is not False

    if update_h:
        init = nmf_kwargs.get("init")
        if init == "custom":
            W0 = _to_checked_custom_factor(
                nmf_kwargs.get("W"),
                (Xcompute.shape[0], rc.k),
                "W",
                host_dtype,
            )
            H0 = _to_checked_custom_factor(
                nmf_kwargs.get("H"),
                (rc.k, Xcompute.shape[1]),
                "H",
                host_dtype,
            )
            Wt0 = np.repeat(W0.T[None, :, :], replicates, axis=0)
            Hs0 = np.repeat(H0[None, :, :], replicates, axis=0)
        else:
            Wt0 = np.empty(
                (replicates, rc.k, Xcompute.shape[0]), dtype=host_dtype
            )
            Hs0 = np.empty(
                (replicates, rc.k, Xcompute.shape[1]), dtype=host_dtype
            )
            for replicate, seed in enumerate(seeds):
                W0, H0 = utils._init_wh(Xcompute, rc.k, seed, init)
                Wt0[replicate] = W0.T
                Hs0[replicate] = H0
    else:
        H0 = utils._to_checked_fixed_h(
            nmf_kwargs.get("H"), rc.k, Xcompute.shape[1]
        )
        H0 = np.ascontiguousarray(H0, dtype=host_dtype)
        # sklearn CD ignores supplied W and starts fixed-H usages at zero.
        Wt0 = np.zeros(
            (replicates, rc.k, Xcompute.shape[0]), dtype=host_dtype
        )
        Hs0 = np.repeat(H0[None, :, :], replicates, axis=0)

    H = torch.as_tensor(Hs0, dtype=rc.dtype, device=rc.device)
    Wt_host, H, _ = _fit_cd(
        torch, rc, Xcompute, Wt0, H, seeds, nmf_kwargs, update_h
    )
    Hc = H.cpu().double().numpy()
    Wc = Wt_host.transpose(0, 2, 1).astype(np.float64, copy=False)
    return [(Hc[r], Wc[r]) for r in range(replicates)]
