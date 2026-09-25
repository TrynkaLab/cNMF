"""Row tiling shared by the GPU NMF solvers.

A solver works on the rows (cells) of X and W either all at once, resident on
the device, or in windows streamed from host memory. This module holds the
solver-neutral parts: turning the row tiling ratio into a window size, and
moving row windows between host and device. Each solver supplies its own
memory model and arithmetic.
"""

import numpy as np

# Automatic sizing leaves max(512 MiB, 10% of free VRAM) unallocated for the
# CUDA context, cuBLAS workspaces and allocator fragmentation.
_VRAM_RESERVE_BYTES = 512 * (1 << 20)
_VRAM_RESERVE_FRACTION = 0.10


def _resolve_window_rows(torch, ratio, device, n_rows, budget, solver):
    """Return the rows per window for a row tiling ratio.

    None or 1 keeps every row resident. A ratio in (0, 1) takes
    floor(n_rows * ratio) rows, at least one. 0 takes the most rows that fit
    in free CUDA memory, so the full matrix stays resident whenever it fits;
    budget() gives the solver's (fixed_bytes, bytes_per_row).
    """
    if ratio is None:
        return n_rows
    if ratio != 0:
        return max(1, int(n_rows * ratio))
    if not device.startswith("cuda"):
        return n_rows

    fixed_bytes, bytes_per_row = budget()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve = max(
        _VRAM_RESERVE_BYTES, int(free_bytes * _VRAM_RESERVE_FRACTION)
    )
    available = free_bytes - reserve - fixed_bytes
    if available < bytes_per_row:
        raise MemoryError(
            f"GPU {solver} does not have enough VRAM for one row; "
            "reduce gpu batch or n_components"
        )
    return min(n_rows, available // bytes_per_row)


def _to_device(torch, array, like):
    """Copy a host array to like's device and dtype as one contiguous tensor."""
    return torch.as_tensor(
        np.ascontiguousarray(array), dtype=like.dtype, device=like.device
    )


class _RowWindows:
    """Host arrays whose rows are processed on the device one window at a time.

    arrays maps a name to (host array, row axis). With one window covering
    every row, the arrays are uploaded once and stay resident. Otherwise each
    window is uploaded when visited, and the writable arrays are copied back
    to the host when the caller moves on.
    """

    def __init__(self, torch, like, n_rows, window_rows, arrays, writable=()):
        self._torch = torch
        self._like = like
        self._arrays = arrays
        self._writable = tuple(writable)
        self.resident = window_rows >= n_rows
        self._bounds = [
            (start, min(start + window_rows, n_rows))
            for start in range(0, n_rows, window_rows)
        ]
        self._resident = (
            {name: self._upload(name, 0, n_rows) for name in arrays}
            if self.resident else None
        )

    def _rows(self, name, start, stop):
        array, axis = self._arrays[name]
        index = [slice(None)] * array.ndim
        index[axis] = slice(start, stop)
        return array, tuple(index)

    def _upload(self, name, start, stop):
        array, index = self._rows(name, start, stop)
        return _to_device(self._torch, array[index], self._like)

    def __iter__(self):
        """Yield (start, stop, tensors), with tensors[name] on the device.

        A streamed window is released when the caller moves on. Callers that
        keep their own references to its tensors must drop them first, or two
        windows are alive during the next upload.
        """
        for start, stop in self._bounds:
            if self.resident:
                yield start, stop, self._resident
                continue
            tensors = {name: self._upload(name, start, stop) for name in self._arrays}
            try:
                yield start, stop, tensors
            finally:
                for name in self._writable:
                    array, index = self._rows(name, start, stop)
                    array[index] = tensors[name].cpu().numpy()
                tensors.clear()

    def host(self, name):
        """Return an array's current values in host memory."""
        if self.resident:
            return self._resident[name].cpu().numpy()
        return self._arrays[name][0]
