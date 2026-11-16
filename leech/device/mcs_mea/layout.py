"""Electrode geometry for MCS MEA2100 60- and 120-channel layouts."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


_CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}


def _plate_positions() -> List[Tuple[float, float]]:
    """Return 60 standard 8x8-minus-corners electrode positions."""
    out: List[Tuple[float, float]] = []
    for row in range(8):
        for col in range(8):
            if (row, col) in _CORNERS:
                continue
            out.append((float(col), float(-row)))
    return out


def electrode_positions(num_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates for the requested 60- or 120-channel MEA.

    Channel order is row-major within each plate. 120-channel devices are
    modelled as two 60-channel plates separated by a small gap.
    """
    if num_channels not in (60, 120):
        raise ValueError("MCS MEA supports 60 or 120 channels")

    base = _plate_positions()
    plates = 1 if num_channels == 60 else 2
    xs: List[float] = []
    ys: List[float] = []
    for plate in range(plates):
        x_offset = plate * 8.5
        for x, y in base:
            xs.append(x + x_offset)
            ys.append(y)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def channel_names(num_channels: int) -> List[str]:
    if num_channels not in (60, 120):
        raise ValueError("MCS MEA supports 60 or 120 channels")
    return [f"CH{i + 1:03d}" for i in range(num_channels)]
