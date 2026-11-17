"""Electrode geometry for MCS MEA2100 60- and 120-channel layouts.

Channel order is the amplifier channel order (1-dimensional MEA pin order).
Labels are the 2-dimensional electrode IDs used by MCS software. Both tables
were verified against MCS DataManager exports.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# 1-dim pin (1..60) -> 2-dim electrode ID. Pin 15 is the internal reference.
_MEA60_LABELS: Tuple[str, ...] = (
    "47", "48", "46", "45", "38", "37", "28", "36", "27", "17",
    "26", "16", "35", "25", "Ref", "14", "24", "34", "13", "23",
    "12", "22", "33", "21", "32", "31", "44", "43", "41", "42",
    "52", "51", "53", "54", "61", "62", "71", "63", "72", "82",
    "73", "83", "64", "74", "84", "85", "75", "65", "86", "76",
    "87", "77", "66", "78", "67", "68", "55", "56", "58", "57",
)

# Headstage channel (1..120) -> 2-dim electrode ID.
_MEA120_LABELS: Tuple[str, ...] = (
    "F7", "F8", "F12", "F11", "F10", "F9", "E12", "E11", "E10", "E9",
    "D12", "D11", "D10", "D9", "C11", "C10", "B10", "E8", "C9", "B9",
    "A9", "D8", "C8", "B8", "A8", "D7", "C7", "B7", "A7", "E7",
    "F6", "E6", "A6", "B6", "C6", "D6", "A5", "B5", "C5", "D5",
    "A4", "B4", "C4", "D4", "B3", "C3", "C2", "E5", "D3", "D2",
    "D1", "E4", "E3", "E2", "E1", "F4", "F3", "F2", "F1", "F5",
    "G6", "G5", "G1", "G2", "G3", "G4", "H1", "H2", "H3", "H4",
    "J1", "J2", "J3", "J4", "K2", "K3", "L3", "H5", "K4", "L4",
    "M4", "J5", "K5", "L5", "M5", "J6", "K6", "L6", "M6", "H6",
    "G7", "H7", "M7", "L7", "K7", "J7", "M8", "L8", "K8", "J8",
    "M9", "L9", "K9", "J9", "L10", "K10", "K11", "H8", "J10", "J11",
    "J12", "H9", "H10", "H11", "H12", "G9", "G10", "G11", "G12", "G8",
)

_MEA120_ROWS = "ABCDEFGHJKLM"

_MEA_NAMES: Dict[int, Tuple[str, str]] = {
    60: ("60MEA200/30iR", "60MEA200_30_iR"),
    120: ("120MEA200/30iR", "120MEA200_30_iR"),
}


def _labels(num_channels: int) -> Tuple[str, ...]:
    if num_channels == 60:
        return _MEA60_LABELS
    if num_channels == 120:
        return _MEA120_LABELS
    raise ValueError("MCS MEA supports 60 or 120 channels")


def _position(label: str) -> Tuple[float, float]:
    """Return (x, y) for a 2-dim electrode ID, row 1 at the top."""
    if label == "Ref":
        return 0.0, -4.0
    if label[0].isdigit():
        col = int(label[0])
        row = int(label[1])
    else:
        col = int(label[1:])
        row = _MEA120_ROWS.index(label[0]) + 1
    return float(col - 1), float(-(row - 1))


def electrode_positions(num_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates aligned with the channel labels."""
    xs: List[float] = []
    ys: List[float] = []
    for label in _labels(num_channels):
        x, y = _position(label)
        xs.append(x)
        ys.append(y)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def channel_names(num_channels: int) -> List[str]:
    """Return electrode IDs in amplifier channel order."""
    return list(_labels(num_channels))


def mea_name(num_channels: int) -> str:
    """Return the MCS product name for the MEA."""
    return _MEA_NAMES[num_channels][0]


def mea_layout(num_channels: int) -> str:
    """Return the MCS layout identifier for the MEA."""
    return _MEA_NAMES[num_channels][1]
