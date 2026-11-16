"""Streaming spike detection for the MCS MEA tab.

The detector follows the Intan detector's core steps: band-pass, robust
z-score, absolute threshold, refractory distance, and amplitude gates. It is
implemented locally because the Intan package is not a dependency of LEECH.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt


HP_SPIKE_BAND = (300.0, 5000.0)
SPIKE_Z_THR = 6.0
REFRACTORY_MS = 1.0
AMP_MIN_UV = 40.0
AMP_MAX_UV = 700.0


def bandpass_sos(fs: float) -> np.ndarray:
    low, high = HP_SPIKE_BAND
    high = min(high, 0.45 * float(fs))
    if high <= low:
        raise ValueError("sample rate too low for spike band")
    return butter(3, [low, high], btype="bandpass", fs=float(fs), output="sos")


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    fallback = np.std(values, axis=1, keepdims=True)
    scale = np.where(mad > 1e-12, mad, np.maximum(fallback, 1e-12))
    return 0.6745 * (values - median) / scale


class StreamingSpikeDetector:
    """Stateful, vectorised detector for consecutive ``(channels, samples)`` chunks."""

    def __init__(
        self,
        sample_rate: float,
        num_channels: int,
        z_thr: float = SPIKE_Z_THR,
        amp_min_uv: float = AMP_MIN_UV,
        amp_max_uv: float = AMP_MAX_UV,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.num_channels = int(num_channels)
        self.z_thr = float(z_thr)
        self.amp_min_uv = float(amp_min_uv)
        self.amp_max_uv = float(amp_max_uv)
        self._sos = bandpass_sos(self.sample_rate)
        self._zi = np.zeros((self._sos.shape[0], self.num_channels, 2), dtype=np.float64)
        self._distance = max(1, int(round(REFRACTORY_MS / 1000.0 * self.sample_rate)))

    def process(self, chunk: np.ndarray) -> Dict[str, object]:
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.num_channels:
            raise ValueError(
                f"expected ({self.num_channels}, samples), got {x.shape}"
            )
        if x.shape[1] < 4:
            return {
                "spikes": [np.empty(0, dtype=np.int64) for _ in range(self.num_channels)],
                "rms": np.zeros(self.num_channels, dtype=np.float32),
                "samples": int(x.shape[1]),
            }

        filtered, self._zi = sosfilt(self._sos, x, axis=1, zi=self._zi)
        z = robust_z(filtered)
        abs_z = np.abs(z)
        spikes: List[np.ndarray] = []
        for ch in range(self.num_channels):
            peaks, _ = find_peaks(abs_z[ch], height=self.z_thr, distance=self._distance)
            if peaks.size:
                amplitude = np.abs(filtered[ch, peaks])
                peaks = peaks[
                    (amplitude >= self.amp_min_uv) & (amplitude <= self.amp_max_uv)
                ]
            spikes.append(peaks.astype(np.int64, copy=False))

        rms = np.sqrt(np.mean(filtered * filtered, axis=1)).astype(np.float32)
        return {"spikes": spikes, "rms": rms, "samples": int(x.shape[1])}
