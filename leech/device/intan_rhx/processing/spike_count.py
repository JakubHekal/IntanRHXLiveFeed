import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

HP_SPIKE_BAND = (300.0, 5000.0)
SPIKE_Z_THR = 6.0
POLARITY = "both"
REFRACTORY_MS = 1.0
AMP_MIN_UV = 40.0
AMP_MAX_UV = 700.0
W_MIN_MS = None
W_MAX_MS = None


def butter_bandpass(lo, hi, fs, order=3):
    ny = 0.5 * fs
    lo = max(lo / ny, 1e-6)
    hi = min(hi / ny, 0.999999)
    b, a = butter(order, [lo, hi], btype="bandpass")
    return b, a


def bandpass_filt(x, fs, band, order=3):
    b, a = butter_bandpass(band[0], band[1], fs, order=order)
    return filtfilt(b, a, x)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x), med, mad
    z = 0.6745 * (x - med) / mad
    return z, med, mad


def apply_refractory(peaks: np.ndarray, refractory_samp: int) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    peaks = np.asarray(peaks, dtype=int)
    peaks.sort()
    kept = [peaks[0]]
    last = peaks[0]
    for p in peaks[1:]:
        if p - last >= refractory_samp:
            kept.append(p)
            last = p
    return np.asarray(kept, dtype=int)


def estimate_fs_from_time(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot estimate sampling rate from time column.")
    med_dt = np.median(dt)
    if med_dt <= 0:
        raise ValueError("Non-positive median dt; time column may be invalid.")
    return 1.0 / med_dt


def width_gate_indices(x_hp: np.ndarray, peaks: np.ndarray, fs: float,
                       wmin_ms=None, wmax_ms=None, polarity="neg"):
    if peaks.size == 0:
        return peaks
    if wmin_ms is None and wmax_ms is None:
        return peaks

    wmin_samp = 0 if wmin_ms is None else int(round((wmin_ms / 1000.0) * fs))
    wmax_samp = int(round((wmax_ms / 1000.0) * fs)) if wmax_ms is not None else None

    if wmax_samp is None:
        search_max = int(round(0.002 * fs))
    else:
        search_max = max(wmax_samp, 1)

    kept = []
    N = len(x_hp)
    for p in peaks:
        a = p
        b = min(p + search_max, N - 1)
        if b <= a + 1:
            continue

        seg = x_hp[a:b]

        if polarity == "neg":
            j = np.argmax(seg)
        elif polarity == "pos":
            j = np.argmin(seg)
        else:
            if x_hp[p] < 0:
                j = np.argmax(seg)
            else:
                j = np.argmin(seg)

        width = j
        if width <= 0:
            continue
        if wmin_ms is not None and width < wmin_samp:
            continue
        if wmax_samp is not None and width > wmax_samp:
            continue
        kept.append(p)

    return np.asarray(kept, dtype=int)


def find_peaks_distance(z: np.ndarray, polarity: str, thr: float, distance: int) -> np.ndarray:
    peaks_all = []

    if polarity in ("pos", "both"):
        p_pos, _ = find_peaks(z, height=thr, distance=distance)
        peaks_all.append(p_pos)

    if polarity in ("neg", "both"):
        p_neg, _ = find_peaks(-z, height=thr, distance=distance)
        peaks_all.append(p_neg)

    if not peaks_all:
        return np.array([], dtype=int)

    peaks = np.unique(np.concatenate(peaks_all)).astype(int)
    peaks.sort()
    return peaks


def detect_spikes(x_uv: np.ndarray, t_s: np.ndarray):
    fs = estimate_fs_from_time(t_s)

    x_hp = bandpass_filt(x_uv.astype(float), fs, HP_SPIKE_BAND, order=3)

    z, _, _ = robust_z(x_hp)
    thr = float(SPIKE_Z_THR)

    refractory = int(round((REFRACTORY_MS / 1000.0) * fs))
    refractory = max(refractory, 1)

    peaks = find_peaks_distance(z, polarity=POLARITY, thr=thr, distance=refractory)

    peaks = apply_refractory(peaks, refractory)

    peaks = width_gate_indices(x_hp, peaks, fs, W_MIN_MS, W_MAX_MS, polarity=POLARITY)

    if AMP_MIN_UV is not None or AMP_MAX_UV is not None:
        amps = np.abs(x_hp[peaks]) if peaks.size else np.array([])
        keep = np.ones_like(peaks, dtype=bool)
        if AMP_MIN_UV is not None:
            keep &= (amps >= float(AMP_MIN_UV))
        if AMP_MAX_UV is not None:
            keep &= (amps <= float(AMP_MAX_UV))
        peaks = peaks[keep]

    spike_times = t_s[peaks] if peaks.size else np.array([], dtype=float)
    return peaks, spike_times, fs
