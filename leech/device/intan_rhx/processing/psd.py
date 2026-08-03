import numpy as np
from scipy.signal import welch

TARGET_NPERSEG_SEC = 2.0
NOVERLAP_RATIO     = 0.5
FMAX               = 150.0


def welch_psd(x: np.ndarray, fs: float, nperseg: int, noverlap: int, fmax: float | None):
    f, Pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant",
        scaling="density",
        average="mean",
    )
    if fmax is not None:
        keep = f <= fmax
        f, Pxx = f[keep], Pxx[keep]
    Pxx_db = 10.0 * np.log10(Pxx + np.finfo(float).eps)
    return f, Pxx, Pxx_db
