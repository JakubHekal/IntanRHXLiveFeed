import numpy as np

DISPLAY_WINDOW_SEC    = 10
DISPLAY_BUFFER_SEC    = 300
DEFAULT_SAMPLING_RATE = 20000
MAX_DISPLAY_POINTS    = 2500

PLOT_UPDATE_FREQ_HZ = 90
RAW_RENDER_HZ       = 30
PSD_RENDER_HZ       = 30
SPIKE_RENDER_HZ     = 30
WAVEFORM_YLIM_ABS_UV = 100
SPIKE_SCROLL_WINDOW_MIN = 10.0
RAW_HISTORY_TARGET_HZ = 100.0
RAW_HISTORY_HIGH_TARGET_HZ = 1000.0
RAW_ADAPTIVE_HIGH_RES_MAX_SPAN_SEC = 30.0
RAW_FULL_RES_MAX_SPAN_SEC = 30.0
RAW_MANUAL_VIEW_MARGIN_SEC = 2.0
MAX_RAW_HISTORY_PLOT_POINTS = 200000

PSD_PLOT_UPDATE_EVERY_N   = 20
SPIKE_PLOT_UPDATE_EVERY_N = 20


def _make_display_buffer(fs: float, num_channels=1) -> np.ndarray:
    n = max(1, int(round(fs * DISPLAY_BUFFER_SEC)))
    return np.zeros((1 + num_channels, n), dtype=np.float64)


def _minmax_downsample(x, y, max_points):
    n = x.size
    if n <= max_points:
        return x, y
    step = n // max_points
    n_bins = n // step
    x = x[:n_bins * step]
    y = y[:n_bins * step]
    x_binned = x.reshape(-1, step)
    y_binned = y.reshape(-1, step)
    x_out = np.repeat(x_binned.mean(axis=1), 2)
    y_out = np.empty(n_bins * 2)
    y_out[0::2] = y_binned.min(axis=1)
    y_out[1::2] = y_binned.max(axis=1)
    return x_out, y_out
