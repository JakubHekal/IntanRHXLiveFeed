from .psd import TARGET_NPERSEG_SEC, NOVERLAP_RATIO, FMAX, welch_psd
from .spike_count import (
    HP_SPIKE_BAND, SPIKE_Z_THR, POLARITY, REFRACTORY_MS,
    AMP_MIN_UV, AMP_MAX_UV, W_MIN_MS, W_MAX_MS,
    bandpass_filt, robust_z, find_peaks_distance, apply_refractory, detect_spikes,
)
from .spike_plot import PRE_MS, POST_MS, extract_waveforms

# ponytail: constants used externally by plot_settings.py, connect_screen.py, canvas.py
PSD_BUFFER_SEC = 10
SPIKE_BIN_SEC = 5
WAVEFORM_BUFFER_SEC = 10
SPIKE_INCREMENTAL_MIN_SAMPLES = 200
SPIKE_OVERLAP_SAMPLES = 400
PSD_YLIM_MIN = -10.0
PSD_YLIM_MAX = 40.0


def configure_processing_windows(psd_buffer_sec: int, waveform_buffer_sec: int, spike_bin_sec: int):
    global PSD_BUFFER_SEC, WAVEFORM_BUFFER_SEC, SPIKE_BIN_SEC
    PSD_BUFFER_SEC = int(psd_buffer_sec)
    WAVEFORM_BUFFER_SEC = int(waveform_buffer_sec)
    SPIKE_BIN_SEC = int(spike_bin_sec)
