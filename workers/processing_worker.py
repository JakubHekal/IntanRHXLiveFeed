from PyQt5 import QtCore
import numpy as np

import processing.psd as psd
import processing.spike_count as spike_count
import processing.spike_plot as spike_plot

PSD_BUFFER_SEC = 10
SPIKE_HISTORY_MIN = 30
SPIKE_BIN_SEC = 5
# Minimum new samples needed before scheduling an incremental spike scan
SPIKE_INCREMENTAL_MIN_SAMPLES = 200
# Overlap kept at scan start to avoid bandpass filter edge artifacts
SPIKE_OVERLAP_SAMPLES = 400

# PSD y-axis limits (dB)
PSD_YLIM_MIN = -10.0
PSD_YLIM_MAX = 40.0

# Window of spike times used for the average waveform
WAVEFORM_BUFFER_SEC = 10


def configure_processing_windows(psd_buffer_sec: int, waveform_buffer_sec: int, spike_bin_sec: int):
    """Update processing window settings at runtime."""
    global PSD_BUFFER_SEC, WAVEFORM_BUFFER_SEC, SPIKE_BIN_SEC
    PSD_BUFFER_SEC = int(psd_buffer_sec)
    WAVEFORM_BUFFER_SEC = int(waveform_buffer_sec)
    SPIKE_BIN_SEC = int(spike_bin_sec)

class _ProcessingResult:
    __slots__ = (
        'has_psd_update', 'has_spike_update',
        'psd_f', 'psd_db',
        'spike_minute_idx', 'spike_counts',
        'wf_t_ms', 'wf_mu', 'wf_sem',
        'spike_times_cache', 'last_scan_sample',
    )
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


def _detect_spike_indices(x: np.ndarray, fs: float) -> np.ndarray:
    if x.size < 10:
        return np.array([], dtype=int)

    x_hp = spike_count.bandpass_filt(x.astype(float), fs, spike_count.HP_SPIKE_BAND, order=3)
    z, _, _ = spike_count.robust_z(x_hp)
    z = np.asarray(z, dtype=np.float64).ravel()
    refractory = max(1, int(round((spike_count.REFRACTORY_MS / 1000.0) * fs)))

    peaks_all = []
    if spike_count.POLARITY in ("pos", "both"):
        p_pos, _ = spike_count.find_peaks(z, height=spike_count.SPIKE_Z_THR, distance=refractory)
        peaks_all.append(p_pos)
    if spike_count.POLARITY in ("neg", "both"):
        p_neg, _ = spike_count.find_peaks(-z, height=spike_count.SPIKE_Z_THR, distance=refractory)
        peaks_all.append(p_neg)

    if not peaks_all:
        return np.array([], dtype=int)

    peaks = np.unique(np.concatenate(peaks_all)).astype(int)
    peaks.sort()
    peaks = spike_count.apply_refractory(peaks, refractory)

    if spike_count.AMP_MIN_UV is not None or spike_count.AMP_MAX_UV is not None:
        amps = np.abs(x_hp[peaks]) if peaks.size else np.array([])
        keep = np.ones_like(peaks, dtype=bool)
        if spike_count.AMP_MIN_UV is not None:
            keep &= amps >= float(spike_count.AMP_MIN_UV)
        if spike_count.AMP_MAX_UV is not None:
            keep &= amps <= float(spike_count.AMP_MAX_UV)
        peaks = peaks[keep]

    return peaks


class ProcessingWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending = None
        self._mutex = QtCore.QMutex()
        self._condition = QtCore.QWaitCondition()
        self._running = True

    def stop(self):
        with QtCore.QMutexLocker(self._mutex):
            self._running = False
            self._condition.wakeAll()
        self.wait(2000)

    def schedule(self, signal_snap, t_snap, fs, spike_cache, last_scan, stored, do_psd=True, do_spike=True):
        with QtCore.QMutexLocker(self._mutex):
            self._pending = (signal_snap, t_snap, fs, spike_cache, last_scan, stored, bool(do_psd), bool(do_spike))
            self._condition.wakeOne()

    def run(self):
        while True:
            with QtCore.QMutexLocker(self._mutex):
                while self._pending is None and self._running:
                    self._condition.wait(self._mutex)
                if not self._running:
                    return
                job = self._pending
                self._pending = None

            signal_snap, t_snap, fs, spike_cache, last_scan, stored, do_psd, do_spike = job
            result = _ProcessingResult()
            result.has_psd_update = do_psd
            result.has_spike_update = do_spike

            # PSD
            if do_psd:
                psd_samples = max(8, int(round(fs * PSD_BUFFER_SEC)))
                psd_sig = signal_snap[-psd_samples:]
                target_nperseg = max(64, int(round(psd.TARGET_NPERSEG_SEC * fs)))
                nperseg = min(target_nperseg, psd_sig.size)
                noverlap = min(int(round(psd.NOVERLAP_RATIO * nperseg)), nperseg - 1)
                if nperseg >= 8:
                    try:
                        f, _, Pxx_db = psd.welch_psd(psd_sig, fs, nperseg, noverlap, fmax=psd.FMAX)
                        result.psd_f = f
                        result.psd_db = Pxx_db
                    except Exception:
                        pass

            # Incremental spike detection
            if do_spike:
                new_spike_times = list(spike_cache)
                if stored > last_scan + SPIKE_INCREMENTAL_MIN_SAMPLES:
                    overlap = SPIKE_OVERLAP_SAMPLES
                    scan_start = max(0, last_scan - overlap)
                    scan_signal = signal_snap[scan_start:stored]
                    if scan_signal.size >= 10:
                        try:
                            rel_peaks = _detect_spike_indices(scan_signal, fs)
                            if rel_peaks.size:
                                abs_times = t_snap[scan_start + rel_peaks]
                                if new_spike_times and last_scan < t_snap.size:
                                    cutoff = t_snap[last_scan]
                                    abs_times = abs_times[abs_times >= cutoff]
                                new_spike_times.extend(abs_times.tolist())
                        except Exception:
                            pass

                # Keep all spike times — full session history for histogram
                result.spike_times_cache = new_spike_times
                result.last_scan_sample = stored

                # Spike count histogram — spans entire session (t=0 to now)
                spike_arr = np.array(new_spike_times, dtype=float)
                if t_snap.size:
                    history_start_t = 0.0
                    first_bin = 0
                    last_bin = int(np.floor(t_snap[-1] / SPIKE_BIN_SEC))
                    bin_edges = np.arange(first_bin, last_bin + 2, dtype=float) * SPIKE_BIN_SEC
                    if bin_edges.size < 2:
                        bin_edges = np.array([history_start_t, history_start_t + SPIKE_BIN_SEC])
                    counts, edges = np.histogram(spike_arr, bins=bin_edges)
                    result.spike_minute_idx = edges[:-1] / 60.0
                    result.spike_counts = counts

                # Waveform mean+-SEM — use only spikes in the last WAVEFORM_BUFFER_SEC to ensure stationarity of waveforms
                if spike_arr.size >= 1 and t_snap.size:
                    current_time = t_snap[-1]
                    wf_spike_arr = spike_arr[spike_arr >= current_time - WAVEFORM_BUFFER_SEC]
                    if wf_spike_arr.size >= 1:
                        # Map spike timestamps back to nearest sample indices.
                        pk_indices = np.searchsorted(t_snap, wf_spike_arr, side='left').astype(int)
                        if pk_indices.size:
                            pk_indices = np.clip(pk_indices, 0, max(0, stored - 1))
                            left_idx = np.maximum(pk_indices - 1, 0)
                            use_left = np.abs(t_snap[left_idx] - wf_spike_arr) <= np.abs(t_snap[pk_indices] - wf_spike_arr)
                            pk_indices = np.where(use_left, left_idx, pk_indices)
                        pk_indices = np.unique(pk_indices)
                        pk_indices = pk_indices[(pk_indices > 0) & (pk_indices < stored)]
                        try:
                            # Use the same spike band for waveform extraction as detection.
                            x_hp = spike_count.bandpass_filt(
                                signal_snap[:stored].astype(float),
                                fs,
                                spike_count.HP_SPIKE_BAND,
                                order=3,
                            )
                            _, W = spike_plot.extract_waveforms(
                                x_hp,
                                pk_indices,
                                fs,
                                pre_ms=spike_plot.PRE_MS,
                                post_ms=spike_plot.POST_MS,
                            )
                            if W.shape[0] >= 1:
                                result.wf_t_ms = (np.arange(W.shape[1], dtype=np.float64) / fs) * 1000.0 - spike_plot.PRE_MS
                                result.wf_mu  = W.mean(axis=0)
                                result.wf_sem = W.std(axis=0) / max(np.sqrt(W.shape[0]), 1)
                        except Exception:
                            pass

            self.result_ready.emit(result)
