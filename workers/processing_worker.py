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

# Waveform averaging buffer duration and forgetting
WAVEFORM_BUFFER_SEC = 60.0
WAVEFORM_FORGETTING_FACTOR = 0.95

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


def _apply_waveform_forgetting(waveform_buffer, current_time, W, spike_time):
    """
    Add new waveforms to buffer and apply exponential forgetting based on age.
    Returns weighted mean and SEM across buffer.
    
    Args:
        waveform_buffer: List of (timestamp, waveform_array) tuples
        current_time: Current timestamp (seconds)
        W: New waveforms array (num_spikes, num_samples)
        spike_time: Time of the spike being added
    """
    # Add new waveforms to buffer with their spike time
    for i in range(W.shape[0]):
        waveform_buffer.append((spike_time + i / (W.shape[0] + 1), W[i, :]))
    
    # Remove waveforms older than buffer duration
    cutoff_time = current_time - WAVEFORM_BUFFER_SEC
    waveform_buffer[:] = [(t, w) for t, w in waveform_buffer if t >= cutoff_time]
    
    if not waveform_buffer:
        return None, None
    
    # Apply exponential forgetting: weight = FORGETTING_FACTOR ^ (age_in_seconds)
    weights = []
    for t, _ in waveform_buffer:
        age = current_time - t
        weight = WAVEFORM_FORGETTING_FACTOR ** age
        weights.append(weight)
    
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    
    # Stack waveforms and apply weights
    W_all = np.array([w for _, w in waveform_buffer], dtype=np.float64)
    wf_mu = (W_all * weights[:, np.newaxis]).sum(axis=0)
    wf_sem = (W_all * weights[:, np.newaxis]).std(axis=0) / max(np.sqrt((weights ** 2).sum()), 1.0)
    
    return wf_mu, wf_sem


class ProcessingWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending = None
        self._mutex = QtCore.QMutex()
        self._condition = QtCore.QWaitCondition()
        self._running = True
        self._waveform_buffer = []  # List of (timestamp, waveform) tuples for forgetting

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

                # Trim history
                if t_snap.size:
                    history_cutoff = t_snap[-1] - SPIKE_HISTORY_MIN * 60.0
                    new_spike_times = [t for t in new_spike_times if t >= history_cutoff]

                result.spike_times_cache = new_spike_times
                result.last_scan_sample = stored

                # Spike count histogram
                spike_arr = np.array(new_spike_times, dtype=float)
                if t_snap.size:
                    history_start_t = max(0.0, t_snap[-1] - SPIKE_HISTORY_MIN * 60.0)
                    first_bin = int(np.floor(history_start_t / SPIKE_BIN_SEC))
                    last_bin = int(np.floor(t_snap[-1] / SPIKE_BIN_SEC))
                    bin_edges = np.arange(first_bin, last_bin + 2, dtype=float) * SPIKE_BIN_SEC
                    if bin_edges.size < 2:
                        bin_edges = np.array([history_start_t, history_start_t + SPIKE_BIN_SEC])
                    counts, edges = np.histogram(spike_arr, bins=bin_edges)
                    result.spike_minute_idx = edges[:-1] / 60.0
                    result.spike_counts = counts

                # Waveform mean+-SEM
                if spike_arr.size >= 1 and t_snap.size:
                    pk_indices = np.searchsorted(t_snap, spike_arr).astype(int)
                    pk_indices = pk_indices[(pk_indices > 0) & (pk_indices < stored)]
                    try:
                        _, W = spike_plot.extract_waveforms(
                            signal_snap[:stored].astype(float),
                            pk_indices,
                            fs,
                            pre_ms=spike_plot.PRE_MS,
                            post_ms=spike_plot.POST_MS,
                        )
                        if W.shape[0] >= 1:
                            t_wf_ms = (np.arange(W.shape[1], dtype=np.float64) / fs) * 1000.0 - spike_plot.PRE_MS
                            result.wf_t_ms = t_wf_ms
                            
                            # Apply waveform forgetting
                            current_time = t_snap[-1] if t_snap.size else 0.0
                            spike_time = spike_arr[0] if spike_arr.size else current_time
                            wf_mu, wf_sem = _apply_waveform_forgetting(
                                self._waveform_buffer, 
                                current_time, 
                                W, 
                                spike_time
                            )
                            
                            if wf_mu is not None:
                                result.wf_mu = wf_mu
                                result.wf_sem = wf_sem
                            else:
                                result.wf_mu = W.mean(axis=0)
                                result.wf_sem = W.std(axis=0) / max(np.sqrt(W.shape[0]), 1)
                    except Exception:
                        pass

            self.result_ready.emit(result)
