import numpy as np

from .processing import psd as _psd
from .processing import spike_count as _spike_count
from .processing import spike_plot as _spike_plot
from . import processing as _proc_cfg
from .processing import SPIKE_INCREMENTAL_MIN_SAMPLES, SPIKE_OVERLAP_SAMPLES


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
    x_hp = _spike_count.bandpass_filt(x.astype(float), fs, _spike_count.HP_SPIKE_BAND, order=3)
    z, _, _ = _spike_count.robust_z(x_hp)
    z = np.asarray(z, dtype=np.float64).ravel()
    refractory = max(1, int(round((_spike_count.REFRACTORY_MS / 1000.0) * fs)))
    peaks_all = []
    if _spike_count.POLARITY in ("pos", "both"):
        p_pos, _ = _spike_count.find_peaks(z, height=_spike_count.SPIKE_Z_THR, distance=refractory)
        peaks_all.append(p_pos)
    if _spike_count.POLARITY in ("neg", "both"):
        p_neg, _ = _spike_count.find_peaks(-z, height=_spike_count.SPIKE_Z_THR, distance=refractory)
        peaks_all.append(p_neg)
    if not peaks_all:
        return np.array([], dtype=int)
    peaks = np.unique(np.concatenate(peaks_all)).astype(int)
    peaks.sort()
    peaks = _spike_count.apply_refractory(peaks, refractory)
    if _spike_count.AMP_MIN_UV is not None or _spike_count.AMP_MAX_UV is not None:
        amps = np.abs(x_hp[peaks]) if peaks.size else np.array([])
        keep = np.ones_like(peaks, dtype=bool)
        if _spike_count.AMP_MIN_UV is not None:
            keep &= amps >= float(_spike_count.AMP_MIN_UV)
        if _spike_count.AMP_MAX_UV is not None:
            keep &= amps <= float(_spike_count.AMP_MAX_UV)
        peaks = peaks[keep]
    return peaks


# ponytail: histogram cache — single-threaded in BackgroundWorker, never concurrently accessed
_hist_counts = np.zeros(0, dtype=np.int64)
_hist_bin_sec = float(_proc_cfg.SPIKE_BIN_SEC)
_hist_spike_count = 0
_hist_last_t = 0.0
_hist_minute_idx_cache = None


def _reset_hist_cache():
    global _hist_counts, _hist_bin_sec, _hist_spike_count, _hist_last_t, _hist_minute_idx_cache
    _hist_counts = np.zeros(0, dtype=np.int64)
    _hist_bin_sec = float(_proc_cfg.SPIKE_BIN_SEC)
    _hist_spike_count = 0
    _hist_last_t = 0.0
    _hist_minute_idx_cache = None


def _update_incremental_histogram(spike_times, last_time_s):
    global _hist_counts, _hist_bin_sec, _hist_spike_count, _hist_last_t, _hist_minute_idx_cache
    bin_sec = float(_proc_cfg.SPIKE_BIN_SEC)
    total_bins = max(1, int(np.floor(float(last_time_s) / bin_sec)) + 1)
    cur_count = len(spike_times)
    need_reset = (
        _hist_counts.size == 0
        or not np.isclose(_hist_bin_sec, bin_sec)
        or cur_count < _hist_spike_count
        or float(last_time_s) + 1e-9 < float(_hist_last_t)
    )
    if need_reset:
        _hist_minute_idx_cache = None
    if need_reset:
        _hist_counts = np.zeros(total_bins, dtype=np.int64)
        if cur_count:
            spike_arr = np.asarray(spike_times, dtype=np.float64)
            bins = np.floor(spike_arr / bin_sec).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < total_bins)]
            if bins.size:
                _hist_counts += np.bincount(bins, minlength=total_bins).astype(np.int64)
    else:
        if total_bins > _hist_counts.size:
            pad_n = total_bins - _hist_counts.size
            _hist_counts = np.pad(_hist_counts, (0, pad_n), mode='constant')
        if cur_count > _hist_spike_count:
            new_spikes = np.asarray(spike_times[_hist_spike_count:], dtype=np.float64)
            bins = np.floor(new_spikes / bin_sec).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < _hist_counts.size)]
            if bins.size:
                _hist_counts += np.bincount(bins, minlength=_hist_counts.size).astype(np.int64)
    _hist_bin_sec = bin_sec
    _hist_spike_count = cur_count
    _hist_last_t = float(last_time_s)
    if _hist_minute_idx_cache is None or _hist_minute_idx_cache.size != _hist_counts.size:
        _hist_minute_idx_cache = (np.arange(_hist_counts.size, dtype=np.float64) * bin_sec) / 60.0
    return _hist_minute_idx_cache, _hist_counts.copy()


def _run_spike_detect(signal_snap, t_snap, fs, spike_cache, last_scan, stored, do_psd, do_spike):
    """Run in BackgroundWorker thread. Returns _ProcessingResult."""
    result = _ProcessingResult()
    result.has_psd_update = do_psd
    result.has_spike_update = do_spike

    if do_psd:
        psd_samples = max(8, int(round(fs * _proc_cfg.PSD_BUFFER_SEC)))
        psd_sig = signal_snap[-psd_samples:]
        target_nperseg = max(64, int(round(_psd.TARGET_NPERSEG_SEC * fs)))
        nperseg = min(target_nperseg, psd_sig.size)
        nperseg = max(64, 2**int(round(np.log2(nperseg))))
        nperseg = min(nperseg, psd_sig.size)
        noverlap = min(int(round(_psd.NOVERLAP_RATIO * nperseg)), nperseg - 1)
        if nperseg >= 8:
            try:
                f, _, Pxx_db = _psd.welch_psd(psd_sig, fs, nperseg, noverlap, fmax=_psd.FMAX)
                result.psd_f = f
                result.psd_db = Pxx_db
            except Exception:
                pass

    if do_spike:
        if stored > last_scan + SPIKE_INCREMENTAL_MIN_SAMPLES:
            overlap = SPIKE_OVERLAP_SAMPLES
            scan_start = max(0, last_scan - overlap)
            scan_signal = signal_snap[scan_start:stored]
            new_spike_times = list(spike_cache)
            if not new_spike_times:
                _reset_hist_cache()
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
        else:
            new_spike_times = spike_cache

        result.spike_times_cache = new_spike_times
        result.last_scan_sample = stored

        spike_arr = np.array(new_spike_times, dtype=float)
        if t_snap.size:
            minute_idx, counts = _update_incremental_histogram(new_spike_times, float(t_snap[-1]))
            result.spike_minute_idx = minute_idx
            result.spike_counts = counts

        if spike_arr.size >= 1 and t_snap.size:
            current_time = t_snap[-1]
            wf_spike_arr = spike_arr[spike_arr >= current_time - _proc_cfg.WAVEFORM_BUFFER_SEC]
            if wf_spike_arr.size >= 1:
                pk_indices = np.searchsorted(t_snap, wf_spike_arr, side='left').astype(int)
                if pk_indices.size:
                    pk_indices = np.clip(pk_indices, 0, max(0, stored - 1))
                    left_idx = np.maximum(pk_indices - 1, 0)
                    use_left = np.abs(t_snap[left_idx] - wf_spike_arr) <= np.abs(t_snap[pk_indices] - wf_spike_arr)
                    pk_indices = np.where(use_left, left_idx, pk_indices)
                pk_indices = np.unique(pk_indices)
                pk_indices = pk_indices[(pk_indices > 0) & (pk_indices < stored)]
                try:
                    pre_samp = int(round((_spike_plot.PRE_MS / 1000.0) * fs))
                    post_samp = int(round((_spike_plot.POST_MS / 1000.0) * fs))
                    seg_start = max(0, int(pk_indices[0]) - pre_samp - 2)
                    seg_end = min(int(stored), int(pk_indices[-1]) + post_samp + 3)
                    if seg_end - seg_start >= 10:
                        x_hp_seg = _spike_count.bandpass_filt(
                            signal_snap[seg_start:seg_end].astype(float),
                            fs,
                            _spike_count.HP_SPIKE_BAND,
                            order=3,
                        )
                        pk_local = pk_indices - seg_start
                        _, W = _spike_plot.extract_waveforms(
                            x_hp_seg, pk_local, fs,
                            pre_ms=_spike_plot.PRE_MS,
                            post_ms=_spike_plot.POST_MS,
                        )
                        if W.shape[0] >= 1:
                            result.wf_t_ms = (np.arange(W.shape[1], dtype=np.float64) / fs) * 1000.0 - _spike_plot.PRE_MS
                            result.wf_mu = W.mean(axis=0)
                            result.wf_sem = W.std(axis=0) / max(np.sqrt(W.shape[0]), 1)
                except Exception:
                    pass

    return result


def _run_spike_rebin(spike_times, bin_sec, last_time_s, task_id=0, session_id=0):
    """Run in BackgroundWorker thread. Returns result dict with task metadata."""
    spike_arr = np.asarray(spike_times, dtype=np.float64)
    if bin_sec <= 0:
        raise ValueError("bin_sec must be > 0")
    total_bins = max(1, int(np.floor(float(last_time_s) / bin_sec)) + 1)
    counts = np.zeros(total_bins, dtype=np.int64)
    if spike_arr.size:
        bins = np.floor(spike_arr / bin_sec).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < total_bins)]
        if bins.size:
            counts += np.bincount(bins, minlength=total_bins).astype(np.int64)
    minute_idx = (np.arange(total_bins, dtype=np.float64) * bin_sec) / 60.0
    return {
        "task_type": "spike_rebin",
        "task_id": task_id,
        "session_id": session_id,
        "status": "ok",
        "data": {
            "minute_idx": minute_idx,
            "counts": counts,
            "bin_sec": bin_sec,
            "last_time_s": last_time_s,
        },
        "error": "",
    }
