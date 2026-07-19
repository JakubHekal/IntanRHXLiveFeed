import numpy as np
from scipy.signal import butter, sosfilt

from .processing import psd as _psd
from .processing import spike_count as _spike_count
from .processing import spike_plot as _spike_plot
from . import processing as _proc_cfg
from .processing import SPIKE_INCREMENTAL_MIN_SAMPLES, SPIKE_OVERLAP_SAMPLES

_SOS_CACHE = {}
_SOS_CACHE_FS = 0.0


def _get_sos(fs):
    if fs != _SOS_CACHE_FS:
        ny = fs / 2.0
        lo = max(_spike_count.HP_SPIKE_BAND[0] / ny, 1e-6)
        hi = min(_spike_count.HP_SPIKE_BAND[1] / ny, 0.999999)
        _SOS_CACHE[0] = butter(3, [lo, hi], btype='bandpass', output='sos')
        global _SOS_CACHE_FS
        _SOS_CACHE_FS = fs
    return _SOS_CACHE[0]


class _ProcessingResult:
    __slots__ = (
        'has_psd_update', 'has_spike_update',
        'selected_ch',
        'psd_f', 'psd_db',
        'wf_t_ms', 'wf_mu', 'wf_sem',
        'spike_times_cache', 'last_scan_sample',
        'hist_minute_idx', 'hist_counts',
        'hist_states', 'last_scans',
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


def _update_histogram_state(spike_times, last_time_s, state):
    bin_sec = float(_proc_cfg.SPIKE_BIN_SEC)
    total_bins = max(1, int(np.floor(float(last_time_s) / bin_sec)) + 1)
    cur_count = len(spike_times)

    counts = state.get('counts', np.zeros(0, dtype=np.int64))
    prev_count = state.get('spike_count', 0)
    prev_last_t = state.get('last_t', 0.0)
    prev_bin_sec = state.get('bin_sec', bin_sec)
    minute_idx_cache = state.get('minute_idx_cache', None)

    need_reset = (
        counts.size == 0
        or not np.isclose(prev_bin_sec, bin_sec)
        or cur_count < prev_count
        or float(last_time_s) + 1e-9 < float(prev_last_t)
    )

    if need_reset:
        minute_idx_cache = None
        counts = np.zeros(total_bins, dtype=np.int64)
        if cur_count:
            spike_arr = np.asarray(spike_times, dtype=np.float64)
            bins = np.floor(spike_arr / bin_sec).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < total_bins)]
            if bins.size:
                counts += np.bincount(bins, minlength=total_bins).astype(np.int64)
    else:
        if total_bins > counts.size:
            counts = np.pad(counts, (0, total_bins - counts.size), mode='constant')
        if cur_count > prev_count:
            new_spikes = np.asarray(spike_times[prev_count:], dtype=np.float64)
            bins = np.floor(new_spikes / bin_sec).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < counts.size)]
            if bins.size:
                counts += np.bincount(bins, minlength=counts.size).astype(np.int64)

    if minute_idx_cache is None or minute_idx_cache.size != counts.size:
        minute_idx_cache = (np.arange(counts.size, dtype=np.float64) * bin_sec) / 60.0

    new_state = {
        'counts': counts,
        'bin_sec': bin_sec,
        'spike_count': cur_count,
        'last_t': float(last_time_s),
        'minute_idx_cache': minute_idx_cache,
    }
    return minute_idx_cache, counts.copy(), new_state


def _run_all_channels(signal_matrix, t, fs, selected_ch, spike_cache,
                      last_scans, do_psd, do_spike, hist_states):
    """Vectorized matrix pipeline: spike detection + histogram for all channels."""
    result = _ProcessingResult()
    result.selected_ch = selected_ch
    result.has_psd_update = do_psd
    result.has_spike_update = do_spike

    n_ch, n_samp = signal_matrix.shape
    stored = n_samp

    new_last_scans = dict(last_scans) if last_scans else {}
    new_hist_states = dict(hist_states) if hist_states else {}
    ch_spike_times = {}

    if do_spike and n_samp >= 10:
        sos = _get_sos(fs)
        x_hp = sosfilt(sos, signal_matrix.astype(np.float64), axis=1)

        med = np.median(x_hp, axis=1, keepdims=True)
        mad = np.median(np.abs(x_hp - med), axis=1, keepdims=True)
        mad = np.maximum(mad, 1e-12)
        z = 0.6745 * (x_hp - med) / mad

        mask = np.abs(z) > _spike_count.SPIKE_Z_THR

        refractory = max(1, int(round((_spike_count.REFRACTORY_MS / 1000.0) * fs)))

        for ch in range(n_ch):
            last_scan = last_scans.get(ch, 0) if last_scans else 0
            gap = stored - last_scan
            if gap < SPIKE_INCREMENTAL_MIN_SAMPLES:
                continue

            overlap = SPIKE_OVERLAP_SAMPLES
            scan_start = max(0, last_scan - overlap)

            ch_mask = mask[ch, scan_start:stored]
            peak_idx = np.where(ch_mask)[0]

            if peak_idx.size > 0:
                peak_idx = peak_idx + scan_start
                keep = np.concatenate([[True], np.diff(peak_idx) >= refractory])
                peak_idx = peak_idx[keep]

                if _spike_count.AMP_MIN_UV is not None or _spike_count.AMP_MAX_UV is not None:
                    amps = np.abs(x_hp[ch, peak_idx])
                    amp_keep = np.ones_like(peak_idx, dtype=bool)
                    if _spike_count.AMP_MIN_UV is not None:
                        amp_keep &= amps >= float(_spike_count.AMP_MIN_UV)
                    if _spike_count.AMP_MAX_UV is not None:
                        amp_keep &= amps <= float(_spike_count.AMP_MAX_UV)
                    peak_idx = peak_idx[amp_keep]

            spike_times = t[peak_idx] if peak_idx.size else np.array([], dtype=float)
            ch_spike_times[ch] = spike_times
            new_last_scans[ch] = stored

            prev_cache = list(spike_cache) if (ch == selected_ch and spike_cache) else []
            if prev_cache:
                if last_scan < t.size:
                    cutoff = t[last_scan]
                    new_cache = [s for s in prev_cache if s < cutoff]
                    new_cache.extend(spike_times.tolist())
                else:
                    new_cache = prev_cache
            else:
                new_cache = spike_times.tolist()

            if ch == selected_ch:
                result.spike_times_cache = new_cache

            state = new_hist_states.get(ch, {})
            minute_idx, counts, new_state = _update_histogram_state(
                new_cache if ch == selected_ch else spike_times.tolist(),
                float(t[-1]) if t.size else 0.0,
                state,
            )
            new_hist_states[ch] = new_state

            if ch == selected_ch:
                result.hist_minute_idx = minute_idx
                result.hist_counts = counts

            # waveforms for selected channel only
            if ch == selected_ch and new_cache:
                spike_arr = np.asarray(new_cache, dtype=float)
                current_time = float(t[-1]) if t.size else 0.0
                wf_spike_arr = spike_arr[spike_arr >= current_time - _proc_cfg.WAVEFORM_BUFFER_SEC]
                if wf_spike_arr.size >= 1 and t.size:
                    pk_indices = np.searchsorted(t, wf_spike_arr, side='left').astype(int)
                    if pk_indices.size:
                        pk_indices = np.clip(pk_indices, 0, max(0, stored - 1))
                        left_idx = np.maximum(pk_indices - 1, 0)
                        use_left = np.abs(t[left_idx] - wf_spike_arr) <= np.abs(t[pk_indices] - wf_spike_arr)
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
                                signal_matrix[selected_ch, seg_start:seg_end].astype(float),
                                fs, _spike_count.HP_SPIKE_BAND, order=3,
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

    elif do_spike:
        for ch in range(n_ch):
            new_last_scans[ch] = stored

    if do_psd and selected_ch < n_ch:
        try:
            psd_samples = max(8, int(round(fs * _proc_cfg.PSD_BUFFER_SEC)))
            psd_sig = signal_matrix[selected_ch, -psd_samples:]
            target_nperseg = max(64, int(round(_psd.TARGET_NPERSEG_SEC * fs)))
            nperseg = min(target_nperseg, psd_sig.size)
            nperseg = max(64, 2 ** int(round(np.log2(nperseg))))
            nperseg = min(nperseg, psd_sig.size)
            noverlap = min(int(round(_psd.NOVERLAP_RATIO * nperseg)), nperseg - 1)
            if nperseg >= 8:
                f, _, Pxx_db = _psd.welch_psd(psd_sig, fs, nperseg, noverlap, fmax=_psd.FMAX)
                result.psd_f = f
                result.psd_db = Pxx_db
        except Exception:
            pass

    result.hist_states = new_hist_states
    result.last_scans = new_last_scans
    return result


def _run_spike_detect(signal_snap, t_snap, fs, spike_cache, last_scan, stored, do_psd, do_spike):
    """Original per-channel pipeline. Kept for backward compat."""
    result = _ProcessingResult()
    result.has_psd_update = do_psd
    result.has_spike_update = do_spike

    if do_psd:
        psd_samples = max(8, int(round(fs * _proc_cfg.PSD_BUFFER_SEC)))
        psd_sig = signal_snap[-psd_samples:]
        target_nperseg = max(64, int(round(_psd.TARGET_NPERSEG_SEC * fs)))
        nperseg = min(target_nperseg, psd_sig.size)
        nperseg = max(64, 2 ** int(round(np.log2(nperseg))))
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
            minute_idx, counts, _ = _update_histogram_state(
                new_spike_times, float(t_snap[-1]), {},
            )
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
                            fs, _spike_count.HP_SPIKE_BAND, order=3,
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
