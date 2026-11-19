import numpy as np
from scipy.signal import butter, sosfilt

from .processing import psd as _psd
from .processing import spike_count as _spike_count
from .processing import spike_plot as _spike_plot
from .processing import SPIKE_INCREMENTAL_MIN_SAMPLES, SPIKE_OVERLAP_SAMPLES

_SOS_CACHE = {}
_SOS_CACHE_FS = 0.0


def _get_sos(fs):
    global _SOS_CACHE_FS
    if fs != _SOS_CACHE_FS:
        ny = fs / 2.0
        lo = max(_spike_count.HP_SPIKE_BAND[0] / ny, 1e-6)
        hi = min(_spike_count.HP_SPIKE_BAND[1] / ny, 0.999999)
        _SOS_CACHE[0] = butter(3, [lo, hi], btype='bandpass', output='sos')
        _SOS_CACHE_FS = fs
    return _SOS_CACHE[0]


class _FilterState:
    """Carried sosfilt state plus the trailing filtered window used for z-scoring.

    Mutated in place by the single analysis worker. The ring holds at most
    ``window_sec`` of samples, which is what keeps median/MAD off the full tail.
    """

    __slots__ = ('num_channels', 'sample_rate', 'cap', 'hist', 'wpos', 'length', 'end_abs', 'zi')

    def __init__(self, num_channels, sample_rate, window_sec):
        self.num_channels = int(num_channels)
        self.sample_rate = float(sample_rate)
        self.cap = max(1, int(round(self.sample_rate * float(window_sec))))
        self.hist = np.empty((self.num_channels, self.cap), dtype=np.float64)
        self.wpos = 0
        self.length = 0
        self.end_abs = 0
        self.zi = None

    def matches(self, num_channels, sample_rate):
        return (self.num_channels == int(num_channels)
                and abs(self.sample_rate - float(sample_rate)) < 1e-9)

    def reset(self, end_abs=0):
        self.wpos = 0
        self.length = 0
        self.end_abs = int(end_abs)
        self.zi = None

    def append(self, x, start_abs):
        n = x.shape[1]
        if n <= 0:
            return
        if n >= self.cap:
            self.hist[:, :] = x[:, -self.cap:]
            self.wpos = 0
            self.length = self.cap
        else:
            space = self.cap - self.wpos
            if n <= space:
                self.hist[:, self.wpos:self.wpos + n] = x
            else:
                self.hist[:, self.wpos:] = x[:, :space]
                self.hist[:, :n - space] = x[:, space:]
            self.wpos = (self.wpos + n) % self.cap
            self.length = min(self.cap, self.length + n)
        self.end_abs = int(start_abs) + n

    def _gather(self, start, n):
        if n <= 0:
            return None
        if start + n <= self.cap:
            return self.hist[:, start:start + n]
        tail = self.cap - start
        out = np.empty((self.num_channels, n), dtype=self.hist.dtype)
        out[:, :tail] = self.hist[:, start:]
        out[:, tail:] = self.hist[:, :n - tail]
        return out

    def last(self):
        if self.length <= 0:
            return None
        return self._gather((self.wpos - self.length) % self.cap, self.length)

    def read_abs(self, start_abs, end_abs):
        """Return (samples, actual_start_abs) clamped to the stored window."""
        oldest = self.end_abs - self.length
        lo = max(int(start_abs), oldest)
        hi = min(int(end_abs), self.end_abs)
        if hi <= lo:
            return None, lo
        pos = (self.wpos - (self.end_abs - lo)) % self.cap
        return self._gather(pos, hi - lo), lo


class _ProcessingResult:
    __slots__ = (
        'has_psd_update', 'has_spike_update',
        'selected_ch',
        'psd_f', 'psd_db',
        'wf_t_ms', 'wf_mu', 'wf_sem',
        'spike_times_cache', 'last_scan_sample',
        'hist_minute_idx', 'hist_counts',
        'hist_states', 'last_scans',
        'all_spike_caches', 'filter_state',
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


def _update_histogram_state(spike_times, last_time_s, state, bin_sec):
    bin_sec = float(bin_sec)
    total_bins = max(1, int(np.floor(float(last_time_s) / bin_sec)) + 1)
    cur_count = len(spike_times)

    counts = state.get('counts', np.zeros(0, dtype=np.int64))
    prev_last_t = state.get('last_t', 0.0)
    prev_bin_sec = state.get('bin_sec', bin_sec)
    minute_idx_cache = state.get('minute_idx_cache', None)

    need_reset = (
        counts.size == 0
        or not np.isclose(prev_bin_sec, bin_sec)
        or float(last_time_s) + 1e-9 < float(prev_last_t)
    )

    if need_reset:
        minute_idx_cache = None
        counts = np.zeros(total_bins, dtype=np.int64)
        spike_arr = np.asarray(spike_times, dtype=np.float64) if cur_count else np.array([], dtype=np.float64)
    else:
        if total_bins > counts.size:
            counts = np.pad(counts, (0, total_bins - counts.size), mode='constant')
        # ponytail: time-based incremental — bin only spikes after prev_last_t.
        # Replaces count-based slice which broke when segment sizes varied.
        spike_arr = np.asarray(spike_times, dtype=np.float64) if cur_count else np.array([], dtype=np.float64)
        if spike_arr.size:
            spike_arr = spike_arr[spike_arr > prev_last_t]

    if spike_arr.size:
        bins = np.floor(spike_arr / bin_sec).astype(np.int64)
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


def _run_all_channels(signal_matrix, t, fs, selected_ch, all_spike_caches,
                      last_scans, do_psd, do_spike, hist_states, abs_start=0,
                      psd_buffer_sec=10, waveform_buffer_sec=10, spike_bin_sec=5,
                      filter_state=None, median_window_sec=5.0):
    """Vectorized matrix pipeline: spike detection + histogram for all channels.

    ``filter_state`` carries the sosfilt delay line and a trailing window of
    filtered samples. Median/MAD for z-scoring comes from that window only,
    which keeps the analysis off the full display tail.
    """
    result = _ProcessingResult()
    result.selected_ch = selected_ch
    result.has_psd_update = do_psd
    result.has_spike_update = do_spike

    n_ch, n_samp = signal_matrix.shape
    stored_abs = abs_start + n_samp

    new_last_scans = dict(last_scans) if last_scans else {}
    new_hist_states = dict(hist_states) if hist_states else {}
    all_caches = {}

    state = filter_state
    if state is not None and not state.matches(n_ch, fs):
        state = None
    if state is not None:
        if state.length == 0:
            if state.end_abs != abs_start:
                state.reset(abs_start)
        elif not (abs_start <= state.end_abs <= stored_abs):
            # Ring was cleared, resized, or a backlog drop skipped samples.
            state.reset(abs_start)
    if state is None:
        state = _FilterState(n_ch, fs, median_window_sec)
        state.reset(abs_start)

    if do_spike and n_samp >= 10:
        sos = _get_sos(fs)
        rel = state.end_abs - abs_start
        if rel < 0 or rel > n_samp:
            state.reset(abs_start)
            rel = 0
        if rel < n_samp:
            x_new = signal_matrix[:, rel:]
            if state.zi is None:
                state.zi = np.zeros((sos.shape[0], n_ch, 2), dtype=np.float64)
            x_filt, state.zi = sosfilt(
                sos, x_new.astype(np.float64, copy=False), axis=1, zi=state.zi,
            )
            state.append(x_filt, abs_start + rel)

        hist = state.last()
        if hist is None or hist.shape[1] == 0:
            hist = None
        if hist is not None:
            med = np.median(hist, axis=1, keepdims=True)
            mad = np.median(np.abs(hist - med), axis=1, keepdims=True)
            mad = np.maximum(mad, 1e-12)

        refractory = max(1, int(round((_spike_count.REFRACTORY_MS / 1000.0) * fs)))

        # One gather for every channel; scanning per channel would copy the
        # wrapped window 128 times.
        scan_floor = max(
            abs_start,
            min(last_scans.get(ch, 0) for ch in range(n_ch)) - SPIKE_OVERLAP_SAMPLES,
        )
        scan_all, scan_base = state.read_abs(scan_floor, stored_abs)
        if scan_all is not None and scan_all.shape[1] <= 0:
            scan_all = None

        for ch in range(n_ch):
            last_scan_abs = last_scans.get(ch, 0) if last_scans else 0
            gap = stored_abs - last_scan_abs
            if gap < SPIKE_INCREMENTAL_MIN_SAMPLES:
                continue

            new_last_scans[ch] = stored_abs
            if hist is None or scan_all is None:
                continue

            overlap = SPIKE_OVERLAP_SAMPLES
            ch_start = max(abs_start, last_scan_abs - overlap)
            off = max(0, ch_start - scan_base)
            if off >= scan_all.shape[1]:
                continue
            t_rel = scan_base - abs_start + off
            if t_rel < 0:
                t_rel = 0
            if t_rel >= t.size:
                t_rel = max(0, t.size - 1)

            z = 0.6745 * (scan_all[ch, off:] - med[ch, 0]) / mad[ch, 0]
            peak_idx = np.where(np.abs(z) > _spike_count.SPIKE_Z_THR)[0]

            if peak_idx.size > 0:
                keep = np.concatenate([[True], np.diff(peak_idx) >= refractory])
                peak_idx = peak_idx[keep]

                if _spike_count.AMP_MIN_UV is not None or _spike_count.AMP_MAX_UV is not None:
                    amps = np.abs(scan_all[ch, off + peak_idx])
                    amp_keep = np.ones_like(peak_idx, dtype=bool)
                    if _spike_count.AMP_MIN_UV is not None:
                        amp_keep &= amps >= float(_spike_count.AMP_MIN_UV)
                    if _spike_count.AMP_MAX_UV is not None:
                        amp_keep &= amps <= float(_spike_count.AMP_MAX_UV)
                    peak_idx = peak_idx[amp_keep]

            matrix_idx = peak_idx + t_rel
            matrix_idx = matrix_idx[(matrix_idx >= 0) & (matrix_idx < t.size)]
            spike_times = t[matrix_idx] if matrix_idx.size else np.array([], dtype=float)

            prev_cache = list(all_spike_caches.get(ch, []))
            if prev_cache:
                cutoff = float(t[t_rel]) if t_rel < t.size else prev_cache[-1] + 1.0
                new_cache = [s for s in prev_cache if s < cutoff]
                new_cache.extend(spike_times.tolist())
            else:
                new_cache = spike_times.tolist()

            all_caches[ch] = new_cache

            hist_state = new_hist_states.get(ch, {})
            minute_idx, counts, new_state = _update_histogram_state(
                new_cache, float(t[-1]) if t.size else 0.0, hist_state,
                spike_bin_sec,
            )
            new_hist_states[ch] = new_state

            if ch == selected_ch:
                result.hist_minute_idx = minute_idx
                result.hist_counts = counts

            # waveforms for selected channel only
            if ch == selected_ch and new_cache:
                spike_arr = np.asarray(new_cache, dtype=float)
                current_time = float(t[-1]) if t.size else 0.0
                wf_spike_arr = spike_arr[spike_arr >= current_time - waveform_buffer_sec]
                if wf_spike_arr.size >= 1 and t.size:
                    pk_indices = np.searchsorted(t, wf_spike_arr, side='left').astype(int)
                    if pk_indices.size:
                        pk_indices = np.clip(pk_indices, 0, max(0, n_samp - 1))
                        left_idx = np.maximum(pk_indices - 1, 0)
                        use_left = np.abs(t[left_idx] - wf_spike_arr) <= np.abs(t[pk_indices] - wf_spike_arr)
                        pk_indices = np.where(use_left, left_idx, pk_indices)
                    pk_indices = np.unique(pk_indices)
                    pk_indices = pk_indices[(pk_indices > 0) & (pk_indices < n_samp)]
                    try:
                        pre_samp = int(round((_spike_plot.PRE_MS / 1000.0) * fs))
                        post_samp = int(round((_spike_plot.POST_MS / 1000.0) * fs))
                        seg_start = max(0, int(pk_indices[0]) - pre_samp - 2)
                        seg_end = min(n_samp, int(pk_indices[-1]) + post_samp + 3)
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
            new_last_scans[ch] = stored_abs

    if do_psd and selected_ch < n_ch:
        try:
            psd_samples = max(8, int(round(fs * psd_buffer_sec)))
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
    result.all_spike_caches = all_caches
    result.filter_state = state
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
