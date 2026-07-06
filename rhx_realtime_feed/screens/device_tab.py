import bisect
import time
from pathlib import Path

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np

from rhx_realtime_feed.plot_settings import load_plot_setting, save_plot_setting, DEFAULT_PSDS, DEFAULT_WAVEFORM, DEFAULT_SPIKE_BIN
from rhx_realtime_feed.workers.processing_worker import (
    ProcessingWorker,
    configure_processing_windows,
    SPIKE_INCREMENTAL_MIN_SAMPLES,
    SPIKE_OVERLAP_SAMPLES,
    PSD_BUFFER_SEC,
    WAVEFORM_BUFFER_SEC,
    SPIKE_BIN_SEC,
    PSD_YLIM_MIN,
    PSD_YLIM_MAX,
)
from rhx_realtime_feed.workers.expensive_task_worker import ExpensiveTaskWorker
from rhx_realtime_feed.screens.plot_helpers import (
    _make_display_buffer, _minmax_downsample,
    DISPLAY_WINDOW_SEC, DISPLAY_BUFFER_SEC, DEFAULT_SAMPLING_RATE,
    MAX_DISPLAY_POINTS, PLOT_UPDATE_FREQ_HZ, RAW_RENDER_HZ, PSD_RENDER_HZ, SPIKE_RENDER_HZ,
    WAVEFORM_YLIM_ABS_UV, SPIKE_SCROLL_WINDOW_MIN,
    RAW_HISTORY_TARGET_HZ, RAW_HISTORY_HIGH_TARGET_HZ,
    RAW_ADAPTIVE_HIGH_RES_MAX_SPAN_SEC, RAW_FULL_RES_MAX_SPAN_SEC,
    RAW_MANUAL_VIEW_MARGIN_SEC, MAX_RAW_HISTORY_PLOT_POINTS,
    PSD_PLOT_UPDATE_EVERY_N, SPIKE_PLOT_UPDATE_EVERY_N,
)
from rhx_realtime_feed.screens.pg_canvas import PgCanvas
from rhx_realtime_feed.screens.smu_canvas import SmuCanvas
from rhx_realtime_feed.telemetry_logger import append_telemetry_line


class DeviceTab(QtWidgets.QWidget):
    """Base class for a device's plot tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clear_requested = False
        self._render_requested = False

    def on_data(self, chunk: np.ndarray):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def shutdown(self) -> bool:
        raise NotImplementedError

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        pass

    def set_receiving_state(self, receiving: bool):
        pass

    def request_render(self):
        self._render_requested = True


# ── NeuralDeviceTab ─────────────────────────────────────────────────────────

class NeuralDeviceTab(DeviceTab):
    def __init__(self, sample_rate=float(DEFAULT_SAMPLING_RATE), num_channels=1, channel_labels=None, parent=None):
        super().__init__(parent)
        self.sampling_rate = sample_rate
        self.num_channels = num_channels
        self._current_channel_idx = 0
        self._channel_labels = channel_labels or [f"CH-{i:03d}" for i in range(num_channels)]
        self.sample_counter = 0
        self.marker_records = []
        self._marker_times_sorted = []
        self.is_receiving = False
        self._receiving_wall_active_sec = 0.0
        self._receiving_wall_run_start = None
        self._pending_channel = 0
        self._channel_results: dict[int, _Result] = {}
        self._spike_times_cache: dict[int, list] = {}
        self._last_spike_scan_sample: dict[int, int] = {}
        self._last_proc_abs_start = 0
        self._psd_buffer_sec = load_plot_setting("psd_buffer_sec", DEFAULT_PSDS)
        self._waveform_buffer_sec = load_plot_setting("waveform_buffer_sec", DEFAULT_WAVEFORM)
        self._spike_bin_sec = load_plot_setting("spike_bin_sec", DEFAULT_SPIKE_BIN)
        self._latest_psd_f = None
        self._latest_psd_db = None
        self._latest_wf_t_ms = None
        self._latest_wf_mu = None
        self._raw_hist_t_low = [[] for _ in range(self.num_channels)]
        self._raw_hist_y_low = [[] for _ in range(self.num_channels)]
        self._raw_hist_t_high = [[] for _ in range(self.num_channels)]
        self._raw_hist_y_high = [[] for _ in range(self.num_channels)]
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0
        self._raw_hist_stride_low = 1
        self._raw_hist_stride_high = 1
        self._follow_axes = {'raw': True, 'psd': True, 'spike': True, 'wf': True}
        self._follow_menu_actions = {}
        self._suspend_follow_detection = False
        self._psd_snapshot_curves = []
        self._wf_snapshot_curves = []
        self._ring = _make_display_buffer(self.sampling_rate, self.num_channels)
        self._cap = self._ring.shape[1]
        self._wpos = 0
        self._total = 0
        self._last_raw_render_t = 0.0
        self._last_psd_render_t = 0.0
        self._last_spike_render_t = 0.0
        self._psd_pending = False
        self._spike_pending = False
        self.psd_update_counter = 0
        self.spike_plot_frame_counter = 0
        self._session_id = 0
        self._task_id_counter = 0
        self._active_expensive_task_id = {}
        self._render_dur_ms = 0.0
        self._telemetry_last_emit = time.perf_counter()
        self._telemetry_emit_interval_sec = 5.0
        self._telemetry_chunks = 0
        self._telemetry_samples = 0
        self._telemetry_render_calls = 0
        self._telemetry_raw_renders = 0
        self._telemetry_render_ms_total = 0.0
        self._telemetry_ingest_ms_total = 0.0
        self._telemetry_latest_chunk_received_t = time.perf_counter()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._channel_combo = QtWidgets.QComboBox()
        self._channel_combo.addItems(self._channel_labels)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        layout.addWidget(self._channel_combo)

        self.canvas = PgCanvas(self)
        layout.addWidget(self.canvas, 1)

        self._proc_worker = ProcessingWorker(self)
        self._proc_worker.result_ready.connect(self._on_processing_result)
        self._proc_worker.start()
        self._exp_worker = ExpensiveTaskWorker(self)
        self._exp_worker.result_ready.connect(self._on_expensive_task_result)
        self._exp_worker.start()

        self._connect_plot_range_signals()
        self._install_plot_follow_context_actions()
        self._install_plot_bin_window_size_context_actions()
        self._update_raw_history_stride()

    def _on_channel_changed(self, idx):
        self._current_channel_idx = idx
        # rebuild raw history for new channel from ring
        ch = idx
        self._raw_hist_t_low[ch].clear()
        self._raw_hist_y_low[ch].clear()
        self._raw_hist_t_high[ch].clear()
        self._raw_hist_y_high[ch].clear()
        if self._total > 0:
            t, y = self._ring_read_channel(ch)
            if t.size > 0:
                self._raw_hist_t_low[ch] = t.tolist()
                self._raw_hist_y_low[ch] = y.tolist()
                if t.size > 1:
                    step = max(1, t.size // MAX_RAW_HISTORY_PLOT_POINTS)
                    self._raw_hist_t_high[ch] = t[::step].tolist()
                    self._raw_hist_y_high[ch] = y[::step].tolist()

    def _emit_telemetry_if_due(self):
        now = time.perf_counter()
        elapsed = now - self._telemetry_last_emit
        if elapsed < self._telemetry_emit_interval_sec:
            return
        chunks = max(1, int(self._telemetry_chunks))
        samples = int(self._telemetry_samples)
        rate_hz = float(samples) / max(elapsed, 1e-6)
        avg_ingest_ms = self._telemetry_ingest_ms_total / chunks
        avg_render_ms = self._telemetry_render_ms_total / max(1, int(self._telemetry_render_calls))
        avg_frame_ms = self._render_dur_ms
        line = (
            "[telemetry][plot] "
            f"window_s={elapsed:.2f} chunks={chunks} samples={samples} rate_hz={rate_hz:.1f} "
            f"ingest_avg_ms={avg_ingest_ms:.3f} render_avg_ms={avg_render_ms:.3f} frame_avg_ms={avg_frame_ms:.3f} "
            f"raw_renders={int(self._telemetry_raw_renders)}"
        )
        append_telemetry_line(line)
        self._telemetry_last_emit = now
        self._telemetry_chunks = 0
        self._telemetry_samples = 0
        self._telemetry_render_calls = 0
        self._telemetry_raw_renders = 0
        self._telemetry_render_ms_total = 0.0
        self._telemetry_ingest_ms_total = 0.0

    def _next_task_id(self):
        self._task_id_counter += 1
        return self._task_id_counter

    def _plot_item_for_key(self, key):
        return getattr(self.canvas, f"{key}_plot", None)

    def _connect_plot_range_signals(self):
        for key in ('raw', 'psd', 'spike', 'wf'):
            plot_item = self._plot_item_for_key(key)
            if plot_item is None:
                continue
            vb = plot_item.vb
            if hasattr(vb, 'sigRangeChangedManually'):
                vb.sigRangeChangedManually.connect(lambda *a, k=key: self._on_manual_plot_range_change(k))
            else:
                vb.sigRangeChanged.connect(lambda *a, k=key: self._on_manual_plot_range_change(k))

    def _on_manual_plot_range_change(self, key):
        if self._suspend_follow_detection:
            return
        if key in self._follow_axes:
            self._follow_axes[key] = False
            action = self._follow_menu_actions.get(key)
            if action:
                action.setChecked(False)

    def _install_plot_follow_context_actions(self):
        for key, label in (('raw', 'Auto-follow Raw'), ('psd', 'Auto-follow PSD'),
                           ('spike', 'Auto-follow Spikes'), ('wf', 'Auto-follow Waveform')):
            plot_item = self._plot_item_for_key(key)
            if plot_item is None:
                continue
            action = QtWidgets.QAction(label)
            action.setCheckable(True)
            action.setChecked(self._follow_axes[key])
            action.toggled.connect(lambda checked, k=key: self._set_follow(k, checked))
            plot_item.vb.menu.addAction(action)
            self._follow_menu_actions[key] = action

    def _set_follow(self, key, enabled):
        self._follow_axes[key] = enabled

    def _install_plot_bin_window_size_context_actions(self):
        spike_plot = self._plot_item_for_key('spike')
        if spike_plot is not None:
            vb = spike_plot.vb
            vb.menu.addSeparator()
            bin_menu = QtWidgets.QMenu("Bin width")
            for sec in (5, 10, 15, 20, 30, 60):
                a = bin_menu.addAction(f"{sec}s bins")
                a.triggered.connect(lambda *a, s=sec: self._set_spike_bin_sec(s))
            vb.menu.addMenu(bin_menu)

    def _set_spike_bin_sec(self, sec):
        self._spike_bin_sec = sec
        save_plot_setting("spike_bin_sec", sec)
        configure_processing_windows(spike_bin_sec=sec)
        self.canvas.spike_plot.setTitle(f"Spike counts ({sec}s bins)")
        self._schedule_spike_rebin_task()

    def _update_raw_history_stride(self):
        self._raw_hist_stride_low = max(1, int(round(self.sampling_rate / RAW_HISTORY_TARGET_HZ)))
        self._raw_hist_stride_high = max(1, int(round(self.sampling_rate / RAW_HISTORY_HIGH_TARGET_HZ)))
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0

    def _ring_write(self, t, y):
        """t: (n,) array, y: (num_channels, n) array."""
        n = y.shape[1] if y.ndim == 2 else y.size
        if n == 0:
            return
        if y.ndim == 1:
            y = y.reshape(1, -1)
        cap = self._cap
        if n >= cap:
            self._ring[0, :] = t[-cap:]
            self._ring[1:, :] = y[:, -cap:]
            self._wpos = 0
            self._total += n
        else:
            space = cap - self._wpos
            if n <= space:
                self._ring[0, self._wpos:self._wpos + n] = t
                self._ring[1:, self._wpos:self._wpos + n] = y
            else:
                self._ring[0, self._wpos:] = t[:space]
                self._ring[1:, self._wpos:] = y[:, :space]
                rem = n - space
                self._ring[0, :rem] = t[space:]
                self._ring[1:, :rem] = y[:, space:]
            self._wpos = (self._wpos + n) % cap
            self._total += n

    def _ring_read_channel(self, ch_idx=0):
        """Return (t, y) for channel ch_idx from ring buffer."""
        if self._total == 0:
            return np.array([]), np.array([])
        cnt = min(self._total, self._cap)
        if cnt >= self._cap:
            idx = self._wpos
            t = np.empty(cnt, dtype=np.float64)
            y = np.empty(cnt, dtype=np.float64)
            tail = cnt - idx
            t[:tail] = self._ring[0, idx:]
            y[:tail] = self._ring[1 + ch_idx, idx:]
            t[tail:] = self._ring[0, :idx]
            y[tail:] = self._ring[1 + ch_idx, :idx]
            return t, y
        return self._ring[0, :cnt].copy(), self._ring[1 + ch_idx, :cnt].copy()

    def _ring_read_tail(self, n, ch_idx=None):
        """Return last n (t, y) for channel ch_idx (or current channel)."""
        if ch_idx is None:
            ch_idx = self._current_channel_idx
        cnt = min(self._total, self._cap)
        n = min(n, cnt)
        if n == 0:
            return np.array([]), np.array([])
        t, y = self._ring_read_channel(ch_idx)
        return t[-n:], y[-n:]

    def _raw_time_bounds(self):
        if self._total == 0:
            return 0.0, 0.0
        t, _ = self._ring_read_channel(0)
        if t.size == 0:
            return 0.0, 0.0
        earliest = float(t[0])
        latest = float(t[-1])
        if self._total > self._cap:
            latest = float(self._total) / float(self.sampling_rate)
        return earliest, latest

    def _resize(self, num_channels: int, channel_labels: list[str] | None = None):
        """Resize ring buffer and per-channel data when channel count changes."""
        if num_channels == self.num_channels and channel_labels is None:
            return
        self.num_channels = num_channels
        if channel_labels is not None:
            self._channel_labels = channel_labels
        self._ring = _make_display_buffer(self.sampling_rate, self.num_channels)
        self._cap = self._ring.shape[1]
        self._wpos = 0
        self._total = 0
        self._raw_hist_t_low = [[] for _ in range(self.num_channels)]
        self._raw_hist_y_low = [[] for _ in range(self.num_channels)]
        self._raw_hist_t_high = [[] for _ in range(self.num_channels)]
        self._raw_hist_y_high = [[] for _ in range(self.num_channels)]
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        self._channel_combo.addItems(self._channel_labels)
        self._channel_combo.blockSignals(False)
        if self._current_channel_idx >= self.num_channels:
            self._current_channel_idx = 0
        self._channel_results.clear()
        self._spike_times_cache.clear()
        self._last_spike_scan_sample.clear()

    def on_data(self, chunk: np.ndarray):
        ingest_t0 = time.perf_counter()
        if chunk is None:
            return
        arr = np.asarray(chunk)
        if arr.ndim != 2 or arr.size == 0:
            return
        n_channels = arr.shape[0]
        n_samples = arr.shape[1]
        if n_samples < 1:
            return

        # lazy resize ring buffer if incoming channel count doesn't match
        if n_channels != self.num_channels:
            _ports = ['A', 'B', 'C', 'D']
            chan_labels = [f"{_ports[i // 32]}-{i % 32:03d}" for i in range(n_channels) if i // 32 < 4]
            self._resize(n_channels, chan_labels)

        t_chunk = (self.sample_counter + np.arange(n_samples, dtype=np.float64)) / self.sampling_rate
        self._ring_write(t_chunk, arr)
        self._append_raw_history(t_chunk, arr)
        self.sample_counter += n_samples
        self._telemetry_chunks += 1
        self._telemetry_samples += int(n_samples)
        self._telemetry_latest_chunk_received_t = time.perf_counter()

        self.psd_update_counter += 1
        self.spike_plot_frame_counter += 1
        do_psd = (self.psd_update_counter % max(1, PSD_PLOT_UPDATE_EVERY_N) == 0)
        do_spike = (self.spike_plot_frame_counter % max(1, SPIKE_PLOT_UPDATE_EVERY_N) == 0)

        if self._channel_results.get(self._current_channel_idx) is None:
            do_psd = do_spike = True

        ch = self._current_channel_idx
        if ch >= self.num_channels:
            ch = 0

        last_scan = self._last_spike_scan_sample.get(ch, 0)
        gap = self._total - last_scan
        should_run_spike = do_spike and (gap >= SPIKE_INCREMENTAL_MIN_SAMPLES)
        should_run_psd = do_psd

        if should_run_psd or should_run_spike:
            if should_run_spike:
                stored = min(self._total, self._cap)
                psd_n = max(8, int(round(self.sampling_rate * self._psd_buffer_sec))) if should_run_psd else 0
                wf_n = max(8, int(round(self.sampling_rate * max(1, self._waveform_buffer_sec + 1))))
                spike_n = int(max(SPIKE_INCREMENTAL_MIN_SAMPLES, gap) + SPIKE_OVERLAP_SAMPLES)
                tail_n = min(stored, max(psd_n, wf_n, spike_n))

                t_tail, sig_tail = self._ring_read_tail(tail_n, ch_idx=ch)
                abs_start = self._total - t_tail.size
                rel_last = max(0, int(last_scan - abs_start))
                rel_last = min(rel_last, t_tail.size)
                self._last_proc_abs_start = abs_start
                self._pending_channel = ch

                self._proc_worker.schedule(
                    sig_tail.copy(), t_tail.copy(), self.sampling_rate,
                    list(self._spike_times_cache.get(ch, [])),
                    rel_last, t_tail.size,
                    do_psd=should_run_psd, do_spike=True,
                )
            else:
                psd_n = max(8, int(round(self.sampling_rate * self._psd_buffer_sec)))
                t_tail, sig_tail = self._ring_read_tail(psd_n, ch_idx=ch)
                self._last_proc_abs_start = self._total - t_tail.size
                self._pending_channel = ch
                self._proc_worker.schedule(
                    sig_tail.copy(), t_tail.copy(), self.sampling_rate,
                    [], 0, sig_tail.size,
                    do_psd=True, do_spike=False,
                )
        self._telemetry_ingest_ms_total += (time.perf_counter() - ingest_t0) * 1000.0
        self._emit_telemetry_if_due()

    def _append_decimated_history(self, t_lists, y_lists, t_chunk, y_chunk, stride, mod_counter):
        """Append decimated data to per-channel history lists.
        t_lists, y_lists: list of lists (one per channel) modified in-place.
        y_chunk: (num_channels, n) array.
        """
        if y_chunk.shape[1] == 0:
            return mod_counter
        buf_stride = stride
        i_start = buf_stride - 1 - mod_counter
        if i_start < 0:
            i_start = buf_stride - 1
        t_dec = t_chunk[i_start::buf_stride]
        if t_dec.size > 0:
            for ch in range(len(t_lists)):
                t_lists[ch].extend(t_dec.tolist())
                y_lists[ch].extend(y_chunk[ch, i_start::buf_stride].tolist())
        next_mod = (mod_counter + y_chunk.shape[1]) % buf_stride
        for ch in range(len(t_lists)):
            total_points = len(t_lists[ch])
            if total_points > MAX_RAW_HISTORY_PLOT_POINTS:
                excess = total_points - MAX_RAW_HISTORY_PLOT_POINTS
                del t_lists[ch][:excess]
                del y_lists[ch][:excess]
        return next_mod

    def _append_raw_history(self, t_chunk, y_chunk):
        """y_chunk: (num_channels, n) or (n,)."""
        if y_chunk.ndim == 1:
            y_chunk = y_chunk.reshape(1, -1)
        self._raw_hist_sample_mod_low = self._append_decimated_history(
            self._raw_hist_t_low, self._raw_hist_y_low,
            t_chunk, y_chunk,
            self._raw_hist_stride_low, self._raw_hist_sample_mod_low,
        )
        self._raw_hist_sample_mod_high = self._append_decimated_history(
            self._raw_hist_t_high, self._raw_hist_y_high,
            t_chunk, y_chunk,
            self._raw_hist_stride_high, self._raw_hist_sample_mod_high,
        )

    def _trim_history_store(self, t_hist, y_hist, max_points):
        if len(t_hist) <= max_points:
            return
        excess = len(t_hist) - max_points
        del t_hist[:excess]
        del y_hist[:excess]

    def _on_processing_result(self, result):
        ch = self._pending_channel
        self._channel_results[ch] = result
        if result.spike_times_cache is not None:
            self._spike_times_cache[ch] = result.spike_times_cache
        if result.last_scan_sample is not None:
            self._last_spike_scan_sample[ch] = self._last_proc_abs_start + int(result.last_scan_sample)
        if ch == self._current_channel_idx:
            if getattr(result, 'has_psd_update', False) and result.psd_f is not None:
                self._psd_pending = True
            if getattr(result, 'has_spike_update', False) and result.spike_minute_idx is not None:
                self._spike_pending = True

    def _on_expensive_task_result(self, result):
        task_type = str(result.get("task_type", ""))
        task_id = int(result.get("task_id", 0))
        session_id = int(result.get("session_id", -1))
        if session_id != int(self._session_id):
            return
        active_task_id = int(self._active_expensive_task_id.get(task_type, 0))
        if task_id != active_task_id:
            return
        if str(result.get("status", "")) != "ok":
            return
        if task_type == "spike_rebin":
            data = result.get("data") or {}
            minute_idx = np.asarray(data.get("minute_idx", []), dtype=np.float64)
            counts = np.asarray(data.get("counts", []), dtype=np.int64)
            if minute_idx.size and counts.size and minute_idx.size == counts.size:
                ch = self._current_channel_idx
                if ch not in self._channel_results:
                    self._channel_results[ch] = _Result()
                self._channel_results[ch].spike_minute_idx = minute_idx
                self._channel_results[ch].spike_counts = counts
                self._spike_pending = True

    def _schedule_spike_rebin_task(self):
        if not hasattr(self, '_exp_worker') or self._exp_worker is None:
            return
        task_id = self._next_task_id()
        self._active_expensive_task_id['spike_rebin'] = task_id
        job = {
            "task_type": "spike_rebin",
            "task_id": task_id,
            "session_id": int(self._session_id),
            "spike_times": list(self._spike_times_cache.get(self._current_channel_idx, [])),
            "bin_sec": float(self._spike_bin_sec),
            "last_time_s": float(self.sample_counter) / max(float(self.sampling_rate), 1e-9),
        }
        self._exp_worker.schedule(job)

    def _sync_marker_lines(self, x_min, x_max):
        lines_to_remove = []
        for line, label in zip(self.canvas._marker_lines, self.canvas._marker_labels):
            pos = line.getPos()
            if pos is not None:
                pos_x = float(pos[0]) if hasattr(pos, '__getitem__') else float(pos)
                if pos_x < x_min or pos_x > x_max:
                    lines_to_remove.append(line)
                    if label in self.canvas._marker_labels:
                        self.canvas._marker_labels.remove(label)
                        self.canvas._marker_lines.remove(line)
                        self.canvas.raw_plot.removeItem(line)
                        label.close()
        for line in lines_to_remove:
            if line in self.canvas._marker_lines:
                self.canvas._marker_lines.remove(line)
            self.canvas.raw_plot.removeItem(line)

        visible_ts = sorted(
            float(m["timestamp_s"]) for m in self.marker_records
            if x_min <= float(m["timestamp_s"]) <= x_max
        )
        for ts in visible_ts:
            if ts in self.canvas._last_marker_set:
                continue
            line = pg.InfiniteLine(pos=ts, angle=90, pen=self.canvas._marker_pen)
            self.canvas.raw_plot.addItem(line)
            self.canvas._marker_lines.append(line)
            label = pg.TextItem(text=f"M@{ts:.1f}s", anchor=(0, 1), color='crimson')
            self.canvas.raw_plot.addItem(label)
            label.setPos(ts, 0)
            self.canvas._marker_labels.append(label)
        self.canvas._last_marker_set = list(visible_ts)

    def _sync_spike_marker_lines(self, x_min, x_max):
        lines_to_remove = []
        for line, label in zip(self.canvas._spike_marker_lines, self.canvas._spike_marker_labels):
            pos = line.getPos()
            if pos is not None:
                pos_x = float(pos[0]) if hasattr(pos, '__getitem__') else float(pos)
                if pos_x < x_min or pos_x > x_max:
                    lines_to_remove.append(line)
                    if label in self.canvas._spike_marker_labels:
                        self.canvas._spike_marker_labels.remove(label)
                        self.canvas._spike_marker_lines.remove(line)
                        self.canvas.spike_plot.removeItem(line)
                        label.close()
        for line in lines_to_remove:
            if line in self.canvas._spike_marker_lines:
                self.canvas._spike_marker_lines.remove(line)
            self.canvas.spike_plot.removeItem(line)

        visible_ts = sorted(
            float(m["timestamp_s"]) for m in self.marker_records
            if x_min <= float(m["timestamp_s"]) <= x_max
        )
        for ts in visible_ts:
            if ts in self.canvas._last_spike_marker_set:
                continue
            pen = pg.mkPen(color='orange', style=QtCore.Qt.DashLine, width=1)
            line = pg.InfiniteLine(pos=ts, angle=90, pen=pen)
            self.canvas.spike_plot.addItem(line)
            self.canvas._spike_marker_lines.append(line)
            label = pg.TextItem(text=f"M@{ts:.1f}s", anchor=(0, 1), color='orange')
            self.canvas.spike_plot.addItem(label)
            label.setPos(ts, 0)
            self.canvas._spike_marker_labels.append(label)
        self.canvas._last_spike_marker_set = list(visible_ts)

    def _clear_marker_lines(self):
        for line in self.canvas._marker_lines:
            self.canvas.raw_plot.removeItem(line)
        self.canvas._marker_lines.clear()
        for label in self.canvas._marker_labels:
            label.close()
        self.canvas._marker_labels.clear()
        self.canvas._last_marker_set = []

    def _clear_spike_marker_lines(self):
        for line in self.canvas._spike_marker_lines:
            self.canvas.spike_plot.removeItem(line)
        self.canvas._spike_marker_lines.clear()
        for label in self.canvas._spike_marker_labels:
            label.close()
        self.canvas._spike_marker_labels.clear()
        self.canvas._last_spike_marker_set = []

    def render(self):
        if self._total == 0:
            return

        t0 = time.perf_counter()
        now = t0
        did_work = False

        if (now - self._last_raw_render_t) >= 1.0 / RAW_RENDER_HZ:
            did_work = True
            self._telemetry_raw_renders += 1
            if self._follow_axes['raw']:
                n_vis = int(self.sampling_rate * DISPLAY_WINDOW_SEC)
                x_vis, y_vis = self._ring_read_tail(n_vis)
                xd, yd = _minmax_downsample(x_vis, y_vis, MAX_DISPLAY_POINTS)
                self.canvas.raw_curve.setData(xd, yd)
                x_end = float(x_vis[-1]) if x_vis.size else 0.0
                x_start = max(0.0, x_end - DISPLAY_WINDOW_SEC)
                x_min_lim, _ = self._raw_time_bounds()
                self.canvas.raw_plot.setLimits(xMin=max(0.0, x_min_lim), xMax=max(0.0, x_end + 0.1))
                self._suspend_follow_detection = True
                try:
                    self.canvas.raw_plot.setXRange(x_start, x_end, padding=0)
                finally:
                    self._suspend_follow_detection = False
                peak = float(np.max(np.abs(y_vis))) if y_vis.size else 0.0
                y_lim = max(0.2, peak * 1.2)
                self._suspend_follow_detection = True
                try:
                    self.canvas.raw_plot.setYRange(-y_lim, y_lim, padding=0)
                finally:
                    self._suspend_follow_detection = False
            else:
                history_end = 0.0
                history_start = 0.0
                ch = self._current_channel_idx
                if ch < len(self._raw_hist_t_low) and self._raw_hist_t_low[ch]:
                    history_start = float(self._raw_hist_t_low[ch][0])
                    history_end = float(self._raw_hist_t_low[ch][-1])
                    self.canvas.raw_plot.setLimits(xMin=max(0.0, history_start), xMax=max(0.0, history_end + 0.1))
                vr = self.canvas.raw_plot.vb.viewRange()[0]
                x_start = float(vr[0]) if len(vr) >= 2 else 0.0
                x_end = float(vr[1]) if len(vr) >= 2 else (history_end if history_end > 0.0 else DISPLAY_WINDOW_SEC)
                if history_end > 0.0 and x_end <= 0:
                    x_end = history_end
                    x_start = max(0.0, x_end - DISPLAY_WINDOW_SEC)
                span = max(0.0, x_end - x_start)
                t_src = None
                y_src = None
                stored = min(self._total, self._cap)
                if stored > 1 and span <= RAW_FULL_RES_MAX_SPAN_SEC:
                    earliest_t, latest_t = self._raw_time_bounds()
                    left_q = max(earliest_t, x_start - RAW_MANUAL_VIEW_MARGIN_SEC)
                    right_q = min(latest_t, x_end + RAW_MANUAL_VIEW_MARGIN_SEC)
                    if right_q > left_q:
                        t_lin, y_lin = self._ring_read_channel(self._current_channel_idx)
                        if t_lin.size:
                            i0 = int(np.searchsorted(t_lin, left_q, side='left'))
                            i1 = int(np.searchsorted(t_lin, right_q, side='right'))
                            if i1 <= i0:
                                i0 = max(0, min(i0, t_lin.size - 1))
                                i1 = min(t_lin.size, i0 + 1)
                            t_src = t_lin[i0:i1]
                            y_src = y_lin[i0:i1]
                if t_src is None or y_src is None:
                    ch = self._current_channel_idx
                    use_high = span <= RAW_ADAPTIVE_HIGH_RES_MAX_SPAN_SEC and ch < len(self._raw_hist_t_high) and bool(self._raw_hist_t_high[ch])
                    if use_high:
                        t_store = self._raw_hist_t_high[ch]
                        y_store = self._raw_hist_y_high[ch]
                    else:
                        t_store = self._raw_hist_t_low[ch] if ch < len(self._raw_hist_t_low) else []
                        y_store = self._raw_hist_y_low[ch] if ch < len(self._raw_hist_y_low) else []
                    if t_store:
                        left_q = max(0.0, x_start - RAW_MANUAL_VIEW_MARGIN_SEC)
                        right_q = max(left_q, x_end + RAW_MANUAL_VIEW_MARGIN_SEC)
                        i0 = bisect.bisect_left(t_store, left_q)
                        i1 = bisect.bisect_right(t_store, right_q)
                        if i1 <= i0:
                            i0 = max(0, min(i0, len(t_store) - 1))
                            i1 = min(len(t_store), i0 + 1)
                        t_src = np.asarray(t_store[i0:i1], dtype=np.float64)
                        y_src = np.asarray(y_store[i0:i1], dtype=np.float64)
                if t_src is not None and y_src is not None and np.asarray(t_src).size:
                    xd, yd = _minmax_downsample(np.asarray(t_src), np.asarray(y_src), MAX_RAW_HISTORY_PLOT_POINTS)
                    self.canvas.raw_curve.setData(xd, yd)
            self._sync_marker_lines(max(0.0, min(x_start, x_end)), max(0.0, max(x_start, x_end)))
            self._last_raw_render_t = now

        r = self._channel_results.get(self._current_channel_idx)
        if r is not None and self._psd_pending and (now - self._last_psd_render_t) >= 1.0 / PSD_RENDER_HZ:
            did_work = True
            if r.psd_f is not None and r.psd_db is not None:
                self.canvas.psd_curve.setData(r.psd_f, r.psd_db)
                self._latest_psd_f = np.asarray(r.psd_f).copy()
                self._latest_psd_db = np.asarray(r.psd_db).copy()
                if self._follow_axes['psd']:
                    self._suspend_follow_detection = True
                    try:
                        if r.psd_f.size:
                            self.canvas.psd_plot.setXRange(float(r.psd_f[0]), float(r.psd_f[-1]), padding=0)
                        self.canvas.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)
                    finally:
                        self._suspend_follow_detection = False
            self._psd_pending = False
            self._last_psd_render_t = now

        if r is not None and self._spike_pending and (now - self._last_spike_render_t) >= 1.0 / SPIKE_RENDER_HZ:
            did_work = True
            if r.spike_minute_idx is not None and r.spike_counts is not None:
                self.canvas.spike_curve.setData(r.spike_minute_idx, r.spike_counts)
                max_count = max(1, int(np.max(r.spike_counts)) if r.spike_counts.size else 1)
                right_min = 0.0
                if r.spike_minute_idx.size:
                    right_min = float(r.spike_minute_idx[-1]) + float(self._spike_bin_sec) / 60.0
                self.canvas.spike_plot.setLimits(xMin=0.0, xMax=max(0.0, right_min + 0.1))
                if self._follow_axes['spike']:
                    self._suspend_follow_detection = True
                    try:
                        self.canvas.spike_plot.setYRange(0, max_count * 1.2, padding=0)
                        if right_min > 0.0:
                            left_min = max(0.0, right_min - SPIKE_SCROLL_WINDOW_MIN)
                            self.canvas.spike_plot.setXRange(left_min, right_min, padding=0)
                    finally:
                        self._suspend_follow_detection = False
            if r.wf_t_ms is not None and r.wf_mu is not None and r.wf_sem is not None:
                self.canvas.wf_curve.setData(r.wf_t_ms, r.wf_mu)
                self.canvas.wf_upper.setData(r.wf_t_ms, r.wf_mu + r.wf_sem)
                self.canvas.wf_lower.setData(r.wf_t_ms, r.wf_mu - r.wf_sem)
                self._latest_wf_t_ms = np.asarray(r.wf_t_ms).copy()
                self._latest_wf_mu = np.asarray(r.wf_mu).copy()
                if self._follow_axes['wf']:
                    self._suspend_follow_detection = True
                    try:
                        self.canvas.wf_plot.setXRange(float(r.wf_t_ms[0]), float(r.wf_t_ms[-1]), padding=0)
                        self.canvas.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)
                    finally:
                        self._suspend_follow_detection = False
            else:
                self.canvas.wf_curve.setData([], [])
                self.canvas.wf_upper.setData([], [])
                self.canvas.wf_lower.setData([], [])
                self._latest_wf_t_ms = None
                self._latest_wf_mu = None
            self._spike_pending = False
            self._last_spike_render_t = now

        spike_vr = self.canvas.spike_plot.vb.viewRange()[0]
        if len(spike_vr) >= 2:
            self._sync_spike_marker_lines(float(spike_vr[0]), float(spike_vr[1]))

        self._render_dur_ms = (time.perf_counter() - t0) * 1000.0
        if did_work:
            self._telemetry_render_calls += 1
            self._telemetry_render_ms_total += self._render_dur_ms
        self._emit_telemetry_if_due()

    def clear(self):
        self.canvas.raw_curve.setData([], [])
        self.canvas.psd_curve.setData([], [])
        self.canvas.spike_curve.setData([], [])
        self.canvas.wf_curve.setData([], [])
        self.canvas.wf_upper.setData([], [])
        self.canvas.wf_lower.setData([], [])
        self._clear_marker_lines()
        self._clear_spike_marker_lines()
        self.clear_snapshots()
        self._ring[:] = 0
        self._wpos = 0
        self._total = 0
        self.sample_counter = 0
        self.marker_records = []
        self._marker_times_sorted = []
        self._channel_results.clear()
        self._spike_times_cache.clear()
        self._last_spike_scan_sample.clear()
        self._last_proc_abs_start = 0
        self._latest_psd_f = None
        self._latest_psd_db = None
        self._latest_wf_t_ms = None
        self._latest_wf_mu = None
        self._raw_hist_t_low = [[] for _ in range(self.num_channels)]
        self._raw_hist_y_low = [[] for _ in range(self.num_channels)]
        self._raw_hist_t_high = [[] for _ in range(self.num_channels)]
        self._raw_hist_y_high = [[] for _ in range(self.num_channels)]
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0
        self._psd_pending = False
        self._spike_pending = False
        self._active_expensive_task_id = {}
        self._last_raw_render_t = 0.0
        self._last_psd_render_t = 0.0
        self._last_spike_render_t = 0.0
        self.psd_update_counter = 0
        self.spike_plot_frame_counter = 0
        self.set_auto_follow(True)
        self._suspend_follow_detection = True
        try:
            self.canvas.raw_plot.setXRange(0, DISPLAY_WINDOW_SEC, padding=0)
            self.canvas.raw_plot.setYRange(-1, 1, padding=0)
            self.canvas.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)
            self.canvas.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)
        finally:
            self._suspend_follow_detection = False
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

    def shutdown(self) -> bool:
        ok_proc = True
        ok_exp = True
        if hasattr(self, '_proc_worker') and self._proc_worker is not None:
            ok_proc = self._proc_worker.stop(timeout_ms=4000)
        if hasattr(self, '_exp_worker') and self._exp_worker is not None:
            ok_exp = self._exp_worker.stop(timeout_ms=4000)
        return bool(ok_proc and ok_exp)

    def set_auto_follow(self, enabled: bool):
        for key in self._follow_axes:
            self._follow_axes[key] = enabled
            action = self._follow_menu_actions.get(key)
            if action:
                action.setChecked(enabled)

    def is_auto_follow_enabled(self) -> bool:
        return any(self._follow_axes.values())

    def add_marker(self, marker):
        if isinstance(marker, dict):
            ts = float(marker.get("timestamp_s", marker.get("timestamp", 0.0)))
            marker_id = int(marker.get("id", len(self.marker_records) + 1))
            name = str(marker.get("name", f"Marker {marker_id}"))
            self.marker_records.append({"id": marker_id, "timestamp_s": ts, "name": name})
        else:
            ts = float(marker)
            marker_id = len(self.marker_records) + 1
            self.marker_records.append({"id": marker_id, "timestamp_s": ts, "name": f"Marker {marker_id}"})
        self._marker_times_sorted = sorted(float(m.get("timestamp_s", 0.0)) for m in self.marker_records)

    def get_markers(self):
        return list(self.marker_records)

    def take_psd_snapshot(self) -> bool:
        if self._latest_psd_f is None or self._latest_psd_db is None:
            return False
        curve = self.canvas.psd_plot.plot(
            self._latest_psd_f, self._latest_psd_db,
            pen=pg.mkPen('#888888', width=1, style=QtCore.Qt.DashLine),
        )
        self._psd_snapshot_curves.append(curve)
        return True

    def take_waveform_snapshot(self) -> bool:
        if self._latest_wf_t_ms is None or self._latest_wf_mu is None:
            return False
        curve = self.canvas.wf_plot.plot(
            self._latest_wf_t_ms, self._latest_wf_mu,
            pen=pg.mkPen('#888888', width=1, style=QtCore.Qt.DashLine),
        )
        self._wf_snapshot_curves.append(curve)
        return True

    def clear_snapshots(self):
        for c in self._psd_snapshot_curves:
            self.canvas.psd_plot.removeItem(c)
        self._psd_snapshot_curves.clear()
        for c in self._wf_snapshot_curves:
            self.canvas.wf_plot.removeItem(c)
        self._wf_snapshot_curves.clear()

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        self.clear()
        self._session_id += 1
        if sample_rate > 0:
            self.sampling_rate = float(sample_rate)
        self._ring = _make_display_buffer(self.sampling_rate, self.num_channels)
        self._cap = self._ring.shape[1]
        self._wpos = 0
        self._total = 0
        self._update_raw_history_stride()
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0
        self.set_receiving_state(True)

    def set_receiving_state(self, receiving: bool):
        receiving = bool(receiving)
        if receiving == self.is_receiving:
            return
        now = time.perf_counter()
        if receiving:
            if self._receiving_wall_run_start is None:
                self._receiving_wall_run_start = now
        else:
            if self._receiving_wall_run_start is not None:
                self._receiving_wall_active_sec += max(0.0, now - float(self._receiving_wall_run_start))
                self._receiving_wall_run_start = None
        self.is_receiving = receiving

    def configure_processing_settings(self, psd_buffer_sec, waveform_buffer_sec, spike_bin_sec):
        prev_spike_bin_sec = int(self._spike_bin_sec)
        self._psd_buffer_sec = int(psd_buffer_sec)
        self._waveform_buffer_sec = int(waveform_buffer_sec)
        self._spike_bin_sec = int(spike_bin_sec)
        save_plot_setting("psd_buffer_sec", self._psd_buffer_sec)
        save_plot_setting("waveform_buffer_sec", self._waveform_buffer_sec)
        save_plot_setting("spike_bin_sec", self._spike_bin_sec)
        configure_processing_windows(
            psd_buffer_sec=self._psd_buffer_sec,
            waveform_buffer_sec=self._waveform_buffer_sec,
            spike_bin_sec=self._spike_bin_sec,
        )
        self.canvas.psd_plot.setTitle(f"Power spectrum (last {self._psd_buffer_sec}s)")
        self.canvas.spike_plot.setTitle(f"Spike counts ({self._spike_bin_sec}s bins)")
        self.canvas.wf_plot.setTitle(f"Averaged spike waveform (last {self._waveform_buffer_sec}s)")
        if self._spike_bin_sec != prev_spike_bin_sec:
            self._schedule_spike_rebin_task()


class _Result:
    pass


# ── SmuDeviceTab ────────────────────────────────────────────────────────────

class SmuDeviceTab(DeviceTab):
    def __init__(self, sample_rate=1000.0, parent=None):
        super().__init__(parent)
        self.sampling_rate = sample_rate
        self.sample_counter = 0
        self.is_receiving = False
        self._render_dur_ms = 0.0
        self._last_render_t = 0.0
        self._ring_cap = max(1000, int(round(sample_rate * 300)))
        self._v_ring = np.zeros(self._ring_cap, dtype=np.float64)
        self._i_ring = np.zeros(self._ring_cap, dtype=np.float64)
        self._t_ring = np.zeros(self._ring_cap, dtype=np.float64)
        self._wpos = 0
        self._total = 0
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

        self._suspend_follow_detection = False
        self.canvas = SmuCanvas(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)

    def on_data(self, chunk: np.ndarray):
        arr = np.asarray(chunk)
        if arr.ndim == 1:
            n = 1
            voltage = float(arr[0]) if arr.size > 0 else 0.0
            current = float(arr[1]) if arr.size > 1 else 0.0
            t = self.sample_counter / self.sampling_rate
            self._ring_write(t, voltage, current)
            self.sample_counter += 1
        elif arr.ndim == 2:
            n = arr.shape[1]
            if arr.shape[0] >= 2:
                voltage = arr[0, :]
                current = arr[1, :]
            elif arr.shape[0] == 1:
                voltage = arr[0, :]
                current = np.zeros(n)
            else:
                return
            t = (self.sample_counter + np.arange(n, dtype=np.float64)) / self.sampling_rate
            for i in range(n):
                self._ring_write(t[i], voltage[i], current[i])
            self.sample_counter += n

    def _ring_write(self, t, v, i):
        pos = self._wpos
        self._t_ring[pos] = t
        self._v_ring[pos] = v
        self._i_ring[pos] = i
        self._wpos = (pos + 1) % self._ring_cap
        self._total += 1

    def _ring_read(self, n):
        n = min(n, min(self._total, self._ring_cap))
        if n == 0:
            return np.array([]), np.array([]), np.array([])
        cnt = min(self._total, self._ring_cap)
        if cnt >= self._ring_cap:
            idx = self._wpos
            t = np.empty(cnt, dtype=np.float64)
            v = np.empty(cnt, dtype=np.float64)
            i = np.empty(cnt, dtype=np.float64)
            tail = cnt - idx
            t[:tail] = self._t_ring[idx:]
            v[:tail] = self._v_ring[idx:]
            i[:tail] = self._i_ring[idx:]
            t[tail:] = self._t_ring[:idx]
            v[tail:] = self._v_ring[:idx]
            i[tail:] = self._i_ring[:idx]
        else:
            t = self._t_ring[:cnt].copy()
            v = self._v_ring[:cnt].copy()
            i = self._i_ring[:cnt].copy()
        return t[-n:], v[-n:], i[-n:]

    def render(self):
        now = time.perf_counter()
        if (now - self._last_render_t) < 1.0 / 30.0:
            return
        t0 = now
        n_vis = min(10000, min(self._total, self._ring_cap))
        if n_vis < 2:
            return
        t, v, i = self._ring_read(n_vis)
        if t.size < 2:
            return
        self.canvas.voltage_curve.setData(t, v)
        self.canvas.current_curve.setData(t, i)
        self.canvas.voltage_plot.autoRange()

        v_range = float(np.max(v)) - float(np.min(v)) if v.size else 1.0
        i_range = float(np.max(i)) - float(np.min(i)) if i.size else 1.0
        v_pad = max(0.1, v_range * 0.1)
        i_pad = max(0.001, i_range * 0.1)
        self._suspend_follow_detection = True
        try:
            self.canvas.voltage_plot.setYRange(float(np.min(v)) - v_pad, float(np.max(v)) + v_pad, padding=0)
            self.canvas.current_plot.setYRange(float(np.min(i)) - i_pad, float(np.max(i)) + i_pad, padding=0)
        finally:
            self._suspend_follow_detection = False

        self._render_dur_ms = (time.perf_counter() - t0) * 1000.0
        self._last_render_t = now
        self._update_fps_label()

    def _update_fps_label(self):
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_t
        if elapsed >= 1.0:
            fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_last_t = now

    def clear(self):
        self._v_ring[:] = 0
        self._i_ring[:] = 0
        self._t_ring[:] = 0
        self._wpos = 0
        self._total = 0
        self.sample_counter = 0
        self.canvas.voltage_curve.setData([], [])
        self.canvas.current_curve.setData([], [])
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

    def shutdown(self) -> bool:
        return True

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        self.clear()
        if sample_rate > 0:
            self.sampling_rate = float(sample_rate)
            self._ring_cap = max(1000, int(round(self.sampling_rate * 300)))
            self._v_ring = np.zeros(self._ring_cap, dtype=np.float64)
            self._i_ring = np.zeros(self._ring_cap, dtype=np.float64)
            self._t_ring = np.zeros(self._ring_cap, dtype=np.float64)
        self.set_receiving_state(True)

    def set_receiving_state(self, receiving: bool):
        self.is_receiving = bool(receiving)
