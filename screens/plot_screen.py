import time
import bisect

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np

from workers.processing_worker import (
    ProcessingWorker,
    configure_processing_windows,
    SPIKE_INCREMENTAL_MIN_SAMPLES,
    PSD_BUFFER_SEC,
    WAVEFORM_BUFFER_SEC,
    SPIKE_BIN_SEC,
    PSD_YLIM_MIN,
    PSD_YLIM_MAX,
)

# ── Display constants ──────────────────────────────────────────────────────────
DISPLAY_WINDOW_SEC    = 10          # raw signal x-axis window
DISPLAY_BUFFER_SEC    = 300         # ring buffer size (5 min)
DEFAULT_SAMPLING_RATE = 20000
MAX_DISPLAY_POINTS    = 15000

# ── Per-subplot render rate limits ────────────────────────────────────────────
PLOT_UPDATE_FREQ_HZ = 120
RAW_RENDER_HZ       = 30
PSD_RENDER_HZ       = 30
SPIKE_RENDER_HZ     = 30
WAVEFORM_YLIM_ABS_UV = 100

# ── Background compute throttle (every N data chunks) ─────────────────────────
PSD_PLOT_UPDATE_EVERY_N   = 15
SPIKE_PLOT_UPDATE_EVERY_N = 15  

# ── Pyqtgraph global style ────────────────────────────────────────────────────
pg.setConfigOption('background', 'transparent')
pg.setConfigOption('foreground', 'black')
pg.setConfigOption('antialias', True)

def _make_display_buffer(fs: float) -> np.ndarray:
    """Pre-allocate fixed-size ring buffer for DISPLAY_BUFFER_SEC of data."""
    n = max(1, int(round(fs * DISPLAY_BUFFER_SEC)))
    return np.zeros((2, n), dtype=np.float64)


class PgCanvas(QtWidgets.QWidget):
    """Four pyqtgraph plots: Raw (top, full width) + PSD / Spike counts / Waveform (bottom row)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        self.glw.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

        # Row 0: Raw signal (spans 3 columns)
        self.raw_plot = self.glw.addPlot(row=0, col=0, colspan=3)
        self.raw_plot.setTitle("Raw signal")
        self.raw_plot.setLabel('left',   'U [uV]')
        self.raw_plot.setLabel('bottom', 't [s]')
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        self.raw_plot.setDownsampling(auto=True, mode='peak')
        self.raw_plot.setClipToView(True)
        self.raw_plot.setMouseEnabled(x=False, y=False)

        # Row 1: PSD
        self.psd_plot = self.glw.addPlot(row=1, col=0)
        self.psd_plot.setTitle(f"Power spectrum (last {PSD_BUFFER_SEC}s)")
        self.psd_plot.setLabel('left',   'P [dB]')
        self.psd_plot.setLabel('bottom', 'f [Hz]')
        self.psd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.psd_plot.setMouseEnabled(x=False, y=False)
        self.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)

        # Row 1: Spike counts
        self.spike_plot = self.glw.addPlot(row=1, col=1)
        self.spike_plot.setTitle(f"Spike counts ({SPIKE_BIN_SEC}s bins)")
        self.spike_plot.setLabel('left',   'Count')
        self.spike_plot.setLabel('bottom', 't [min]')
        self.spike_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spike_plot.setMouseEnabled(x=False, y=False)

        # Row 1: Waveform
        self.wf_plot = self.glw.addPlot(row=1, col=2)
        self.wf_plot.setTitle(f"Averaged spike waveform (last {WAVEFORM_BUFFER_SEC}s)")
        self.wf_plot.setLabel('left',   'U [uV]')
        self.wf_plot.setLabel('bottom', 't [ms]')
        self.wf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wf_plot.setMouseEnabled(x=False, y=False)
        self.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)

        # ── Curves ────────────────────────────────────────────────────────────
        self.raw_curve   = self.raw_plot.plot(  pen=pg.mkPen("#1D20ED", width=1))
        self.psd_curve   = self.psd_plot.plot(  pen=pg.mkPen("#f62d2d", width=2))
        self.spike_curve = self.spike_plot.plot(pen=pg.mkPen("#20C814", width=2))
        self.wf_curve    = self.wf_plot.plot(   pen=pg.mkPen("#e840b3", width=2))

        # SEM fill (FillBetweenItem requires two PlotDataItems)
        self.wf_upper = self.wf_plot.plot(pen=None)
        self.wf_lower = self.wf_plot.plot(pen=None)
        self.wf_fill  = pg.FillBetweenItem(
            self.wf_upper, self.wf_lower,
            brush=pg.mkBrush(80, 200, 120, 50),
        )
        self.wf_plot.addItem(self.wf_fill)

        # Marker lines (InfiniteLine pool)
        self._marker_lines    = []
        self._marker_pen      = pg.mkPen(color='crimson', style=QtCore.Qt.DashLine, width=1)
        self._last_marker_set = []   # timestamps of currently displayed lines


class PlotScreen(QtWidgets.QWidget):

    toggle_receiving_request_signal = QtCore.pyqtSignal(bool)
    save_disconnect_request_signal  = QtCore.pyqtSignal()
    marker_request_signal           = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # ── Session state ──────────────────────────────────────────────────────
        self.sampling_rate    = float(DEFAULT_SAMPLING_RATE)
        self.sample_counter   = 0       # total samples ever received (absolute)
        self.marker_times     = []      # full-session marker timestamps
        self.is_receiving     = False
        self.base_connection_details = ""
        self.base_project_line = ""

        self._spike_times_cache      = []
        self._last_spike_scan_sample = 0  # absolute sample count (never resets)
        self._proc_result            = None
        self._psd_buffer_sec         = int(PSD_BUFFER_SEC)
        self._waveform_buffer_sec    = int(WAVEFORM_BUFFER_SEC)
        self._spike_bin_sec          = int(SPIKE_BIN_SEC)
        self._latest_psd_f           = None
        self._latest_psd_db          = None
        self._latest_wf_t_ms         = None
        self._latest_wf_mu           = None

        # Snapshot overlays (underlaid on top of corresponding plots)
        self._psd_snapshot_curves = []
        self._wf_snapshot_curves  = []

        # Ring buffer (fixed-size, never grows)
        self._ring  = _make_display_buffer(DEFAULT_SAMPLING_RATE)
        self._cap   = self._ring.shape[1]
        self._wpos  = 0   # next write position
        self._total = 0   # total samples written

        # Lazy render state
        self._last_raw_render_t   = 0.0
        self._last_psd_render_t   = 0.0
        self._last_spike_render_t = 0.0
        self._psd_pending         = False
        self._spike_pending       = False

        # Compute throttle counters
        self.psd_update_counter       = 0
        self.spike_plot_frame_counter = 0

        # ── Background worker ──────────────────────────────────────────────────
        self._proc_worker = ProcessingWorker(self)
        self._proc_worker.result_ready.connect(self._on_processing_result)
        self._proc_worker.start()

        # ── Layout ─────────────────────────────────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        self.connection_details_label = QtWidgets.QLabel()
        self.connection_details_label.setContentsMargins(0, 0, 12, 0)
        self.fps_label = QtWidgets.QLabel()
        self.fps_label.setContentsMargins(12, 0, 12, 0)

        self.canvas = PgCanvas(self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(1000 // PLOT_UPDATE_FREQ_HZ)
        self.render_timer.timeout.connect(self._render_plot)
        self.render_timer.start()

        self._fps_frame_count = 0
        self._fps_last_t      = time.perf_counter()
        self._render_dur_ms   = 0.0

    def configure_processing_settings(self, psd_buffer_sec: int, waveform_buffer_sec: int, spike_bin_sec: int):
        """Apply runtime processing settings and refresh plot labels."""
        self._psd_buffer_sec = int(psd_buffer_sec)
        self._waveform_buffer_sec = int(waveform_buffer_sec)
        self._spike_bin_sec = int(spike_bin_sec)

        configure_processing_windows(
            psd_buffer_sec=self._psd_buffer_sec,
            waveform_buffer_sec=self._waveform_buffer_sec,
            spike_bin_sec=self._spike_bin_sec,
        )

        self.canvas.psd_plot.setTitle(f"Power spectrum (last {self._psd_buffer_sec}s)")
        self.canvas.spike_plot.setTitle(f"Spike counts ({self._spike_bin_sec}s bins)")
        self.canvas.wf_plot.setTitle(f"Averaged spike waveform (last {self._waveform_buffer_sec}s)")

    # ── Connection management ──────────────────────────────────────────────────

    def set_connection_details(self, host, command_port, data_port, sample_rate, project_name):
        self.clear_project_buffers()
        self.base_connection_details = f"Connected to {host}:{command_port}/{data_port} at {sample_rate} Hz"
        self.base_project_line = f"Project: {project_name}"
        self.set_connection_status("Receiving")
        self.sampling_rate = float(sample_rate)
        # Re-allocate ring buffer for actual sample rate
        self._ring  = _make_display_buffer(self.sampling_rate)
        self._cap   = self._ring.shape[1]
        self._wpos  = 0
        self._total = 0
        self.set_receiving_state(True)

    def set_connection_status(self, status_text=""):
        if not self.base_connection_details:
            self.connection_details_label.clear()
            return
        text = self.base_connection_details
        if status_text:
            text += f" \u2014 {status_text}"
        if self.base_project_line:
            text += f"\n{self.base_project_line}"
        self.connection_details_label.setText(text)

    def clear_project_buffers(self):
        # Clear all curves
        self.canvas.raw_curve.setData([], [])
        self.canvas.psd_curve.setData([], [])
        self.canvas.spike_curve.setData([], [])
        self.canvas.wf_curve.setData([], [])
        self.canvas.wf_upper.setData([], [])
        self.canvas.wf_lower.setData([], [])
        self._clear_marker_lines()
        self.clear_snapshots()

        # Reset ring buffer
        self._ring[:] = 0
        self._wpos    = 0
        self._total   = 0
        self.sample_counter = 0

        # Reset all state
        self.marker_times            = []
        self._spike_times_cache      = []
        self._last_spike_scan_sample = 0
        self._proc_result            = None
        self._latest_psd_f           = None
        self._latest_psd_db          = None
        self._latest_wf_t_ms         = None
        self._latest_wf_mu           = None
        self._psd_pending            = False
        self._spike_pending          = False
        self._last_raw_render_t      = 0.0
        self._last_psd_render_t      = 0.0
        self._last_spike_render_t    = 0.0
        self.psd_update_counter      = 0
        self.spike_plot_frame_counter = 0

        # Reset axes
        self.canvas.raw_plot.setXRange(0, DISPLAY_WINDOW_SEC, padding=0)
        self.canvas.raw_plot.setYRange(-1, 1, padding=0)
        self.canvas.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)
        self.canvas.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)

        self.base_connection_details = ""
        self.base_project_line = ""
        self.connection_details_label.clear()
        self.set_receiving_state(False)
        self._fps_frame_count = 0
        self._fps_last_t      = time.perf_counter()
        self._render_dur_ms   = 0.0
        self.fps_label.setText("FPS: 0.0\nFrame time: 0.0 ms")

    def set_receiving_state(self, receiving: bool):
        self.is_receiving = bool(receiving)

    def take_psd_snapshot(self) -> bool:
        """Capture current PSD curve and underlay it on the PSD plot."""
        if self._latest_psd_f is None or self._latest_psd_db is None:
            return False
        if self._latest_psd_f.size < 1 or self._latest_psd_db.size < 1:
            return False

        # Keep only the newest snapshot.
        for old_curve in self._psd_snapshot_curves:
            self.canvas.psd_plot.removeItem(old_curve)
        self._psd_snapshot_curves.clear()

        curve = self.canvas.psd_plot.plot(
            self._latest_psd_f.copy(),
            self._latest_psd_db.copy(),
            pen=pg.mkPen(80, 80, 80, 130, width=1),
        )
        curve.setZValue(-10)
        self._psd_snapshot_curves.append(curve)
        return True

    def take_waveform_snapshot(self) -> bool:
        """Capture current average waveform and underlay it on the waveform plot."""
        if self._latest_wf_t_ms is None or self._latest_wf_mu is None:
            return False
        if self._latest_wf_t_ms.size < 1 or self._latest_wf_mu.size < 1:
            return False

        # Keep only the newest snapshot.
        for old_curve in self._wf_snapshot_curves:
            self.canvas.wf_plot.removeItem(old_curve)
        self._wf_snapshot_curves.clear()

        curve = self.canvas.wf_plot.plot(
            self._latest_wf_t_ms.copy(),
            self._latest_wf_mu.copy(),
            pen=pg.mkPen(80, 80, 80, 130, width=1),
        )
        curve.setZValue(-10)
        self._wf_snapshot_curves.append(curve)
        return True

    def clear_snapshots(self):
        """Remove all snapshot overlays from PSD and waveform plots."""
        for curve in self._psd_snapshot_curves:
            self.canvas.psd_plot.removeItem(curve)
        self._psd_snapshot_curves.clear()

        for curve in self._wf_snapshot_curves:
            self.canvas.wf_plot.removeItem(curve)
        self._wf_snapshot_curves.clear()

    def add_marker(self, timestamp_s: float):
        self.marker_times.append(float(timestamp_s))

    # ── Ring buffer ────────────────────────────────────────────────────────────

    def _ring_write(self, t_chunk: np.ndarray, signal: np.ndarray):
        n   = signal.size
        cap = self._cap
        if n >= cap:
            # Chunk larger than entire buffer: keep the newest cap samples
            self._ring[0, :] = t_chunk[-cap:]
            self._ring[1, :] = signal[-cap:]
            self._wpos   = 0
            self._total += n
            return
        end = self._wpos + n
        if end <= cap:
            self._ring[0, self._wpos:end] = t_chunk
            self._ring[1, self._wpos:end] = signal
        else:
            first = cap - self._wpos
            self._ring[0, self._wpos:] = t_chunk[:first]
            self._ring[1, self._wpos:] = signal[:first]
            self._ring[0, :end - cap]  = t_chunk[first:]
            self._ring[1, :end - cap]  = signal[first:]
        self._wpos   = end % cap
        self._total += n

    def _ring_read(self) -> tuple:
        """Return (t, y) arrays in chronological order (full 5-min window)."""
        n = min(self._total, self._cap)
        if self._total <= self._cap:
            return self._ring[0, :n], self._ring[1, :n]
        s = self._wpos
        t = np.concatenate([self._ring[0, s:], self._ring[0, :s]])
        y = np.concatenate([self._ring[1, s:], self._ring[1, :s]])
        return t, y

    def _ring_read_tail(self, n: int) -> tuple:
        """Return the last n samples in chronological order without a full buffer copy."""
        stored = min(self._total, self._cap)
        n = min(n, stored)
        if n == 0:
            return self._ring[0, :0], self._ring[1, :0]
        end = self._wpos  # one past the newest sample
        start = end - n
        if start >= 0:
            # Contiguous slice
            return self._ring[0, start:end], self._ring[1, start:end]
        # Wraps around: two pieces
        t = np.concatenate([self._ring[0, start:], self._ring[0, :end]])
        y = np.concatenate([self._ring[1, start:], self._ring[1, :end]])
        return t, y

    # ── Data ingestion ─────────────────────────────────────────────────────────

    def _on_data_received(self, chunk: np.ndarray):
        if chunk is None:
            return
        arr = np.asarray(chunk)
        if arr.ndim != 2 or arr.size == 0:
            return
        if arr.shape[0] != 1:
            print(f"[Plot Warning] Expected (1, N) chunk, got {arr.shape}")
            return
        n_samples = arr.shape[1]
        if n_samples < 1:
            return

        signal  = arr[0, :]
        t_chunk = (self.sample_counter + np.arange(n_samples, dtype=np.float64)) / self.sampling_rate
        self._ring_write(t_chunk, signal)
        self.sample_counter += n_samples

        # Throttle compute
        self.psd_update_counter       += 1
        self.spike_plot_frame_counter += 1
        do_psd   = (self.psd_update_counter       % max(1, PSD_PLOT_UPDATE_EVERY_N)   == 0)
        do_spike = (self.spike_plot_frame_counter % max(1, SPIKE_PLOT_UPDATE_EVERY_N) == 0)

        if self._proc_result is None:
            do_psd = do_spike = True

        gap = self._total - self._last_spike_scan_sample
        should_run_spike = do_spike and (gap >= SPIKE_INCREMENTAL_MIN_SAMPLES)
        should_run_psd   = do_psd

        if should_run_psd or should_run_spike:
            t_lin, sig_lin = self._ring_read()
            # absolute sample index of the first element in the linear window
            abs_start = self._total - t_lin.size

            if should_run_spike:
                rel_last = max(0, int(self._last_spike_scan_sample - abs_start))
                self._proc_worker.schedule(
                    sig_lin.copy(), t_lin.copy(), self.sampling_rate,
                    list(self._spike_times_cache),
                    rel_last, t_lin.size,
                    do_psd=should_run_psd, do_spike=True,
                )
            else:
                psd_n = max(8, int(round(self.sampling_rate * self._psd_buffer_sec)))
                t_tail, sig_tail = self._ring_read_tail(psd_n)
                self._proc_worker.schedule(
                    sig_tail.copy(), t_tail.copy(),
                    self.sampling_rate,
                    [], 0, sig_tail.size,
                    do_psd=True, do_spike=False,
                )

    def _on_processing_result(self, result):
        self._proc_result = result
        if result.spike_times_cache is not None:
            self._spike_times_cache = result.spike_times_cache
        if result.last_scan_sample is not None:
            # Convert relative index back to absolute
            t_lin_size = min(self._total, self._cap)
            abs_start  = self._total - t_lin_size
            self._last_spike_scan_sample = abs_start + result.last_scan_sample
        if getattr(result, 'has_psd_update', False) and result.psd_f is not None:
            self._psd_pending = True
        if getattr(result, 'has_spike_update', False) and result.spike_minute_idx is not None:
            self._spike_pending = True

    # ── Render ─────────────────────────────────────────────────────────────────

    def _render_plot(self):
        if self._total == 0:
            return

        t0 = time.perf_counter()
        now = t0

        # ── Raw signal @ RAW_RENDER_HZ ─────────────────────────────────────────
        if (now - self._last_raw_render_t) >= 1.0 / RAW_RENDER_HZ:
            n_vis = int(self.sampling_rate * DISPLAY_WINDOW_SEC)
            x_vis, y_vis = self._ring_read_tail(n_vis)

            step = max(1, x_vis.size // MAX_DISPLAY_POINTS)
            self.canvas.raw_curve.setData(x_vis[::step], y_vis[::step])

            x_end   = float(x_vis[-1]) if x_vis.size else 0.0
            x_start = max(0.0, x_end - DISPLAY_WINDOW_SEC)
            self.canvas.raw_plot.setXRange(x_start, x_end, padding=0)

            peak  = float(np.max(np.abs(y_vis))) if y_vis.size else 0.0
            y_lim = max(0.2, peak * 1.2)
            self.canvas.raw_plot.setYRange(-y_lim, y_lim, padding=0)

            self._sync_marker_lines(x_start, x_end)
            self._last_raw_render_t = now

        r = self._proc_result

        # ── PSD @ PSD_RENDER_HZ ────────────────────────────────────────────────
        if r is not None and self._psd_pending and (now - self._last_psd_render_t) >= 1.0 / PSD_RENDER_HZ:
            if r.psd_f is not None and r.psd_db is not None:
                self.canvas.psd_curve.setData(r.psd_f, r.psd_db)
                self._latest_psd_f = np.asarray(r.psd_f).copy()
                self._latest_psd_db = np.asarray(r.psd_db).copy()
                if r.psd_f.size:
                    self.canvas.psd_plot.setXRange(float(r.psd_f[0]), float(r.psd_f[-1]), padding=0)
                self.canvas.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)
            self._psd_pending       = False
            self._last_psd_render_t = now

        # ── Spike histogram + waveform @ SPIKE_RENDER_HZ ──────────────────────
        if r is not None and self._spike_pending and (now - self._last_spike_render_t) >= 1.0 / SPIKE_RENDER_HZ:
            if r.spike_minute_idx is not None and r.spike_counts is not None:
                self.canvas.spike_curve.setData(r.spike_minute_idx, r.spike_counts)
                max_count = max(1, int(np.max(r.spike_counts)) if r.spike_counts.size else 1)
                self.canvas.spike_plot.setYRange(0, max_count * 1.2, padding=0)
                if r.spike_minute_idx.size:
                    right_min = float(r.spike_minute_idx[-1]) + SPIKE_BIN_SEC / 60.0
                    self.canvas.spike_plot.setXRange(0.0, right_min + 0.1, padding=0)

            if r.wf_t_ms is not None and r.wf_mu is not None and r.wf_sem is not None:
                self.canvas.wf_curve.setData(r.wf_t_ms, r.wf_mu)
                self.canvas.wf_upper.setData(r.wf_t_ms, r.wf_mu + r.wf_sem)
                self.canvas.wf_lower.setData(r.wf_t_ms, r.wf_mu - r.wf_sem)
                self._latest_wf_t_ms = np.asarray(r.wf_t_ms).copy()
                self._latest_wf_mu = np.asarray(r.wf_mu).copy()
                self.canvas.wf_plot.setXRange(float(r.wf_t_ms[0]), float(r.wf_t_ms[-1]), padding=0)
                self.canvas.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)
            else:
                self.canvas.wf_curve.setData([], [])
                self.canvas.wf_upper.setData([], [])
                self.canvas.wf_lower.setData([], [])
                self._latest_wf_t_ms = None
                self._latest_wf_mu = None

            self._spike_pending       = False
            self._last_spike_render_t = now

        self._render_dur_ms = (time.perf_counter() - t0) * 1000.0
        self._update_fps_label()

    # ── Marker helpers ─────────────────────────────────────────────────────────

    def _sync_marker_lines(self, x_start: float, x_end: float):
        """Show InfiniteLines only for markers visible in [x_start, x_end]."""
        left_i  = bisect.bisect_left(self.marker_times, x_start)
        right_i = bisect.bisect_right(self.marker_times, x_end)
        visible = self.marker_times[left_i:right_i]

        if visible == self.canvas._last_marker_set:
            return  # nothing changed

        self._clear_marker_lines()
        for t in visible:
            line = pg.InfiniteLine(pos=t, angle=90, pen=self.canvas._marker_pen)
            self.canvas.raw_plot.addItem(line)
            self.canvas._marker_lines.append(line)
        self.canvas._last_marker_set = list(visible)

    def _clear_marker_lines(self):
        for line in self.canvas._marker_lines:
            self.canvas.raw_plot.removeItem(line)
        self.canvas._marker_lines.clear()
        self.canvas._last_marker_set = []

    # ── FPS ────────────────────────────────────────────────────────────────────

    def _update_fps_label(self):
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_t
        if elapsed >= 1.0:
            fps = self._fps_frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}\nFrame time: {self._render_dur_ms:.1f}ms")
            self._fps_frame_count = 0
            self._fps_last_t      = now
