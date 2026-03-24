import time
import bisect
from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from telemetry_logger import append_telemetry_line

from workers.processing_worker import (
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

# ── Display constants ──────────────────────────────────────────────────────────
DISPLAY_WINDOW_SEC    = 10          # raw signal x-axis window
DISPLAY_BUFFER_SEC    = 300         # ring buffer size (5 min)
DEFAULT_SAMPLING_RATE = 20000
MAX_DISPLAY_POINTS    = 15000

# ── Per-subplot render rate limits ────────────────────────────────────────────
PLOT_UPDATE_FREQ_HZ = 120
RAW_RENDER_HZ       = 60
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

# ── Background compute throttle (every N data chunks) ─────────────────────────
PSD_PLOT_UPDATE_EVERY_N   = 15
SPIKE_PLOT_UPDATE_EVERY_N = 15  

# ── Pyqtgraph global style ────────────────────────────────────────────────────
pg.setConfigOption('background', 'white')
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
        self.glw.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.glw.setRenderHint(QtGui.QPainter.Antialiasing)
        layout.addWidget(self.glw)

        # Keep default context menu handling so each plot ViewBox can show its menu.
        self.glw.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)

        # Row 0: Raw signal (spans 3 columns)
        self.raw_plot = self.glw.addPlot(row=0, col=0, colspan=3)
        self.raw_plot.setTitle("Raw signal")
        self.raw_plot.setLabel('left',   'U [uV]')
        self.raw_plot.setLabel('bottom', 't [s]')
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        self.raw_plot.setDownsampling(auto=True, mode='peak')
        self.raw_plot.setClipToView(True)
        self.raw_plot.setMouseEnabled(x=True, y=True)

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
        # Allow horizontal pan/zoom so users can review full spike history.
        self.spike_plot.setMouseEnabled(x=True, y=True)

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
        self._marker_labels = []
        self._spike_marker_lines = []
        self._last_spike_marker_set = []
        self._spike_marker_labels = []


class PlotScreen(QtWidgets.QWidget):

    toggle_receiving_request_signal = QtCore.pyqtSignal(bool)
    save_disconnect_request_signal  = QtCore.pyqtSignal()
    marker_request_signal           = QtCore.pyqtSignal()
    auto_follow_changed_signal      = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # ── Session state ──────────────────────────────────────────────────────
        self.sampling_rate    = float(DEFAULT_SAMPLING_RATE)
        self.sample_counter   = 0       # total samples ever received (absolute)
        self.marker_records   = []      # marker dicts: id,timestamp_s,name,active
        self._marker_times_sorted = []
        self.is_receiving     = False
        self._receiving_wall_active_sec = 0.0
        self._receiving_wall_run_start = None
        self.base_connection_details = ""
        self.base_project_line = ""
        self.project_run_dir = ""
        self.project_snapshots_dir = ""

        self._spike_times_cache      = []
        self._last_spike_scan_sample = 0  # absolute sample count (never resets)
        self._last_proc_abs_start    = 0
        self._proc_result            = None
        self._psd_buffer_sec         = int(PSD_BUFFER_SEC)
        self._waveform_buffer_sec    = int(WAVEFORM_BUFFER_SEC)
        self._spike_bin_sec          = int(SPIKE_BIN_SEC)
        self._latest_psd_f           = None
        self._latest_psd_db          = None
        self._latest_wf_t_ms         = None
        self._latest_wf_mu           = None

        # Full-session raw history (dual-resolution for adaptive manual browsing).
        self._raw_hist_t_low = []
        self._raw_hist_y_low = []
        self._raw_hist_t_high = []
        self._raw_hist_y_high = []
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0
        self._raw_hist_stride_low = 1
        self._raw_hist_stride_high = 1

        # Axis follow state: when disabled, user-adjusted ranges are preserved.
        self._follow_axes = {
            'raw': True,
            'psd': True,
            'spike': True,
            'wf': True,
        }
        self._follow_menu_actions = {}
        self._suspend_follow_detection = False

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

        self._connect_plot_range_signals()
        self._install_plot_follow_context_actions()

        self._update_raw_history_stride()

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(1000 // PLOT_UPDATE_FREQ_HZ)
        self.render_timer.timeout.connect(self._render_plot)
        self.render_timer.start()

        self._fps_frame_count = 0
        self._fps_last_t      = time.perf_counter()
        self._render_dur_ms   = 0.0

        # Lightweight runtime telemetry for Phase 1 baselining.
        now = time.perf_counter()
        self._telemetry_last_emit = now
        self._telemetry_emit_interval_sec = 5.0
        self._telemetry_chunks = 0
        self._telemetry_samples = 0
        self._telemetry_render_calls = 0
        self._telemetry_raw_renders = 0
        self._telemetry_render_ms_total = 0.0
        self._telemetry_ingest_ms_total = 0.0
        self._telemetry_latest_chunk_received_t = now

    def _emit_telemetry_if_due(self):
        now = time.perf_counter()
        elapsed = now - self._telemetry_last_emit
        if elapsed < self._telemetry_emit_interval_sec:
            return

        render_calls = max(1, int(self._telemetry_render_calls))
        raw_renders = max(1, int(self._telemetry_raw_renders))
        avg_render_ms = self._telemetry_render_ms_total / render_calls
        avg_ingest_ms = self._telemetry_ingest_ms_total / max(1, int(self._telemetry_chunks))
        chunk_rate = float(self._telemetry_chunks) / max(elapsed, 1e-6)
        sample_rate = float(self._telemetry_samples) / max(elapsed, 1e-6)
        render_rate = float(self._telemetry_render_calls) / max(elapsed, 1e-6)
        raw_render_rate = float(self._telemetry_raw_renders) / max(elapsed, 1e-6)
        data_to_render_ms = (now - self._telemetry_latest_chunk_received_t) * 1000.0
        sample_clock_sec = float(self.sample_counter) / max(float(self.sampling_rate), 1e-9)
        wall_elapsed_sec = float(self._receiving_wall_active_sec)
        if self._receiving_wall_run_start is not None:
            wall_elapsed_sec += max(0.0, now - float(self._receiving_wall_run_start))
        drift_pct = 0.0
        if wall_elapsed_sec > 0.25:
            drift_pct = ((sample_clock_sec / wall_elapsed_sec) - 1.0) * 100.0

        line = (
            "[telemetry][plot] "
            f"window_s={elapsed:.2f} chunks={int(self._telemetry_chunks)} samples={int(self._telemetry_samples)} "
            f"chunk_hz={chunk_rate:.2f} sample_hz={sample_rate:.1f} render_hz={render_rate:.2f} raw_render_hz={raw_render_rate:.2f} "
            f"avg_ingest_ms={avg_ingest_ms:.3f} avg_render_ms={avg_render_ms:.3f} data_to_render_ms={data_to_render_ms:.3f} "
            f"sample_clock_s={sample_clock_sec:.1f} wall_s={wall_elapsed_sec:.1f} drift_pct={drift_pct:.2f}"
        )
        print(line)
        append_telemetry_line(line)

        self._telemetry_last_emit = now
        self._telemetry_chunks = 0
        self._telemetry_samples = 0
        self._telemetry_render_calls = 0
        self._telemetry_raw_renders = 0
        self._telemetry_render_ms_total = 0.0
        self._telemetry_ingest_ms_total = 0.0

    def _plot_item_for_key(self, key: str):
        if key == 'raw':
            return self.canvas.raw_plot
        if key == 'psd':
            return self.canvas.psd_plot
        if key == 'spike':
            return self.canvas.spike_plot
        if key == 'wf':
            return self.canvas.wf_plot
        raise KeyError(key)

    def _update_raw_history_stride(self):
        target_low = max(1.0, float(RAW_HISTORY_TARGET_HZ))
        target_high = max(1.0, float(RAW_HISTORY_HIGH_TARGET_HZ))
        self._raw_hist_stride_low = max(1, int(round(float(self.sampling_rate) / target_low)))
        self._raw_hist_stride_high = max(1, int(round(float(self.sampling_rate) / target_high)))

    def _append_decimated_history(self, t_chunk: np.ndarray, signal: np.ndarray, stride: int, sample_mod: int, t_store: list, y_store: list) -> int:
        n = int(signal.size)
        if n <= 0:
            return int(sample_mod)
        stride = max(1, int(stride))
        start = (stride - int(sample_mod)) % stride
        if start < n:
            idx = np.arange(start, n, stride, dtype=int)
            if idx.size:
                t_store.extend(np.asarray(t_chunk[idx], dtype=np.float64).tolist())
                y_store.extend(np.asarray(signal[idx], dtype=np.float64).tolist())
        return (int(sample_mod) + n) % stride

    def _append_raw_history(self, t_chunk: np.ndarray, signal: np.ndarray):
        self._raw_hist_sample_mod_low = self._append_decimated_history(
            t_chunk,
            signal,
            self._raw_hist_stride_low,
            self._raw_hist_sample_mod_low,
            self._raw_hist_t_low,
            self._raw_hist_y_low,
        )
        self._raw_hist_sample_mod_high = self._append_decimated_history(
            t_chunk,
            signal,
            self._raw_hist_stride_high,
            self._raw_hist_sample_mod_high,
            self._raw_hist_t_high,
            self._raw_hist_y_high,
        )

        # Keep only the last DISPLAY_BUFFER_SEC seconds in decimated history.
        latest_t = float(t_chunk[-1]) if np.asarray(t_chunk).size else 0.0
        cutoff_t = latest_t - float(DISPLAY_BUFFER_SEC)
        self._trim_history_store(self._raw_hist_t_low, self._raw_hist_y_low, cutoff_t)
        self._trim_history_store(self._raw_hist_t_high, self._raw_hist_y_high, cutoff_t)

    def _trim_history_store(self, t_store: list, y_store: list, cutoff_t: float):
        if not t_store:
            return
        keep_from = bisect.bisect_left(t_store, float(cutoff_t))
        if keep_from <= 0:
            return
        del t_store[:keep_from]
        del y_store[:keep_from]

    def _raw_time_bounds(self):
        stored = min(self._total, self._cap)
        if stored <= 0:
            return 0.0, 0.0
        latest_t = (self.sample_counter - 1) / float(self.sampling_rate)
        earliest_t = latest_t - ((stored - 1) / float(self.sampling_rate))
        return float(earliest_t), float(latest_t)

    def _connect_plot_range_signals(self):
        for key in ('raw', 'psd', 'spike', 'wf'):
            vb = self._plot_item_for_key(key).vb
            if hasattr(vb, 'sigRangeChangedManually'):
                vb.sigRangeChangedManually.connect(lambda *_args, k=key: self._on_manual_plot_range_change(k))
            else:
                vb.sigRangeChanged.connect(lambda *_args, k=key: self._on_manual_plot_range_change(k))

    def _install_plot_follow_context_actions(self):
        for key in ('raw', 'psd', 'spike', 'wf'):
            vb = self._plot_item_for_key(key).vb
            menu = getattr(vb, 'menu', None)
            if menu is None:
                continue
            menu.addSeparator()
            action = QtWidgets.QAction("Auto-follow", self)
            action.setCheckable(True)
            action.setChecked(bool(self._follow_axes.get(key, True)))
            action.toggled.connect(lambda checked, k=key: self.set_plot_auto_follow(k, checked))
            menu.addAction(action)
            self._follow_menu_actions[key] = action

    def _set_follow_menu_action_state(self, key: str, enabled: bool):
        action = self._follow_menu_actions.get(key)
        if action is None:
            return
        blocked = action.blockSignals(True)
        try:
            action.setChecked(bool(enabled))
        finally:
            action.blockSignals(blocked)

    def _on_manual_plot_range_change(self, key: str):
        if self._suspend_follow_detection:
            return
        if self._follow_axes.get(key, True):
            self._follow_axes[key] = False
            self._set_follow_menu_action_state(key, False)
            self.auto_follow_changed_signal.emit(self.is_auto_follow_enabled())

    def is_auto_follow_enabled(self) -> bool:
        return all(bool(v) for v in self._follow_axes.values())

    def set_plot_auto_follow(self, key: str, enabled: bool):
        if key not in self._follow_axes:
            return
        enabled = bool(enabled)
        if self._follow_axes[key] == enabled:
            self._set_follow_menu_action_state(key, enabled)
            return
        self._follow_axes[key] = enabled
        self._set_follow_menu_action_state(key, enabled)
        self.auto_follow_changed_signal.emit(self.is_auto_follow_enabled())

    def set_auto_follow(self, enabled: bool):
        enabled = bool(enabled)
        for key in self._follow_axes:
            self._follow_axes[key] = enabled
            self._set_follow_menu_action_state(key, enabled)
        self.auto_follow_changed_signal.emit(self.is_auto_follow_enabled())

    def reset_plot_views(self):
        self._suspend_follow_detection = True
        try:
            self.canvas.raw_plot.setXRange(0, DISPLAY_WINDOW_SEC, padding=0)
            self.canvas.raw_plot.setYRange(-1, 1, padding=0)
            self.canvas.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)
            self.canvas.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)
        finally:
            self._suspend_follow_detection = False
        self.set_auto_follow(True)

    def changeEvent(self, event):
        # 13 is the raw integer ID for WindowChangeInternal in PyQt5
        # WindowStateChange (21) handles maximizing/restoring
        if event.type() in [13, QtCore.QEvent.WindowStateChange]:
            if hasattr(self, 'canvas') and self.canvas.glw:
                # Force the internal GraphicsView to recalculate its scene
                self.canvas.glw.update()
                
                # Re-sync every plot's ViewBox
                for plot in [self.canvas.raw_plot, self.canvas.psd_plot, 
                            self.canvas.spike_plot, self.canvas.wf_plot]:
                    # This 'fake' range set forces the background grid to re-anchor
                    self._suspend_follow_detection = True
                    try:
                        plot.vb.setRange(plot.vb.viewRect(), padding=0)
                    finally:
                        self._suspend_follow_detection = False
                    plot.update()
                    
        super().changeEvent(event)

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
        self._update_raw_history_stride()
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0
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

    def set_project_storage_paths(self, run_dir: str, snapshots_dir: str):
        self.project_run_dir = str(run_dir or "")
        self.project_snapshots_dir = str(snapshots_dir or "")

    def clear_project_buffers(self):
        # Clear all curves
        self.canvas.raw_curve.setData([], [])
        self.canvas.psd_curve.setData([], [])
        self.canvas.spike_curve.setData([], [])
        self.canvas.wf_curve.setData([], [])
        self.canvas.wf_upper.setData([], [])
        self.canvas.wf_lower.setData([], [])
        self._clear_marker_lines()
        self._clear_spike_marker_lines()
        self.clear_snapshots()

        # Reset ring buffer
        self._ring[:] = 0
        self._wpos    = 0
        self._total   = 0
        self.sample_counter = 0

        # Reset all state
        self.marker_records          = []
        self._marker_times_sorted    = []
        self._spike_times_cache      = []
        self._last_spike_scan_sample = 0
        self._last_proc_abs_start    = 0
        self._proc_result            = None
        self._latest_psd_f           = None
        self._latest_psd_db          = None
        self._latest_wf_t_ms         = None
        self._latest_wf_mu           = None
        self._raw_hist_t_low         = []
        self._raw_hist_y_low         = []
        self._raw_hist_t_high        = []
        self._raw_hist_y_high        = []
        self._raw_hist_sample_mod_low = 0
        self._raw_hist_sample_mod_high = 0
        self._psd_pending            = False
        self._spike_pending          = False
        self._last_raw_render_t      = 0.0
        self._last_psd_render_t      = 0.0
        self._last_spike_render_t    = 0.0
        self.psd_update_counter      = 0
        self.spike_plot_frame_counter = 0
        self.set_auto_follow(True)

        # Reset axes
        self._suspend_follow_detection = True
        try:
            self.canvas.raw_plot.setXRange(0, DISPLAY_WINDOW_SEC, padding=0)
            self.canvas.raw_plot.setYRange(-1, 1, padding=0)
            self.canvas.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)
            self.canvas.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)
        finally:
            self._suspend_follow_detection = False

        self.base_connection_details = ""
        self.base_project_line = ""
        self.project_run_dir = ""
        self.project_snapshots_dir = ""
        self.connection_details_label.clear()
        self.set_receiving_state(False)
        self._receiving_wall_active_sec = 0.0
        self._receiving_wall_run_start = None
        self._fps_frame_count = 0
        self._fps_last_t      = time.perf_counter()
        self._render_dur_ms   = 0.0
        self.fps_label.setText("FPS: 0.0\nFrame time: 0.0 ms")

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

    def shutdown_workers(self) -> bool:
        if not hasattr(self, '_proc_worker') or self._proc_worker is None:
            return True
        return self._proc_worker.stop(timeout_ms=4000)

    def _save_plot_snapshot_png(self, plot_item, prefix: str) -> bool:
        if not self.project_snapshots_dir:
            return False
        try:
            out_dir = Path(self.project_snapshots_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = out_dir / f"{prefix}_{stamp}.png"

            try:
                exporter = pg.exporters.ImageExporter(plot_item)
                exporter.parameters()['width'] = 1920
                exporter.export(str(out_path))
            except Exception:
                # Fallback: grab full widget if plot exporter fails.
                pix = self.canvas.glw.grab()
                if pix.isNull() or not pix.save(str(out_path), "PNG"):
                    return False

            return out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            return False

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

        # Saving to disk is best-effort; snapshot availability should depend on data.
        self._save_plot_snapshot_png(self.canvas.psd_plot, "psd_snapshot")
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

        # Saving to disk is best-effort; snapshot availability should depend on data.
        self._save_plot_snapshot_png(self.canvas.wf_plot, "waveform_snapshot")
        return True

    def clear_snapshots(self):
        """Remove all snapshot overlays from PSD and waveform plots."""
        for curve in self._psd_snapshot_curves:
            self.canvas.psd_plot.removeItem(curve)
        self._psd_snapshot_curves.clear()

        for curve in self._wf_snapshot_curves:
            self.canvas.wf_plot.removeItem(curve)
        self._wf_snapshot_curves.clear()

    def add_marker(self, marker):
        if isinstance(marker, dict):
            ts = float(marker.get("timestamp_s", marker.get("timestamp", 0.0)))
            marker_id = int(marker.get("id", len(self.marker_records) + 1))
            name = str(marker.get("name", f"Marker {marker_id}"))
            self.marker_records.append(
                {
                    "id": marker_id,
                    "timestamp_s": ts,
                    "name": name,
                }
            )
        else:
            ts = float(marker)
            marker_id = len(self.marker_records) + 1
            self.marker_records.append(
                {
                    "id": marker_id,
                    "timestamp_s": ts,
                    "name": f"Marker {marker_id}",
                }
            )
        self._marker_times_sorted = sorted(float(m.get("timestamp_s", 0.0)) for m in self.marker_records)

    def set_marker_catalog(self, markers):
        safe = []
        for item in markers or []:
            try:
                safe.append(
                    {
                        "id": int(item.get("id", 0)),
                        "timestamp_s": float(item.get("timestamp_s", 0.0)),
                        "name": str(item.get("name", "")),
                    }
                )
            except Exception:
                continue
        self.marker_records = sorted(safe, key=lambda m: float(m.get("timestamp_s", 0.0)))
        self._marker_times_sorted = [float(m.get("timestamp_s", 0.0)) for m in self.marker_records]

    def get_markers(self):
        return [dict(m) for m in self.marker_records]

    def _active_marker_times(self):
        return list(self._marker_times_sorted)

    def _sorted_markers(self):
        return sorted(self.marker_records, key=lambda m: float(m.get("timestamp_s", 0.0)))

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
        ingest_t0 = time.perf_counter()
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
        self._append_raw_history(t_chunk, signal)
        self.sample_counter += n_samples
        self._telemetry_chunks += 1
        self._telemetry_samples += int(n_samples)
        self._telemetry_latest_chunk_received_t = time.perf_counter()

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
            if should_run_spike:
                stored = min(self._total, self._cap)
                psd_n = max(8, int(round(self.sampling_rate * self._psd_buffer_sec))) if should_run_psd else 0
                wf_n = max(8, int(round(self.sampling_rate * max(1, self._waveform_buffer_sec + 1))))
                spike_n = int(max(SPIKE_INCREMENTAL_MIN_SAMPLES, gap) + SPIKE_OVERLAP_SAMPLES)
                tail_n = min(stored, max(psd_n, wf_n, spike_n))

                t_tail, sig_tail = self._ring_read_tail(tail_n)
                abs_start = self._total - t_tail.size
                rel_last = max(0, int(self._last_spike_scan_sample - abs_start))
                rel_last = min(rel_last, t_tail.size)
                self._last_proc_abs_start = abs_start

                self._proc_worker.schedule(
                    sig_tail.copy(), t_tail.copy(), self.sampling_rate,
                    list(self._spike_times_cache),
                    rel_last, t_tail.size,
                    do_psd=should_run_psd, do_spike=True,
                )
            else:
                psd_n = max(8, int(round(self.sampling_rate * self._psd_buffer_sec)))
                t_tail, sig_tail = self._ring_read_tail(psd_n)
                self._last_proc_abs_start = self._total - t_tail.size
                self._proc_worker.schedule(
                    sig_tail.copy(), t_tail.copy(),
                    self.sampling_rate,
                    [], 0, sig_tail.size,
                    do_psd=True, do_spike=False,
                )
        self._telemetry_ingest_ms_total += (time.perf_counter() - ingest_t0) * 1000.0
        self._emit_telemetry_if_due()

    def _on_processing_result(self, result):
        self._proc_result = result
        if result.spike_times_cache is not None:
            self._spike_times_cache = result.spike_times_cache
        if result.last_scan_sample is not None:
            # Convert relative scan index back to absolute using the scheduled tail window.
            self._last_spike_scan_sample = self._last_proc_abs_start + int(result.last_scan_sample)
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
        self._telemetry_render_calls += 1

        # ── Raw signal @ RAW_RENDER_HZ ─────────────────────────────────────────
        if (now - self._last_raw_render_t) >= 1.0 / RAW_RENDER_HZ:
            self._telemetry_raw_renders += 1
            x_start = 0.0
            x_end = 0.0
            if self._follow_axes['raw']:
                n_vis = int(self.sampling_rate * DISPLAY_WINDOW_SEC)
                x_vis, y_vis = self._ring_read_tail(n_vis)

                step = max(1, x_vis.size // MAX_DISPLAY_POINTS)
                self.canvas.raw_curve.setData(x_vis[::step], y_vis[::step])

                x_end = float(x_vis[-1]) if x_vis.size else 0.0
                x_start = max(0.0, x_end - DISPLAY_WINDOW_SEC)
                x_min_lim, _x_max_data = self._raw_time_bounds()
                self.canvas.raw_plot.setLimits(xMin=max(0.0, x_min_lim), xMax=max(0.0, x_end + 0.1))

                self._suspend_follow_detection = True
                try:
                    self.canvas.raw_plot.setXRange(x_start, x_end, padding=0)
                finally:
                    self._suspend_follow_detection = False

                peak  = float(np.max(np.abs(y_vis))) if y_vis.size else 0.0
                y_lim = max(0.2, peak * 1.2)
                self._suspend_follow_detection = True
                try:
                    self.canvas.raw_plot.setYRange(-y_lim, y_lim, padding=0)
                finally:
                    self._suspend_follow_detection = False
            else:
                history_end = 0.0
                history_start = 0.0
                if self._raw_hist_t_low:
                    history_start = float(self._raw_hist_t_low[0])
                    history_end = float(self._raw_hist_t_low[-1])
                    self.canvas.raw_plot.setLimits(xMin=max(0.0, history_start), xMax=max(0.0, history_end + 0.1))

                vr = self.canvas.raw_plot.vb.viewRange()[0]
                if len(vr) >= 2:
                    x_start = float(vr[0])
                    x_end = float(vr[1])
                elif history_end > 0.0:
                    x_end = history_end
                    x_start = max(0.0, x_end - DISPLAY_WINDOW_SEC)

                span = max(0.0, x_end - x_start)
                t_src = None
                y_src = None

                # For recent/small windows, render from full-resolution ring data
                # so zooming back in restores fine waveform detail.
                stored = min(self._total, self._cap)
                if stored > 1 and span <= RAW_FULL_RES_MAX_SPAN_SEC:
                    earliest_t, latest_t = self._raw_time_bounds()
                    left_q = max(earliest_t, x_start - RAW_MANUAL_VIEW_MARGIN_SEC)
                    right_q = min(latest_t, x_end + RAW_MANUAL_VIEW_MARGIN_SEC)
                    if right_q > left_q:
                        t_lin, y_lin = self._ring_read()
                        if t_lin.size:
                            i0 = int(np.searchsorted(t_lin, left_q, side='left'))
                            i1 = int(np.searchsorted(t_lin, right_q, side='right'))
                            if i1 <= i0:
                                i0 = max(0, min(i0, t_lin.size - 1))
                                i1 = min(t_lin.size, i0 + 1)
                            t_src = t_lin[i0:i1]
                            y_src = y_lin[i0:i1]

                # Otherwise use adaptive decimated full-session history.
                if t_src is None or y_src is None:
                    use_high = span <= RAW_ADAPTIVE_HIGH_RES_MAX_SPAN_SEC and bool(self._raw_hist_t_high)
                    if use_high:
                        t_store = self._raw_hist_t_high
                        y_store = self._raw_hist_y_high
                    else:
                        t_store = self._raw_hist_t_low
                        y_store = self._raw_hist_y_low

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
                    x_seg = np.asarray(t_src, dtype=np.float64)
                    y_seg = np.asarray(y_src, dtype=np.float64)
                    step = max(1, x_seg.size // MAX_RAW_HISTORY_PLOT_POINTS)
                    self.canvas.raw_curve.setData(x_seg[::step], y_seg[::step])

            self._sync_marker_lines(max(0.0, min(x_start, x_end)), max(0.0, max(x_start, x_end)))
            self._last_raw_render_t = now

        r = self._proc_result

        # ── PSD @ PSD_RENDER_HZ ────────────────────────────────────────────────
        if r is not None and self._psd_pending and (now - self._last_psd_render_t) >= 1.0 / PSD_RENDER_HZ:
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
            self._psd_pending       = False
            self._last_psd_render_t = now

        # ── Spike histogram + waveform @ SPIKE_RENDER_HZ ──────────────────────
        if r is not None and self._spike_pending and (now - self._last_spike_render_t) >= 1.0 / SPIKE_RENDER_HZ:
            if r.spike_minute_idx is not None and r.spike_counts is not None:
                self.canvas.spike_curve.setData(r.spike_minute_idx, r.spike_counts)
                max_count = max(1, int(np.max(r.spike_counts)) if r.spike_counts.size else 1)
                right_min = 0.0
                if r.spike_minute_idx.size:
                    right_min = float(r.spike_minute_idx[-1]) + SPIKE_BIN_SEC / 60.0
                # Keep x-limits synced with available history so manual panning can
                # browse the full session while preventing negative time.
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

            self._spike_pending       = False
            self._last_spike_render_t = now

        spike_vr = self.canvas.spike_plot.vb.viewRange()[0]
        if len(spike_vr) >= 2:
            self._sync_spike_marker_lines(float(spike_vr[0]), float(spike_vr[1]))

        self._render_dur_ms = (time.perf_counter() - t0) * 1000.0
        self._telemetry_render_ms_total += self._render_dur_ms
        self._emit_telemetry_if_due()
        self._update_fps_label()

    # ── Marker helpers ─────────────────────────────────────────────────────────

    def _sync_marker_lines(self, x_start: float, x_end: float):
        """Show InfiniteLines only for markers visible in [x_start, x_end]."""
        markers = self._sorted_markers()
        marker_times = [float(m.get("timestamp_s", 0.0)) for m in markers]
        left_i  = bisect.bisect_left(marker_times, x_start)
        right_i = bisect.bisect_right(marker_times, x_end)
        visible = markers[left_i:right_i]
        visible_key = [(int(m.get("id", 0)), float(m.get("timestamp_s", 0.0)), str(m.get("name", ""))) for m in visible]

        if visible_key == self.canvas._last_marker_set:
            return  # nothing changed

        self._clear_marker_lines()
        y_top = float(self.canvas.raw_plot.vb.viewRange()[1][1]) if self.canvas.raw_plot.vb.viewRange() else 1.0
        for m in visible:
            t = float(m.get("timestamp_s", 0.0))
            name = str(m.get("name", ""))
            line = pg.InfiniteLine(pos=t, angle=90, pen=self.canvas._marker_pen)
            self.canvas.raw_plot.addItem(line)
            self.canvas._marker_lines.append(line)
            if name:
                label = pg.TextItem(text=name, color=(220, 20, 60), anchor=(0, 1))
                label.setPos(t, y_top)
                self.canvas.raw_plot.addItem(label)
                self.canvas._marker_labels.append(label)
        self.canvas._last_marker_set = list(visible_key)

    def _sync_spike_marker_lines(self, x_start_min: float, x_end_min: float):
        markers = self._sorted_markers()
        marker_times_min = [float(m.get("timestamp_s", 0.0)) / 60.0 for m in markers]
        left_i = bisect.bisect_left(marker_times_min, x_start_min)
        right_i = bisect.bisect_right(marker_times_min, x_end_min)
        visible = markers[left_i:right_i]
        visible_key = [(int(m.get("id", 0)), float(m.get("timestamp_s", 0.0)), str(m.get("name", ""))) for m in visible]

        if visible_key == self.canvas._last_spike_marker_set:
            return

        self._clear_spike_marker_lines()
        y_top = float(self.canvas.spike_plot.vb.viewRange()[1][1]) if self.canvas.spike_plot.vb.viewRange() else 1.0
        for m in visible:
            t_min = float(m.get("timestamp_s", 0.0)) / 60.0
            name = str(m.get("name", ""))
            line = pg.InfiniteLine(pos=t_min, angle=90, pen=self.canvas._marker_pen)
            self.canvas.spike_plot.addItem(line)
            self.canvas._spike_marker_lines.append(line)
            if name:
                label = pg.TextItem(text=name, color=(220, 20, 60), anchor=(0, 1))
                label.setPos(t_min, y_top)
                self.canvas.spike_plot.addItem(label)
                self.canvas._spike_marker_labels.append(label)
        self.canvas._last_spike_marker_set = list(visible_key)

    def _clear_marker_lines(self):
        for line in self.canvas._marker_lines:
            self.canvas.raw_plot.removeItem(line)
        self.canvas._marker_lines.clear()
        for label in self.canvas._marker_labels:
            self.canvas.raw_plot.removeItem(label)
        self.canvas._marker_labels.clear()
        self.canvas._last_marker_set = []

    def _clear_spike_marker_lines(self):
        for line in self.canvas._spike_marker_lines:
            self.canvas.spike_plot.removeItem(line)
        self.canvas._spike_marker_lines.clear()
        for label in self.canvas._spike_marker_labels:
            self.canvas.spike_plot.removeItem(label)
        self.canvas._spike_marker_labels.clear()
        self.canvas._last_spike_marker_set = []

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
