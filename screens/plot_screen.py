import time
import bisect

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

import numpy as np

from workers.processing_worker import (
    ProcessingWorker,
    SPIKE_INCREMENTAL_MIN_SAMPLES,
    PSD_BUFFER_SEC,
    SPIKE_BIN_SEC,
    SPIKE_HISTORY_MIN,
    PSD_YLIM_MIN,
    PSD_YLIM_MAX,
)

DISPLAY_WINDOW_SEC = 10
DEFAULT_SAMPLING_RATE = 20000
PLOT_UPDATE_FREQ_HZ = 10

SPIKE_PLOT_UPDATE_EVERY_N = 5
PSD_PLOT_UPDATE_EVERY_N = 5
MAX_DISPLAY_POINTS = 3000

INITIAL_BUFFER_SEC = 30

def default_data_buffer(fs: float) -> np.ndarray:
    n = max(1, int(round(fs * INITIAL_BUFFER_SEC)))
    return np.zeros((2, n), dtype=np.float64)


def ensure_data_buffer_capacity(buffer: np.ndarray, stored_samples: int, extra_samples: int) -> np.ndarray:
    needed = stored_samples + extra_samples
    if needed <= buffer.shape[1]:
        return buffer
    new_size = max(needed, buffer.shape[1] * 2)
    new_buffer = np.zeros((buffer.shape[0], new_size), dtype=buffer.dtype)
    new_buffer[:, :stored_samples] = buffer[:, :stored_samples]
    return new_buffer


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.subplot_mosaic('RRR;PSA')
        self.axes['R'].set_title("Raw signal")
        self.axes['R'].set_xlabel("t [s]")
        self.axes['R'].set_ylabel("U [uV]")
        self.axes['R'].grid(True, alpha=0.3)
        self.axes['P'].set_title(f"Power spectrum (last {PSD_BUFFER_SEC}s)")
        self.axes['P'].set_xlabel("f [Hz]")
        self.axes['P'].set_ylabel("P [dB]")
        self.axes['P'].grid(True, alpha=0.3)
        self.axes['S'].set_title(f"Spike counts ({SPIKE_BIN_SEC}s bins)")
        self.axes['S'].set_xlabel("t [min]")
        self.axes['S'].set_ylabel("Count")
        self.axes['S'].grid(True, alpha=0.3)
        self.axes['A'].set_title(f"Averaged spike waveform")
        self.axes['A'].set_xlabel("t [ms]")
        self.axes['A'].set_ylabel("U [uV]")
        self.axes['A'].grid(True, alpha=0.3)
        super().__init__(fig)
        fig.tight_layout(pad=0.8)

        self.raw_signal_line, = self.axes['R'].plot([], [], color='blue', linewidth=0.5)
        self.psd_line, = self.axes['P'].plot([], [], color='orange', linewidth=1.0)
        self.spike_count_line, = self.axes['S'].plot([], [], color='red', linewidth=1.0)
        self.waveform_mean_line, = self.axes['A'].plot([], [], color='tab:green', linewidth=2.0)
       

class PlotScreen(QtWidgets.QWidget):
    
    toggle_receiving_request_signal = QtCore.pyqtSignal(bool)
    save_disconnect_request_signal = QtCore.pyqtSignal()
    marker_request_signal = QtCore.pyqtSignal()

    data_buffer = default_data_buffer(DEFAULT_SAMPLING_RATE)
    sampling_rate = DEFAULT_SAMPLING_RATE
    stored_samples = 0
    sample_counter = 0
    render_pending = False
    psd_update_counter = 0
    spike_plot_frame_counter = 0
    waveform_sem_fill = None
    is_receiving = True
    marker_times = []
    marker_artists = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.marker_times = []
        self.marker_artists = []
        self.base_connection_details = ""
        self._spike_times_cache = []
        self._last_spike_scan_sample = 0
        self._proc_result = None
        self._proc_worker = ProcessingWorker(self)
        self._proc_worker.result_ready.connect(self._on_processing_result)
        self._proc_worker.start()

        self._marker_collection = LineCollection(
            [],
            colors='crimson',
            linestyles='--',
            linewidths=1.0,
            alpha=0.75,
        )
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Header
        header_layout = QtWidgets.QHBoxLayout()

        self.toggle_receiving_button = QtWidgets.QPushButton("Stop receiving")
        self.toggle_receiving_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.toggle_receiving_button.clicked.connect(self._on_toggle_receiving_clicked)

        self.marker_button = QtWidgets.QPushButton("Add marker")
        self.marker_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.marker_button.clicked.connect(self._on_marker_clicked)

        self.save_disconnect_button = QtWidgets.QPushButton("Save and Disconnect")
        self.save_disconnect_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.save_disconnect_button.clicked.connect(self._on_save_disconnect_clicked)

        self.connection_details_label = QtWidgets.QLabel()
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        
        header_layout.addWidget(self.toggle_receiving_button)
        header_layout.addWidget(self.marker_button)
        header_layout.addWidget(self.save_disconnect_button)
        header_layout.addWidget(self.connection_details_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.fps_label)

        layout.addLayout(header_layout)

        self.canvas = MplCanvas(self, width=16, height=9, dpi=100)
        self.canvas.axes['R'].add_collection(self._marker_collection)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(1000 // PLOT_UPDATE_FREQ_HZ)
        self.render_timer.timeout.connect(self._render_plot)
        self.render_timer.start()

        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        layout.addWidget(self.canvas, 1)

    def set_connection_details(self, host, command_port, data_port, sample_rate, project_name):
        self.clear_project_buffers()
        self.base_connection_details = f"Connected to {host}:{command_port}/{data_port} at {sample_rate} Hz - Project: {project_name}"
        self.set_connection_status("Receiving")
        self.sampling_rate = float(sample_rate)
        self.set_receiving_state(True)

    def set_connection_status(self, status_text=""):
        if not self.base_connection_details:
            self.connection_details_label.clear()
            return
        if status_text:
            self.connection_details_label.setText(f"{self.base_connection_details} - {status_text}")
        else:
            self.connection_details_label.setText(self.base_connection_details)

    def clear_project_buffers(self):
        self.render_pending = False
        self.canvas.raw_signal_line.set_data([], [])
        self.canvas.psd_line.set_data([], [])
        self.canvas.spike_count_line.set_data([], [])
        self.canvas.waveform_mean_line.set_data([], [])
        self._clear_marker_artists()
        self.marker_times = []
        self._spike_times_cache = []
        self._last_spike_scan_sample = 0
        self._proc_result = None
        self.canvas.axes['R'].set_xlim(0, DISPLAY_WINDOW_SEC)
        self.canvas.axes['R'].set_ylim(-1, 1)
        self.canvas.axes['P'].relim()
        self.canvas.axes['P'].autoscale_view()
        self.canvas.axes['S'].relim()
        self.canvas.axes['S'].autoscale_view()
        self.canvas.axes['A'].relim()
        self.canvas.axes['A'].autoscale_view()
        self.canvas.draw_idle()
        self.data_buffer = default_data_buffer(self.sampling_rate)
        self.stored_samples = 0
        self.sample_counter = 0
        self.psd_update_counter = 0
        self.spike_plot_frame_counter = 0
        self.base_connection_details = ""
        self.connection_details_label.clear()
        self.set_receiving_state(False)
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()
        self.fps_label.setText("FPS: 0.0")

    def _update_fps_label(self):
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_t
        if elapsed >= 1.0:
            fps = self._fps_frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._fps_frame_count = 0
            self._fps_last_t = now

    def _on_disconnect_clicked(self):
        self.disconnect_request_signal.emit()

    def _on_toggle_receiving_clicked(self):
        self.toggle_receiving_request_signal.emit(not self.is_receiving)

    def _on_save_disconnect_clicked(self):
        self.save_disconnect_request_signal.emit()

    def _on_marker_clicked(self):
        self.marker_request_signal.emit()

    def set_receiving_state(self, receiving):
        self.is_receiving = bool(receiving)
        if self.is_receiving:
            self.toggle_receiving_button.setText("Stop receiving")
            self.marker_button.setEnabled(True)
        else:
            self.toggle_receiving_button.setText("Start receiving")
            self.marker_button.setEnabled(False)

    def add_marker(self, timestamp_s):
        self.marker_times.append(float(timestamp_s))
        self.render_pending = True

    def _clear_marker_artists(self):
        self.marker_artists = []
        self._marker_collection.set_segments([])

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
        
        signal = arr[0, :]

        self.data_buffer = ensure_data_buffer_capacity(self.data_buffer, self.stored_samples, n_samples)

        t_chunk = (self.sample_counter + np.arange(n_samples, dtype=np.float64)) / self.sampling_rate
        start = self.stored_samples
        end = start + n_samples

        self.data_buffer[0, start:end] = t_chunk
        self.data_buffer[1, start:end] = signal

        self.stored_samples = end
        self.sample_counter += n_samples
        self.render_pending = True

        # Throttle expensive processing: raw plot can update in realtime while PSD/spikes run every N chunks.
        self.psd_update_counter += 1
        self.spike_plot_frame_counter += 1
        do_psd = (self.psd_update_counter % max(1, PSD_PLOT_UPDATE_EVERY_N) == 0)
        do_spike = (self.spike_plot_frame_counter % max(1, SPIKE_PLOT_UPDATE_EVERY_N) == 0)

        # Force a first pass so PSD/spike plots initialize quickly.
        if self._proc_result is None:
            do_psd = True
            do_spike = True

        # Schedule background processing only when requested by Every-N policy.
        gap_since_spike_scan = self.stored_samples - self._last_spike_scan_sample
        should_run_spike = do_spike and (gap_since_spike_scan >= SPIKE_INCREMENTAL_MIN_SAMPLES)
        should_run_psd = do_psd
        if should_run_psd or should_run_spike:
            if should_run_spike:
                # Spike path still needs the full history window currently used by histogram/waveform logic.
                signal_snap = self.data_buffer[1, :self.stored_samples].copy()
                t_snap = self.data_buffer[0, :self.stored_samples].copy()
                stored_for_job = self.stored_samples
                last_scan_for_job = self._last_spike_scan_sample
                spike_cache_for_job = list(self._spike_times_cache)
            else:
                # PSD-only path: copy only what the PSD calculation needs.
                psd_samples = max(8, int(round(self.sampling_rate * PSD_BUFFER_SEC)))
                psd_start = max(0, self.stored_samples - psd_samples)
                signal_snap = self.data_buffer[1, psd_start:self.stored_samples].copy()
                t_snap = self.data_buffer[0, psd_start:self.stored_samples].copy()
                stored_for_job = signal_snap.size
                last_scan_for_job = 0
                spike_cache_for_job = []

            self._proc_worker.schedule(
                signal_snap, t_snap, self.sampling_rate,
                spike_cache_for_job,
                last_scan_for_job,
                stored_for_job,
                do_psd=should_run_psd,
                do_spike=should_run_spike,
            )

    def _on_processing_result(self, result):
        self._proc_result = result
        if result.spike_times_cache is not None:
            self._spike_times_cache = result.spike_times_cache
        if result.last_scan_sample is not None:
            self._last_spike_scan_sample = result.last_scan_sample
        self.render_pending = True

    def _render_plot(self):
        if not self.render_pending or self.stored_samples < 1:
            return

        visible_start_idx = max(0, self.stored_samples - int(self.sampling_rate * DISPLAY_WINDOW_SEC))
        x_visible = self.data_buffer[0, visible_start_idx:self.stored_samples]
        y_visible = self.data_buffer[1, visible_start_idx:self.stored_samples]
        if x_visible.size < 1:
            return

        step = max(1, x_visible.size // MAX_DISPLAY_POINTS)
        self.canvas.raw_signal_line.set_data(x_visible[::step], y_visible[::step])

        x_end = x_visible[-1]
        x_start = max(0.0, x_end - DISPLAY_WINDOW_SEC)
        self.canvas.axes['R'].set_xlim(x_start, x_end)

        peak = float(np.max(np.abs(y_visible))) if y_visible.size else 0.0
        y_lim = max(0.2, peak * 1.2)
        self.canvas.axes['R'].set_ylim(-y_lim, y_lim)

        # Keep marker rendering O(visible_markers) and avoid per-frame artist creation.
        left_i = bisect.bisect_left(self.marker_times, x_start)
        right_i = bisect.bisect_right(self.marker_times, x_end)
        visible_markers = self.marker_times[left_i:right_i]
        if visible_markers:
            y0, y1 = self.canvas.axes['R'].get_ylim()
            segments = [((t, y0), (t, y1)) for t in visible_markers]
            self._marker_collection.set_segments(segments)
        else:
            self._marker_collection.set_segments([])

        # Apply pre-computed results from the background ProcessingWorker
        r = self._proc_result
        if r is not None:
            # PSD
            if getattr(r, 'has_psd_update', True) and r.psd_f is not None and r.psd_db is not None:
                self.canvas.psd_line.set_data(r.psd_f, r.psd_db)
                if r.psd_f.size:
                    self.canvas.axes['P'].set_xlim(float(r.psd_f[0]), float(r.psd_f[-1]))
                self.canvas.axes['P'].set_ylim(PSD_YLIM_MIN, PSD_YLIM_MAX)

            # Spike count histogram
            if getattr(r, 'has_spike_update', True) and r.spike_minute_idx is not None and r.spike_counts is not None:
                self.canvas.spike_count_line.set_data(r.spike_minute_idx, r.spike_counts)
                max_count = max(1, int(np.max(r.spike_counts)) if r.spike_counts.size else 1)
                self.canvas.axes['S'].set_ylim(0, max_count * 1.2)
                if r.spike_minute_idx.size:
                    right_min = float(r.spike_minute_idx[-1]) + SPIKE_BIN_SEC / 60.0
                    left_min = max(0.0, right_min - SPIKE_HISTORY_MIN)
                    self.canvas.axes['S'].set_xlim(left_min, right_min + 0.1)

            # Waveform mean+-SEM
            if getattr(r, 'has_spike_update', True):
                if r.wf_t_ms is not None and r.wf_mu is not None and r.wf_sem is not None:
                    self.canvas.waveform_mean_line.set_data(r.wf_t_ms, r.wf_mu)
                    if self.waveform_sem_fill is not None:
                        self.waveform_sem_fill.remove()
                    self.waveform_sem_fill = self.canvas.axes['A'].fill_between(
                        r.wf_t_ms,
                        r.wf_mu - r.wf_sem,
                        r.wf_mu + r.wf_sem,
                        color='tab:green',
                        alpha=0.3,
                    )
                    self.canvas.axes['A'].set_xlim(float(r.wf_t_ms[0]), float(r.wf_t_ms[-1]))
                    wf_peak = float(np.max(np.abs(r.wf_mu))) if r.wf_mu.size else 0.0
                    self.canvas.axes['A'].set_ylim(-max(1.0, wf_peak * 1.5), max(1.0, wf_peak * 1.5))
                else:
                    self.canvas.waveform_mean_line.set_data([], [])
                    if self.waveform_sem_fill is not None:
                        self.waveform_sem_fill.remove()
                        self.waveform_sem_fill = None

        

        self.canvas.draw_idle()
        self.render_pending = False
        self._update_fps_label()

    

