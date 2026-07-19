import time

from PyQt5 import QtWidgets
import numpy as np

from leech.device.tabs.base import DeviceTab
from leech.device.ring_buffer import RingBuffer
from leech.screens.plot_helpers import MAX_DISPLAY_POINTS, _minmax_downsample
from leech.telemetry_logger import append_telemetry_line
from .canvas import SmuCanvas

SMU_DISPLAY_WINDOW_SEC = 30.0


class SmuDeviceTab(DeviceTab):
    def __init__(self, sample_rate=1000.0, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.sampling_rate = sample_rate
        self.sample_counter = 0
        self.is_receiving = False
        self._render_dur_ms = 0.0
        self._last_render_t = 0.0
        self._ring = RingBuffer(sample_rate, 2, duration_sec=300)
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

        self._suspend_follow_detection = False
        self._follow_axes = {'voltage': True, 'current': True}
        self._follow_menu_actions = {}

        self.canvas = SmuCanvas(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)

        self._connect_plot_range_signals()
        self._install_plot_follow_context_actions()

    def _connect_plot_range_signals(self):
        for key, plot_item in [('voltage', self.canvas.voltage_plot),
                               ('current', self.canvas.current_plot)]:
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
        for key, label in [('voltage', 'Auto-follow Voltage'), ('current', 'Auto-follow Current')]:
            plot_item = self.canvas.voltage_plot if key == 'voltage' else self.canvas.current_plot
            action = QtWidgets.QAction(label)
            action.setCheckable(True)
            action.setChecked(self._follow_axes[key])
            action.toggled.connect(lambda checked, k=key: self._set_follow(k, checked))
            plot_item.vb.menu.addAction(action)
            self._follow_menu_actions[key] = action

    def _set_follow(self, key, enabled):
        self._follow_axes[key] = enabled
        action = self._follow_menu_actions.get(key)
        if action:
            action.setChecked(enabled)

    def on_data(self, chunk: np.ndarray):
        arr = np.asarray(chunk)
        if arr.ndim == 1:
            voltage = float(arr[0]) if arr.size > 0 else 0.0
            current = float(arr[1]) if arr.size > 1 else 0.0
            t = self.sample_counter / self.sampling_rate
            y = np.array([[voltage], [current]], dtype=np.float64)
            self._ring.write(np.array([t], dtype=np.float64), y)
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
            y = np.vstack([voltage, current])
            self._ring.write(t, y)
            self.sample_counter += n
            if self.sample_counter % 500 < n:
                append_telemetry_line(
                    f"smutab | on_data | shape={arr.shape} "
                    f"v=[{float(voltage[0]):.6f}..{float(voltage[-1]):.6f}] "
                    f"i=[{float(current[0]):.9f}..{float(current[-1]):.9f}] "
                    f"total={self._ring.total}"
                )

    def render(self):
        now = time.perf_counter()
        if (now - self._last_render_t) < 1.0 / 30.0:
            return
        t0 = now
        total = min(self._ring.total, self._ring.cap)
        if total < 2:
            return

        for ch_idx, key, curve, plot_item in [
            (0, 'voltage', self.canvas.voltage_curve, self.canvas.voltage_plot),
            (1, 'current', self.canvas.current_curve, self.canvas.current_plot),
        ]:
            if self._follow_axes[key]:
                n_vis = int(self.sampling_rate * SMU_DISPLAY_WINDOW_SEC)
                t, y = self._ring.read_tail(n_vis, ch_idx=ch_idx)
                if t.size < 2:
                    curve.setData([], [])
                    continue
                step = max(1, t.size // MAX_DISPLAY_POINTS)
                xd, yd = t[::step], y[::step]
                curve.setData(xd, yd)
                x_end = float(t[-1])
                x_start = max(0.0, x_end - SMU_DISPLAY_WINDOW_SEC)
                x_min_lim, _ = self._ring.raw_time_bounds()
                self._suspend_follow_detection = True
                try:
                    plot_item.setLimits(xMin=max(0.0, x_min_lim), xMax=max(0.0, x_end + 0.1))
                    plot_item.setXRange(x_start, x_end, padding=0)
                    peak = max(abs(float(y.min())), abs(float(y.max()))) if y.size else 0.0
                    y_lim = max(0.1, peak * 1.2)
                    plot_item.setYRange(-y_lim, y_lim, padding=0)
                finally:
                    self._suspend_follow_detection = False
            else:
                vr = plot_item.vb.viewRange()[0]
                x_start = float(vr[0])
                x_end = float(vr[1])
                if x_end <= x_start:
                    x_end = x_start + SMU_DISPLAY_WINDOW_SEC
                span = x_end - x_start
                t_full, y_full = self._ring.read_channel(ch_idx)
                if t_full.size < 2:
                    curve.setData([], [])
                    continue
                margin = span * 0.05
                i0 = int(np.searchsorted(t_full, x_start - margin, side='left'))
                i1 = int(np.searchsorted(t_full, x_end + margin, side='right'))
                i0 = max(0, min(i0, t_full.size))
                i1 = max(i0, min(i1, t_full.size))
                t_src = t_full[i0:i1]
                y_src = y_full[i0:i1]
                if t_src.size > 2:
                    xd, yd = _minmax_downsample(t_src, y_src, MAX_DISPLAY_POINTS)
                else:
                    xd, yd = t_src, y_src
                curve.setData(xd, yd)
                latest_t = float(t_full[-1]) if t_full.size else 0.0
                self._suspend_follow_detection = True
                try:
                    plot_item.setLimits(xMin=0.0, xMax=max(0.1, latest_t + 0.1))
                    peak = max(abs(float(y_src.min())), abs(float(y_src.max()))) if y_src.size else 0.0
                    y_lim = max(0.1, peak * 1.2)
                    plot_item.setYRange(-y_lim, y_lim, padding=0)
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
        self._ring.clear()
        self.sample_counter = 0
        self.canvas.voltage_curve.setData([], [])
        self.canvas.current_curve.setData([], [])
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()
        self.set_auto_follow(True)

    def shutdown(self) -> bool:
        return True

    def set_auto_follow(self, enabled: bool):
        for key in self._follow_axes:
            self._follow_axes[key] = enabled
            action = self._follow_menu_actions.get(key)
            if action:
                action.setChecked(enabled)

    def is_auto_follow_enabled(self) -> bool:
        return any(self._follow_axes.values())

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        self.clear()
        if sample_rate > 0:
            self.sampling_rate = float(sample_rate)
            self._ring = RingBuffer(self.sampling_rate, 2, duration_sec=300)
        self.set_receiving_state(True)

    def set_receiving_state(self, receiving: bool):
        self.is_receiving = bool(receiving)
