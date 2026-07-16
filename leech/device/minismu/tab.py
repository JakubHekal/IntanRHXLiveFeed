import time

from PyQt5 import QtWidgets
import numpy as np

from leech.device.tabs.base import DeviceTab
from leech.device.ring_buffer import RingBuffer
from leech.screens.plot_helpers import MAX_DISPLAY_POINTS, _minmax_downsample
from leech.telemetry_logger import append_telemetry_line
from .canvas import SmuCanvas


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
        self.canvas = SmuCanvas(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)

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
        n_vis = min(10000, min(self._ring.total, self._ring.cap))
        if n_vis < 2:
            return
        t, v = self._ring.read_channel(0)
        _, i = self._ring.read_channel(1)
        if t.size < 2:
            return
        td, vd = _minmax_downsample(t, v, MAX_DISPLAY_POINTS)
        _, id = _minmax_downsample(t, i, MAX_DISPLAY_POINTS)
        self.canvas.voltage_curve.setData(td, vd)
        self.canvas.current_curve.setData(td, id)

        v_range = float(np.max(vd)) - float(np.min(vd)) if vd.size else 1.0
        i_range = float(np.max(id)) - float(np.min(id)) if id.size else 1.0
        v_pad = max(0.1, v_range * 0.1)
        i_pad = max(0.001, i_range * 0.1)
        self._suspend_follow_detection = True
        try:
            self.canvas.voltage_plot.setYRange(float(np.min(vd)) - v_pad, float(np.max(vd)) + v_pad, padding=0)
            self.canvas.current_plot.setYRange(float(np.min(id)) - i_pad, float(np.max(id)) + i_pad, padding=0)
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

    def shutdown(self) -> bool:
        return True

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        self.clear()
        if sample_rate > 0:
            self.sampling_rate = float(sample_rate)
            self._ring = RingBuffer(self.sampling_rate, 2, duration_sec=300)
        self.set_receiving_state(True)

    def set_receiving_state(self, receiving: bool):
        self.is_receiving = bool(receiving)
