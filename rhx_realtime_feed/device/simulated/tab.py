import time

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

from rhx_realtime_feed.device.tabs.base import DeviceTab
from rhx_realtime_feed.device.ring_buffer import RingBuffer
from rhx_realtime_feed.screens.plot_helpers import MAX_DISPLAY_POINTS, _minmax_downsample


class SimpleDeviceTab(DeviceTab):
    def __init__(self, sample_rate=1000.0, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.sampling_rate = sample_rate
        self.sample_counter = 0
        self.is_receiving = False
        self._render_dur_ms = 0.0
        self._last_render_t = 0.0
        self._ring = RingBuffer(sample_rate, 1, duration_sec=300)
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.glw.setRenderHint(QtGui.QPainter.Antialiasing)
        self.glw.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw, 1)

        self.plot = self.glw.addPlot()
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 't [s]')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setDownsampling(auto=True, mode='peak')
        self.plot.setClipToView(True)
        self.plot.setMouseEnabled(x=True, y=True)

        self.curve = self.plot.plot(pen=pg.mkPen("#1D20ED", width=1))

    def on_data(self, chunk: np.ndarray):
        arr = np.asarray(chunk)
        if arr.ndim == 1:
            val = float(arr[0]) if arr.size > 0 else 0.0
            t = self.sample_counter / self.sampling_rate
            self._ring.write(np.array([t], dtype=np.float64), np.array([[val]], dtype=np.float64))
            self.sample_counter += 1
        elif arr.ndim == 2 and arr.shape[0] >= 1:
            n = arr.shape[1]
            ch0 = arr[0, :]
            t = (self.sample_counter + np.arange(n, dtype=np.float64)) / self.sampling_rate
            self._ring.write(t, ch0[np.newaxis, :])
            self.sample_counter += n

    def render(self):
        now = time.perf_counter()
        if (now - self._last_render_t) < 1.0 / 30.0:
            return
        t0 = now
        n_vis = min(10000, min(self._ring.total, self._ring.cap))
        if n_vis < 2:
            return
        t, y = self._ring.read_channel(0)
        if t.size < 2:
            return
        td, yd = _minmax_downsample(t, y, MAX_DISPLAY_POINTS)
        self.curve.setData(td, yd)

        y_range = float(np.max(yd)) - float(np.min(yd)) if yd.size else 1.0
        y_pad = max(0.1, y_range * 0.1)
        self.plot.setYRange(float(np.min(yd)) - y_pad, float(np.max(yd)) + y_pad, padding=0)

        self._render_dur_ms = (time.perf_counter() - t0) * 1000.0
        self._last_render_t = now
        self._update_fps_label()

    def _update_fps_label(self):
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_t
        if elapsed >= 1.0:
            self._fps_frame_count = 0
            self._fps_last_t = now

    def clear(self):
        self._ring.clear()
        self.sample_counter = 0
        self.curve.setData([], [])
        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()

    def shutdown(self) -> bool:
        return True

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        self.clear()
        if sample_rate > 0:
            self.sampling_rate = float(sample_rate)
            self._ring = RingBuffer(self.sampling_rate, 1, duration_sec=300)
        self.set_receiving_state(True)

    def set_receiving_state(self, receiving: bool):
        self.is_receiving = bool(receiving)
