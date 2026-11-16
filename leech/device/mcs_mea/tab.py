"""MCS MEA tab: electrode map, selected-channel trace, spike counts."""

from __future__ import annotations

import queue
import time
from collections import deque
from typing import Any, List, Optional

import numpy as np
from PyQt5 import QtWidgets

from leech.device.background_worker import BackgroundWorker
from leech.gui.tabs.base import DataTab

from .canvas import McsMeaCanvas
from .layout import channel_names, electrode_positions
from .processing import StreamingSpikeDetector


DISPLAY_WINDOW_SEC = 10.0
SPIKE_WINDOW_SEC = 60.0
SPIKE_BIN_SEC = 5.0
RENDER_HZ = 30.0
MAX_TRACE_POINTS = 2500


class McsMeaTab(DataTab):
    # The MCS tab keeps only a selected-channel trace. This tells TabHost not
    # to replace _ring with a 300 s all-channel buffer (several GB at 120 ch).
    owns_ring = True

    def __init__(
        self,
        sample_rate: float = 20000.0,
        num_channels: int = 1,
        channel_labels: Optional[List[str]] = None,
        parent=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self.sampling_rate = float(sample_rate or 20000.0)
        self._num_channels = max(1, int(num_channels or 1))
        self._channel_labels = list(channel_labels or [])
        self._selected = 0
        self._sample_counter = 0
        self._trace: deque = deque()
        self._trace_total = 0
        self._spike_times: List[deque] = []
        self._rates = np.zeros(self._num_channels, dtype=np.float32)
        self._rms = np.zeros(self._num_channels, dtype=np.float32)
        self._pending: queue.Queue = queue.Queue(maxsize=8)
        self._detector: Optional[StreamingSpikeDetector] = None
        self._session_id = 0
        self._xs = np.empty(0, dtype=np.float32)
        self._ys = np.empty(0, dtype=np.float32)
        self._last_render = 0.0
        self._dropped_chunks = 0
        self._build_ui()
        self._proc_worker = BackgroundWorker(self)
        self._proc_worker.result_ready.connect(self._on_processed)
        self._proc_worker.start()
        self._resize(self._num_channels, self._channel_labels)

    # ── UI ─────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("MCS MEA2100")
        self._title.setStyleSheet("font-weight: 600;")
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(QtWidgets.QLabel("Channel"))
        self._combo = QtWidgets.QComboBox()
        self._combo.setMinimumWidth(110)
        self._combo.currentIndexChanged.connect(self._select_channel)
        top.addWidget(self._combo)
        layout.addLayout(top)

        self.canvas = McsMeaCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.canvas.mea_plot.scene().sigMouseClicked.connect(self._on_mea_click)

    def _resize(self, num_channels: int, channel_labels: Optional[List[str]] = None) -> None:
        count = max(1, int(num_channels or 1))
        self._num_channels = count
        labels = list(channel_labels or [])
        if len(labels) != count:
            labels = channel_names(count) if count in (60, 120) else [
                f"CH{i + 1:03d}" for i in range(count)
            ]
        self._channel_labels = labels
        self._selected = min(self._selected, count - 1)
        self._session_id += 1
        self._pending = queue.Queue(maxsize=8)
        self._spike_times = [deque(maxlen=5000) for _ in range(count)]
        self._rates = np.zeros(count, dtype=np.float32)
        self._rms = np.zeros(count, dtype=np.float32)
        self._trace.clear()
        self._trace_total = 0
        self._sample_counter = 0
        self._detector = None
        try:
            self._detector = StreamingSpikeDetector(self.sampling_rate, count)
        except Exception:
            self._detector = None

        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(labels)
        self._combo.setCurrentIndex(self._selected)
        self._combo.blockSignals(False)

        self.canvas.clear()
        if count in (60, 120):
            self._xs, self._ys = electrode_positions(count)
            self.canvas.set_geometry(self._xs, self._ys)
        else:
            self._xs = np.empty(0, dtype=np.float32)
            self._ys = np.empty(0, dtype=np.float32)

    # ── Data path ──────────────────────────────────────────────────────
    def on_data(self, chunk: np.ndarray) -> None:
        t0 = time.perf_counter()
        arr = np.asarray(chunk)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return
        if arr.shape[0] != self._num_channels:
            self._resize(arr.shape[0])

        start = self._sample_counter
        selected = arr[self._selected].astype(np.float32, copy=True)
        self._trace.append(selected)
        self._trace_total += selected.size
        max_samples = max(1, int(self.sampling_rate * DISPLAY_WINDOW_SEC))
        while self._trace_total > max_samples and self._trace:
            self._trace_total -= self._trace.popleft().size
        self._sample_counter += arr.shape[1]

        try:
            self._pending.put_nowait((self._session_id, arr.astype(np.float32, copy=True), start))
        except queue.Full:
            self._dropped_chunks += 1
        self._proc_worker.schedule(self._drain_pending)

        self._tel_chunks += 1
        self._tel_samples += arr.shape[1]
        self._tel_ingest_ms_total += (time.perf_counter() - t0) * 1000.0

    def _drain_pending(self):
        detector = self._detector
        if detector is None:
            return []
        results = []
        while True:
            try:
                session, chunk, start = self._pending.get_nowait()
            except queue.Empty:
                break
            results.append((session, start, detector.process(chunk)))
        return results

    def _on_processed(self, result: Any) -> None:
        if isinstance(result, Exception) or not result:
            return
        for session, start, processed in result:
            if session != self._session_id:
                continue
            spikes = processed["spikes"]
            samples = max(1, int(processed["samples"]))
            duration = samples / max(self.sampling_rate, 1e-9)
            counts = np.asarray([len(s) for s in spikes], dtype=np.float32)
            self._rates = 0.8 * self._rates + 0.2 * (counts / max(duration, 1e-9))
            self._rms = 0.8 * self._rms + 0.2 * np.asarray(processed["rms"], dtype=np.float32)
            for ch, idx in enumerate(spikes):
                if idx.size:
                    self._spike_times[ch].extend((start + idx).tolist())
        self.request_render()

    # ── Interaction ────────────────────────────────────────────────────
    def _select_channel(self, index: int) -> None:
        if 0 <= index < self._num_channels and index != self._selected:
            self._selected = int(index)
            self._trace.clear()
            self._trace_total = 0
            self.request_render()

    def _on_mea_click(self, event) -> None:
        if not self._xs.size:
            return
        pos = event.scenePos()
        if not self.canvas.mea_plot.sceneBoundingRect().contains(pos):
            return
        point = self.canvas.mea_plot.vb.mapSceneToView(pos)
        dist2 = (self._xs - point.x()) ** 2 + (self._ys - point.y()) ** 2
        idx = int(np.argmin(dist2))
        if dist2[idx] <= 1.5:
            self._combo.setCurrentIndex(idx)

    # ── Rendering ──────────────────────────────────────────────────────
    def render(self) -> None:
        now = time.perf_counter()
        if now - self._last_render < (1.0 / RENDER_HZ):
            return
        self._last_render = now
        t0 = time.perf_counter()

        if self._trace:
            y = np.concatenate(list(self._trace))
            count = y.size
            x = (self._sample_counter - count + np.arange(count)) / max(self.sampling_rate, 1e-9)
            if count > MAX_TRACE_POINTS:
                step = int(np.ceil(count / MAX_TRACE_POINTS))
                x = x[::step]
                y = y[::step]
            self.canvas.set_signal(x, y)

        window = int(SPIKE_WINDOW_SEC * self.sampling_rate)
        start = max(0, self._sample_counter - window)
        times = np.asarray(self._spike_times[self._selected], dtype=np.int64)
        if times.size:
            times = times[times >= start]
        bins = np.linspace(start, max(start + 1, self._sample_counter), 13)
        heights, edges = np.histogram(times, bins=bins)
        centres = ((edges[:-1] + edges[1:]) * 0.5 - self._sample_counter) / max(self.sampling_rate, 1e-9)
        self.canvas.set_spikes(centres, heights)
        self.canvas.spike_plot.setXRange(-SPIKE_WINDOW_SEC, 0, padding=0)

        values = self._rates
        if float(np.max(values)) <= 0:
            values = self._rms
        self.canvas.set_mea_activity(values)

        self._tel_render_calls += 1
        self._tel_render_ms_total += (time.perf_counter() - t0) * 1000.0
        self._tel_raw_renders += 1
        self._emit_telemetry_if_due("mcs_mea")

    # ── Lifecycle ──────────────────────────────────────────────────────
    def clear(self) -> None:
        self._session_id += 1
        self._trace.clear()
        self._trace_total = 0
        self._sample_counter = 0
        self._spike_times = [deque(maxlen=5000) for _ in range(self._num_channels)]
        self._rates = np.zeros(self._num_channels, dtype=np.float32)
        self._rms = np.zeros(self._num_channels, dtype=np.float32)
        self._pending = queue.Queue(maxsize=8)
        self._detector = None
        try:
            self._detector = StreamingSpikeDetector(self.sampling_rate, self._num_channels)
        except Exception:
            self._detector = None
        self.canvas.clear()
        self.update()

    def shutdown(self) -> bool:
        self._proc_worker.stop(timeout_ms=3000)
        return True
