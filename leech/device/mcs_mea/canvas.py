"""Canvas for the MCS MEA tab."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor


def _activity_brush(value: float, maximum: float) -> QColor:
    norm = 0.0 if maximum <= 0 else max(0.0, min(1.0, float(value) / float(maximum)))
    return QColor.fromHsvF(0.66 * (1.0 - norm), 0.9, 0.55 + 0.45 * norm)


class McsMeaCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        self.mea_plot = self.glw.addPlot(row=0, col=0, rowspan=2)
        self.mea_plot.setAspectLocked(True)
        self.mea_plot.setLabel("left", "MEA")
        self.mea_plot.hideAxis("bottom")
        self.mea_plot.hideAxis("left")

        self.signal_plot = self.glw.addPlot(row=0, col=1)
        self.signal_plot.setLabel("left", "uV")
        self.signal_plot.setLabel("bottom", "time (s)")
        self.signal_plot.showGrid(x=True, y=True, alpha=0.2)

        self.spike_plot = self.glw.addPlot(row=1, col=1)
        self.spike_plot.setLabel("left", "spikes")
        self.spike_plot.setLabel("bottom", "time (s)")
        self.spike_plot.showGrid(x=True, y=True, alpha=0.2)

        self.mea_scatter = pg.ScatterPlotItem(size=18, pxMode=True)
        self.mea_plot.addItem(self.mea_scatter)
        self.signal_curve = self.signal_plot.plot(pen=pg.mkPen("#4fc3f7", width=1))
        self.spike_bars = pg.BarGraphItem(x=[], height=[], width=4.8, brush="#ffb74d")
        self.spike_plot.addItem(self.spike_bars)
        self._xs = np.empty(0)
        self._ys = np.empty(0)

    def set_geometry(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        self._xs = np.asarray(xs, dtype=float)
        self._ys = np.asarray(ys, dtype=float)
        self.mea_scatter.setData(self._xs, self._ys, brush=pg.mkBrush("#546e7a"), pen=None)
        if self._xs.size:
            self.mea_plot.setXRange(float(self._xs.min()) - 0.7, float(self._xs.max()) + 0.7, padding=0)
            self.mea_plot.setYRange(float(self._ys.min()) - 0.7, float(self._ys.max()) + 0.7, padding=0)

    def set_mea_activity(self, values: Sequence[float]) -> None:
        values = np.asarray(values, dtype=float)
        if values.size != self._xs.size:
            return
        maximum = float(np.max(values)) if values.size else 0.0
        brushes = [_activity_brush(v, maximum) for v in values]
        self.mea_scatter.setData(self._xs, self._ys, brush=brushes, pen=None)

    def set_signal(self, t: np.ndarray, y: np.ndarray) -> None:
        self.signal_curve.setData(t, y)

    def set_spikes(self, centres: np.ndarray, heights: np.ndarray) -> None:
        self.spike_bars.setOpts(x=centres, height=heights, width=4.8)

    def clear(self) -> None:
        self.signal_curve.setData([], [])
        self.spike_bars.setOpts(x=[], height=[], width=4.8)
        self.mea_scatter.setData([], [])
