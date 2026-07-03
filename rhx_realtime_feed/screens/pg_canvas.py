import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

from rhx_realtime_feed.workers.processing_worker import (
    PSD_BUFFER_SEC,
    SPIKE_BIN_SEC,
    WAVEFORM_BUFFER_SEC,
    PSD_YLIM_MIN,
    PSD_YLIM_MAX,
)
from rhx_realtime_feed.screens.plot_helpers import WAVEFORM_YLIM_ABS_UV


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

        self.glw.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)

        self.raw_plot = self.glw.addPlot(row=0, col=0, colspan=3)
        self.raw_plot.setTitle("Raw signal")
        self.raw_plot.setLabel('left',   'U [uV]')
        self.raw_plot.setLabel('bottom', 't [s]')
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        self.raw_plot.setDownsampling(auto=True, mode='peak')
        self.raw_plot.setClipToView(True)
        self.raw_plot.setMouseEnabled(x=True, y=True)

        self.psd_plot = self.glw.addPlot(row=1, col=0)
        self.psd_plot.setTitle(f"Power spectrum (last {PSD_BUFFER_SEC}s)")
        self.psd_plot.setLabel('left',   'P [dB]')
        self.psd_plot.setLabel('bottom', 'f [Hz]')
        self.psd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.psd_plot.setMouseEnabled(x=False, y=False)
        self.psd_plot.setYRange(PSD_YLIM_MIN, PSD_YLIM_MAX, padding=0)

        self.spike_plot = self.glw.addPlot(row=1, col=1)
        self.spike_plot.setTitle(f"Spike counts ({SPIKE_BIN_SEC}s bins)")
        self.spike_plot.setLabel('left',   'Count')
        self.spike_plot.setLabel('bottom', 't [min]')
        self.spike_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spike_plot.setMouseEnabled(x=True, y=True)

        self.wf_plot = self.glw.addPlot(row=1, col=2)
        self.wf_plot.setTitle(f"Averaged spike waveform (last {WAVEFORM_BUFFER_SEC}s)")
        self.wf_plot.setLabel('left',   'U [uV]')
        self.wf_plot.setLabel('bottom', 't [ms]')
        self.wf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wf_plot.setMouseEnabled(x=False, y=True)
        self.wf_plot.setYRange(-WAVEFORM_YLIM_ABS_UV, WAVEFORM_YLIM_ABS_UV, padding=0)

        self.raw_curve   = self.raw_plot.plot(  pen=pg.mkPen("#1D20ED", width=1))
        self.psd_curve   = self.psd_plot.plot(  pen=pg.mkPen("#f62d2d", width=2))
        self.spike_curve = self.spike_plot.plot(pen=pg.mkPen("#20C814", width=2))
        self.wf_curve    = self.wf_plot.plot(   pen=pg.mkPen("#e840b3", width=2))

        self.wf_upper = self.wf_plot.plot(pen=None)
        self.wf_lower = self.wf_plot.plot(pen=None)
        self.wf_fill  = pg.FillBetweenItem(
            self.wf_upper, self.wf_lower,
            brush=pg.mkBrush(80, 200, 120, 50),
        )
        self.wf_plot.addItem(self.wf_fill)

        self._marker_lines    = []
        self._marker_pen      = pg.mkPen(color='crimson', style=QtCore.Qt.DashLine, width=1)
        self._last_marker_set = []
        self._marker_labels = []
        self._spike_marker_lines = []
        self._last_spike_marker_set = []
        self._spike_marker_labels = []
