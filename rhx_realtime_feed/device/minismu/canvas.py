import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui


class SmuCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.glw.setRenderHint(QtGui.QPainter.Antialiasing)
        self.glw.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        layout.addWidget(self.glw)

        self.voltage_plot = self.glw.addPlot(row=0, col=0)
        self.voltage_plot.setTitle("Voltage")
        self.voltage_plot.setLabel('left', 'U [V]')
        self.voltage_plot.setLabel('bottom', 't [s]')
        self.voltage_plot.showGrid(x=True, y=True, alpha=0.3)
        self.voltage_plot.setDownsampling(auto=True, mode='peak')
        self.voltage_plot.setClipToView(True)
        self.voltage_plot.setMouseEnabled(x=True, y=True)

        self.current_plot = self.glw.addPlot(row=1, col=0)
        self.current_plot.setTitle("Current")
        self.current_plot.setLabel('left', 'I [A]')
        self.current_plot.setLabel('bottom', 't [s]')
        self.current_plot.showGrid(x=True, y=True, alpha=0.3)
        self.current_plot.setDownsampling(auto=True, mode='peak')
        self.current_plot.setClipToView(True)
        self.current_plot.setMouseEnabled(x=True, y=True)

        self.voltage_curve = self.voltage_plot.plot(pen=pg.mkPen("#1D20ED", width=1))
        self.current_curve = self.current_plot.plot(pen=pg.mkPen("#f62d2d", width=1))
