import sys

from PyQt5 import QtCore, QtWidgets

from screens.connect_screen import ConnectScreen
from screens.plot_screen import PlotScreen
from workers.rhx_worker import RHXWorker

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.rhx_worker = None

        # Set up main window
        self.setWindowTitle("Intan RHX Device Monitor")

        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar.addAction("Connect", lambda: self.set_screen("connect"))
        toolbar.addAction("Plot", lambda: self.set_screen("plot"))
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        # Screens defitions
        self.connect_screen = ConnectScreen()
        self.connect_screen.connection_request_signal.connect(self._on_connection_requested)
        self.stack.addWidget(self.connect_screen)

        self.plot_screen = PlotScreen()
        self.plot_screen.toggle_receiving_request_signal.connect(self._on_toggle_receiving_requested)
        self.plot_screen.save_disconnect_request_signal.connect(self._on_save_disconnect_requested)
        self.plot_screen.marker_request_signal.connect(self._on_marker_requested)
        self.stack.addWidget(self.plot_screen)

        self.set_screen("connect")

    def set_screen(self, screen_name):
        if screen_name == "connect":
            self.stack.setCurrentWidget(self.connect_screen)
            self.setFixedSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
            connect_size = self.connect_screen.sizeHint()
            self.stack.setMinimumSize(self.connect_screen.minimumSizeHint())
            self.stack.setMaximumSize(QtCore.QSize(16777215, 16777215))
            if self.isMaximized() or self.isFullScreen():
                self.showNormal()
            self.resize(connect_size)
        elif screen_name == "plot":
            self.stack.setCurrentWidget(self.plot_screen)
            self.setFixedSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
            self.stack.setMinimumSize(0, 0)
            self.stack.setMaximumSize(QtCore.QSize(16777215, 16777215))
            self.setMinimumSize(800, 600)
            self.setMaximumSize(QtCore.QSize(16777215, 16777215))
            self.showMaximized()
        else:
            raise ValueError(f"Unknown screen name: {screen_name}")

    def _on_connection_requested(self, host, command_port, data_port, sample_rate, project_name, port, channel):
        self.connect_screen.set_busy(True)
        if self.rhx_worker is not None:
            self.rhx_worker.stop()
        self.rhx_worker = RHXWorker(host, command_port, data_port, sample_rate, project_name, port, channel)
        self.rhx_worker.connection_request_result_signal.connect(self._on_connection_result)
        self.rhx_worker.start()

    def _on_connection_result(self, successful, message=None):
        self.connect_screen.set_busy(False)
        if successful:
            self.set_screen("plot")
            self.plot_screen.set_connection_details(
                host=self.rhx_worker.host,
                command_port=self.rhx_worker.command_port,
                data_port=self.rhx_worker.data_port,
                sample_rate=self.rhx_worker.sample_rate,
                project_name=self.rhx_worker.project_name,
            )
            self.rhx_worker.data_received_signal.connect(self.plot_screen._on_data_received)
            self.rhx_worker.marker_added_signal.connect(self.plot_screen.add_marker)
            self.rhx_worker.acquisition_state_signal.connect(self._on_acquisition_state_changed)
        else:
            self.connect_screen.set_status_message(f"Connection failed! Make sure the Intan RHX program is running and has TCP server running.\n\n{message}", error=True)

    def _on_toggle_receiving_requested(self, should_receive):
        if self.rhx_worker is None:
            return
        if should_receive:
            self.rhx_worker.resume_receiving()
        else:
            self.rhx_worker.pause_receiving()

    def _on_marker_requested(self):
        if self.rhx_worker is None:
            return
        self.rhx_worker.request_marker()

    def _on_save_disconnect_requested(self):
        if self.rhx_worker is None:
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Save and Disconnect",
            "Save recorded output data and disconnect now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            if self.rhx_worker is not None:
                self.rhx_worker.stop()
                self.rhx_worker = None
            self.plot_screen.clear_project_buffers()
            self.set_screen("connect")

    def _on_acquisition_state_changed(self, state):
        if state == "running":
            self.plot_screen.set_receiving_state(True)
            self.plot_screen.set_connection_status("Receiving")
        elif state == "waiting":
            self.plot_screen.set_receiving_state(True)
            self.plot_screen.set_connection_status("Waiting for transmission...")
        elif state in ("paused", "stopped"):
            self.plot_screen.set_receiving_state(False)
            if state == "paused":
                self.plot_screen.set_connection_status("Paused")
        elif state == "connection_lost":
            self.plot_screen.set_receiving_state(False)
            self.plot_screen.set_connection_status("Connection lost during receive")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())