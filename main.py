import sys

from PyQt5 import QtCore, QtWidgets, QtGui

from screens.connect_screen import ConnectDialog
from screens.plot_screen import PlotScreen
from workers.rhx_worker import RHXWorker
from state_manager import StateManager, AppState

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.rhx_worker = None
        self.connect_dialog = None
        
        # State machine integration
        self.state_manager = StateManager.get_instance()
        
        # Map states to UI button configurations
        # Format: {AppState: {button_name: enabled, ...}}
        self.UI_STATE_CONFIG = {
            AppState.IDLE: {
                "connect": True,
                "toggle_receive": False,
                "marker": False,
                "save": False,
                "snapshot": False,
            },
            AppState.CONNECTING: {
                "connect": False,
                "toggle_receive": False,
                "marker": False,
                "save": False,
                "snapshot": False,
            },
            AppState.CONNECTED: {
                "connect": False,
                "toggle_receive": True,
                "marker": False,
                "save": False,
                "snapshot": True,
            },
            AppState.STREAMING: {
                "connect": False,
                "toggle_receive": True,
                "marker": True,
                "save": True,
                "snapshot": True,
            },
            AppState.WAITING_FOR_DATA: {
                "connect": False,
                "toggle_receive": True,
                "marker": True,
                "save": True,
                "snapshot": True,
            },
            AppState.PAUSED: {
                "connect": False,
                "toggle_receive": True,
                "marker": False,
                "save": True,
                "snapshot": True,
            },
            AppState.CONNECTION_LOST: {
                "connect": False,
                "toggle_receive": False,
                "marker": False,
                "save": False,
                "snapshot": False,
            },
            AppState.DISCONNECTING: {
                "connect": False,
                "toggle_receive": False,
                "marker": False,
                "save": False,
                "snapshot": False,
            },
        }

        # Set up main window
        self.setWindowTitle("Intan RHX Device Monitor")

        self.plot_screen = PlotScreen()
        self.setCentralWidget(self.plot_screen)
        self.setMinimumSize(800, 600)

        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.connect_action = self.toolbar.addAction("Connect")
        self.connect_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DriveNetIcon))
        self.connect_action.triggered.connect(self.open_connect_dialog)

        self.toggle_receiving_action = self.toolbar.addAction("Start receiving")
        self.toggle_receiving_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.toggle_receiving_action.triggered.connect(self._on_toggle_receiving_requested)

        self.toolbar.addSeparator()

        self.marker_action = self.toolbar.addAction("Add marker")
        self.marker_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowDown))
        self.marker_action.triggered.connect(self._on_marker_requested)

        self.snapshot_menu = QtWidgets.QMenu(self)
        self.snapshot_psd_action = self.snapshot_menu.addAction("Take PSD snapshot")
        self.snapshot_psd_action.triggered.connect(self._on_snapshot_psd_requested)
        self.snapshot_waveform_action = self.snapshot_menu.addAction("Take waveform snapshot")
        self.snapshot_waveform_action.triggered.connect(self._on_snapshot_waveform_requested)
        self.snapshot_menu.addSeparator()
        self.clear_snapshots_action = self.snapshot_menu.addAction("Clear snapshots")
        self.clear_snapshots_action.triggered.connect(self._on_clear_snapshots_requested)

        self.snapshot_button = QtWidgets.QToolButton(self)
        self.snapshot_button.setText("Snapshots")
        self.snapshot_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.snapshot_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.snapshot_button.setMenu(self.snapshot_menu)
        self.snapshot_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DesktopIcon))
        self.toolbar.addWidget(self.snapshot_button)

        self.toolbar.addSeparator()
        self.save_disconnect_action = self.toolbar.addAction("Save and Disconnect")
        self.save_disconnect_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_disconnect_action.triggered.connect(self._on_save_disconnect_requested)

        spacer = QtWidgets.QWidget(self)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        self.toolbar.addWidget(self.plot_screen.connection_details_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.plot_screen.fps_label)

        # Connect state machine signals
        self.state_manager.get_state_changed_signal().connect(self._on_state_changed)
        
        # Initialize UI based on current state
        self._update_ui_from_state()
        
        self.showMaximized()
        QtCore.QTimer.singleShot(0, self.open_connect_dialog)

    def _update_ui_from_state(self):
        """
        Update all UI elements based on the current state.
        Uses UI_STATE_CONFIG to determine button enable states.
        """
        current_state = self.state_manager.get_current_state()
        config = self.UI_STATE_CONFIG.get(current_state, {})
        
        # Apply button enable states from config
        self.connect_action.setEnabled(config.get("connect", False))
        self.toggle_receiving_action.setEnabled(config.get("toggle_receive", False))
        self.marker_action.setEnabled(config.get("marker", False))
        self.save_disconnect_action.setEnabled(config.get("save", False))
        self.snapshot_button.setEnabled(config.get("snapshot", False))
        
        # Update button text based on state
        if current_state in (AppState.STREAMING, AppState.WAITING_FOR_DATA):
            self.toggle_receiving_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self.toggle_receiving_action.setText("Stop receiving")
            self.plot_screen.set_receiving_state(True)
            self.plot_screen.set_connection_status("Receiving" if current_state == AppState.STREAMING else "Waiting for transmission...")
        elif current_state == AppState.PAUSED:
            self.toggle_receiving_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.toggle_receiving_action.setText("Start receiving")
            self.plot_screen.set_receiving_state(False)
            self.plot_screen.set_connection_status("Paused")
        elif current_state == AppState.CONNECTED:
            self.toggle_receiving_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.toggle_receiving_action.setText("Start receiving")
            self.plot_screen.set_receiving_state(False)
        elif current_state == AppState.CONNECTION_LOST:
            self.plot_screen.set_receiving_state(False)
            self.plot_screen.set_connection_status("Connection lost during receive")
        elif current_state == AppState.IDLE:
            self.toggle_receiving_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.toggle_receiving_action.setText("Start receiving")
            self.plot_screen.set_receiving_state(False)
    
    def _on_state_changed(self, new_state: AppState):
        """Called when state machine transitions to a new state."""
        self._update_ui_from_state()
        print(f"[UI] State changed to: {new_state.value}")

    def _ensure_connect_dialog(self):
        if self.connect_dialog is None:
            self.connect_dialog = ConnectDialog(self)
            self.connect_dialog.connection_request_signal.connect(self._on_connection_requested)

    def open_connect_dialog(self):
        self._ensure_connect_dialog()
        if self.connect_dialog.isVisible():
            self.connect_dialog.raise_()
            self.connect_dialog.activateWindow()
            return
        self.connect_dialog.reset_status_message()
        self.connect_dialog.show()

    def _on_connection_requested(
        self,
        host,
        command_port,
        data_port,
        sample_rate,
        project_name,
        port,
        channel,
        psd_buffer_sec,
        waveform_buffer_sec,
        spike_bin_sec,
    ):
        self._ensure_connect_dialog()
        
        # Transition to CONNECTING state
        self.state_manager.request_connect()
        self.connect_dialog.set_busy(True)
        
        if self.rhx_worker is not None:
            self.rhx_worker.stop()

        self.plot_screen.configure_processing_settings(
            psd_buffer_sec=psd_buffer_sec,
            waveform_buffer_sec=waveform_buffer_sec,
            spike_bin_sec=spike_bin_sec,
        )
        
        self.rhx_worker = RHXWorker(host, command_port, data_port, sample_rate, project_name, port, channel)
        self.rhx_worker.connection_request_result_signal.connect(self._on_connection_result)
        # Connect state/data signals before start so initial worker emits are not missed.
        self.rhx_worker.data_received_signal.connect(self.plot_screen._on_data_received)
        self.rhx_worker.marker_added_signal.connect(self.plot_screen.add_marker)
        self.rhx_worker.acquisition_state_signal.connect(self._on_acquisition_state_changed)
        self.rhx_worker.start()

    def _on_connection_result(self, successful, message=None):
        if self.connect_dialog is not None:
            self.connect_dialog.set_busy(False)
        
        if successful:
            # Transition to CONNECTED state
            self.state_manager.connection_succeeded()
            if self.connect_dialog is not None:
                self.connect_dialog.accept()
            self.plot_screen.set_connection_details(
                host=self.rhx_worker.host,
                command_port=self.rhx_worker.command_port,
                data_port=self.rhx_worker.data_port,
                sample_rate=self.rhx_worker.sample_rate,
                project_name=self.rhx_worker.project_name,
            )
        else:
            # Transition back to IDLE state
            self.state_manager.connection_failed()
            QtWidgets.QMessageBox.critical(
                self,
                "Connection Failed",
                "Connection failed! Make sure the Intan RHX program is running and has TCP server running.\n\n"
                f"{message}",
            )

    def _on_toggle_receiving_requested(self):
        """Handle start/stop receiving button press."""
        if self.rhx_worker is None:
            return
        
        current_state = self.state_manager.get_current_state()
        
        # If currently receiving (STREAMING or WAITING_FOR_DATA), pause
        if current_state in (AppState.STREAMING, AppState.WAITING_FOR_DATA):
            if not self.state_manager.user_pause():
                print("[UI] Failed to pause receiving")
                return
        # If in PAUSED state, resume using user_resume
        elif current_state == AppState.PAUSED:
            if not self.state_manager.user_resume():
                print("[UI] Failed to resume from paused state")
                return
        # If in CONNECTED state, start streaming using request_stream
        elif current_state == AppState.CONNECTED:
            if not self.state_manager.request_stream():
                print("[UI] Failed to start streaming")
                return

    def _on_marker_requested(self):
        if self.rhx_worker is None:
            return
        self.rhx_worker.request_marker()

    def _on_snapshot_psd_requested(self):
        if not self.plot_screen.take_psd_snapshot():
            QtWidgets.QMessageBox.information(
                self,
                "Snapshot unavailable",
                "No PSD data available yet. Start receiving and wait for PSD updates.",
            )

    def _on_snapshot_waveform_requested(self):
        if not self.plot_screen.take_waveform_snapshot():
            QtWidgets.QMessageBox.information(
                self,
                "Snapshot unavailable",
                "No average waveform available yet. Start receiving and wait for spike waveform updates.",
            )

    def _on_clear_snapshots_requested(self):
        self.plot_screen.clear_snapshots()

    def _on_save_disconnect_requested(self):
        """Handle save and disconnect button press."""
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
            # Transition to DISCONNECTING state
            self.state_manager.request_disconnect()
            
            # Perform cleanup
            if self.rhx_worker is not None:
                self.rhx_worker.stop()
                self.rhx_worker = None
            
            self.plot_screen.clear_project_buffers()
            
            # Complete the disconnect and return to IDLE
            self.state_manager.disconnect_complete()
            self.open_connect_dialog()

    def _on_acquisition_state_changed(self, state):
        """
        Called when RHXWorker acquisition state changes.
        Maps worker states to FSM transitions.
        """
        if state == "running":
            # Data is flowing
            self.state_manager.data_arrived()
        elif state == "waiting":
            # No data yet, but still trying to stream
            self.state_manager.no_data_available()
        elif state == "paused":
            # User paused or naturally paused
            self.state_manager.user_pause()
        elif state == "stopped":
            # Streaming stopped normally
            pass
        elif state == "connection_lost":
            # Device disconnected unexpectedly
            self.state_manager.device_disconnected()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())