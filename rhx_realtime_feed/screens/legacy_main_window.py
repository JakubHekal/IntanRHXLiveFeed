import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets, QtGui

from rhx_realtime_feed import __version__
from rhx_realtime_feed.screens.connect_screen import ConnectDialog
from rhx_realtime_feed.updater import UpdateCheckThread, UpdateInfo
from rhx_realtime_feed.screens.marker_dialog import MarkerDialog
from rhx_realtime_feed.screens.plot_screen import PlotScreen
from rhx_realtime_feed.device import IntanRHXDevice
from rhx_realtime_feed.workers.rhx_worker import RHXWorker
from rhx_realtime_feed.workers.chunk_writer import ChunkWriter
from rhx_realtime_feed.workers.rhx_worker import RAW_CHUNK_SEC, CSV_FILE_BUFFER_BYTES, CSV_FLUSH_INTERVAL_SEC
from rhx_realtime_feed.state_manager import StateManager, AppState
from rhx_realtime_feed.telemetry_logger import set_telemetry_file, append_telemetry_line


class LegacyMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.rhx_worker = None
        self.connect_dialog = None
        self.marker_dialog = None
        self._handling_connection_lost = False
        self._waiting_connection_lost_worker = False
        self._update_thread = None

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
        self.setWindowTitle("RHX Realtime Feed")

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

        self.marker_manager_action = self.toolbar.addAction("Markers")
        self.marker_manager_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.marker_manager_action.triggered.connect(self._on_open_marker_manager)

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
        self.toolbar.addWidget(self.snapshot_button)

        self.toolbar.addSeparator()
        self.save_disconnect_action = self.toolbar.addAction("Save and Disconnect")
        self.save_disconnect_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_disconnect_action.triggered.connect(self._on_save_disconnect_requested)

        spacer = QtWidgets.QWidget(self)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        self.toolbar.addSeparator()
        self.check_update_action = self.toolbar.addAction("Check for Updates")
        self.check_update_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.check_update_action.triggered.connect(self._check_for_updates_manual)

        self.toolbar.addWidget(self.plot_screen.connection_details_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.plot_screen.fps_label)

        # Connect state machine signals
        self.state_manager.get_state_changed_signal().connect(self._on_state_changed)

        # Initialize UI based on current state
        self._update_ui_from_state()

        self.showMaximized()
        QtCore.QTimer.singleShot(0, self.open_connect_dialog)
        QtCore.QTimer.singleShot(3000, self._check_for_updates_auto)

    def _check_for_updates_auto(self):
        settings = QtCore.QSettings("RHX", "RealtimeFeed")
        last_check = settings.value("update/last_check_time", 0, type=int)
        if time.time() - last_check < 86400:
            return
        self._run_update_check()

    def _check_for_updates_manual(self):
        self._run_update_check(force_show=True)

    def _run_update_check(self, force_show=False):
        if self._update_thread is not None and self._update_thread.isRunning():
            if force_show:
                QtWidgets.QMessageBox.information(
                    self, "Checking", "Update check is already in progress."
                )
            return

        self._update_force_show = force_show
        self._update_thread = UpdateCheckThread(__version__, self)
        self._update_thread.result_ready.connect(self._on_update_result)
        self._update_thread.start()

    def _on_update_result(self, result):
        self._update_thread = None
        force_show = getattr(self, "_update_force_show", False)

        if isinstance(result, UpdateInfo):
            QtCore.QSettings("RHX", "RealtimeFeed").setValue(
                "update/last_check_time", int(time.time())
            )
            if result.available:
                msg = (
                    f"Version {result.latest_version} is now available "
                    f"(you have {result.current_version}).\n\n"
                    "Download the latest release from GitHub?"
                )
                reply = QtWidgets.QMessageBox.question(
                    self, "Update Available", msg,
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    url = result.download_url or result.release_url
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
            elif force_show:
                QtWidgets.QMessageBox.information(
                    self, "Up to Date",
                    f"You are running the latest version ({result.current_version}).",
                )
        elif force_show:
            QtWidgets.QMessageBox.warning(
                self, "Update Check Failed",
                "Could not check for updates.\n\n"
                "Make sure you have an internet connection and try again.",
            )

    def _update_ui_from_state(self):
        current_state = self.state_manager.get_current_state()
        config = self.UI_STATE_CONFIG.get(current_state, {})

        self.connect_action.setEnabled(config.get("connect", False))
        self.toggle_receiving_action.setEnabled(config.get("toggle_receive", False))
        self.marker_action.setEnabled(config.get("marker", False))
        self.save_disconnect_action.setEnabled(config.get("save", False))
        self.snapshot_button.setEnabled(config.get("snapshot", False))
        self.marker_manager_action.setEnabled(
            current_state in (AppState.CONNECTED, AppState.STREAMING, AppState.WAITING_FOR_DATA, AppState.PAUSED)
        )

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
        self._update_ui_from_state()
        line = f"[UI] State changed to: {new_state.value}"
        print(line)
        append_telemetry_line(line)
        if new_state == AppState.CONNECTION_LOST and not self._handling_connection_lost:
            QtCore.QTimer.singleShot(0, self._handle_connection_lost)

    def _ensure_connect_dialog(self):
        if self.connect_dialog is None:
            self.connect_dialog = ConnectDialog(self)
            self.connect_dialog.connection_request_signal.connect(self._on_connection_requested)

    def _stop_rhx_worker(self, timeout_ms: int = 3000) -> bool:
        if self.rhx_worker is None:
            return True
        worker = self.rhx_worker
        stopped = worker.stop(timeout_ms=timeout_ms)
        if not stopped:
            print("[UI] RHX worker did not stop cleanly")
            return False
        self.rhx_worker = None
        return True

    def _handle_connection_lost(self):
        if self._handling_connection_lost:
            return
        self._handling_connection_lost = True
        if self.state_manager.get_current_state() == AppState.CONNECTION_LOST:
            self.state_manager.request_disconnect()

        worker = self.rhx_worker
        if worker is None:
            self._finalize_connection_lost_cleanup()
            return

        if self._stop_rhx_worker(timeout_ms=4000):
            self._finalize_connection_lost_cleanup()
            return

        if not self._waiting_connection_lost_worker:
            self._waiting_connection_lost_worker = True
            try:
                worker.finished.connect(self._on_connection_lost_worker_finished)
            except Exception:
                pass

        QtWidgets.QMessageBox.warning(
            self,
            "Connection Lost",
            "Connection was lost and worker shutdown is still finishing in the background. Cleanup will complete automatically.",
        )

    def _on_connection_lost_worker_finished(self):
        self._waiting_connection_lost_worker = False
        self.rhx_worker = None
        self._finalize_connection_lost_cleanup()

    def _finalize_connection_lost_cleanup(self):
        if self.marker_dialog is not None:
            self.marker_dialog.hide()

        self.plot_screen.clear_project_buffers()

        if self.state_manager.get_current_state() == AppState.DISCONNECTING:
            self.state_manager.disconnect_complete()

        QtWidgets.QMessageBox.warning(
            self,
            "Connection Lost",
            "The stream connection was lost. The session was safely reset and is ready to reconnect.",
        )
        set_telemetry_file("")
        self.open_connect_dialog()
        self._handling_connection_lost = False

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
        project_name,
        project_path,
        port,
        channel,
        psd_buffer_sec,
        waveform_buffer_sec,
        spike_bin_sec,
    ):
        self._ensure_connect_dialog()

        self.state_manager.request_connect()
        self.connect_dialog.set_busy(True)
        set_telemetry_file("")

        if not self._stop_rhx_worker(timeout_ms=3000):
            self.connect_dialog.set_busy(False)
            self.state_manager.connection_failed()
            QtWidgets.QMessageBox.warning(
                self,
                "Worker Busy",
                "The previous stream worker is still shutting down. Please wait a moment and try again.",
            )
            return

        self.plot_screen.configure_processing_settings(
            psd_buffer_sec=psd_buffer_sec,
            waveform_buffer_sec=waveform_buffer_sec,
            spike_bin_sec=spike_bin_sec,
        )

        device = IntanRHXDevice(
            host=host,
            command_port=command_port,
            data_port=data_port,
            num_channels=1,
        )
        if not device.connect():
            self.connect_dialog.set_busy(False)
            self.state_manager.connection_failed()
            QtWidgets.QMessageBox.critical(
                self,
                "Connection Failed",
                "Could not connect to Intan RHX at "
                f"{host}:{command_port}/{data_port}.",
            )
            return
        device.configure(enable_wide_channel=[channel], port=port, blocks_per_write=1)

        effective_fs = float(device.sample_rate)
        self.plot_screen.add_device("Intan RHX", "rhx", sample_rate=effective_fs, num_channels=1)
        sink = ChunkWriter(
            sample_rate=effective_fs,
            num_channels=1,
            chunk_max_sec=RAW_CHUNK_SEC,
            buffer_bytes=CSV_FILE_BUFFER_BYTES,
            flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
        )

        self.rhx_worker = RHXWorker(
            device=device,
            output_sink=sink,
            project_name=project_name,
            project_path=project_path,
        )
        self.rhx_worker.connection_request_result_signal.connect(self._on_connection_result)
        self.rhx_worker.data_received_signal.connect(self.plot_screen._on_data_received)
        self.rhx_worker.marker_added_signal.connect(self.plot_screen.add_marker)
        self.rhx_worker.marker_catalog_signal.connect(self.plot_screen.set_marker_catalog)
        self.rhx_worker.marker_catalog_signal.connect(self._on_marker_catalog_updated)
        self.rhx_worker.acquisition_state_signal.connect(self._on_acquisition_state_changed)
        self.rhx_worker.start()

    def _on_connection_result(self, successful, message=None):
        if self.connect_dialog is not None:
            self.connect_dialog.set_busy(False)

        if successful:
            self.state_manager.connection_succeeded()
            if self.connect_dialog is not None:
                self.connect_dialog.accept()
            paths = self.rhx_worker.get_project_paths() if self.rhx_worker is not None else {}
            run_dir = str(paths.get("run_dir", "") or "")
            if run_dir:
                set_telemetry_file(str(Path(run_dir) / "telemetry.txt"))
            dev = self.rhx_worker.device
            self.plot_screen.set_connection_details(
                host=getattr(dev, "host", "N/A"),
                command_port=getattr(dev, "command_port", 0),
                data_port=getattr(dev, "data_port", 0),
                sample_rate=self.rhx_worker.sample_rate,
                project_name=self.rhx_worker.project_name,
            )
            self.plot_screen.set_project_storage_paths(
                run_dir=paths.get("run_dir", ""),
                snapshots_dir=paths.get("snapshots_dir", ""),
            )
        else:
            self.state_manager.connection_failed()
            set_telemetry_file("")
            QtWidgets.QMessageBox.critical(
                self,
                "Connection Failed",
                "Connection failed! Make sure the Intan RHX program is running and has TCP server running.\n\n"
                f"{message}",
            )

    def _on_toggle_receiving_requested(self):
        if self.rhx_worker is None:
            return

        current_state = self.state_manager.get_current_state()

        if current_state in (AppState.STREAMING, AppState.WAITING_FOR_DATA):
            if not self.state_manager.user_pause():
                print("[UI] Failed to pause receiving")
                return
            self.rhx_worker.pause_receiving()
        elif current_state == AppState.PAUSED:
            if not self.state_manager.user_resume():
                print("[UI] Failed to resume from paused state")
                return
            self.rhx_worker.resume_receiving()
        elif current_state == AppState.CONNECTED:
            if not self.state_manager.request_stream():
                print("[UI] Failed to start streaming")
                return

    def _on_marker_requested(self):
        if self.rhx_worker is None:
            return

        default_name = f"Marker {len(self.plot_screen.get_markers()) + 1}"
        self.rhx_worker.request_marker(default_name)

    def _ensure_marker_dialog(self):
        if self.marker_dialog is None:
            self.marker_dialog = MarkerDialog(self)
            self.marker_dialog.rename_requested.connect(self._on_marker_rename_requested)
            self.marker_dialog.delete_requested.connect(self._on_marker_delete_requested)

    def _on_open_marker_manager(self):
        self._ensure_marker_dialog()
        self.marker_dialog.set_markers(self.plot_screen.get_markers())
        self.marker_dialog.show()
        self.marker_dialog.raise_()
        self.marker_dialog.activateWindow()

    def _on_marker_rename_requested(self, marker_id: int, new_name: str):
        if self.rhx_worker is None:
            return
        if not self.rhx_worker.request_rename_marker(marker_id, new_name):
            QtWidgets.QMessageBox.warning(self, "Markers", "Rename failed. The marker may no longer exist.")
            return

    def _on_marker_delete_requested(self, marker_id: int):
        if self.rhx_worker is None:
            return
        if not self.rhx_worker.request_delete_marker(marker_id):
            QtWidgets.QMessageBox.warning(self, "Markers", "Delete failed. The marker may no longer exist.")
            return

    def _on_marker_catalog_updated(self, markers):
        if self.marker_dialog is not None and self.marker_dialog.isVisible():
            self.marker_dialog.set_markers(markers)

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
            self.state_manager.request_disconnect()

            if not self._stop_rhx_worker(timeout_ms=4000):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Disconnect Incomplete",
                    "Worker did not stop cleanly yet. Please try disconnect again.",
                )
                return

            if self.marker_dialog is not None:
                self.marker_dialog.hide()

            self.plot_screen.clear_project_buffers()

            self.state_manager.disconnect_complete()
            set_telemetry_file("")
            self.open_connect_dialog()

    def _on_acquisition_state_changed(self, state):
        current_state = self.state_manager.get_current_state()

        if state == "running":
            if current_state == AppState.CONNECTED:
                self.state_manager.request_stream()
            elif current_state == AppState.WAITING_FOR_DATA:
                self.state_manager.data_arrived()
        elif state == "waiting":
            if current_state == AppState.STREAMING:
                self.state_manager.no_data_available()
        elif state == "paused":
            if current_state in (AppState.STREAMING, AppState.WAITING_FOR_DATA):
                self.state_manager.user_pause()
        elif state == "stopped":
            pass
        elif state == "connection_lost":
            if current_state in (AppState.STREAMING, AppState.WAITING_FOR_DATA, AppState.PAUSED):
                self.state_manager.device_disconnected()

    def closeEvent(self, event):
        answer = QtWidgets.QMessageBox.question(
            self,
            "Confirm Exit",
            "Close the application? Any active stream will be disconnected.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            event.ignore()
            return

        if not self._stop_rhx_worker(timeout_ms=4000):
            QtWidgets.QMessageBox.warning(
                self,
                "Close Blocked",
                "Background worker is still shutting down. Please try closing again in a moment.",
            )
            event.ignore()
            return

        if not self.plot_screen.shutdown_workers():
            QtWidgets.QMessageBox.warning(
                self,
                "Close Blocked",
                "Processing worker is still shutting down. Please try closing again in a moment.",
            )
            event.ignore()
            return

        if self.marker_dialog is not None:
            self.marker_dialog.hide()
        if self.connect_dialog is not None:
            self.connect_dialog.hide()

        super().closeEvent(event)
