import time

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from rhx_realtime_feed.screens.plot_helpers import PLOT_UPDATE_FREQ_HZ
from rhx_realtime_feed.screens.device_tab import NeuralDeviceTab, SmuDeviceTab
from rhx_realtime_feed.screens.marker_dialog import MarkerDialog


class PlotScreen(QtWidgets.QWidget):

    toggle_receiving_request_signal = QtCore.pyqtSignal(bool)
    save_disconnect_request_signal  = QtCore.pyqtSignal()
    marker_request_signal           = QtCore.pyqtSignal()
    auto_follow_changed_signal      = QtCore.pyqtSignal(bool)
    fps_updated = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # ponytail: multi-device tab container, stores DeviceTab instances keyed by name
        self._tabs = {}  # name -> DeviceTab

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toolbar = QtWidgets.QWidget()
        self._toolbar.setVisible(False)
        toolbar_layout = QtWidgets.QHBoxLayout(self._toolbar)
        toolbar_layout.setContentsMargins(4, 4, 4, 0)
        toolbar_layout.setSpacing(4)

        self.btn_psd_snapshot = QtWidgets.QToolButton()
        self.btn_psd_snapshot.setText("PSD Snap")
        self.btn_psd_snapshot.setToolTip("Capture PSD snapshot as dashed overlay")
        self.btn_psd_snapshot.clicked.connect(self.take_psd_snapshot)

        self.btn_wf_snapshot = QtWidgets.QToolButton()
        self.btn_wf_snapshot.setText("WF Snap")
        self.btn_wf_snapshot.setToolTip("Capture waveform snapshot as dashed overlay")
        self.btn_wf_snapshot.clicked.connect(self.take_waveform_snapshot)

        self.btn_clear_snapshots = QtWidgets.QToolButton()
        self.btn_clear_snapshots.setText("Clear Snap")
        self.btn_clear_snapshots.setToolTip("Remove all snapshot overlays")
        self.btn_clear_snapshots.clicked.connect(self.clear_snapshots)

        toolbar_layout.addWidget(self.btn_psd_snapshot)
        toolbar_layout.addWidget(self.btn_wf_snapshot)
        toolbar_layout.addWidget(self.btn_clear_snapshots)

        toolbar_layout.addSpacing(16)

        self.btn_add_marker = QtWidgets.QToolButton()
        self.btn_add_marker.setText("Add Marker")
        self.btn_add_marker.setToolTip("Add a visual marker near the right edge of the visible plot")
        self.btn_add_marker.clicked.connect(self._add_marker)

        self.btn_markers = QtWidgets.QToolButton()
        self.btn_markers.setText("Markers\u2026")
        self.btn_markers.setToolTip("View, rename, or delete markers")
        self.btn_markers.clicked.connect(self._open_marker_dialog)

        toolbar_layout.addWidget(self.btn_add_marker)
        toolbar_layout.addWidget(self.btn_markers)

        layout.addWidget(self._toolbar)

        self._marker_dialog = None

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setDocumentMode(True)
        # ponytail: tabs are closeable only when multi-device, single-device hides bar entirely
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tab_widget.currentChanged.connect(self._update_toolbar_visibility)
        layout.addWidget(self.tab_widget, 1)

        self._empty_label = QtWidgets.QLabel("No data sources.\nAdd a device to begin.")
        self._empty_label.setAlignment(QtCore.Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: #6C6C6C; padding: 40px 16px; font-size: 13px;")
        layout.addWidget(self._empty_label, 1)

        pg.setConfigOption('background', 'white')
        pg.setConfigOption('foreground', 'black')
        pg.setConfigOption('antialias', True)

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(1000 // PLOT_UPDATE_FREQ_HZ)
        self.render_timer.timeout.connect(self._render_all)
        self.render_timer.start()

        self._fps_frame_count = 0
        self._fps_last_t = time.perf_counter()
        self._render_dur_ms = 0.0

        self._empty_label.show()
        self.tab_widget.hide()
        self._update_toolbar_visibility()

    def _on_tab_close_requested(self, index):
        widget = self.tab_widget.widget(index)
        name = self.tab_widget.tabText(index)
        if widget is not None:
            widget.shutdown()
        self.tab_widget.removeTab(index)
        self._tabs.pop(name, None)
        self._update_tab_bar_visibility()

    def _update_tab_bar_visibility(self):
        visible = len(self._tabs) > 1
        self.tab_widget.tabBar().setVisible(visible)

    def _update_toolbar_visibility(self):
        tab = self._active_tab()
        self._toolbar.setVisible(tab is not None and hasattr(tab, 'take_psd_snapshot'))

    def add_device(self, name, device_type, sample_rate=20000.0, num_channels=1, channel_labels=None):
        if name in self._tabs:
            return
        if device_type == "smu":
            tab = SmuDeviceTab(sample_rate=sample_rate, parent=self)
        else:
            tab = NeuralDeviceTab(sample_rate=sample_rate, num_channels=num_channels, channel_labels=channel_labels, parent=self)
        self._tabs[name] = tab
        self.tab_widget.addTab(tab, name)
        self._empty_label.hide()
        self.tab_widget.show()
        self._update_tab_bar_visibility()

    def remove_device(self, name):
        tab = self._tabs.pop(name, None)
        if tab is None:
            return
        idx = self.tab_widget.indexOf(tab)
        if idx >= 0:
            self.tab_widget.removeTab(idx)
        tab.shutdown()
        tab.deleteLater()
        self._update_tab_bar_visibility()
        if not self._tabs:
            self._empty_label.show()
            self.tab_widget.hide()

    def on_device_configured(self, device_name: str, num_channels: int, channel_labels: list[str]):
        tab = self._tabs.get(device_name)
        if tab is not None and hasattr(tab, '_resize'):
            tab._resize(num_channels, channel_labels)

    def clear_all(self):
        for name in list(self._tabs):
            self.remove_device(name)
        if not self._tabs:
            self._empty_label.show()
            self.tab_widget.hide()

    def _active_tab(self):
        widget = self.tab_widget.currentWidget()
        if widget is None and self._tabs:
            return next(iter(self._tabs.values()))
        return widget

    def on_data(self, device_name, chunk):
        tab = self._tabs.get(device_name)
        if tab is not None:
            tab.on_data(chunk)

    # ── Render ─────────────────────────────────────────────────────────────

    def _render_all(self):
        t0 = time.perf_counter()
        did_work = False
        for tab in self._tabs.values():
            if hasattr(tab, 'render'):
                tab.render()
                did_work = True
        now = time.perf_counter()
        self._render_dur_ms = (now - t0) * 1000.0
        if did_work:
            self._fps_frame_count += 1
            elapsed = now - self._fps_last_t
            if elapsed >= 1.0:
                fps = self._fps_frame_count / elapsed
                self.fps_updated.emit(f"FPS: {fps:.1f}  Frame: {self._render_dur_ms:.1f} ms")
                self._fps_frame_count = 0
                self._fps_last_t = now

    # ── Connection details display ─────────────────────────────────────────

    def set_connection_details(self, host, command_port, data_port, sample_rate, project_name):
        tab = self._active_tab()
        if tab is not None:
            tab.set_connection_details(
                host=host, command_port=command_port,
                data_port=data_port, sample_rate=sample_rate,
                project_name=project_name,
            )

    def set_connection_status(self, status_text=""):
        pass  # delegating to tab-based layout; label kept for backward compat

    def set_project_storage_paths(self, run_dir, snapshots_dir):
        pass

    def set_receiving_state(self, receiving: bool):
        tab = self._active_tab()
        if tab is not None:
            tab.set_receiving_state(receiving)

    # ── Markers ────────────────────────────────────────────────────────────

    def add_marker(self, marker):
        tab = self._active_tab()
        if tab is not None and hasattr(tab, 'add_marker'):
            tab.add_marker(marker)

    def set_marker_catalog(self, markers):
        tab = self._active_tab()
        if tab is not None and hasattr(tab, 'set_marker_catalog'):
            tab.set_marker_catalog(markers)

    def get_markers(self):
        tab = self._active_tab()
        if tab is not None and hasattr(tab, 'get_markers'):
            return tab.get_markers()
        return []

    # ── Processing settings ────────────────────────────────────────────────

    def configure_processing_settings(self, psd_buffer_sec, waveform_buffer_sec, spike_bin_sec):
        for tab in self._tabs.values():
            if hasattr(tab, 'configure_processing_settings'):
                tab.configure_processing_settings(
                    psd_buffer_sec=psd_buffer_sec,
                    waveform_buffer_sec=waveform_buffer_sec,
                    spike_bin_sec=spike_bin_sec,
                )

    # ── Session state ──────────────────────────────────────────────────────

    def clear_project_buffers(self):
        for tab in self._tabs.values():
            if hasattr(tab, 'clear'):
                tab.clear()

    def shutdown_workers(self) -> bool:
        ok = True
        for tab in self._tabs.values():
            if hasattr(tab, 'shutdown'):
                if not tab.shutdown():
                    ok = False
        return ok

    # ── Snapshots ──────────────────────────────────────────────────────────

    def take_psd_snapshot(self) -> bool:
        tab = self._active_tab()
        if tab is not None and hasattr(tab, 'take_psd_snapshot'):
            return tab.take_psd_snapshot()
        return False

    def take_waveform_snapshot(self) -> bool:
        tab = self._active_tab()
        if tab is not None and hasattr(tab, 'take_waveform_snapshot'):
            return tab.take_waveform_snapshot()
        return False

    def clear_snapshots(self):
        for tab in self._tabs.values():
            if hasattr(tab, 'clear_snapshots'):
                tab.clear_snapshots()

    # ── Auto-follow ────────────────────────────────────────────────────────

    def is_auto_follow_enabled(self) -> bool:
        tab = self._active_tab()
        if tab is not None and hasattr(tab, 'is_auto_follow_enabled'):
            return tab.is_auto_follow_enabled()
        return True

    def set_auto_follow(self, enabled: bool):
        for tab in self._tabs.values():
            if hasattr(tab, 'set_auto_follow'):
                tab.set_auto_follow(enabled)

    # ── Legacy compat (single-device data routing) ────────────────────────

    def _on_data_received(self, chunk):
        if self._tabs:
            name = next(iter(self._tabs))
            self._tabs[name].on_data(chunk)

    def changeEvent(self, event):
        if event.type() in (13, QtCore.QEvent.WindowStateChange):
            self.tab_widget.update()
        super().changeEvent(event)

    def _add_marker(self):
        tab = self._active_tab()
        if tab is None:
            return
        vb = tab.canvas.raw_plot.getViewBox()
        if vb is None:
            return
        x_range = vb.viewRange()[0]
        ts = x_range[0] + (x_range[1] - x_range[0]) * 0.8
        self.add_marker(ts)

    def _open_marker_dialog(self):
        if self._marker_dialog is None:
            self._marker_dialog = MarkerDialog(self)
            self._marker_dialog.rename_requested.connect(self._on_marker_rename)
            self._marker_dialog.delete_requested.connect(self._on_marker_delete)
        markers = self.get_markers()
        self._marker_dialog.set_markers(markers)
        self._marker_dialog.show()
        self._marker_dialog.raise_()

    def _on_marker_rename(self, marker_id, new_name):
        markers = self.get_markers()
        for m in markers:
            if m.get("id") == marker_id:
                m["name"] = new_name
                break
        self.set_marker_catalog(markers)
        if self._marker_dialog is not None:
            self._marker_dialog.set_markers(markers)

    def _on_marker_delete(self, marker_id):
        markers = self.get_markers()
        markers = [m for m in markers if m.get("id") != marker_id]
        self.set_marker_catalog(markers)
        if self._marker_dialog is not None:
            self._marker_dialog.set_markers(markers)
