import time

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

from leech.gui.plot_helpers import PLOT_UPDATE_FREQ_HZ
from leech.gui.ring_buffer import RingBuffer


PLANNING_EMPTY_MESSAGE = (
    "No live plots yet\n\n"
    "Use + Add Device in the timeline header, build your plan,\n"
    "then choose Run Experiment to start live data."
)


class TabHost(QtWidgets.QWidget):

    toggle_receiving_request_signal = QtCore.pyqtSignal(bool)
    save_disconnect_request_signal  = QtCore.pyqtSignal()
    marker_request_signal           = QtCore.pyqtSignal()
    auto_follow_changed_signal      = QtCore.pyqtSignal(bool)
    fps_updated = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # ponytail: multi-tab container, stores DataTab instances keyed by name
        self._tabs = {}   # tab key -> DataTab
        self._route = {}  # source (device) name -> [DataTab, ...]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setDocumentMode(True)
        # toolbar visibility handled per-tab
        layout.addWidget(self.tab_widget, 1)

        self._empty_label = QtWidgets.QLabel(PLANNING_EMPTY_MESSAGE)
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

    def _update_tab_bar_visibility(self):
        visible = len(self._tabs) > 1
        self.tab_widget.tabBar().setVisible(visible)

    def set_planning_state(self):
        self._empty_label.setText(PLANNING_EMPTY_MESSAGE)
        self._empty_label.show()
        self.tab_widget.hide()

    def has_tab(self, key):
        return key in self._tabs

    def add_tab(self, key, tab_cls, sample_rate, num_channels=1, channel_labels=None, sources=None):
        if key in self._tabs or tab_cls is None:
            return
        tab = tab_cls(sample_rate=sample_rate, num_channels=num_channels,
                      channel_labels=channel_labels, parent=self)
        tab.sources = list(sources) if sources else [key]
        self._tabs[key] = tab
        for source in tab.sources:
            self._route.setdefault(source, []).append(tab)
        self.tab_widget.addTab(tab, key)
        self._empty_label.hide()
        self.tab_widget.show()
        self._update_tab_bar_visibility()

    def remove_device(self, name):
        tab = self._tabs.pop(name, None)
        if tab is None:
            return
        for source in tab.sources:
            tabs = self._route.get(source, [])
            if tab in tabs:
                tabs.remove(tab)
            if not tabs:
                self._route.pop(source, None)
        idx = self.tab_widget.indexOf(tab)
        if idx >= 0:
            self.tab_widget.removeTab(idx)
        tab.shutdown()
        tab.deleteLater()
        self._update_tab_bar_visibility()
        if not self._tabs:
            self.set_planning_state()

    def on_device_configured(self, source: str, num_channels: int, channel_labels: list[str], sample_rate: float = 0.0):
        for tab in self._route.get(source, ()):
            if hasattr(tab, 'clear'):
                tab.clear()
            if hasattr(tab, '_resize'):
                tab._resize(num_channels, channel_labels)
            if sample_rate > 0 and hasattr(tab, 'sampling_rate'):
                tab.sampling_rate = float(sample_rate)
                tab._ring = RingBuffer(tab.sampling_rate, tab._ring.num_channels, duration_sec=300)

    def clear_all(self):
        for name in list(self._tabs):
            self.remove_device(name)
        if not self._tabs:
            self.set_planning_state()

    def _active_tab(self):
        widget = self.tab_widget.currentWidget()
        if widget is None and self._tabs:
            return next(iter(self._tabs.values()))
        return widget

    def on_data(self, source, chunk):
        for tab in self._route.get(source, ()):
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

    def set_receiving_state(self, receiving: bool):
        for tab in self._tabs.values():
            if hasattr(tab, 'set_receiving_state'):
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

    def shutdown_workers(self) -> bool:
        ok = True
        for tab in self._tabs.values():
            if hasattr(tab, 'shutdown'):
                if not tab.shutdown():
                    ok = False
        return ok

    # ── Snapshots ──────────────────────────────────────────────────────────

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

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.WindowStateChange:
            self.tab_widget.update()
        super().changeEvent(event)
