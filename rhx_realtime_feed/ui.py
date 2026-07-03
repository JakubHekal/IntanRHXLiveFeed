import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt5.QtCore import pyqtSignal, QPropertyAnimation, QRect, QSize, Qt, QEasingCurve
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QAction,
    QMenu,
    QSlider,
    QTabWidget,
    QToolButton,
)

import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette 
from qdarkstyle.light.palette import LightPalette

from rhx_realtime_feed.experiment import ExperimentManager, ExperimentDialog, RunExperimentDialog
from rhx_realtime_feed.experiment.experiment import ExperimentConfig, SequenceStep
from rhx_realtime_feed.experiment.experiment_runner import ExperimentRunner
from rhx_realtime_feed.screens.legacy_main_window import LegacyMainWindow
from rhx_realtime_feed.screens.plot_screen import PlotScreen

BG_DARK = "#1E1E1E"
BG_SURFACE = "#252526"
BG_HEADER = "#2D2D2D"
TEXT_PRIMARY = "#EDEBE9"
ACCENT_BLUE = "#0078D4"

from rhx_realtime_feed.device import (
    IntanRHXDevice, SimulatedRecordingDevice, SimulatedActorDevice,
    SimulatedCombinedDevice, MiniSMUDevice, DeviceOperation, ParamDef,
)

_DEVICE_CLASSES = {}
def _build_registry():
    for cls in [IntanRHXDevice, SimulatedRecordingDevice, SimulatedActorDevice, SimulatedCombinedDevice, MiniSMUDevice]:
        _DEVICE_CLASSES[cls.device_type] = cls
        
_build_registry()

_SYSTEM_OPERATIONS = [
    DeviceOperation("wait_input", "Wait for User Input", instantaneous=True, default_duration=0, color="#FFD700"),
    DeviceOperation("log_event", "Log Event", instantaneous=True, default_duration=0, color="#FFA500"),
    DeviceOperation("start_recording", "Start Recording", instantaneous=True, default_duration=0, color="#00CC66"),
    DeviceOperation("stop_recording", "Stop Recording", instantaneous=True, default_duration=0, color="#CC3333"),
    DeviceOperation("pause", "Pause", default_duration=1.0, color="#FF8C00"),
]


class FluentExpander(QFrame):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._expanded = expanded
        self._content = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QFrame()
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet(
            f"QFrame {{ background-color: {BG_HEADER}; }}"
            f"QLabel {{ color: {TEXT_PRIMARY}; font-weight: 600; }}"
        )
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(4)

        self._chevron = QLabel("\u25BC" if expanded else "\u25B6")
        header_layout.addWidget(self._chevron)

        self._title_label = QLabel(title)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()

        layout.addWidget(self._header)

        self._content_frame = QFrame()
        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(8, 4, 8, 8)
        self._content_layout.setSpacing(6)
        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        self._header.mousePressEvent = self._on_header_click
        self._animation = None

    def setContentWidget(self, widget: QWidget):
        self._content = widget
        self._content_layout.addWidget(widget)

    def _on_header_click(self, event):
        self._toggle()

    def _toggle(self):
        self._expanded = not self._expanded
        self._chevron.setText("\u25BC" if self._expanded else "\u25B6")
        self._animate_content(self._expanded)

    def _animate_content(self, show: bool):
        target_height = self._content_frame.sizeHint().height() if show else 0
        if self._animation:
            self._animation.stop()
        self._animation = QPropertyAnimation(self._content_frame, b"maximumHeight")
        self._animation.setDuration(150)
        self._animation.setStartValue(self._content_frame.maximumHeight() if self._content_frame.isVisible() else 0)
        self._animation.setEndValue(target_height)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        if show:
            self._content_frame.setVisible(True)
            self._content_frame.setMaximumHeight(0)
        self._animation.finished.connect(lambda: self._content_frame.setMaximumHeight(target_height) if not show else None)
        self._animation.start()
        if not show:
            self._animation.finished.connect(lambda: self._content_frame.setVisible(False))
        try:
            self._animation.finished.disconnect()
        except TypeError:
            pass
        if show:
            self._content_frame.setVisible(True)
            self._content_frame.setMaximumHeight(0)
            self._animation.setStartValue(0)
            self._animation.setEndValue(target_height)
        else:
            self._animation.setStartValue(self._content_frame.height())
            self._animation.setEndValue(0)
            self._animation.finished.connect(lambda: self._content_frame.setVisible(False))

        self._content_frame.setMaximumHeight(target_height)





class LeftSidebar(QFrame):
    run_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Run History")
        layout.addWidget(header)

        self.run_list = QListWidget()
        self.run_list.setAlternatingRowColors(True)
        self.run_list.itemDoubleClicked.connect(self._on_run_activated)
        layout.addWidget(self.run_list, 1)

        self.setMinimumWidth(260)

    def _on_run_activated(self, item):
        run_path = item.data(Qt.UserRole)
        if run_path:
            self.run_selected.emit(run_path)

    def reload_runs(self, runs_dir):
        self.run_list.clear()
        if not runs_dir or not Path(runs_dir).exists():
            return
        run_paths = sorted(Path(runs_dir).iterdir(), reverse=True)
        for rp in run_paths:
            if not rp.is_dir() or rp.name.startswith('.'):
                continue
            meta_file = rp / "metadata.json"
            meta = {}
            if meta_file.exists():
                try:
                    import json
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    pass
            name = meta.get("name", rp.name)
            ts = meta.get("timestamp", "")
            status = meta.get("status", "unknown")
            label = f"[{ts}] {name} \u2014 {status}" if ts else f"{name} \u2014 {status}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(rp))
            self.run_list.addItem(item)


class RightSidebar(QFrame):
    block_change_requested = pyqtSignal(int, int, str, float, float, dict)
    device_config_changed = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._block_dev_idx = -1
        self._block_idx = -1
        self._device_idx = -1
        self._block_op_name = ""
        self._block_param_widgets = []
        self._device_param_widgets = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        scroll.setWidget(container)
        form_layout = QVBoxLayout(container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(4)

        self.filter_expander = FluentExpander("Filter", expanded=True)
        filter_widget = QWidget()
        filter_form = QFormLayout(filter_widget)
        filter_form.setContentsMargins(0, 4, 0, 0)
        filter_form.setSpacing(6)

        self.cutoff_hz = QSpinBox()
        self.cutoff_hz.setRange(0, 1000)
        self.cutoff_hz.setValue(200)

        self.cutoff_secondary = QDoubleSpinBox()
        self.cutoff_secondary.setValue(1.0)

        self.method = QComboBox()
        self.method.addItems(["Default method"])

        filter_form.addRow("Cut-off (Hz)", self.cutoff_hz)
        filter_form.addRow("Secondary", self.cutoff_secondary)
        filter_form.addRow("Method", self.method)
        self.filter_expander.setContentWidget(filter_widget)

        self.analytics_expander = FluentExpander("Analytics", expanded=True)
        analytics_widget = QWidget()
        analytics_form = QFormLayout(analytics_widget)
        analytics_form.setContentsMargins(0, 4, 0, 0)
        analytics_form.setSpacing(6)

        self.function_name = QLineEdit("Power spectrum")
        self.sessions = QComboBox()
        self.sessions.addItems(["Session 1", "Session 2", "Session 3"])
        self.features = QLineEdit("Spike amplitude")
        self.memory = QLineEdit("320 GB")

        analytics_form.addRow("Function", self.function_name)
        analytics_form.addRow("Session", self.sessions)
        analytics_form.addRow("Features", self.features)
        analytics_form.addRow("Memory", self.memory)
        self.analytics_expander.setContentWidget(analytics_widget)

        # ── Block Properties with dynamic operation params ──
        self._block_expander = FluentExpander("Block Properties", expanded=True)
        self._bp_widget = QWidget()
        self._bp_layout = QVBoxLayout(self._bp_widget)
        self._bp_layout.setContentsMargins(0, 4, 0, 0)
        self._bp_layout.setSpacing(4)

        self._bp_form = QFormLayout()
        self._bp_form.setSpacing(6)
        self._block_name = QLineEdit("")
        self._block_start = QDoubleSpinBox()
        self._block_start.setRange(0.0, 9999.0)
        self._block_start.setSuffix(" min")
        self._block_start.setSingleStep(0.5)
        self._block_start.setDecimals(1)
        self._block_dur = QDoubleSpinBox()
        self._block_dur.setRange(0.0, 9999.0)
        self._block_dur.setSuffix(" min")
        self._block_dur.setSingleStep(0.5)
        self._block_dur.setDecimals(1)
        self._bp_form.addRow("Name", self._block_name)
        self._bp_form.addRow("Start", self._block_start)
        self._bp_form.addRow("Duration", self._block_dur)
        self._bp_layout.addLayout(self._bp_form)

        self._bp_sep = QFrame()
        self._bp_sep.setFrameShape(QFrame.HLine)
        self._bp_layout.addWidget(self._bp_sep)
        self._bp_dynamic_widget = QWidget()
        self._bp_dynamic_layout = QFormLayout(self._bp_dynamic_widget)
        self._bp_dynamic_layout.setContentsMargins(0, 4, 0, 0)
        self._bp_dynamic_layout.setSpacing(6)
        self._bp_layout.addWidget(self._bp_dynamic_widget)

        self._block_expander.setContentWidget(self._bp_widget)

        self._block_name.textChanged.connect(self._on_block_field_changed)
        self._block_start.valueChanged.connect(self._on_block_field_changed)
        self._block_dur.valueChanged.connect(self._on_block_field_changed)

        # ── Device Properties with dynamic config params ──
        self._device_expander = FluentExpander("Device Properties", expanded=True)
        self._dp_widget = QWidget()
        self._dp_layout = QVBoxLayout(self._dp_widget)
        self._dp_layout.setContentsMargins(0, 4, 0, 0)
        self._dp_layout.setSpacing(4)
        self._dp_form = QFormLayout()
        self._dp_form.setSpacing(6)
        self._dp_layout.addLayout(self._dp_form)
        self._device_expander.setContentWidget(self._dp_widget)

        form_layout.addWidget(self.filter_expander)
        form_layout.addWidget(self.analytics_expander)
        form_layout.addWidget(self._block_expander)
        form_layout.addWidget(self._device_expander)
        form_layout.addStretch()

        layout.addWidget(scroll)
        self.setMinimumWidth(260)
        self._device_expander.hide()

    # ── param widget helpers ──

    def _build_param_widget(self, param_def, current_value):
        value = current_value if current_value is not None else param_def.default
        if param_def.dtype == "float":
            w = QDoubleSpinBox()
            w.setRange(param_def.min_val or -1e6, param_def.max_val or 1e6)
            w.setDecimals(3)
            w.setValue(float(value or 0.0))
            return w
        elif param_def.dtype == "int":
            w = QSpinBox()
            w.setRange(int(param_def.min_val or 0), int(param_def.max_val or 999999))
            w.setValue(int(value or 0))
            return w
        elif param_def.dtype == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        elif param_def.dtype == "choice":
            w = QComboBox()
            w.addItems(param_def.choices or [])
            if value in (param_def.choices or []):
                w.setCurrentText(value)
            return w
        else:
            w = QLineEdit(str(value or ""))
            return w

    def _read_param_widget(self, param_def, widget):
        if param_def.dtype == "float":
            return widget.value()
        elif param_def.dtype == "int":
            return widget.value()
        elif param_def.dtype == "bool":
            return widget.isChecked()
        elif param_def.dtype == "choice":
            return widget.currentText()
        else:
            return widget.text()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_dynamic_form(self, form_layout, param_defs, current_values, store):
        self._clear_layout(form_layout)
        store.clear()
        for pd in param_defs:
            val = current_values.get(pd.name, pd.default)
            w = self._build_param_widget(pd, val)
            form_layout.addRow(pd.label, w)
            store.append((pd.name, w, pd))

    def _connect_param_signals(self, store, slot):
        for name, w, pd in store:
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.valueChanged.connect(slot)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(slot)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(slot)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(slot)

    def _gather_params(self, store):
        return {name: self._read_param_widget(pd, w) for name, w, pd in store}

    def _emit_block_params(self):
        if self._block_dev_idx < 0 or self._block_idx < 0:
            return
        params = self._gather_params(self._block_param_widgets)
        self.block_change_requested.emit(
            self._block_dev_idx, self._block_idx,
            self._block_name.text(),
            self._block_start.value(),
            self._block_dur.value(),
            params,
        )

    def _emit_device_params(self):
        if self._device_idx < 0:
            return
        params = self._gather_params(self._device_param_widgets)
        self.device_config_changed.emit(self._device_idx, params)

    # ── public methods ──

    def _set_duration_visible(self, visible):
        for i in range(self._bp_form.rowCount()):
            label_item = self._bp_form.itemAt(i, QFormLayout.LabelRole)
            if label_item and label_item.widget() and label_item.widget().text() == "Duration":
                label_item.widget().setVisible(visible)
                field_item = self._bp_form.itemAt(i, QFormLayout.FieldRole)
                if field_item and field_item.widget():
                    field_item.widget().setVisible(visible)
                break
        self._bp_sep.setVisible(visible)

    def set_block_info(self, dev_idx, block_idx, display_name, op_name, start, duration, params=None, device_type=""):
        self._updating = True
        self._block_dev_idx = dev_idx
        self._block_idx = block_idx
        self._block_op_name = op_name
        self._device_idx = -1
        self._block_name.setText(display_name)
        self._block_start.setValue(start)
        self._block_dur.setValue(duration)
        label = f"Block: {display_name}" if display_name else "Block Properties"
        self._block_expander._title_label.setText(label)

        is_instant = (duration == 0)
        self._set_duration_visible(not is_instant)

        param_defs = []
        if op_name and device_type:
            ops = _SYSTEM_OPERATIONS if device_type == "__system__" else []
            if not ops:
                cls = _DEVICE_CLASSES.get(device_type)
                ops = cls.get_operations() if cls else []
            for op in ops:
                if op.name == op_name:
                    param_defs = op.params
                    break

        self._populate_dynamic_form(self._bp_dynamic_layout, param_defs, params or {}, self._block_param_widgets)
        self._connect_param_signals(self._block_param_widgets, self._emit_block_params)

        self._device_expander.hide()
        self._block_expander.show()
        self._updating = False

    def set_device_info(self, dev_idx, device_type, config):
        self._updating = True
        self._device_idx = dev_idx
        self._block_dev_idx = -1
        self._block_idx = -1
        label = f"Device: {device_type}" if device_type else "Device Properties"
        self._device_expander._title_label.setText(label)

        cls = _DEVICE_CLASSES.get(device_type)
        param_defs = cls.get_config_params() if cls else []
        self._populate_dynamic_form(self._dp_form, param_defs, config or {}, self._device_param_widgets)
        self._connect_param_signals(self._device_param_widgets, self._emit_device_params)

        self._block_expander.hide()
        self._device_expander.show()
        self._updating = False

    def _on_block_field_changed(self, *args):
        if self._updating:
            return
        if self._block_dev_idx < 0 or self._block_idx < 0:
            return
        self._emit_block_params()


class ExperimentTimeline(QWidget):
    ROW_HEIGHT = 36
    FLAG_HEIGHT = 18
    LABEL_WIDTH = 130
    HEADER_HEIGHT = 28
    RESIZE_THRESHOLD = 6
    SNAP = 0.5
    _BLOCK_COLORS = ["#0078D4", "#2B88D8", "#4BA3E3", "#107C10", "#498205",
                     "#D13438", "#E74856", "#F1707A", "#8764B8", "#B146C2", "#C239B3"]

    block_selected = pyqtSignal(int, int, str, str, float, float, dict, str)
    data_changed = pyqtSignal()
    device_selected = pyqtSignal(int, str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._device_counter = 0
        self._devices = [
            ["Intan RHX (Port A)", [
                ["Configure", 0.0, 0, "#2B88D8", "Configure", {}],
                ["Stream", 1.0, 14.0, "#4BA3E3", "Stream", {}],
                ["Stimulus", 8.0, 3.0, "#D13438", "Stimulus", {"channel": 1, "amplitude": 500.0, "waveform": "biphasic", "frequency": 100.0}],
            ], "rhx", {"host": "127.0.0.1", "command_port": 5000, "data_port": 5001, "num_channels": 128, "buffer_duration_sec": 5.0}],
            ["Intan RHX (Port B)", [
                ["Configure", 0.0, 0, "#2B88D8", "Configure", {}],
                ["Stream", 1.0, 15.0, "#4BA3E3", "Stream", {}],
            ], "rhx", {"host": "127.0.0.1", "command_port": 5000, "data_port": 5001, "num_channels": 64, "buffer_duration_sec": 5.0}],
            ["miniSMU MS01", [
                ["Configure", 0.0, 0, "#E74856", "Configure", {}],
                ["Stimulus", 2.0, 6.0, "#F1707A", "Stimulus", {"voltage": 5.0, "current_limit": 0.1, "duration_s": 1.0}],
                ["Measure", 8.0, 0, "#E74856", "Measure", {}],
            ], "smu", {"connection_type": "usb", "port": "COM3", "host": "192.168.1.1", "tcp_port": 3333, "mode": "FVMI"}],
            ["Simulated Actor", [
                ["Configure", 0.0, 0, "#B146C2", "Configure", {}],
                ["Write", 3.0, 3.0, "#C239B3", "Write", {"channel": 1, "value": 5.0}],
                ["Trigger", 6.0, 0.5, "#8764B8", "Trigger", {"channel": 1}],
            ], "simulated_actor", {"num_outputs": 2}],
        ]
        self._devices.append(["__System__", [], "__system__", {}])
        self._sel_dev = None
        self._sel_block = None
        self._drag_state = None
        self._drag_dev = None
        self._drag_block = None
        self._drag_press_x = 0.0
        self._drag_orig_start = 0.0
        self._drag_orig_dur = 0.0
        self._update_total_time()
        self._update_height()

    def _update_height(self):
        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        h = self.HEADER_HEIGHT + 6 + len(self._devices) * per_device + 16
        self.setMinimumHeight(h)

    def _row_at(self, my):
        if my < self.HEADER_HEIGHT + 6:
            return None
        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        dev_idx = (my - self.HEADER_HEIGHT - 6) // per_device
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return None
        row_origin = self.HEADER_HEIGHT + 6 + dev_idx * per_device
        if not (row_origin <= my < row_origin + per_device):
            return None
        return dev_idx

    def add_device(self, name=None, device_type=None):
        if not name:
            self._device_counter += 1
            name = f"Device {self._device_counter}"
        if device_type is None:
            device_type = "rhx"
        cls = _DEVICE_CLASSES.get(device_type)
        config = {}
        if cls:
            config = {p.name: p.default for p in cls.get_config_params()}
        self._devices.append([name, [], device_type, config])
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
        self.update()

    def remove_device(self, dev_idx):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        if self._devices[dev_idx][2] == "__system__":
            return
        del self._devices[dev_idx]
        if self._sel_dev == dev_idx:
            self._sel_dev = None
            self._sel_block = None
        elif self._sel_dev is not None and self._sel_dev > dev_idx:
            self._sel_dev -= 1
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
        self.update()

    def clear_all(self):
        system_entry = None
        for d in self._devices:
            if d[2] == "__system__":
                system_entry = [d[0], [], d[2], d[3] if len(d) >= 4 else {}]
                break
        self._devices.clear()
        if system_entry:
            self._devices.append(system_entry)
        self._sel_dev = None
        self._sel_block = None
        self._update_total_time()
        self._update_height()
        self.data_changed.emit()
        self.update()

    def add_block(self, dev_idx, op_name="New Block", start=None, duration=None, params=None):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if start is None:
            start = max(0.0, self._total_time - 2.0)
        label = op_name
        color = self._BLOCK_COLORS[len(blocks) % len(self._BLOCK_COLORS)]
        device_type = self._devices[dev_idx][2]
        ops = _SYSTEM_OPERATIONS if device_type == "__system__" else (getattr(_DEVICE_CLASSES.get(device_type), 'get_operations', lambda: [])())
        op_duration = duration
        for op in ops:
            if op.name == op_name:
                label = op.label
                color = op.color
                if op.instantaneous:
                    op_duration = 0
                elif op_duration is None:
                    op_duration = op.default_duration
                if params is None:
                    params = {p.name: p.default for p in op.params}
                break
        if op_duration is None:
            op_duration = 2.0
        if params is None:
            params = {}
        blocks.append([label, start, op_duration, color, op_name, params])
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def remove_block(self, dev_idx, block_idx):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if block_idx < 0 or block_idx >= len(blocks):
            return
        del blocks[block_idx]
        if self._sel_dev == dev_idx and self._sel_block == block_idx:
            self._sel_block = None
        elif self._sel_dev == dev_idx and self._sel_block is not None and self._sel_block > block_idx:
            self._sel_block -= 1
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def contextMenuEvent(self, event):
        mx, my = event.x(), event.y()
        menu = QMenu(self)
        dev_idx, block_idx, _ = self._block_at(mx, my)
        row_idx = self._row_at(my)

        def _build_add_block_menu(parent_menu, target_dev):
            sub = QMenu("Add Block", parent_menu)
            device_type = self._devices[target_dev][2]
            ops = _SYSTEM_OPERATIONS if device_type == "__system__" else []
            if not ops:
                cls = _DEVICE_CLASSES.get(device_type)
                ops = cls.get_operations() if cls else []
            actions = {}
            for op in ops:
                a = sub.addAction(op.label)
                actions[a] = op.name
            if not actions:
                a = sub.addAction("Generic Block")
                actions[a] = "New Block"
            return sub, actions

        is_system_row = row_idx is not None and self._devices[row_idx][2] == "__system__"
        is_system_dev = dev_idx is not None and self._devices[dev_idx][2] == "__system__"

        if block_idx is not None:
            a_del = menu.addAction(f"Remove  «{self._devices[dev_idx][1][block_idx][0]}»")
            menu.addSeparator()
            add_menu, add_actions = _build_add_block_menu(menu, dev_idx)
            menu.addMenu(add_menu)
            if not is_system_dev:
                a_del_d = menu.addAction(f"Remove  «{self._devices[dev_idx][0]}»")
            action = menu.exec_(event.globalPos())
            if action == a_del:
                self.remove_block(dev_idx, block_idx)
            elif action in add_actions:
                self.add_block(dev_idx, add_actions[action])
            elif not is_system_dev and action == a_del_d:
                self.remove_device(dev_idx)
        elif row_idx is not None:
            add_menu, add_actions = _build_add_block_menu(menu, row_idx)
            menu.addMenu(add_menu)
            if not is_system_row:
                a_del_d = menu.addAction(f"Remove  «{self._devices[row_idx][0]}»")
            action = menu.exec_(event.globalPos())
            if action in add_actions:
                self.add_block(row_idx, add_actions[action])
            elif not is_system_row and action == a_del_d:
                self.remove_device(row_idx)
        else:
            add_dev = menu.addAction("Add Device")
            sys_menu = QMenu("Add System Block", menu)
            sys_actions = {}
            for op in _SYSTEM_OPERATIONS:
                a = sys_menu.addAction(op.label)
                sys_actions[a] = op.name
            menu.addMenu(sys_menu)
            action = menu.exec_(event.globalPos())
            if action is not None:
                if action == add_dev:
                    types = sorted(_DEVICE_CLASSES.keys())
                    type_str, ok = QInputDialog.getItem(self, "Add Device", "Device type:", types, 0, False)
                    if ok and type_str:
                        cls = _DEVICE_CLASSES[type_str]
                        name, ok2 = QInputDialog.getText(self, "Add Device", "Name:", text=cls.name)
                        if ok2:
                            self.add_device(name.strip() or cls.name, type_str)
                elif action in sys_actions:
                    self.add_block(len(self._devices) - 1, sys_actions[action])

    def _update_total_time(self):
        self._total_time = max(
            (s + d for row in self._devices for _, s, d, *_ in row[1]),
            default=20,
        )
        if self._total_time <= 0:
            self._total_time = 20

    def _plot_left(self):
        return self.LABEL_WIDTH

    def _plot_w(self):
        return max(1, self.width() - self._plot_left() - 12)

    def _x_from_time(self, t):
        return self._plot_left() + int((t / self._total_time) * self._plot_w())

    def _snap(self, t):
        return round(t / self.SNAP) * self.SNAP

    def _block_at(self, mx, my):
        if my < self.HEADER_HEIGHT + 6:
            return None, None, None
        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        dev_idx = (my - self.HEADER_HEIGHT - 6) // per_device
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return None, None, None
        blocks = self._devices[dev_idx][1]
        row_origin = self.HEADER_HEIGHT + 6 + dev_idx * per_device
        if not (row_origin <= my < row_origin + per_device):
            return None, None, None
        in_flag = my < row_origin + self.FLAG_HEIGHT
        if in_flag:
            by = row_origin
            bh = self.FLAG_HEIGHT
            fm = QFontMetrics(QFont("Segoe UI", 8))
        else:
            by = row_origin + self.FLAG_HEIGHT + 3
            bh = self.ROW_HEIGHT - 6
        for bi in range(len(blocks) - 1, -1, -1):
            func, start, dur, *_ = blocks[bi]
            is_instant = dur == 0
            if in_flag != is_instant:
                continue
            bx = self._x_from_time(start)
            if is_instant:
                pill_w = max(24, fm.horizontalAdvance(func) + 12)
                pill_x = max(self._plot_left(), bx - pill_w // 2)
                if pill_x <= mx <= pill_x + pill_w and by <= my < by + bh:
                    return dev_idx, bi, "body"
            else:
                bw = max(4, int((dur / self._total_time) * self._plot_w()))
                if (bx - self.RESIZE_THRESHOLD <= mx <= bx + bw + self.RESIZE_THRESHOLD
                        and by <= my < by + bh):
                    if dur > 0 and abs(mx - bx) <= self.RESIZE_THRESHOLD:
                        return dev_idx, bi, "left"
                    if dur > 0 and abs(mx - (bx + bw)) <= self.RESIZE_THRESHOLD:
                        return dev_idx, bi, "right"
                    return dev_idx, bi, "body"
        return None, None, None

    def _select(self, dev_idx, block_idx):
        self._sel_dev = dev_idx
        self._sel_block = block_idx
        if dev_idx is not None and block_idx is not None and block_idx < len(self._devices[dev_idx][1]):
            blocks = self._devices[dev_idx][1]
            b = blocks[block_idx]
            func = b[0]
            start = b[1]
            dur = b[2]
            op_name = b[4] if len(b) >= 5 else ""
            params = b[5] if len(b) >= 6 else {}
            device_type = self._devices[dev_idx][2]
            self.block_selected.emit(dev_idx, block_idx, func, op_name, start, dur, params, device_type)
        elif dev_idx is not None:
            device_type = self._devices[dev_idx][2]
            config = self._devices[dev_idx][3] if len(self._devices[dev_idx]) >= 4 else {}
            self.device_selected.emit(dev_idx, device_type, config)
        else:
            self.block_selected.emit(-1, -1, "", "", 0.0, 0.0, {}, "")
            self.device_selected.emit(-1, "", {})
        self.update()

    def update_block(self, dev_idx, block_idx, func_name, start, duration, params=None):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        blocks = self._devices[dev_idx][1]
        if block_idx < 0 or block_idx >= len(blocks):
            return
        is_instant = (duration == 0)
        if not is_instant:
            duration = max(0.5, duration)
        start = max(0.0, start)
        if is_instant:
            duration = 0
        blocks[block_idx][0] = func_name
        blocks[block_idx][1] = start
        blocks[block_idx][2] = duration
        if params is not None:
            if len(blocks[block_idx]) < 6:
                blocks[block_idx].append("")
                blocks[block_idx].append({})
            blocks[block_idx][5] = params
        self._update_total_time()
        self.data_changed.emit()
        self.update()

    def update_device_config(self, dev_idx, config):
        if dev_idx < 0 or dev_idx >= len(self._devices):
            return
        while len(self._devices[dev_idx]) < 4:
            self._devices[dev_idx].append({})
        self._devices[dev_idx][3] = config
        self.data_changed.emit()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        mx, my = event.x(), event.y()
        dev_idx, block_idx, edge = self._block_at(mx, my)
        if dev_idx is not None and block_idx is not None:
            self._select(dev_idx, block_idx)
            blocks = self._devices[dev_idx][1]
            self._drag_state = f"resize_{edge}" if edge in ("left", "right") else "move"
            self._drag_dev = dev_idx
            self._drag_block = block_idx
            self._drag_press_x = mx
            self._drag_orig_start = blocks[block_idx][1]
            self._drag_orig_dur = blocks[block_idx][2]
        else:
            row_idx = self._row_at(my)
            if row_idx is not None:
                self._select(row_idx, None)
            else:
                self._select(None, None)

    def mouseMoveEvent(self, event):
        mx = event.x()
        my = event.y()
        if self._drag_state:
            if self._drag_state in ("resize_left", "resize_right"):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.SizeAllCursor)
            blocks = self._devices[self._drag_dev][1]
            block = blocks[self._drag_block]
            total = self._total_time
            pw = self._plot_w()
            dt = ((mx - self._drag_press_x) / pw) * total if pw > 0 else 0.0

            if self._drag_state == "move":
                new_start = self._snap(max(0.0, min(
                    self._drag_orig_start + dt,
                    total - block[2],
                )))
                block[1] = new_start
            elif self._drag_state == "resize_left":
                new_start = self._snap(max(0.0, min(
                    self._drag_orig_start + dt,
                    self._drag_orig_start + self._drag_orig_dur - self.SNAP,
                )))
                new_dur = self._snap(max(
                    self.SNAP,
                    self._drag_orig_start + self._drag_orig_dur - new_start,
                ))
                block[1] = new_start
                block[2] = new_dur
            elif self._drag_state == "resize_right":
                new_dur = self._snap(max(
                    self.SNAP,
                    min(self._drag_orig_dur + dt, total - self._drag_orig_start),
                ))
                block[2] = new_dur

            self._update_total_time()
            self.data_changed.emit()
            block_op = block[4] if len(block) >= 5 else ""
            block_params = block[5] if len(block) >= 6 else {}
            device_type = self._devices[self._drag_dev][2] if self._drag_dev is not None and self._drag_dev < len(self._devices) else ""
            self.block_selected.emit(
                self._drag_dev, self._drag_block,
                block[0], block_op, block[1], block[2], block_params, device_type,
            )
            self.update()
        else:
            _, _, edge = self._block_at(mx, my)
            if edge in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
            elif edge == "body":
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._drag_state = None
        self._drag_dev = None
        self._drag_block = None
        self._drag_press_x = 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()

        painter.fillRect(rect, QColor("#1E1E1E"))

        plot_left = self.LABEL_WIDTH
        plot_w = max(1, w - plot_left - 12)

        # Time header
        painter.setPen(QPen(QColor("#EDEBE9"), 1))
        painter.setFont(QFont("Segoe UI", 9))
        painter.fillRect(0, 0, w, self.HEADER_HEIGHT, QColor("#2D2D2D"))

        num_ticks = max(2, int(self._total_time / 2))
        for i in range(num_ticks + 1):
            x = plot_left + int((i / num_ticks) * plot_w)
            painter.drawLine(x, self.HEADER_HEIGHT, x, self.HEADER_HEIGHT + 4)
            painter.drawText(x - 10, self.HEADER_HEIGHT + 16, str(i * 2))

        per_device = self.FLAG_HEIGHT + self.ROW_HEIGHT
        rows_top = self.HEADER_HEIGHT + 6
        rows_height = len(self._devices) * per_device

        # Phase 2: Draw non-system device rows (flag area + main row)
        for i, row in enumerate(self._devices):
            name, blocks, device_type = row[0], row[1], row[2]
            if device_type == "__system__":
                continue
            flag_y = rows_top + i * per_device
            row_y = flag_y + self.FLAG_HEIGHT

            bg = QColor("#252526") if i % 2 == 0 else QColor("#1E1E1E")
            painter.fillRect(0, flag_y, w, self.FLAG_HEIGHT, bg)
            painter.fillRect(0, row_y, w, self.ROW_HEIGHT, bg)

            # Separator
            painter.setPen(QPen(QColor("#3E3E3E"), 1))
            painter.drawLine(self.LABEL_WIDTH, row_y, w, row_y)

            # Device name in main row
            painter.setPen(QPen(QColor(TEXT_PRIMARY), 1))
            painter.drawText(8, row_y + self.ROW_HEIGHT // 2 + 4, name)

            for bi, block in enumerate(blocks):
                func, start, dur, color_str, *_ = block
                color = QColor(color_str)
                bx = plot_left + int((start / self._total_time) * plot_w)
                is_instant = dur == 0

                if is_instant:
                    fm = painter.fontMetrics()
                    pill_w = max(24, fm.horizontalAdvance(func) + 12)
                    pill_h = self.FLAG_HEIGHT - 2
                    pill_x = max(plot_left, bx - pill_w // 2)
                    pill_y = flag_y + 1

                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    stem_bot = row_y + self.ROW_HEIGHT - 3
                    painter.drawRect(bx - 1, pill_y + pill_h, 2, stem_bot - pill_y - pill_h)
                    painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 4, 4)
                    painter.setPen(QPen(QColor("#FFFFFF"), 1))
                    old_font = painter.font()
                    painter.setFont(QFont("Segoe UI", 8))
                    painter.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, func)
                    painter.setFont(old_font)
                else:
                    by = row_y + 3
                    bh = self.ROW_HEIGHT - 6
                    bw = max(4, int((dur / self._total_time) * plot_w))
                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(bx, by, bw, bh, 4, 4)
                    if bw > 40:
                        painter.setPen(QPen(QColor("#FFFFFF"), 1))
                        painter.drawText(bx + 4, by + bh // 2 + 4, func)

                # Selection highlight
                if self._sel_dev == i and self._sel_block == bi:
                    painter.setBrush(Qt.NoBrush)
                    pen = QPen(QColor("#FFFFFF"), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    if is_instant:
                        painter.drawRoundedRect(pill_x - 1, pill_y - 1, pill_w + 2, pill_h + 2, 4, 4)
                    else:
                        bw_sel = max(4, int((dur / self._total_time) * plot_w))
                        painter.drawRoundedRect(bx - 1, by - 1, bw_sel + 2, bh + 2, 4, 4)
                        handle_w = 3
                        handle_h = 10
                        handle_y = by + (bh - handle_h) // 2
                        painter.setBrush(QColor(255, 255, 255, 160))
                        painter.setPen(Qt.NoPen)
                        painter.drawRect(bx - 1, handle_y, handle_w, handle_h)
                        painter.drawRect(bx + bw - handle_w + 1, handle_y, handle_w, handle_h)

        # Phase 3: System block full-height bands (overlay across all rows)
        for row in self._devices:
            if row[2] != "__system__":
                continue
            for block in row[1]:
                _, start, dur, color_str = block[0], block[1], block[2], block[3]
                bx = plot_left + int((start / self._total_time) * plot_w)
                if dur == 0:
                    painter.fillRect(bx, rows_top, 2, rows_height, QColor(255, 255, 255, 40))
                else:
                    bw = max(4, int((dur / self._total_time) * plot_w))
                    band = QColor(color_str)
                    band.setAlpha(20)
                    painter.fillRect(bx, rows_top, bw, rows_height, band)
                    painter.setPen(QPen(QColor(color_str).lighter(120), 1))
                    painter.drawLine(bx, rows_top, bx, rows_top + rows_height)
                    painter.drawLine(bx + bw, rows_top, bx + bw, rows_top + rows_height)

        # Phase 4: System row (on top of bands)
        for i, row in enumerate(self._devices):
            name, blocks, device_type = row[0], row[1], row[2]
            if device_type != "__system__":
                continue
            flag_y = rows_top + i * per_device
            row_y = flag_y + self.FLAG_HEIGHT

            painter.fillRect(0, flag_y, w, self.FLAG_HEIGHT, QColor("#2A2A2A"))
            painter.fillRect(0, row_y, w, self.ROW_HEIGHT, QColor("#2A2A2A"))

            painter.setPen(QPen(QColor("#3E3E3E"), 1))
            painter.drawLine(self.LABEL_WIDTH, row_y, w, row_y)

            font = QFont("Segoe UI", 9, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QPen(QColor(TEXT_PRIMARY), 1))
            painter.drawText(8, row_y + self.ROW_HEIGHT // 2 + 4, name)
            painter.setFont(QFont("Segoe UI", 9))

            for bi, block in enumerate(blocks):
                func, start, dur, color_str, *_ = block
                color = QColor(color_str)
                bx = plot_left + int((start / self._total_time) * plot_w)
                is_instant = dur == 0

                if is_instant:
                    fm = painter.fontMetrics()
                    pill_w = max(24, fm.horizontalAdvance(func) + 12)
                    pill_h = self.FLAG_HEIGHT - 2
                    pill_x = max(plot_left, bx - pill_w // 2)
                    pill_y = flag_y + 1

                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    stem_bot = row_y + self.ROW_HEIGHT - 3
                    painter.drawRect(bx - 1, pill_y + pill_h, 2, stem_bot - pill_y - pill_h)
                    painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 4, 4)
                    painter.setPen(QPen(QColor("#FFFFFF"), 1))
                    old_font = painter.font()
                    painter.setFont(QFont("Segoe UI", 8))
                    painter.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, func)
                    painter.setFont(old_font)
                else:
                    by = row_y + 3
                    bh = self.ROW_HEIGHT - 6
                    bw = max(4, int((dur / self._total_time) * plot_w))
                    painter.setBrush(color)
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(bx, by, bw, bh, 4, 4)
                    if bw > 40:
                        painter.setPen(QPen(QColor("#FFFFFF"), 1))
                        painter.drawText(bx + 4, by + bh // 2 + 4, func)

                if self._sel_dev == i and self._sel_block == bi:
                    painter.setBrush(Qt.NoBrush)
                    pen = QPen(QColor("#FFFFFF"), 2)
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    if is_instant:
                        painter.drawRoundedRect(pill_x - 1, pill_y - 1, pill_w + 2, pill_h + 2, 4, 4)
                    else:
                        bw_sel = max(4, int((dur / self._total_time) * plot_w))
                        painter.drawRoundedRect(bx - 1, by - 1, bw_sel + 2, bh + 2, 4, 4)
                        handle_w = 3
                        handle_h = 10
                        handle_y = by + (bh - handle_h) // 2
                        painter.setBrush(QColor(255, 255, 255, 160))
                        painter.setPen(Qt.NoPen)
                        painter.drawRect(bx - 1, handle_y, handle_w, handle_h)
                        painter.drawRect(bx + bw - handle_w + 1, handle_y, handle_w, handle_h)


class MainStage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.left_sidebar = LeftSidebar()
        self.right_sidebar = RightSidebar()

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(4, 4, 4, 4)
        center_layout.setSpacing(4)

        self.plot_screen = PlotScreen()

        timeline_bar = QHBoxLayout()
        timeline_bar.setContentsMargins(4, 4, 4, 4)
        timeline_bar.setSpacing(4)

        btn_play = QToolButton()
        btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        btn_pause = QToolButton()
        btn_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        btn_prev = QToolButton()
        btn_prev.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        btn_next = QToolButton()
        btn_next.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 120)
        self.timeline_slider.setValue(15)

        timeline_bar.addWidget(btn_play)
        timeline_bar.addWidget(btn_pause)
        timeline_bar.addWidget(btn_prev)
        timeline_bar.addWidget(btn_next)
        timeline_bar.addWidget(QLabel("0"))
        timeline_bar.addWidget(self.timeline_slider)
        timeline_bar.addWidget(QLabel("120"))

        center_layout.addWidget(self.plot_screen, 1)

        self.timeline = ExperimentTimeline()
        center_layout.addWidget(self.timeline)
        self.timeline.block_selected.connect(self.right_sidebar.set_block_info)
        self.right_sidebar.block_change_requested.connect(self.timeline.update_block)
        self.timeline.device_selected.connect(self.right_sidebar.set_device_info)
        self.right_sidebar.device_config_changed.connect(self.timeline.update_device_config)

        center_layout.addLayout(timeline_bar)

        layout.addWidget(self.left_sidebar, 1)
        layout.addWidget(center_widget, 4)
        layout.addWidget(self.right_sidebar, 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroSense Data")
        self.resize(1700, 980)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Add text-based top toolbar
        self._create_text_toolbar(root_layout)

        self._current_experiment_path = None
        self._current_run_path = None
        self._run_device_configs = {}
        self._run_device_instances = []
        self.main_stage = MainStage()
        root_layout.addWidget(self.main_stage, 1)

        self._wire_behavior()
        self._setup_status_bar()
        self.main_stage.plot_screen.fps_updated.connect(self._fps_status_label.setText)

    def _create_text_toolbar(self, parent_layout):
        """Create a menu bar style toolbar at the top."""
        menubar_frame = QFrame()
        menubar_layout = QHBoxLayout(menubar_frame)
        menubar_layout.setContentsMargins(0, 0, 0, 0)
        menubar_layout.setSpacing(0)

        # File menu
        file_menu = QMenu("File", self)
        file_menu.addAction("New Session")
        file_menu.addAction("Open Session")
        file_menu.addAction("Save Session")
        file_menu.addAction("Save As...")
        file_menu.addSeparator()
        file_menu.addAction("Exit")
        file_btn = self._create_menu_button("File", file_menu)
        menubar_layout.addWidget(file_btn)

        # Edit menu
        edit_menu = QMenu("Edit", self)
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")
        edit_menu.addSeparator()
        edit_menu.addAction("Clear Markers")
        edit_menu.addAction("Reset Filters")
        edit_btn = self._create_menu_button("Edit", edit_menu)
        menubar_layout.addWidget(edit_btn)

        # View menu
        view_menu = QMenu("View", self)
        view_menu.addAction("Show Left Sidebar")
        view_menu.addAction("Show Right Sidebar")
        view_menu.addSeparator()
        view_menu.addAction("Full Screen")
        view_menu.addAction("Reset Layout")
        view_menu.addSeparator()
        self._legacy_ui_action = view_menu.addAction("Legacy UI\u2026")
        view_btn = self._create_menu_button("View", view_menu)
        menubar_layout.addWidget(view_btn)

        # Tools menu
        tools_menu = QMenu("Tools", self)
        tools_menu.addAction("Device Settings")
        tools_menu.addAction("Preferences")
        tools_menu.addSeparator()
        tools_menu.addAction("Check for Updates")
        tools_btn = self._create_menu_button("Tools", tools_menu)
        menubar_layout.addWidget(tools_btn)

        # Experiment menu
        experiment_menu = QMenu("Experiment", self)
        self._exp_new_action = experiment_menu.addAction("New Experiment")
        self._exp_open_action = experiment_menu.addAction("Open Experiment")
        experiment_menu.addSeparator()
        self._exp_save_action = experiment_menu.addAction("Save Experiment")
        experiment_menu.addSeparator()
        self._exp_run_action = experiment_menu.addAction("Run Experiment\u2026")
        experiment_btn = self._create_menu_button("Experiment", experiment_menu)
        menubar_layout.addWidget(experiment_btn)

        # Help menu
        help_menu = QMenu("Help", self)
        help_menu.addAction("About")
        help_menu.addAction("Documentation")
        help_btn = self._create_menu_button("Help", help_menu)
        menubar_layout.addWidget(help_btn)

        menubar_layout.addStretch()

        parent_layout.addWidget(menubar_frame)

    def _create_menu_button(self, text: str, menu: QMenu) -> QToolButton:
        """Create a styled menu button for the menu bar."""
        button = QToolButton()
        button.setText(text)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 4px 8px;
                color: #CCCCCC;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #3E3E42;
            }
            QToolButton:pressed {
                background-color: #007ACC;
            }
            QToolButton::menu-indicator { image: none; }
        """)
        return button

    def _wire_behavior(self):
        self._exp_new_action.triggered.connect(self._on_experiment_new)
        self._exp_open_action.triggered.connect(self._on_experiment_open)
        self._exp_save_action.triggered.connect(self._on_experiment_save)
        self._exp_run_action.triggered.connect(self._on_experiment_run)
        self._legacy_ui_action.triggered.connect(self._on_open_legacy_ui)
        self.main_stage.left_sidebar.run_selected.connect(self._on_replay_run)
        self.main_stage.right_sidebar.cutoff_hz.valueChanged.connect(self._on_parameter_change)

    def _setup_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(120)
        status.addPermanentWidget(self.progress)
        self._fps_status_label = QLabel("FPS: 0.0")
        status.addPermanentWidget(self._fps_status_label)

    def _on_parameter_change(self, *args):
        pass

    def _on_experiment_new(self):
        dialog = ExperimentDialog(self)
        if dialog.exec_():
            path = dialog.result_path()
            if path:
                self._current_experiment_path = path
                config = ExperimentManager.load(path)
                self._populate_timeline_from_config(config)
                self.setWindowTitle(f"NeuroSense Data \u2014 {config.metadata.experiment_name}")
                self.main_stage.left_sidebar.reload_runs(str(Path(path) / "runs"))

    def _on_experiment_open(self):
        default_dir = str(self._current_experiment_path) if self._current_experiment_path else str(Path.cwd() / "experiments")
        path = QFileDialog.getExistingDirectory(
            self, "Open Experiment", default_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not path:
            return
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            QMessageBox.warning(self, "Invalid Experiment",
                                "Selected directory does not contain a config.json file.")
            return
        self._current_experiment_path = path
        config = ExperimentManager.load(path)
        self._populate_timeline_from_config(config)
        self.setWindowTitle(f"NeuroSense Data \u2014 {config.metadata.experiment_name}")
        self.main_stage.left_sidebar.reload_runs(str(Path(path) / "runs"))

    def _on_experiment_save(self):
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment",
                                    "No experiment is open. Create or open one first.")
            return
        config = ExperimentManager.load(self._current_experiment_path)
        timeline = self.main_stage.timeline
        devs = timeline._devices

        config.devices = [
            {"name": d[0], "device_type": d[2]}
            for d in devs if d[2] != "__system__"
        ]
        config.execution_control.required_devices = list(set(
            d[2] for d in devs if d[2] != "__system__"
        ))

        sequence = []
        step_id = 1
        for dev in devs:
            if dev[2] == "__system__":
                continue
            for block in dev[1]:
                op_name = block[4] if len(block) >= 5 else block[0]
                params = block[5] if len(block) >= 6 else {}
                sequence.append(SequenceStep(
                    step_id=step_id,
                    action=op_name,
                    parameters=dict(params),
                    device_name=dev[0],
                ))
                step_id += 1
        config.sequence = sequence
        ExperimentManager.save(self._current_experiment_path, config)
        QMessageBox.information(self, "Saved", f"Experiment saved to {self._current_experiment_path}")

    def _on_open_legacy_ui(self):
        if not hasattr(self, '_legacy_window') or self._legacy_window is None:
            self._legacy_window = LegacyMainWindow()
        self._legacy_window.show()
        self._legacy_window.raise_()
        self._legacy_window.activateWindow()

    def _on_experiment_run(self):
        if not self._current_experiment_path:
            QMessageBox.information(self, "No Experiment",
                                    "Open or create an experiment first.")
            return

        exp_name = Path(self._current_experiment_path).name
        device_groups = []
        for d in self.main_stage.timeline._devices:
            if d[2] == "__system__":
                continue
            device_type = d[2]
            cls = _DEVICE_CLASSES.get(device_type)
            param_defs = cls.get_config_params() if cls else []
            current_config = d[3] if len(d) >= 4 else {}
            device_groups.append({
                "name": d[0],
                "device_type": device_type,
                "device_class": cls,
                "param_defs": param_defs,
                "current_config": current_config,
            })

        dialog = RunExperimentDialog(
            experiment_name=exp_name,
            experiment_path=self._current_experiment_path,
            device_groups=device_groups,
            parent=self,
        )

        if dialog.exec_():
            self._current_run_path = dialog.run_path()
            self._run_device_configs = dialog.device_configs()
            self._run_device_instances = dialog.device_instances()
            run_name = Path(self._current_run_path).name
            self.setWindowTitle(
                f"NeuroSense Data \u2014 {exp_name} \u2014 Run: {run_name}"
            )

            self._start_experiment_sequence(exp_name)

    def _build_sequence_for_runner(self):
        devs = self.main_stage.timeline._devices
        sequence = []
        step_id = 1
        for dev in devs:
            if dev[2] == "__system__":
                for block in dev[1]:
                    op_name = block[4] if len(block) >= 5 else block[0]
                    params = block[5] if len(block) >= 6 else {}
                    p = dict(params)
                    p.setdefault("duration_s", block[2] * 60.0)
                    sequence.append(SequenceStep(
                        step_id=step_id,
                        action=op_name,
                        parameters=p,
                        device_name="",
                    ))
                    step_id += 1
                continue
            for block in dev[1]:
                op_name = block[4] if len(block) >= 5 else block[0]
                params = block[5] if len(block) >= 6 else {}
                p = dict(params)
                p.setdefault("duration_s", block[2] * 60.0)
                sequence.append(SequenceStep(
                    step_id=step_id,
                    action=op_name,
                    parameters=p,
                    device_name=dev[0],
                ))
                step_id += 1
        return sequence

    def _devices_with_instances(self):
        result = []
        for d in self.main_stage.timeline._devices:
            if d[2] == "__system__":
                result.append(d)
                continue
            inst = None
            for runner_inst in self._run_device_instances:
                if hasattr(runner_inst, 'name') and runner_inst.name == d[0]:
                    inst = runner_inst
                    break
                if hasattr(runner_inst, 'device_type') and runner_inst.device_type == d[2]:
                    inst = runner_inst
            result.append(list(d) + [inst])
        return result

    def _start_experiment_sequence(self, exp_name):
        if hasattr(self, '_experiment_runner') and self._experiment_runner is not None:
            if self._experiment_runner.is_running():
                QMessageBox.information(self, "Already Running", "An experiment is already in progress.")
                return

        timeline_devs = self._devices_with_instances()
        sequence = self._build_sequence_for_runner()

        if not sequence:
            QMessageBox.information(self, "No Steps", "The experiment has no sequence steps to run.")
            return

        self._experiment_runner = ExperimentRunner(
            devices=timeline_devs,
            sequence=sequence,
            run_path=self._current_run_path,
            parent=self,
        )
        self._experiment_runner.step_started.connect(self._on_exp_step_started)
        self._experiment_runner.step_completed.connect(self._on_exp_step_completed)
        self._experiment_runner.experiment_finished.connect(self._on_exp_finished)
        self._experiment_runner.error_occurred.connect(self._on_exp_error)
        self._experiment_runner.start()

        self._exp_run_action.setEnabled(False)
        self.statusBar().showMessage(f"Running: {exp_name}")

    def _on_exp_step_started(self, step_index, device_name, action, duration):
        self.statusBar().showMessage(
            f"Step {step_index + 1}: {device_name} → {action} ({duration:.1f}s)"
        )
        if hasattr(self, 'progress'):
            total = len(self._build_sequence_for_runner())
            self.progress.setMaximum(total)
            self.progress.setValue(step_index)

    def _on_exp_step_completed(self, step_index, device_name, action):
        if hasattr(self, 'progress'):
            self.progress.setValue(step_index + 1)

    def _on_exp_finished(self, success, message):
        self._exp_run_action.setEnabled(True)
        self.progress.setValue(0)
        self._save_run_metadata(success)
        if self._current_experiment_path:
            self.main_stage.left_sidebar.reload_runs(
                str(Path(self._current_experiment_path) / "runs")
            )
        if success:
            self.statusBar().showMessage(f"Experiment finished: {message}")
        else:
            self.statusBar().showMessage(f"Experiment aborted: {message}")
        self._experiment_runner = None

    def _save_run_metadata(self, success):
        if not self._current_run_path:
            return
        import json, datetime
        meta_path = Path(self._current_run_path) / "metadata.json"
        meta = {
            "name": Path(self._current_experiment_path).name if self._current_experiment_path else "",
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "success" if success else "failed",
            "device_type": "rhx",
            "sample_rate": 20000.0,
            "num_channels": 1,
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass

    def _on_replay_run(self, run_path):
        from rhx_realtime_feed.workers.replay_worker import ReplayWorker
        meta_path = Path(run_path) / "metadata.json"
        if not meta_path.exists():
            QMessageBox.warning(self, "Replay", f"No metadata in {run_path}")
            return
        import json
        meta = json.loads(meta_path.read_text())
        device_type = meta.get("device_type", "rhx")
        sr = meta.get("sample_rate", 20000.0)
        nc = meta.get("num_channels", 1)
        replay_name = f"Replay: {Path(run_path).name}"
        self.main_stage.plot_screen.add_device(replay_name, device_type, sample_rate=sr, num_channels=nc)
        worker = ReplayWorker(run_path, replay_name, self)
        worker.data_received.connect(self.main_stage.plot_screen.on_data)
        worker.error.connect(lambda msg: self.statusBar().showMessage(f"Replay error: {msg}"))
        worker.finished.connect(lambda: self.statusBar().showMessage("Replay finished"))
        worker.start()
        # ponytail: keep worker alive via attribute; add worker registry if multiple replays needed
        self._replay_worker = worker

    def _on_exp_error(self, device_name, error_message):
        print(f"[UI] Experiment error: {device_name}: {error_message}")

    def _populate_timeline_from_config(self, config: ExperimentConfig):
        timeline = self.main_stage.timeline
        timeline.clear_all()

        # restore devices from stored name + type
        if config.devices:
            for d in config.devices:
                timeline.add_device(name=d["name"], device_type=d.get("device_type", "rhx"))
        else:
            # legacy: create one device per required_device type
            for dev_type in config.execution_control.required_devices:
                timeline.add_device(name=dev_type, device_type=dev_type)

        # map device name → index
        name_to_idx = {
            d[0]: i for i, d in enumerate(timeline._devices)
            if d[2] != "__system__"
        }

        # ensure at least one device exists
        if not name_to_idx:
            timeline.add_device(name="Default", device_type="rhx")
            name_to_idx = {
                d[0]: i for i, d in enumerate(timeline._devices)
                if d[2] != "__system__"
            }

        current_time = 0.0
        for step in config.sequence:
            duration = step.parameters.get("duration_s", 2.0)
            dev_idx = name_to_idx.get(step.device_name) if step.device_name else None
            if dev_idx is None:
                dev_idx = next(iter(name_to_idx.values()))
            timeline.add_block(dev_idx, step.action, start=current_time, duration=duration, params=dict(step.parameters))
            current_time += duration


def main():
    import importlib, os

    if '_PYI_SPLASH_IPC' in os.environ and importlib.util.find_spec("pyi_splash"):
        import pyi_splash
        pyi_splash.update_text('UI Loaded ...')
        pyi_splash.close()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    if hasattr(Qt, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)
    base_ss = qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=DarkPalette)
    override_ss = f"""
        QMainWindow {{ background-color: {BG_DARK}; }}
        QWidget {{ background-color: {BG_DARK}; }}
        QLabel {{ color: {TEXT_PRIMARY}; background: transparent; }}
        QTreeView {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                      border: 1px solid {BG_HEADER}; }}
        QTextEdit {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY};
                     border: 1px solid {BG_HEADER}; }}
        QSplitter::handle {{ background-color: {BG_HEADER}; }}
        QScrollArea {{ background: {BG_DARK}; }}
        QStatusBar {{ background-color: {BG_HEADER}; color: {TEXT_PRIMARY}; }}
        QProgressBar {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                        border: 1px solid {BG_HEADER}; text-align: center; }}
        QProgressBar::chunk {{ background-color: {ACCENT_BLUE}; }}
        QTabWidget::pane {{ background-color: {BG_DARK}; border: 1px solid {BG_HEADER}; }}
        QTabBar::tab {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                        border: 1px solid {BG_HEADER}; padding: 4px 8px; }}
        QTabBar::tab:selected {{ background-color: {BG_HEADER}; }}
        QComboBox {{ background-color: {BG_SURFACE}; color: {TEXT_PRIMARY};
                     border: 1px solid {BG_HEADER}; padding: 2px 4px; }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY};
                                       selection-background-color: {BG_HEADER}; }}
        QLineEdit, QSpinBox, QDoubleSpinBox {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY};
                                               border: 1px solid {BG_HEADER}; padding: 2px 4px; }}
        QToolBar {{ background-color: {BG_HEADER}; border: none; spacing: 4px; }}
        QToolButton {{ color: {TEXT_PRIMARY}; background: transparent; border: none; padding: 2px 6px; }}
        QToolButton:hover {{ background-color: {BG_SURFACE}; }}
        QToolButton:pressed, QToolButton:checked {{ background-color: {ACCENT_BLUE}; }}
        QMenu {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY}; border: 1px solid {BG_HEADER}; }}
        QMenu::item:selected {{ background-color: {BG_HEADER}; }}
    """
    app.setStyleSheet(base_ss + override_ss)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
