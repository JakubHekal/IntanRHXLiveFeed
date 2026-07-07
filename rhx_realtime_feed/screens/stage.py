from PyQt5.QtCore import QPropertyAnimation, QRect, QSize, Qt, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QPushButton, QScrollArea, QSpinBox, QStyle, QToolButton, QVBoxLayout,
    QWidget,
)

from ._registry import _DEVICE_CLASSES, _SYSTEM_OPERATIONS
from .channel_selector import ChannelSelector
from .timeline import ExperimentTimeline
from .plot_screen import PlotScreen


BG_HEADER = "#2D2D2D"
TEXT_PRIMARY = "#EDEBE9"
ACCENT_BLUE = "#0078D4"


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
    run_action = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("Run History")
        header.setStyleSheet("padding: 8px;")
        layout.addWidget(header)

        self._empty_label = QLabel("No runs yet.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: #6C6C6C; padding: 40px 16px; border: 1px solid #3E3E3E; border-radius: 4px; margin: 8px;")
        layout.addWidget(self._empty_label, 1)

        self.run_list = QListWidget()
        self.run_list.setAlternatingRowColors(True)
        self.run_list.itemDoubleClicked.connect(self._on_run_activated)
        self.run_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.run_list.customContextMenuRequested.connect(self._on_run_context_menu)
        self.run_list.hide()
        layout.addWidget(self.run_list, 1)

        self.setMinimumWidth(260)

    def _on_run_activated(self, item):
        run_path = item.data(Qt.UserRole)
        if run_path:
            self.run_selected.emit(run_path)

    def _on_run_context_menu(self, pos):
        item = self.run_list.itemAt(pos)
        if item is None:
            return
        run_path = item.data(Qt.UserRole)
        if not run_path:
            return
        menu = QMenu(self)
        menu.addAction("Rerun", lambda: self.run_action.emit("rerun", run_path))
        menu.addSeparator()
        menu.addAction("Rename\u2026", lambda: self.run_action.emit("rename", run_path))
        menu.addAction("Delete\u2026", lambda: self.run_action.emit("delete", run_path))
        menu.exec_(self.run_list.viewport().mapToGlobal(pos))

    def reload_runs(self, runs_dir):
        self.run_list.clear()
        if not runs_dir or not __import__('pathlib').Path(runs_dir).exists():
            self._empty_label.setVisible(True)
            self.run_list.setVisible(False)
            return
        from pathlib import Path
        run_paths = sorted(Path(runs_dir).iterdir(), reverse=True)
        import json
        for rp in run_paths:
            if not rp.is_dir() or rp.name.startswith('.'):
                continue
            meta_file = rp / "run.json"
            if not meta_file.exists():
                meta_file = rp / "metadata.json"
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    pass
            if "run" in meta:
                run_data = meta["run"]
                exp_data = meta.get("experiment", {})
                name = exp_data.get("name", run_data.get("name", rp.name))
                ts = run_data.get("start_time", "")
                status = run_data.get("status", "unknown")
            else:
                name = meta.get("name", rp.name)
                ts = meta.get("timestamp", "")
                status = meta.get("status", "unknown")
            label = f"[{ts}] {name} \u2014 {status}" if ts else f"{name} \u2014 {status}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(rp))
            if status == "running":
                f = item.font()
                f.setBold(True)
                item.setFont(f)
                item.setForeground(QColor("#00FF00"))
            self.run_list.addItem(item)
        has_runs = self.run_list.count() > 0
        self._empty_label.setVisible(not has_runs)
        self.run_list.setVisible(has_runs)


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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("Properties")
        header.setStyleSheet("padding: 8px;")
        layout.addWidget(header)

        self._placeholder = QLabel("Select a block or device\nto edit its properties.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: #6C6C6C; padding: 40px 16px; border: 1px solid #3E3E3E; border-radius: 4px; margin: 8px;")
        layout.addWidget(self._placeholder, 1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.hide()
        container = QWidget()
        self._scroll.setWidget(container)
        form_layout = QVBoxLayout(container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(4)

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

        self._device_expander = FluentExpander("Device Properties", expanded=True)
        self._dp_widget = QWidget()
        self._dp_layout = QVBoxLayout(self._dp_widget)
        self._dp_layout.setContentsMargins(0, 4, 0, 0)
        self._dp_layout.setSpacing(4)
        self._dp_form = QFormLayout()
        self._dp_form.setSpacing(6)
        self._dp_layout.addLayout(self._dp_form)
        self._device_expander.setContentWidget(self._dp_widget)

        form_layout.addWidget(self._block_expander)
        form_layout.addWidget(self._device_expander)
        form_layout.addStretch()

        layout.addWidget(self._scroll, 1)
        self.setMinimumWidth(260)
        self._device_expander.hide()
        self._update_visibility()

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
        elif param_def.dtype == "channel_list":
            w = ChannelSelector()
            w.setValue(str(value or ""))
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
        elif param_def.dtype == "channel_list":
            return widget.value()
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
            elif isinstance(w, ChannelSelector):
                w.valueChanged.connect(slot)

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

    def _update_visibility(self):
        has_block = self._block_dev_idx >= 0 and self._block_idx >= 0
        has_device = self._device_idx >= 0
        visible = has_block or has_device
        self._placeholder.setVisible(not visible)
        self._scroll.setVisible(visible)
        self._block_expander.setVisible(has_block)
        self._device_expander.setVisible(has_device)

    def set_block_info(self, dev_idx, block_idx, display_name, op_name, start, duration, params=None, device_type=""):
        self._updating = True
        self._block_dev_idx = dev_idx
        self._block_idx = block_idx
        self._device_idx = -1

        if dev_idx < 0 or block_idx < 0:
            self._update_visibility()
            self._updating = False
            return

        self._block_op_name = op_name
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

        self._update_visibility()
        self._updating = False

    def set_device_info(self, dev_idx, device_type, config):
        self._updating = True
        self._device_idx = dev_idx
        self._block_dev_idx = -1
        self._block_idx = -1

        if dev_idx < 0:
            self._update_visibility()
            self._updating = False
            return

        label = f"Device: {device_type}" if device_type else "Device Properties"
        self._device_expander._title_label.setText(label)

        cls = _DEVICE_CLASSES.get(device_type)
        param_defs = cls.get_config_params() if cls else []
        self._populate_dynamic_form(self._dp_form, param_defs, config or {}, self._device_param_widgets)
        self._connect_param_signals(self._device_param_widgets, self._emit_device_params)

        self._update_visibility()
        self._updating = False

    def _on_block_field_changed(self, *args):
        if self._updating:
            return
        if self._block_dev_idx < 0 or self._block_idx < 0:
            return
        self._emit_block_params()


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

        btn_style = "QPushButton { border: 1px solid #555; border-radius: 3px; padding: 4px 10px; } QPushButton:hover { background: #3A3A3A; }"

        self.btn_play = QPushButton("Play")
        self.btn_play.setStyleSheet(btn_style)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(btn_style)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        timeline_bar.addWidget(self.btn_play)
        timeline_bar.addWidget(self.btn_pause)
        timeline_bar.addWidget(self.btn_stop)

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


