import traceback
from datetime import datetime
from pathlib import Path

from PyQt5 import QtWidgets, QtCore

from ..device.widget_builder import build_param_widget, read_param_widget
from .experiment import ExperimentManager


class ExperimentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("New Experiment")
        self.setModal(True)

        self._result_path = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

        # Experiment details
        details_group = QtWidgets.QGroupBox("Experiment details")
        details_form = QtWidgets.QFormLayout()
        details_form.setSpacing(10)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("My Experiment")
        details_form.addRow("Name:", self.name_edit)

        self.author_edit = QtWidgets.QLineEdit()
        self.author_edit.setPlaceholderText("Dr. Jane Doe")
        details_form.addRow("Author:", self.author_edit)

        self.version_edit = QtWidgets.QLineEdit("1.0.0")
        details_form.addRow("Version:", self.version_edit)

        self.description_edit = QtWidgets.QTextEdit()
        self.description_edit.setPlaceholderText("Optional description...")
        self.description_edit.setMaximumHeight(80)
        details_form.addRow("Description:", self.description_edit)

        details_group.setLayout(details_form)
        layout.addWidget(details_group)

        # Storage location
        storage_group = QtWidgets.QGroupBox("Storage location")
        storage_form = QtWidgets.QFormLayout()
        storage_form.setSpacing(10)

        default_path = str((Path.cwd() / "experiments").resolve())
        self.path_edit = QtWidgets.QLineEdit(default_path)
        self.path_browse_button = QtWidgets.QPushButton("Browse...")
        self.path_browse_button.clicked.connect(self._on_browse)

        path_row = QtWidgets.QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.path_browse_button)
        storage_form.addRow("Base path:", path_row)

        storage_group.setLayout(storage_form)
        layout.addWidget(storage_group)

        # Buttons
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)

        self.create_button = QtWidgets.QPushButton("Create")
        self.create_button.clicked.connect(self._on_create)
        self.create_button.setDefault(True)
        button_row.addWidget(self.create_button)

        layout.addLayout(button_row)

    def _on_browse(self):
        current = self.path_edit.text().strip() or str(Path.cwd())
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select experiments directory",
            current,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if selected:
            self.path_edit.setText(selected)

    def _on_create(self):
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Experiment name is required.")
            return

        base_path = self.path_edit.text().strip()
        if not base_path:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Base path is required.")
            return

        author = self.author_edit.text().strip()
        description = self.description_edit.toPlainText().strip()

        try:
            self._result_path = ExperimentManager.create(base_path, name, author, description)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create experiment:\n{e}")
            return

        self.accept()

    def result_path(self) -> str:
        return str(self._result_path) if self._result_path else ""


class _DeviceConnectionFrame(QtWidgets.QFrame):
    status_changed = QtCore.pyqtSignal()

    def __init__(self, device_group: dict, parent=None):
        super().__init__(parent)
        self._dg = device_group
        self._device_instance = None
        self._param_widgets = []

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QtWidgets.QLabel(device_group["name"])
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        if not device_group["device_class"] or not device_group["param_defs"]:
            layout.addWidget(QtWidgets.QLabel("(no configurable parameters)"))
        else:
            form = QtWidgets.QFormLayout()
            form.setSpacing(4)
            for pd in device_group["param_defs"]:
                val = device_group["current_config"].get(pd.name, pd.default)
                w = build_param_widget(pd, val)
                self._param_widgets.append((pd.name, w, pd))
                form.addRow(pd.label + ":", w)
            layout.addLayout(form)

        status_row = QtWidgets.QHBoxLayout()
        status_row.addStretch()
        self._connect_button = QtWidgets.QPushButton("Connect")
        self._connect_button.clicked.connect(self._on_connect)
        self._connect_button.setFixedWidth(100)
        status_row.addWidget(self._connect_button)
        self._status_label = QtWidgets.QLabel("\u26A0 Disconnected")
        status_row.addWidget(self._status_label)
        layout.addLayout(status_row)

    def _read_params(self):
        from ..device.widget_builder import gather_params
        return gather_params(self._param_widgets)

    def _on_connect(self):
        self._connect_button.setEnabled(False)
        self._status_label.setText("\U0001F504 Connecting...")
        QtWidgets.QApplication.processEvents()

        cls = self._dg["device_class"]
        if cls is None:
            self._status_label.setText("\u26A0 No driver available")
            self._connect_button.setEnabled(True)
            self.status_changed.emit()
            return

        params = self._read_params()
        try:
            device = cls(**params)
            ok = device.connect()
            if not ok:
                self._status_label.setText("\u2717 Connection failed")
                self._connect_button.setEnabled(True)
                self.status_changed.emit()
                return
            self._device_instance = device
            self._status_label.setText("\u2713 Connected")
            self._connect_button.setText("Connected")
            self._connect_button.setEnabled(False)
        except Exception as e:
            self._status_label.setText(f"\u2717 Failed: {e}")
            tb = traceback.format_exc()
            print(f"[RunDialog] Connection failed for {self._dg['name']}: {tb}")
            self._connect_button.setEnabled(True)

        self.status_changed.emit()

    def is_connected(self) -> bool:
        return self._device_instance is not None and getattr(self._device_instance, 'connected', False)

    def get_params(self) -> dict:
        return self._read_params()

    def get_device_type(self) -> str:
        return self._dg["device_type"]

    def get_device_name(self) -> str:
        return self._dg["name"]

    def get_device_instance(self):
        return self._device_instance


class RunExperimentDialog(QtWidgets.QDialog):
    def __init__(self, experiment_name: str, experiment_path: str, device_groups: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(f"Run Experiment \u2014 {experiment_name}")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._experiment_path = experiment_path
        self._device_groups = device_groups
        self._device_sections = []
        self._run_path = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        exp_label = QtWidgets.QLabel(f"Experiment:  {experiment_name}")
        exp_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(exp_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        for dg in device_groups:
            section = _DeviceConnectionFrame(dg, self)
            self._device_sections.append(section)
            scroll_layout.addWidget(section)
            section.status_changed.connect(self._update_start_button)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(sep)

        name_row = QtWidgets.QHBoxLayout()
        name_row.setSpacing(8)
        name_row.addWidget(QtWidgets.QLabel("Run name:"))
        self._run_name_edit = QtWidgets.QLineEdit()
        self._run_name_edit.setPlaceholderText("auto-timestamp")
        name_row.addWidget(self._run_name_edit, 1)
        layout.addLayout(name_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)
        self._start_btn = QtWidgets.QPushButton("Start Run")
        self._start_btn.clicked.connect(self._on_start)
        self._start_btn.setEnabled(False)
        self._start_btn.setDefault(True)
        btn_row.addWidget(self._start_btn)
        layout.addLayout(btn_row)

    def _all_connected(self) -> bool:
        return all(s.is_connected() for s in self._device_sections)

    def _update_start_button(self):
        name = self._run_name_edit.text().strip() or "auto-timestamp"
        self._start_btn.setEnabled(self._all_connected() and bool(name))

    def _on_start(self):
        run_name = self._run_name_edit.text().strip()
        if not run_name:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            self._run_path = ExperimentManager.start_run(self._experiment_path, run_name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create run:\n{e}")
            return

        self.accept()

    def run_path(self) -> str:
        return str(self._run_path) if self._run_path else ""

    def device_configs(self) -> dict:
        return {s.get_device_type(): s.get_params() for s in self._device_sections}

    def device_instances(self) -> list:
        return [s.get_device_instance() for s in self._device_sections if s.is_connected()]

    def device_names(self) -> list[str]:
        return [s.get_device_name() for s in self._device_sections]
