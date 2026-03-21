from pathlib import Path

from PyQt5 import QtWidgets, QtCore
from workers.processing_worker import PSD_BUFFER_SEC, WAVEFORM_BUFFER_SEC, SPIKE_BIN_SEC

class ConnectDialog(QtWidgets.QDialog):
    connection_request_signal = QtCore.pyqtSignal(str, int, int, int, str, str, str, int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Connect to RHX")
        self.setModal(True)

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        # Group boxes for organization of the input fields
        connection_group_box = QtWidgets.QGroupBox("Connection details")
        project_group_box = QtWidgets.QGroupBox("Project details")

        layout.addWidget(connection_group_box)
        layout.addWidget(project_group_box)

        # Connection details form
        connection_form = QtWidgets.QFormLayout()
        connection_form.setSpacing(10)

        self.host_edit = QtWidgets.QLineEdit("127.0.0.1")
        connection_form.addRow("Host:", self.host_edit)

        self.command_port_edit = QtWidgets.QLineEdit("5000")
        connection_form.addRow("Command port:", self.command_port_edit)

        self.data_port_edit = QtWidgets.QLineEdit("5001")
        connection_form.addRow("Data port:", self.data_port_edit)

        self.sample_rate_edit = QtWidgets.QLineEdit("20000")
        connection_form.addRow("Sample rate (Hz):", self.sample_rate_edit)

        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.addItems(["A", "B", "C", "D"])
        self.port_combo.setCurrentText("B")
        connection_form.addRow("Port:", self.port_combo)

        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems([str(i) for i in range(32)])
        self.channel_combo.setCurrentText("0")
        connection_form.addRow("Channel:", self.channel_combo)
       
        connection_group_box.setLayout(connection_form)

        # Project details form
        project_form = QtWidgets.QFormLayout()
        project_form.setSpacing(10)

        self.project_name_edit = QtWidgets.QLineEdit("New project")
        project_form.addRow("Project name:", self.project_name_edit)

        self.project_path_edit = QtWidgets.QLineEdit(str((Path.cwd() / "recordings").resolve()))
        self.project_path_browse_button = QtWidgets.QPushButton("Browse...")
        self.project_path_browse_button.clicked.connect(self._on_browse_project_path)
        path_row = QtWidgets.QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.addWidget(self.project_path_edit, 1)
        path_row.addWidget(self.project_path_browse_button)
        project_form.addRow("Project path:", path_row)

        self.psd_buffer_edit = QtWidgets.QLineEdit(str(int(PSD_BUFFER_SEC)))
        project_form.addRow("PSD buffer duration (s):", self.psd_buffer_edit)

        self.waveform_buffer_edit = QtWidgets.QLineEdit(str(int(WAVEFORM_BUFFER_SEC)))
        project_form.addRow("Waveform buffer duration (s):", self.waveform_buffer_edit)

        self.spike_bin_edit = QtWidgets.QLineEdit(str(int(SPIKE_BIN_SEC)))
        project_form.addRow("Spike count bin duration (s):", self.spike_bin_edit)

        project_group_box.setLayout(project_form)

        # Status + action buttons
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.PlainText)
        self.status_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.status_label.setMinimumHeight(0)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        self.connect_button.setDefault(True)
        button_row.addWidget(self.connect_button)

        layout.addLayout(button_row)
    
    def set_busy(self, busy: bool):
        if busy:
            self.set_status_message("Connecting...")
        else:
            self.reset_status_message()
    
        self.connect_button.setDisabled(busy)
        self.host_edit.setDisabled(busy)
        self.command_port_edit.setDisabled(busy)
        self.data_port_edit.setDisabled(busy)
        self.sample_rate_edit.setDisabled(busy)
        self.project_name_edit.setDisabled(busy)
        self.project_path_edit.setDisabled(busy)
        self.project_path_browse_button.setDisabled(busy)
        self.psd_buffer_edit.setDisabled(busy)
        self.waveform_buffer_edit.setDisabled(busy)
        self.spike_bin_edit.setDisabled(busy)
        self.port_combo.setDisabled(busy)
        self.channel_combo.setDisabled(busy)
        self.cancel_button.setDisabled(busy)

    def set_status_message(self, message: str, error: bool = False):
        self.status_label.setVisible(bool(message))
        if error:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("")

        self._adjust_window_height_only()

    def reset_status_message(self):
        self.status_label.setText("")
        self.status_label.setStyleSheet("")
        self.status_label.hide()

        self._adjust_window_height_only()

    def _adjust_window_height_only(self):
        window = self.window()
        if window is None or window.isMaximized():
            return

        current_width = window.width()

        def _resize():
            if window.isMaximized():
                return
            hint_h = max(window.minimumHeight(), window.sizeHint().height())
            window.resize(current_width, hint_h)

        QtCore.QTimer.singleShot(0, _resize)

    def _on_browse_project_path(self):
        current = self.project_path_edit.text().strip() or str(Path.cwd())
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select project storage path",
            current,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if selected:
            self.project_path_edit.setText(selected)

    def _on_connect_clicked(self):
        self.reset_status_message()

        host = self.host_edit.text().strip()
        if not host:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Host is required.")
            return

        try:
            command_port = int(self.command_port_edit.text().strip())
            data_port = int(self.data_port_edit.text().strip())
            if not (0 < command_port < 65536) or not (0 < data_port < 65536):
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Ports must be valid integers from 1 to 65535.")
            return

        try:
            sample_rate = int(self.sample_rate_edit.text().strip())
            if sample_rate <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Sample rate must be a positive integer.")
            return

        project_name = self.project_name_edit.text().strip()
        if not project_name:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Project name is required.")
            return

        project_path = self.project_path_edit.text().strip()
        if not project_path:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", "Project path is required.")
            return
        try:
            Path(project_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", f"Cannot create/access project path:\n{e}")
            return

        try:
            psd_buffer_sec = int(self.psd_buffer_edit.text().strip())
            waveform_buffer_sec = int(self.waveform_buffer_edit.text().strip())
            spike_bin_sec = int(self.spike_bin_edit.text().strip())
            if psd_buffer_sec <= 0 or waveform_buffer_sec <= 0 or spike_bin_sec <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Input",
                "PSD buffer, waveform buffer, and spike bin must be positive integers.",
            )
            return

        port = self.port_combo.currentText()
        channel = int(self.channel_combo.currentText())

        self.connection_request_signal.emit(
            host,
            command_port,
            data_port,
            sample_rate,
            project_name,
            project_path,
            port,
            channel,
            psd_buffer_sec,
            waveform_buffer_sec,
            spike_bin_sec,
        )

    

