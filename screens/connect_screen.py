from PyQt5 import QtWidgets, QtCore

class ConnectScreen(QtWidgets.QWidget):
    connection_request_signal = QtCore.pyqtSignal(str, int, int, int, str, str, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        # Title
        title = QtWidgets.QLabel("RHX connection setup")
        title.setWordWrap(False)
        title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        title.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(title)

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

        project_group_box.setLayout(project_form)

        # Connect button and error label

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.PlainText)
        self.status_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.status_label.setMinimumHeight(0)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_button)
    
    def set_busy(self, busy: bool):
        if busy:
            self.set_status_message("Connecting...")
        else:
            self.reset_status_message()
    
        self.connect_button.setDisabled(busy)
        self.host_edit.setDisabled(busy)
        self.command_port_edit.setDisabled(busy)
        self.data_port_edit.setDisabled(busy)
        self.project_name_edit.setDisabled(busy)

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

    def _on_connect_clicked(self):
        self.reset_status_message()

        host = self.host_edit.text().strip()
        if not host:
            self.set_status_message("Host is required.", error=True)
            return

        try:
            command_port = int(self.command_port_edit.text().strip())
            data_port = int(self.data_port_edit.text().strip())
            if not (0 < command_port < 65536) or not (0 < data_port < 65536):
                raise ValueError
        except ValueError:
            self.set_status_message("Ports must be valid integers from 1 to 65535.", error=True)
            return

        try:
            sample_rate = int(self.sample_rate_edit.text().strip())
            if sample_rate <= 0:
                raise ValueError
        except ValueError:
            self.set_status_message("Sample rate must be a positive integer.", error=True)
            return

        project_name = self.project_name_edit.text().strip()
        if not project_name:
            self.set_status_message("Project name is required.", error=True)
            return

        port = self.port_combo.currentText()
        channel = int(self.channel_combo.currentText())

        self.connection_request_signal.emit(host, command_port, data_port, sample_rate, project_name, port, channel)

    

