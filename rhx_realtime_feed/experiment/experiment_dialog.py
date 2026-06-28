from pathlib import Path

from PyQt5 import QtWidgets, QtCore

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
