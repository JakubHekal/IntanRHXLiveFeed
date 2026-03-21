from PyQt5 import QtCore, QtWidgets


class MarkerDialog(QtWidgets.QDialog):
    rename_requested = QtCore.pyqtSignal(int, str)
    delete_requested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Markers")
        self.setModal(False)
        self.resize(620, 360)

        self._markers = []

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["ID", "Timestamp (s)", "Name"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)

        self.rename_button = QtWidgets.QPushButton("Rename")
        self.rename_button.clicked.connect(self._on_rename)
        buttons.addWidget(self.rename_button)

        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete)
        buttons.addWidget(self.delete_button)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(lambda: self.set_markers(self._markers))
        buttons.addWidget(self.refresh_button)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)

        layout.addLayout(buttons)

    def set_markers(self, markers):
        self._markers = list(markers or [])
        self.table.setRowCount(0)

        ordered = sorted(self._markers, key=lambda m: float(m.get("timestamp_s", 0.0)))
        for m in ordered:
            row = self.table.rowCount()
            self.table.insertRow(row)

            marker_id = int(m.get("id", 0))
            timestamp_s = float(m.get("timestamp_s", 0.0))
            name = str(m.get("name", ""))
            id_item = QtWidgets.QTableWidgetItem(str(marker_id))
            id_item.setData(QtCore.Qt.UserRole, marker_id)
            ts_item = QtWidgets.QTableWidgetItem(f"{timestamp_s:.6f}")
            name_item = QtWidgets.QTableWidgetItem(name)

            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, ts_item)
            self.table.setItem(row, 2, name_item)

        self.table.resizeColumnsToContents()

    def _selected_marker_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        value = item.data(QtCore.Qt.UserRole)
        if value is None:
            return None
        return int(value)

    def _on_rename(self):
        marker_id = self._selected_marker_id()
        if marker_id is None:
            QtWidgets.QMessageBox.information(self, "Markers", "Select a marker to rename.")
            return
        row = self.table.currentRow()
        current_name = self.table.item(row, 2).text() if row >= 0 and self.table.item(row, 2) else ""
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Marker", "New marker name:", text=current_name)
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QtWidgets.QMessageBox.warning(self, "Markers", "Marker name cannot be empty.")
            return
        self.rename_requested.emit(marker_id, new_name)

    def _on_delete(self):
        marker_id = self._selected_marker_id()
        if marker_id is None:
            QtWidgets.QMessageBox.information(self, "Markers", "Select a marker to delete.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Marker",
            "Delete selected marker? This will update marker files and raw chunk marker fields.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self.delete_requested.emit(marker_id)
