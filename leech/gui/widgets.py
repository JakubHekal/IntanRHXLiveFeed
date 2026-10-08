from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

_PORTS = ['A', 'B', 'C', 'D']


class _ChannelDialog(QDialog):
    def __init__(self, current_value="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        self._rows = []
        for port in _PORTS:
            row = QHBoxLayout()
            cb = QCheckBox(f"Port {port}")
            lo = QSpinBox()
            lo.setRange(0, 31)
            lo.setValue(0)
            hi = QSpinBox()
            hi.setRange(0, 31)
            hi.setValue(31)
            row.addWidget(cb)
            row.addWidget(QLabel("Ch"))
            row.addWidget(lo)
            row.addWidget(QLabel("to"))
            row.addWidget(hi)
            self._rows.append((port, cb, lo, hi))
            layout.addLayout(row)

        self._set_from_value(current_value)

        self._summary = QLabel()
        layout.addWidget(self._summary)

        btns = QHBoxLayout()
        btns.addStretch()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

        for _, cb, lo, hi in self._rows:
            cb.stateChanged.connect(self._update_summary)
            lo.valueChanged.connect(self._update_summary)
            hi.valueChanged.connect(self._update_summary)
        self._update_summary()

    def _parse_value(self, value):
        try:
            indices = set()
            for part in value.split(','):
                part = part.strip()
                if not part:
                    continue
                if '-' in part:
                    a, b = part.split('-', 1)
                    indices.update(range(int(a), int(b) + 1))
                else:
                    indices.add(int(part))
            return indices
        except Exception:
            return set()

    def _set_from_value(self, value):
        if not value:
            return
        indices = self._parse_value(value)
        port_map = {p: set() for p in _PORTS}
        for idx in indices:
            p_idx = idx // 32
            ch = idx % 32
            if p_idx < 4:
                port_map[_PORTS[p_idx]].add(ch)
        for port, cb, lo, hi in self._rows:
            chs = port_map[port]
            if chs:
                cb.setChecked(True)
                lo.setValue(min(chs))
                hi.setValue(max(chs))

    def _update_summary(self):
        total = 0
        parts = []
        for port, cb, lo, hi in self._rows:
            if cb.isChecked():
                mn, mx = lo.value(), hi.value()
                n = mx - mn + 1
                total += n
                parts.append(f"{port}:{mn}-{mx}")
        self._summary.setText(
            f"{', '.join(parts)}  ({total} ch)" if parts else "No channels selected"
        )

    def value(self):
        ranges = []
        for port, cb, lo, hi in self._rows:
            if cb.isChecked():
                offset = _PORTS.index(port) * 32
                ranges.append(f"{offset + lo.value()}-{offset + hi.value()}")
        return ",".join(ranges)


class ChannelSelector(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = ""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._btn = QPushButton("Select channels...")
        self._btn.clicked.connect(self._open_dialog)
        layout.addWidget(self._btn)

    def value(self):
        return self._value

    def setValue(self, val):
        self._value = str(val or "")
        self._btn.setText(self._display_text())

    def _display_text(self):
        if not self._value:
            return "Select channels..."
        try:
            total = 0
            for part in self._value.split(','):
                if '-' in part:
                    a, b = part.split('-', 1)
                    total += int(b) - int(a) + 1
                else:
                    total += 1
            return f"{self._value}  ({total} ch)"
        except Exception:
            return self._value

    def _open_dialog(self):
        dlg = _ChannelDialog(self._value, self)
        if dlg.exec_() == QDialog.Accepted:
            new_val = dlg.value()
            if new_val != self._value:
                self._value = new_val
                self._btn.setText(self._display_text())
                self.valueChanged.emit()


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
