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
