import json
from pathlib import Path

import numpy as np
from PyQt5 import QtCore


class ReplayWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(str, object)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, run_path, device_name, parent=None):
        super().__init__(parent)
        self._run_path = Path(run_path)
        self._device_name = device_name
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._emit_next_chunk)
        self._chunk_files = []
        self._chunk_idx = 0

    def start(self):
        meta_file = self._run_path / "metadata.json"
        if not meta_file.exists():
            self.error.emit(f"No metadata.json in {self._run_path}")
            return

        chunks_dir = self._run_path / "raw_chunks"
        if not chunks_dir.exists():
            self.error.emit(f"No raw_chunks/ dir in {self._run_path}")
            return

        self._chunk_files = sorted(chunks_dir.rglob("*.csv"))
        if not self._chunk_files:
            self.error.emit(f"No CSV chunks in {chunks_dir}")
            return

        self._chunk_idx = 0
        # ponytail: fixed 500ms interval, roughly matches original chunk timing
        self._timer.start(500)

    def _emit_next_chunk(self):
        if self._chunk_idx >= len(self._chunk_files):
            self._timer.stop()
            self.finished.emit()
            return

        csv_path = self._chunk_files[self._chunk_idx]
        self._chunk_idx += 1
        try:
            # ponytail: skip header row, transpose to (channels, samples)
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1).T
            self.data_received.emit(self._device_name, data.astype(np.float32))
        except Exception:
            pass
