import csv
import json
import re
from pathlib import Path

import numpy as np
from PyQt5 import QtCore


class ReplayWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(str, object)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        run_path,
        device_name,
        parent=None,
        device_dir=None,
        device_id="",
        source_device_name="",
    ):
        super().__init__(parent)
        self._run_path = Path(run_path)
        self._device_name = device_name
        self._device_dir = Path(device_dir) if device_dir else None
        self._device_id = device_id
        self._source_device_name = source_device_name
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._emit_next_chunk)
        self._chunk_files = []
        self._chunk_idx = 0

    def start(self):
        chunks_dir = self._device_dir
        if chunks_dir is None:
            chunks_dir = self._run_path / "raw"
        if not chunks_dir.exists():
            self.error.emit(f"No raw/ dir in {chunks_dir}")
            return

        self._chunk_files = sorted(chunks_dir.rglob("chunk_*.csv"), key=self._chunk_sort_key)
        if not self._chunk_files:
            self.error.emit(f"No CSV chunks in {chunks_dir}")
            return

        self._chunk_idx = 0
        self._timer.start(500)

    def _metadata(self):
        for name in ("run.json", "metadata.json"):
            path = self._run_path / name
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return {}
        return {}

    def _label_order(self):
        metadata = self._metadata()
        devices = metadata.get("devices", [])
        if not isinstance(devices, list):
            return {}
        device = next(
            (
                item
                for item in devices
                if isinstance(item, dict) and item.get("device_id") == self._device_id
            ),
            None,
        )
        if device is None and self._source_device_name:
            device = next(
                (
                    item
                    for item in devices
                    if isinstance(item, dict) and item.get("name") == self._source_device_name
                ),
                None,
            )
        if device is None and len(devices) == 1 and isinstance(devices[0], dict):
            device = devices[0]
        blocks = device.get("blocks", []) if device else []
        if not isinstance(blocks, list):
            return {}
        return {
            block["label"].casefold(): index
            for index, block in enumerate(blocks)
            if isinstance(block, dict) and isinstance(block.get("label"), str)
        }

    def _chunk_sort_key(self, path):
        match = re.match(r"^chunk_(\d{6})_", path.name)
        if match:
            return 0, int(match.group(1)), path.name
        match = re.match(r"^chunk_(.*)_(\d{6})\.csv$", path.name)
        if match:
            label_order = self._label_order()
            label, chunk_index = match.groups()
            return 1, label_order.get(label.casefold(), len(label_order)), int(chunk_index), path.name
        return 2, path.stat().st_mtime_ns, path.name

    @staticmethod
    def _read_chunk(path):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        excluded = {"time_s", "marker_id", "marker_name"}
        columns = tuple(
            index for index, name in enumerate(header) if name.strip().lower() not in excluded
        )
        if not columns:
            raise ValueError(f"No channel columns in {path.name}")
        data = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=columns,
            dtype=np.float32,
            ndmin=2,
        )
        if data.size == 0:
            raise ValueError(f"No samples in {path.name}")
        return data.T

    def _emit_next_chunk(self):
        if self._chunk_idx >= len(self._chunk_files):
            self._timer.stop()
            self.finished.emit()
            return

        csv_path = self._chunk_files[self._chunk_idx]
        self._chunk_idx += 1
        try:
            data = self._read_chunk(csv_path)
            self.data_received.emit(self._device_name, data)
        except Exception as exc:
            self.error.emit(f"Could not replay {csv_path.name}: {exc}")
