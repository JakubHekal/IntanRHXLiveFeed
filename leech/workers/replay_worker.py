import csv
import json
import re
from pathlib import Path

import numpy as np
from PyQt5 import QtCore

from leech.experiment.experiment import resolve_storage
from leech.workers.replay_source import make_source


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
        self._source = None
        self._chunk_files = []
        self._chunk_idx = 0

    def _device_record(self):
        metadata = self._metadata()
        devices = metadata.get("devices")
        if not isinstance(devices, list):
            return {}
        for item in devices:
            if not isinstance(item, dict):
                continue
            if self._device_id and item.get("device_id") == self._device_id:
                return item
            if self._source_device_name and item.get("name") == self._source_device_name:
                return item
        return devices[0] if devices and isinstance(devices[0], dict) else {}

    def start(self):
        record = self._device_record()
        try:
            fmt, chunks_dir = resolve_storage(record, self._run_path, self._device_dir)
        except (OSError, ValueError, FileNotFoundError) as exc:
            self.error.emit(str(exc))
            return
        device_type = str(record.get("device_type", ""))
        if fmt == "csv" and ("rhx" in device_type.lower() or "intan" in device_type.lower()):
            self.error.emit(
                "Old Intan CSV run — replay supports native .rhd/.rhs or HDF5."
            )
            return
        try:
            self._source = make_source(
                chunks_dir, fmt,
                label_order=self._label_order(),
                device_type=device_type,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.error.emit(str(exc))
            return
        self._chunk_idx = 0
        self._timer.start(max(1, int(1000 * 0.05)))

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
        if path.stat().st_size == 0:
            return None
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
            return None
        return data.T

    def _finish(self):
        self._timer.stop()
        source = self._source
        self._source = None
        closer = getattr(source, "close", None)
        if callable(closer):
            closer()

    def _emit_next_chunk(self):
        if self._source is None:
            self._finish()
            self.finished.emit()
            return
        try:
            n = max(1, int(self._source.sample_rate * 0.05))
            data, _markers = self._source.read(n)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self._finish()
            self.error.emit(f"Could not replay {self._device_name}: {exc}")
            return
        if data is None or data.shape[1] == 0:
            self._finish()
            self.finished.emit()
            return
        self.data_received.emit(self._device_name, data)
