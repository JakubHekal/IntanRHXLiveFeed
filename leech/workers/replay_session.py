"""One shared-clock replay session for an entire recorded run."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5 import QtCore

from leech.experiment.experiment import resolve_storage
from leech.workers.replay_source import make_source

_ACTIVE_SESSION = None


class ReplaySession(QtCore.QObject):
    """Drives every replayable device in a run from one timer.

    Only one session may be active at a time. Callers must stop it before
    starting another, and must not record while it runs.
    """

    device_configured = QtCore.pyqtSignal(str, int, list, float)
    data_received = QtCore.pyqtSignal(str, object)
    markers_received = QtCore.pyqtSignal(str, list)
    progress = QtCore.pyqtSignal(float, float)  # elapsed_sec, total_sec (0 = unknown)
    unsupported = QtCore.pyqtSignal(str, str)  # device_name, message
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()

    BLOCK_SEC = 0.05

    def __init__(self, run_path, parent=None):
        super().__init__(parent)
        self._run_path = Path(run_path)
        self._speed = 1.0
        self._instant = False
        self._sources = {}
        self._emitted = {}
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._elapsed = 0.0
        self._total = 0.0
        self._running = False
        self._finished = False

    # ── lifecycle ───────────────────────────────────────────────────────
    @staticmethod
    def active():
        return _ACTIVE_SESSION

    def start(self, speed=1.0) -> bool:
        global _ACTIVE_SESSION
        if _ACTIVE_SESSION is not None:
            self.error.emit("A replay is already running.")
            return False
        if self._running or self._finished:
            return False
        self._elapsed = 0.0
        self._total = 0.0
        self._emitted = {}
        metadata = self._metadata()
        records = self._device_records(metadata)
        if not records:
            self.error.emit("Run contains no devices to replay.")
            return False

        failures = []
        for record in records:
            name = str(record.get("name") or record.get("device_name") or "Device")
            config = record.get("config") if isinstance(record.get("config"), dict) else {}
            try:
                fmt, chunks_dir = resolve_storage(record, self._run_path)
            except (OSError, ValueError, FileNotFoundError) as exc:
                failures.append(f"{name}: {exc}")
                continue
            if fmt == "rhs":
                self.unsupported.emit(
                    name, "Native RHX recording — open this run in Intan RHX."
                )
                continue
            try:
                source = make_source(
                    chunks_dir, fmt,
                    sample_rate=float(config.get("sample_rate", 20000.0)),
                    label_order=self._label_order(record),
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                failures.append(f"{name}: {exc}")
                continue
            labels = source.labels or [
                f"CH{i:03d}" for i in range(max(1, source.num_channels))
            ]
            if source.num_channels <= 0:
                failures.append(f"{name}: no channels in {chunks_dir}")
                continue
            self._sources[name] = source
            self.device_configured.emit(
                name, source.num_channels, labels, float(source.sample_rate)
            )
            if source.total_samples:
                self._total = max(
                    self._total, float(source.total_samples) / source.sample_rate
                )

        if not self._sources:
            details = "\n".join(failures) if failures else "No replayable data found."
            self.error.emit(details)
            return False
        for failure in failures:
            self.error.emit(failure)

        _ACTIVE_SESSION = self
        self._running = True
        self._set_speed(speed)
        self._sync_interval()
        self._timer.start()
        return True

    def stop(self):
        global _ACTIVE_SESSION
        was_running = self._running
        self._running = False
        self._timer.stop()
        self._close_sources()
        if _ACTIVE_SESSION is self:
            _ACTIVE_SESSION = None
        if was_running:
            self.stopped.emit()

    # ── speed ───────────────────────────────────────────────────────────
    def set_speed(self, speed):
        self._set_speed(speed)
        self._sync_interval()

    def _set_speed(self, speed):
        if isinstance(speed, str) and speed.lower() in ("instant", "max"):
            self._instant = True
            self._speed = 1.0
            return
        try:
            value = float(speed)
        except (TypeError, ValueError):
            value = 1.0
        self._instant = False
        self._speed = max(0.1, value)

    def _sync_interval(self):
        if self._instant:
            self._timer.setInterval(0)
        else:
            self._timer.setInterval(max(1, int(self.BLOCK_SEC * 1000.0 / self._speed)))

    # ── clock ───────────────────────────────────────────────────────────
    def _tick(self):
        if not self._running:
            return
        blocks = 20 if self._instant else 1
        alive = False
        for name, source in list(self._sources.items()):
            n = max(1, int(source.sample_rate * self.BLOCK_SEC)) * blocks
            try:
                chunk, markers = source.read(n)
            except Exception as exc:  # noqa: BLE001 - report and drop this source
                self.error.emit(f"{name}: {exc}")
                self._sources.pop(name, None)
                continue
            if chunk is None or chunk.shape[1] == 0:
                self._sources.pop(name, None)
                continue
            alive = True
            base = self._emitted.get(name, 0)
            self._emitted[name] = base + chunk.shape[1]
            self.data_received.emit(name, chunk)
            if markers:
                self.markers_received.emit(
                    name,
                    [
                        {
                            "id": marker_id,
                            "name": marker_name,
                            "timestamp_s": (base + offset) / source.sample_rate,
                        }
                        for offset, marker_id, marker_name in markers
                    ],
                )
        self._elapsed += self.BLOCK_SEC * (20 if self._instant else 1) * self._speed
        self.progress.emit(self._elapsed, self._total)
        if not alive and not self._sources:
            self._finish()

    def _finish(self):
        self._timer.stop()
        self._running = False
        self._finished = True
        self._close_sources()
        global _ACTIVE_SESSION
        if _ACTIVE_SESSION is self:
            _ACTIVE_SESSION = None
        self.finished.emit()

    def _close_sources(self):
        for source in self._sources.values():
            closer = getattr(source, "close", None)
            if callable(closer):
                closer()
        self._sources.clear()

    # ── metadata ────────────────────────────────────────────────────────
    @staticmethod
    def _label_order(record):
        blocks = record.get("blocks") if isinstance(record, dict) else None
        if not isinstance(blocks, list):
            return {}
        return {
            block["label"].casefold(): index
            for index, block in enumerate(blocks)
            if isinstance(block, dict) and isinstance(block.get("label"), str)
        }

    def _metadata(self):
        for name in ("run.json", "metadata.json"):
            path = self._run_path / name
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return {}
        return {}

    @staticmethod
    def _device_records(metadata):
        devices = metadata.get("devices")
        if isinstance(devices, list) and devices:
            return [item for item in devices if isinstance(item, dict)]
        if metadata.get("device_name") or metadata.get("device_type"):
            return [metadata]
        return []
