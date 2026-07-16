import time

from PyQt5 import QtWidgets

from leech.telemetry_logger import append_telemetry_line


class DeviceTab(QtWidgets.QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self._clear_requested = False
        self._render_requested = False
        self._tel_last_emit = time.perf_counter()
        self._tel_interval_sec = 5.0
        self._tel_chunks = 0
        self._tel_samples = 0
        self._tel_render_calls = 0
        self._tel_render_ms_total = 0.0
        self._tel_ingest_ms_total = 0.0
        self._tel_raw_renders = 0

    def on_data(self, chunk):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def shutdown(self) -> bool:
        raise NotImplementedError

    def set_connection_details(self, host="", command_port=0, data_port=0, sample_rate=0, project_name=""):
        pass

    def set_receiving_state(self, receiving: bool):
        pass

    def request_render(self):
        self._render_requested = True

    def _emit_telemetry_if_due(self, prefix="tab"):
        now = time.perf_counter()
        elapsed = now - self._tel_last_emit
        if elapsed < self._tel_interval_sec:
            return
        chunks = max(1, int(self._tel_chunks))
        samples = int(self._tel_samples)
        rate_hz = float(samples) / max(elapsed, 1e-6)
        avg_ingest_ms = self._tel_ingest_ms_total / chunks
        avg_render_ms = self._tel_render_ms_total / max(1, int(self._tel_render_calls))
        line = (
            f"[telemetry][{prefix}] "
            f"window_s={elapsed:.2f} chunks={chunks} samples={samples} rate_hz={rate_hz:.1f} "
            f"ingest_avg_ms={avg_ingest_ms:.3f} render_avg_ms={avg_render_ms:.3f} "
            f"raw_renders={int(self._tel_raw_renders)}"
        )
        append_telemetry_line(line)
        self._tel_last_emit = now
        self._tel_chunks = 0
        self._tel_samples = 0
        self._tel_render_calls = 0
        self._tel_render_ms_total = 0.0
        self._tel_ingest_ms_total = 0.0
        self._tel_raw_renders = 0
