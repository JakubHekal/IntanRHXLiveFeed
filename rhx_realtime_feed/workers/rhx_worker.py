import time
from datetime import datetime

from PyQt5 import QtCore
import numpy as np

from rhx_realtime_feed.device import Device, OutputSink
from rhx_realtime_feed.state_manager import StateManager, AppState
from rhx_realtime_feed.telemetry_logger import append_telemetry_line
from rhx_realtime_feed.workers.marker_manager import MarkerManager


CHANNELS_PORT = "B"
CHANNELS_TO_PLOT = [0]

PLOT_UPDATE_HZ = 20
PLOT_INTERVAL_MS = int(1000 / PLOT_UPDATE_HZ)
ACQUIRE_CHUNK_MS = max(20, PLOT_INTERVAL_MS)
NO_DATA_SLEEP_MS = max(1, PLOT_INTERVAL_MS // 4)
READ_WAIT_SLICE_MS = 2
WAITING_CHUNK_THRESHOLD = 10
NO_DATA_LOSS_TIMEOUT_SEC = 60.0
RAW_CHUNK_SEC = 300
TELEMETRY_EMIT_INTERVAL_SEC = 5.0
CSV_FLUSH_INTERVAL_SEC = 1.0
CSV_FILE_BUFFER_BYTES = 1024 * 1024
STARTUP_ZERO_SUPPRESS_SEC = 1.0


class RHXWorker(QtCore.QThread):
    connection_request_result_signal = QtCore.pyqtSignal(bool, str)
    data_received_signal = QtCore.pyqtSignal(object)
    acquisition_state_signal = QtCore.pyqtSignal(str)
    marker_added_signal = QtCore.pyqtSignal(float)
    marker_catalog_signal = QtCore.pyqtSignal(object)

    def __init__(self, device: Device, output_sink: OutputSink,
                 project_name, project_path=None):
        super().__init__()
        self.device = device
        self.output_sink = output_sink
        self.project_name = project_name
        self.project_path = project_path

        self.marker_manager = MarkerManager()

        self.project_run_dir = None

        self.sample_rate = device.sample_rate if device.sample_rate else 20000.0
        self._chunk_max_samples = max(1, int(round(self.sample_rate * RAW_CHUNK_SEC)))

        self.empty_chunk_count = 0
        self.repeated_chunk_count = 0
        self._last_chunk_data = None
        self._seen_nonzero_chunk = False
        self._stream_started_t = 0.0
        self._last_fresh_data_monotonic = 0.0

        now = time.perf_counter()
        self._telemetry_last_emit = now
        self._telemetry_chunk_count = 0
        self._telemetry_sample_count = 0
        self._telemetry_acquire_ms_total = 0.0
        self._telemetry_csv_ms_total = 0.0
        self._telemetry_emit_ms_total = 0.0
        self._telemetry_empty_chunks = 0
        self._telemetry_repeated_chunks = 0
        self._telemetry_invalid_chunks = 0
        self._telemetry_zero_suppressed = 0

        self.state_manager = StateManager.get_instance()
        self._last_emitted_state = None
        self._paused = False

    def __del__(self):
        self.stop()

    def stop(self, timeout_ms: int = 3000) -> bool:
        if not self.isRunning():
            return True
        self.requestInterruption()
        try:
            if self.device is not None:
                try:
                    self.device.stop_acquisition()
                except Exception:
                    pass
                try:
                    self.device.close()
                except Exception:
                    pass
        except Exception:
            pass
        if self.wait(max(0, int(timeout_ms))):
            return True
        print(f"[WORKER] RHXWorker stop timeout after {int(timeout_ms)} ms")
        return False

    def pause_receiving(self):
        self._paused = True

    def resume_receiving(self):
        self._paused = False

    def request_marker(self, marker_name: str = ""):
        marker_id, marker_ts, markers_snapshot = self.marker_manager.add_marker(
            sample_index=self.output_sink.sample_index,
            sample_rate=self.sample_rate,
            marker_name=marker_name,
        )
        self.marker_added_signal.emit(marker_ts)
        self.marker_catalog_signal.emit(markers_snapshot)

    def get_project_paths(self):
        return {
            "run_dir": str(self.project_run_dir) if self.project_run_dir else "",
            "raw_chunks_dir": "",
            "markers_csv": str(self.marker_manager.markers_csv_path) if self.marker_manager.markers_csv_path else "",
        }

    def get_markers(self):
        return self.marker_manager.get_markers()

    def request_rename_marker(self, marker_id: int, new_name: str):
        return self.marker_manager.request_rename(marker_id, new_name)

    def request_delete_marker(self, marker_id: int):
        return self.marker_manager.request_delete(marker_id)

    def _emit_telemetry_if_due(self):
        now = time.perf_counter()
        elapsed = now - self._telemetry_last_emit
        if elapsed < TELEMETRY_EMIT_INTERVAL_SEC:
            return

        chunks = max(1, int(self._telemetry_chunk_count))
        samples = int(self._telemetry_sample_count)
        rate_hz = float(samples) / max(elapsed, 1e-6)
        avg_acquire_ms = self._telemetry_acquire_ms_total / chunks
        avg_csv_ms = self._telemetry_csv_ms_total / chunks
        avg_emit_ms = self._telemetry_emit_ms_total / chunks

        line = (
            "[telemetry][rhx] "
            f"window_s={elapsed:.2f} chunks={chunks} samples={samples} rate_hz={rate_hz:.1f} "
            f"avg_acquire_ms={avg_acquire_ms:.3f} avg_csv_ms={avg_csv_ms:.3f} avg_emit_ms={avg_emit_ms:.3f} "
            f"empty={int(self._telemetry_empty_chunks)} repeated={int(self._telemetry_repeated_chunks)} invalid={int(self._telemetry_invalid_chunks)} "
            f"zero_suppressed={int(self._telemetry_zero_suppressed)}"
        )
        print(line)
        append_telemetry_line(line)

        self._telemetry_last_emit = now
        self._telemetry_chunk_count = 0
        self._telemetry_sample_count = 0
        self._telemetry_acquire_ms_total = 0.0
        self._telemetry_csv_ms_total = 0.0
        self._telemetry_emit_ms_total = 0.0
        self._telemetry_empty_chunks = 0
        self._telemetry_repeated_chunks = 0
        self._telemetry_invalid_chunks = 0
        self._telemetry_zero_suppressed = 0

    def run(self):
        """Main device worker thread loop."""
        try:
            if not self.device.connected:
                if not self.device.connect():
                    self.connection_request_result_signal.emit(False, "Device connection failed")
                    return

            if self.device.sample_rate and self.device.sample_rate > 0:
                self.sample_rate = float(self.device.sample_rate)
                self._chunk_max_samples = max(1, int(round(self.sample_rate * RAW_CHUNK_SEC)))

            self.device.start_acquisition()

            self.project_run_dir, _ = self.output_sink.start_session(
                self.project_name, self.project_path
            )
            self.marker_manager.initialize(self.project_run_dir)
        except Exception as e:
            self.connection_request_result_signal.emit(False, str(e))
            return

        self.connection_request_result_signal.emit(True, None)
        self.acquisition_state_signal.emit("running")
        self._last_fresh_data_monotonic = QtCore.QElapsedTimer()
        self._last_fresh_data_monotonic.start()
        self._stream_started_t = time.perf_counter()
        self._seen_nonzero_chunk = False

        try:
            is_currently_streaming = True
            read_period_s = float(ACQUIRE_CHUNK_MS) / 1000.0
            next_read_t = time.perf_counter()

            while not self.isInterruptionRequested():
                if self.device is not None and not self.device.connected:
                    break

                marker_catalog, marker_csv_path = self.marker_manager.process_commands()
                if marker_catalog is not None:
                    self.marker_catalog_signal.emit(marker_catalog)

                should_stream = not self._paused

                if not should_stream and is_currently_streaming:
                    try:
                        self.device.stop_acquisition()
                    except Exception:
                        pass
                    is_currently_streaming = False
                    self.empty_chunk_count = 0
                    self.repeated_chunk_count = 0
                    self._last_chunk_data = None
                    self._seen_nonzero_chunk = False
                    self._stream_started_t = time.perf_counter()
                    if isinstance(self._last_fresh_data_monotonic, QtCore.QElapsedTimer):
                        self._last_fresh_data_monotonic.restart()
                    self.acquisition_state_signal.emit("paused")

                if should_stream and not is_currently_streaming:
                    try:
                        self.device.start_acquisition()
                        is_currently_streaming = True
                        self.empty_chunk_count = 0
                        self.repeated_chunk_count = 0
                        self._last_chunk_data = None
                        self._seen_nonzero_chunk = False
                        self._stream_started_t = time.perf_counter()
                        if isinstance(self._last_fresh_data_monotonic, QtCore.QElapsedTimer):
                            self._last_fresh_data_monotonic.restart()
                        self.acquisition_state_signal.emit("running")
                    except Exception as e:
                        print(f"[WORKER] Error starting acquisition on resume: {e}")
                        is_currently_streaming = False
                        self.msleep(PLOT_INTERVAL_MS)
                        continue

                if not is_currently_streaming:
                    next_read_t = time.perf_counter()
                    self.msleep(PLOT_INTERVAL_MS)
                    continue

                now_t = time.perf_counter()
                if now_t < next_read_t:
                    wait_ms = int((next_read_t - now_t) * 1000.0)
                    self.msleep(max(1, min(READ_WAIT_SLICE_MS, wait_ms)))
                    continue

                try:
                    acq_t0 = time.perf_counter()
                    data = self.device.read_data()
                    self._telemetry_acquire_ms_total += (time.perf_counter() - acq_t0) * 1000.0

                    if data is None or not hasattr(data, 'shape') or len(data.shape) != 2 or data.shape[1] < 1:
                        self.empty_chunk_count += 1
                        self._telemetry_empty_chunks += 1
                        self._last_chunk_data = None
                        self.repeated_chunk_count = 0
                        next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)

                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD:
                            if self.state_manager.get_current_state() == AppState.STREAMING:
                                self.acquisition_state_signal.emit("waiting")

                        if self._last_fresh_data_monotonic.elapsed() >= int(NO_DATA_LOSS_TIMEOUT_SEC * 1000):
                            if self.device is not None and not self.device.connected:
                                break

                        self.msleep(NO_DATA_SLEEP_MS)
                        continue

                    arr = np.asarray(data)
                    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
                        self._telemetry_invalid_chunks += 1
                        next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)
                        self.msleep(NO_DATA_SLEEP_MS)
                        self._emit_telemetry_if_due()
                        continue

                    if not self._seen_nonzero_chunk:
                        if np.any(arr):
                            self._seen_nonzero_chunk = True
                        elif (time.perf_counter() - self._stream_started_t) <= STARTUP_ZERO_SUPPRESS_SEC:
                            self._telemetry_zero_suppressed += 1
                            self.empty_chunk_count += 1
                            next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)
                            self.msleep(NO_DATA_SLEEP_MS)
                            self._emit_telemetry_if_due()
                            continue

                    is_repeated_chunk = (
                        self._last_chunk_data is not None
                        and self._last_chunk_data.shape == arr.shape
                        and self._last_chunk_data[0, 0] == arr[0, 0]
                        and self._last_chunk_data[0, -1] == arr[0, -1]
                        and (self.repeated_chunk_count < WAITING_CHUNK_THRESHOLD
                             or np.array_equal(self._last_chunk_data, arr))
                    )

                    if is_repeated_chunk:
                        self.repeated_chunk_count += 1
                        self._telemetry_repeated_chunks += 1
                        self.empty_chunk_count += 1
                        next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)

                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD:
                            if self.state_manager.get_current_state() == AppState.STREAMING:
                                self.acquisition_state_signal.emit("waiting")

                        if self._last_fresh_data_monotonic.elapsed() >= int(NO_DATA_LOSS_TIMEOUT_SEC * 1000):
                            if self.device is not None and not self.device.connected:
                                break

                        self.msleep(NO_DATA_SLEEP_MS)
                        continue

                    self._last_chunk_data = np.array(arr, copy=True)
                    self.repeated_chunk_count = 0
                    self.empty_chunk_count = 0
                    self._last_fresh_data_monotonic.restart()
                    next_read_t = max(next_read_t + read_period_s, time.perf_counter())

                    if self.state_manager.get_current_state() == AppState.WAITING_FOR_DATA:
                        self.acquisition_state_signal.emit("running")

                    pending_markers = self.marker_manager.get_pending_markers()
                    marker_dict = {int(m.get("sample_index", -1)): m for m in pending_markers}

                    csv_t0 = time.perf_counter()
                    self.output_sink.append_data(arr, marker_info=marker_dict if marker_dict else None)
                    self._telemetry_csv_ms_total += (time.perf_counter() - csv_t0) * 1000.0

                    emit_t0 = time.perf_counter()
                    self.data_received_signal.emit(arr)
                    self._telemetry_emit_ms_total += (time.perf_counter() - emit_t0) * 1000.0

                    self._telemetry_chunk_count += 1
                    self._telemetry_sample_count += int(arr.shape[1])
                    self._emit_telemetry_if_due()

                except Exception as e:
                    print(f"Error reading data: {e}")
                    if self.device is not None and not self.device.connected:
                        break

            if self.device is not None and not self.device.connected and not self.isInterruptionRequested():
                self.acquisition_state_signal.emit("connection_lost")

        finally:
            markers_snapshot, markers_csv = self.marker_manager.finalize_session()
            self.output_sink.close()

            if self.device is not None:
                try:
                    if is_currently_streaming:
                        self.device.stop_acquisition()
                except Exception:
                    pass

            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass

            self.acquisition_state_signal.emit("stopped")
