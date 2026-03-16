import sys
import importlib
import io
import re
from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore
import numpy as np

def load_intan_device_class():
    """
    Load the IntanRHXDevice class from the intan.interface module.
    Use the installed version if available, otherwise load from the local python-intan package.
    """
    try:
        module = importlib.import_module("intan.interface")
        return module.IntanRHXDevice
    except ImportError:
        pass

    root = Path(sys.path[0])
    local_pkg = root / "python-intan"
    if local_pkg.exists():
        sys.path.insert(0, str(local_pkg))
    module = importlib.import_module("intan.interface")
    return module.IntanRHXDevice


IntanRHXDevice = load_intan_device_class()

CHANNELS_PORT = "B"
CHANNELS_TO_PLOT = [0]

PLOT_UPDATE_HZ = 20
PLOT_INTERVAL_MS = int(1000 / PLOT_UPDATE_HZ)
ACQUIRE_CHUNK_MS = max(20, PLOT_INTERVAL_MS)
WAITING_CHUNK_THRESHOLD = 10

class RHXWorker(QtCore.QThread):
    connection_request_result_signal = QtCore.pyqtSignal(bool, str)
    data_received_signal = QtCore.pyqtSignal(object)
    acquisition_state_signal = QtCore.pyqtSignal(str)
    marker_added_signal = QtCore.pyqtSignal(float)

    def __init__(self, host, command_port, data_port, sample_rate, project_name, port="B", channel=0):
        super().__init__()
        self.host = host
        self.command_port = command_port
        self.data_port = data_port
        self.sample_rate = sample_rate
        self.project_name = project_name
        self.port = port
        self.channel = channel
        self.device = None
        self.empty_chunk_count = 0
        self.last_chunk_signature = None
        self.repeated_chunk_count = 0
        self.csv_file_handle = None
        self.csv_sample_counter = 0
        self._desired_streaming = True
        self._is_streaming = False
        self._pending_marker_indices = []
        self._waiting_for_data = False
        self._chunks_since_flush = 0

    def __del__(self):
        self.stop()

    def stop(self):
        self._desired_streaming = False
        self.requestInterruption()
        self.wait(1000)

    def pause_receiving(self):
        self._desired_streaming = False

    def resume_receiving(self):
        self._desired_streaming = True

    def request_marker(self):
        marker_index = int(self.csv_sample_counter)
        self._pending_marker_indices.append(marker_index)
        self.marker_added_signal.emit(marker_index / float(self.sample_rate))

    def _start_csv_writer(self):
        safe_project = re.sub(r"[^A-Za-z0-9._-]+", "_", str(self.project_name).strip()) or "project"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recordings_dir = Path.cwd() / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        csv_path = recordings_dir / f"{safe_project}_{timestamp}.csv"

        self.csv_file_handle = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_file_handle.write(f"time_s,ch_{self.channel}_uV,marker\r\n")
        self.csv_sample_counter = 0
        self._pending_marker_indices = []
        self._chunks_since_flush = 0

    def _append_chunk_to_csv(self, arr: np.ndarray, force_flush: bool = False):
        if self.csv_file_handle is None or arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            return

        n_samples = arr.shape[1]
        times = (self.csv_sample_counter + np.arange(n_samples, dtype=np.float64)) / float(self.sample_rate)
        values = arr[0, :].astype(np.float64, copy=False)
        markers = np.zeros(n_samples, dtype=np.int8)

        chunk_start = int(self.csv_sample_counter)
        chunk_end = int(self.csv_sample_counter + n_samples)
        has_marker = False
        still_pending = []
        for marker_index in self._pending_marker_indices:
            if marker_index < chunk_start:
                continue
            if marker_index >= chunk_end:
                still_pending.append(marker_index)
                continue
            markers[marker_index - chunk_start] = 1
            has_marker = True
        self._pending_marker_indices = still_pending

        # Build CSV text in memory — avoids Python list boxing from tolist()
        buf = io.StringIO()
        for i in range(n_samples):
            buf.write(f"{times[i]:.6f},{values[i]:.4f},{markers[i]}\r\n")
        self.csv_file_handle.write(buf.getvalue())

        self._chunks_since_flush += 1
        if has_marker or force_flush or self._chunks_since_flush >= 5:
            self.csv_file_handle.flush()
            self._chunks_since_flush = 0

        self.csv_sample_counter += n_samples

    def _close_csv_writer(self):
        if self.csv_file_handle is not None:
            try:
                self.csv_file_handle.flush()
                self.csv_file_handle.close()
            except Exception:
                pass
        self.csv_file_handle = None
    
    def run(self):
        try:
            self.device = IntanRHXDevice(
                host=self.host, 
                command_port=self.command_port, 
                data_port=self.data_port, 
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            self.device.enable_wide_channel([self.channel], port=self.port)
            self.device.set_blocks_per_write(1)
            self.device.start_streaming()
            self._is_streaming = True
            self._start_csv_writer()
        except Exception as e:
            self.connection_request_result_signal.emit(False, str(e))
            return

        self.connection_request_result_signal.emit(self.device.connected, None)
        if self.device.connected:
            self.acquisition_state_signal.emit("running")

        try:
            connection_lost = False
            while self.device.connected and not self.isInterruptionRequested():
                if not self._desired_streaming and self._is_streaming:
                    try:
                        self.device.stop_streaming()
                    except Exception:
                        pass
                    self._is_streaming = False
                    self._waiting_for_data = False
                    self.acquisition_state_signal.emit("paused")

                if self._desired_streaming and not self._is_streaming:
                    try:
                        self.device.start_streaming()
                        self._is_streaming = True
                        self.empty_chunk_count = 0
                        self.repeated_chunk_count = 0
                        self.last_chunk_signature = None
                        self._waiting_for_data = False
                        self.acquisition_state_signal.emit("running")
                    except Exception as e:
                        print(f"Error resuming stream: {e}")
                        break

                if not self._is_streaming:
                    self.msleep(PLOT_INTERVAL_MS)
                    continue

                try:
                    data = self.device.get_latest_window(duration_ms=ACQUIRE_CHUNK_MS)

                    if data is None or not hasattr(data, 'shape') or len(data.shape) != 2:
                        self.empty_chunk_count += 1
                        self.last_chunk_signature = None
                        self.repeated_chunk_count = 0
                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD and not self._waiting_for_data:
                            self._waiting_for_data = True
                            self.acquisition_state_signal.emit("waiting")
                        self.msleep(PLOT_INTERVAL_MS)
                        continue
                    elif data.shape[1] < 1:
                        self.empty_chunk_count += 1
                        self.last_chunk_signature = None
                        self.repeated_chunk_count = 0
                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD and not self._waiting_for_data:
                            self._waiting_for_data = True
                            self.acquisition_state_signal.emit("waiting")
                        self.msleep(PLOT_INTERVAL_MS)
                        continue
                    else:
                        arr = np.asarray(data)
                        signature = (
                            arr.shape[0],
                            arr.shape[1],
                            float(np.round(arr[0, 0], 6)),
                            float(np.round(arr[0, -1], 6)),
                            float(np.round(np.mean(arr), 6)),
                        )

                        if signature == self.last_chunk_signature:
                            self.repeated_chunk_count += 1
                            if self.repeated_chunk_count > 5:
                                self.empty_chunk_count += 1
                                if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD and not self._waiting_for_data:
                                    self._waiting_for_data = True
                                    self.acquisition_state_signal.emit("waiting")
                            self.msleep(PLOT_INTERVAL_MS)
                            continue

                        self.last_chunk_signature = signature
                        self.repeated_chunk_count = 0
                        self.empty_chunk_count = 0
                        if self._waiting_for_data:
                            self._waiting_for_data = False
                            self.acquisition_state_signal.emit("running")
                        self._append_chunk_to_csv(arr)
                        self.data_received_signal.emit(arr)

                    # Match the proven cadence from connection_test.py and avoid flooding Qt.
                    self.msleep(PLOT_INTERVAL_MS)
                except Exception as e:
                    print(f"Error reading data: {e}")
                    if self.device is not None and not self.device.connected:
                        connection_lost = True
                    break

            if self.device is not None and not self.device.connected and not self.isInterruptionRequested():
                connection_lost = True

            if connection_lost:
                self.acquisition_state_signal.emit("connection_lost")
        finally:
            if self.device is not None:
                try:
                    if self._is_streaming:
                        self.device.stop_streaming()
                except Exception:
                    pass
                self._is_streaming = False

            self._close_csv_writer()

            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
            self.acquisition_state_signal.emit("stopped")