import sys
import importlib
import csv
import re
import time
from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore
import numpy as np

from state_manager import StateManager, AppState

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
NO_DATA_LOSS_TIMEOUT_SEC = 3.0
RAW_CHUNK_SEC = 300
TELEMETRY_EMIT_INTERVAL_SEC = 5.0
CSV_FLUSH_INTERVAL_SEC = 1.0
CSV_FILE_BUFFER_BYTES = 1024 * 1024

class RHXWorker(QtCore.QThread):
    connection_request_result_signal = QtCore.pyqtSignal(bool, str)
    data_received_signal = QtCore.pyqtSignal(object)
    acquisition_state_signal = QtCore.pyqtSignal(str)
    marker_added_signal = QtCore.pyqtSignal(float)
    marker_catalog_signal = QtCore.pyqtSignal(object)

    def __init__(self, host, command_port, data_port, sample_rate, project_name, project_path=None, port="B", channel=0):
        super().__init__()
        self.host = host
        self.command_port = command_port
        self.data_port = data_port
        self.sample_rate = sample_rate
        self.project_name = project_name
        self.project_path = project_path
        self.port = port
        self.channel = channel
        self.device = None
        self.empty_chunk_count = 0
        self.last_chunk_signature = None
        self.repeated_chunk_count = 0
        self.csv_file_handle = None
        self.csv_writer = None
        self.csv_sample_counter = 0
        self._chunk_samples_written = 0
        self._chunk_max_samples = max(1, int(round(float(self.sample_rate) * RAW_CHUNK_SEC)))
        self._chunk_index = 0
        self._raw_chunk_paths = []
        self._current_chunk_path = None
        self._pending_markers = []
        self._markers = []
        self._next_marker_id = 1
        self.project_run_dir = None
        self.raw_chunks_dir = None
        self.snapshots_dir = None
        self.markers_dir = None
        self.markers_csv_path = None
        self._chunks_since_flush = 0
        self._last_csv_flush_t = 0.0
        self._last_fresh_data_monotonic = 0.0
        self._io_mutex = QtCore.QMutex()
        self._marker_cmd_queue = []
        self._marker_rewrite_pending = False

        # Lightweight runtime telemetry for Phase 1 baselining.
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
        
        # State machine integration
        self.state_manager = StateManager.get_instance()
        self._last_emitted_state = None

    def __del__(self):
        self.stop()

    def stop(self, timeout_ms: int = 3000) -> bool:
        if not self.isRunning():
            return True
        self.requestInterruption()
        if self.wait(max(0, int(timeout_ms))):
            return True
        print(f"[WORKER] RHXWorker stop timeout after {int(timeout_ms)} ms")
        return False

    def pause_receiving(self):
        """Pause receiving (state machine transition handled by caller)."""
        pass

    def resume_receiving(self):
        """Resume receiving (state machine transition handled by caller)."""
        pass

    def request_marker(self, marker_name: str = ""):
        markers_snapshot = None
        markers_csv_path = None
        marker_ts = 0.0
        with QtCore.QMutexLocker(self._io_mutex):
            marker_index = int(self.csv_sample_counter)
            marker_id = int(self._next_marker_id)
            self._next_marker_id += 1

            safe_name = str(marker_name).strip() or f"Marker {marker_id}"
            marker_record = {
                "id": marker_id,
                "sample_index": marker_index,
                "timestamp_s": marker_index / float(self.sample_rate),
                "name": safe_name,
            }

            self._markers.append(dict(marker_record))
            self._pending_markers.append(dict(marker_record))
            markers_snapshot = self._copy_markers_locked()
            markers_csv_path = self.markers_csv_path
            marker_ts = float(marker_record["timestamp_s"])

        self._write_markers_csv(markers_csv_path, markers_snapshot)
        self.marker_added_signal.emit(marker_ts)
        self.marker_catalog_signal.emit(markers_snapshot)

    def _copy_markers_locked(self):
        return [dict(m) for m in self._markers]

    def get_project_paths(self):
        with QtCore.QMutexLocker(self._io_mutex):
            return {
                "run_dir": str(self.project_run_dir) if self.project_run_dir is not None else "",
                "snapshots_dir": str(self.snapshots_dir) if self.snapshots_dir is not None else "",
                "markers_csv": str(self.markers_csv_path) if self.markers_csv_path is not None else "",
                "raw_chunks_dir": str(self.raw_chunks_dir) if self.raw_chunks_dir is not None else "",
            }

    def get_markers(self):
        with QtCore.QMutexLocker(self._io_mutex):
            return self._copy_markers_locked()

    def request_rename_marker(self, marker_id: int, new_name: str):
        with QtCore.QMutexLocker(self._io_mutex):
            clean_name = str(new_name).strip()
            if not clean_name:
                return False
            self._marker_cmd_queue.append(("rename", int(marker_id), clean_name))
            return True

    def request_delete_marker(self, marker_id: int):
        with QtCore.QMutexLocker(self._io_mutex):
            self._marker_cmd_queue.append(("delete", int(marker_id), ""))
            return True

    def _process_marker_commands_locked(self):
        if not self._marker_cmd_queue:
            return None, None

        changed = False
        for cmd in self._marker_cmd_queue:
            op = cmd[0]
            marker_id = int(cmd[1])

            if op == "rename":
                new_name = str(cmd[2]).strip()
                if not new_name:
                    continue
                updated = False
                for m in self._markers:
                    if int(m.get("id", -1)) == marker_id:
                        m["name"] = new_name
                        updated = True
                if updated:
                    for m in self._pending_markers:
                        if int(m.get("id", -1)) == marker_id:
                            m["name"] = new_name
                    changed = True

            elif op == "delete":
                before = len(self._markers)
                self._markers = [m for m in self._markers if int(m.get("id", -1)) != marker_id]
                self._pending_markers = [m for m in self._pending_markers if int(m.get("id", -1)) != marker_id]
                if len(self._markers) != before:
                    changed = True

        self._marker_cmd_queue.clear()

        if changed:
            # Defer expensive chunk rewrites until save/disconnect.
            self._marker_rewrite_pending = True
            return self._copy_markers_locked(), self.markers_csv_path

        return None, None

    def _start_csv_writer(self):
        with QtCore.QMutexLocker(self._io_mutex):
            self._start_csv_writer_locked()

    def _start_csv_writer_locked(self):
        safe_project = re.sub(r"[^A-Za-z0-9._-]+", "_", str(self.project_name).strip()) or "project"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_dir = Path(self.project_path).expanduser() if self.project_path else (Path.cwd() / "recordings")
        base_dir.mkdir(parents=True, exist_ok=True)

        self.project_run_dir = base_dir / f"{safe_project}_{timestamp}"
        self.raw_chunks_dir = self.project_run_dir / "raw_chunks"
        self.snapshots_dir = self.project_run_dir / "snapshots"
        self.project_run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.markers_csv_path = self.project_run_dir / "markers.csv"

        self._chunk_index = 0
        self._chunk_samples_written = 0
        self._raw_chunk_paths = []
        self._current_chunk_path = None
        self.csv_sample_counter = 0
        self._pending_markers = []
        self._markers = []
        self._next_marker_id = 1
        self._chunks_since_flush = 0
        self._last_csv_flush_t = time.perf_counter()
        self._marker_rewrite_pending = False
        self._open_new_chunk_locked()
        self._write_markers_csv_locked()

    def _open_new_chunk_locked(self):
        if self.raw_chunks_dir is None:
            return
        self._chunk_index += 1
        path = self.raw_chunks_dir / f"chunk_{self._chunk_index:06d}.csv"
        self.csv_file_handle = open(
            path,
            "w",
            newline="",
            encoding="utf-8",
            buffering=CSV_FILE_BUFFER_BYTES,
        )
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow(["time_s", f"ch_{self.channel}_uV", "marker_id", "marker_name"])
        self._chunk_samples_written = 0
        self._current_chunk_path = path
        self._raw_chunk_paths.append(path)

    def _write_markers_csv_locked(self):
        if self.markers_csv_path is None:
            return
        self._write_markers_csv(self.markers_csv_path, self._markers)

    def _write_markers_csv(self, csv_path, markers_snapshot):
        if csv_path is None:
            return
        safe_markers = [dict(m) for m in (markers_snapshot or [])]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "timestamp_s", "name"])
            for m in safe_markers:
                w.writerow([
                    int(m.get("id", 0)),
                    f"{float(m.get('timestamp_s', 0.0)):.6f}",
                    str(m.get("name", "")),
                ])

    def _append_chunk_to_csv(self, arr: np.ndarray, force_flush: bool = False):
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            return

        with QtCore.QMutexLocker(self._io_mutex):
            if self.csv_file_handle is None:
                return

            values_all = arr[0, :].astype(np.float64, copy=False)
            n_samples = int(values_all.size)
            offset = 0
            has_marker = False
            inv_fs = 1.0 / float(self.sample_rate)

            while offset < n_samples:
                if self.csv_file_handle is None:
                    self._open_new_chunk_locked()
                space = max(0, int(self._chunk_max_samples - self._chunk_samples_written))
                if space <= 0:
                    self._rotate_chunk_locked()
                    continue

                take = min(space, n_samples - offset)
                seg_start = int(self.csv_sample_counter + offset)
                seg_end = int(seg_start + take)
                seg_vals = values_all[offset:offset + take]

                if not self._pending_markers:
                    rows = [
                        f"{(seg_start + i) * inv_fs:.6f},{float(seg_vals[i]):.4f},0,\n"
                        for i in range(take)
                    ]
                    self.csv_file_handle.writelines(rows)

                    self._chunk_samples_written += take
                    offset += take

                    if self._chunk_samples_written >= self._chunk_max_samples:
                        self._rotate_chunk_locked()
                    continue

                marker_ids = np.zeros(take, dtype=np.int64)
                marker_names = [""] * take
                still_pending = []
                for marker in self._pending_markers:
                    idx = int(marker.get("sample_index", -1))
                    if idx < seg_start:
                        continue
                    if idx >= seg_end:
                        still_pending.append(marker)
                        continue
                    local = idx - seg_start
                    marker_ids[local] = int(marker.get("id", 0))
                    marker_names[local] = str(marker.get("name", ""))
                    has_marker = True
                self._pending_markers = still_pending

                # Batch CSV writes to reduce Python call overhead in the hot path.
                rows = [
                    [
                        f"{(seg_start + i) * inv_fs:.6f}",
                        f"{float(seg_vals[i]):.4f}",
                        int(marker_ids[i]),
                        marker_names[i],
                    ]
                    for i in range(take)
                ]
                self.csv_writer.writerows(rows)

                self._chunk_samples_written += take
                offset += take

                if self._chunk_samples_written >= self._chunk_max_samples:
                    self._rotate_chunk_locked()

            self._chunks_since_flush += 1
            now = time.perf_counter()
            if has_marker or force_flush or (now - self._last_csv_flush_t) >= CSV_FLUSH_INTERVAL_SEC:
                self.csv_file_handle.flush()
                self._chunks_since_flush = 0
                self._last_csv_flush_t = now

            self.csv_sample_counter += n_samples

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

        print(
            "[telemetry][rhx] "
            f"window_s={elapsed:.2f} chunks={chunks} samples={samples} rate_hz={rate_hz:.1f} "
            f"avg_acquire_ms={avg_acquire_ms:.3f} avg_csv_ms={avg_csv_ms:.3f} avg_emit_ms={avg_emit_ms:.3f} "
            f"empty={int(self._telemetry_empty_chunks)} repeated={int(self._telemetry_repeated_chunks)} invalid={int(self._telemetry_invalid_chunks)}"
        )

        self._telemetry_last_emit = now
        self._telemetry_chunk_count = 0
        self._telemetry_sample_count = 0
        self._telemetry_acquire_ms_total = 0.0
        self._telemetry_csv_ms_total = 0.0
        self._telemetry_emit_ms_total = 0.0
        self._telemetry_empty_chunks = 0
        self._telemetry_repeated_chunks = 0
        self._telemetry_invalid_chunks = 0

    def _rotate_chunk_locked(self):
        if self.csv_file_handle is not None:
            try:
                self.csv_file_handle.flush()
                self.csv_file_handle.close()
            except Exception:
                pass
        self.csv_file_handle = None
        self.csv_writer = None
        self._current_chunk_path = None
        self._chunk_samples_written = 0
        self._open_new_chunk_locked()
        self._last_csv_flush_t = time.perf_counter()

    def _close_csv_writer(self):
        with QtCore.QMutexLocker(self._io_mutex):
            if self.csv_file_handle is not None:
                try:
                    self.csv_file_handle.flush()
                    self.csv_file_handle.close()
                except Exception:
                    pass
            self.csv_file_handle = None
            self.csv_writer = None
            self._current_chunk_path = None
            if self._marker_rewrite_pending:
                self._rewrite_marker_fields_in_chunks_locked()
                self._marker_rewrite_pending = False
            self._write_markers_csv_locked()

    def _rewrite_marker_fields_in_chunks_locked(self):
        if not self._raw_chunk_paths:
            return

        lookup = {int(m.get("id", -1)): m for m in self._markers}

        reopen_path = self._current_chunk_path
        if self.csv_file_handle is not None:
            try:
                self.csv_file_handle.flush()
                self.csv_file_handle.close()
            except Exception:
                pass
            self.csv_file_handle = None
            self.csv_writer = None

        for path in list(self._raw_chunk_paths):
            if not Path(path).exists():
                continue
            tmp_path = Path(path).with_suffix(".tmp")
            with open(path, "r", newline="", encoding="utf-8") as src, open(tmp_path, "w", newline="", encoding="utf-8") as dst:
                reader = csv.DictReader(src)
                if not reader.fieldnames:
                    continue
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    try:
                        marker_id = int(str(row.get("marker_id", "0") or "0"))
                    except Exception:
                        marker_id = 0

                    if marker_id > 0:
                        marker_info = lookup.get(marker_id)
                        if marker_info is None:
                            row["marker_id"] = "0"
                            row["marker_name"] = ""
                        else:
                            row["marker_id"] = str(marker_id)
                            row["marker_name"] = str(marker_info.get("name", ""))

                    writer.writerow(row)
            tmp_path.replace(path)

        if reopen_path is not None and Path(reopen_path).exists():
            self.csv_file_handle = open(reopen_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file_handle)
            self._current_chunk_path = reopen_path
    
    def run(self):
        """Main device worker thread loop."""
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
            self._start_csv_writer()
        except Exception as e:
            self.connection_request_result_signal.emit(False, str(e))
            return

        self.connection_request_result_signal.emit(self.device.connected, None)
        if self.device.connected:
            self.acquisition_state_signal.emit("running")
            self._last_fresh_data_monotonic = QtCore.QElapsedTimer()
            self._last_fresh_data_monotonic.start()

        try:
            connection_lost = False
            is_currently_streaming = True  # Started above at device init
            
            while not self.isInterruptionRequested():
                marker_catalog = None
                marker_csv_path = None
                with QtCore.QMutexLocker(self._io_mutex):
                    marker_catalog, marker_csv_path = self._process_marker_commands_locked()

                if marker_catalog is not None:
                    self._write_markers_csv(marker_csv_path, marker_catalog)
                    self.marker_catalog_signal.emit(marker_catalog)

                # Check for hard link loss where the device worker thread died
                if is_currently_streaming and hasattr(self.device, "streaming_thread"):
                    stream_thread = self.device.streaming_thread
                    if stream_thread is not None and not stream_thread.is_alive():
                        connection_lost = True
                        break
                
                # Query state machine to determine desired action
                should_stream = self.state_manager.get_current_state() in (
                    AppState.STREAMING,
                    AppState.WAITING_FOR_DATA,
                )
                
                # Stop streaming if state machine says we shouldn't stream
                if not should_stream and is_currently_streaming:
                    try:
                        self.device.stop_streaming()
                    except Exception:
                        pass
                    is_currently_streaming = False
                    self.acquisition_state_signal.emit("paused")
                
                # Start streaming if state machine says we should stream
                if should_stream and not is_currently_streaming:
                    try:
                        self.device.start_streaming()
                        is_currently_streaming = True
                        self.empty_chunk_count = 0
                        self.repeated_chunk_count = 0
                        self.last_chunk_signature = None
                        self.acquisition_state_signal.emit("running")
                    except Exception as e:
                        print(f"[WORKER] Error starting stream on resume: {e}")
                        is_currently_streaming = False
                        # Don't break - allow retry on next iteration
                        self.msleep(PLOT_INTERVAL_MS)
                        continue
                
                # If not streaming, just sleep
                if not is_currently_streaming:
                    self.msleep(PLOT_INTERVAL_MS)
                    continue
                
                # Try to get data
                try:
                    acq_t0 = time.perf_counter()
                    data = self.device.get_latest_window(duration_ms=ACQUIRE_CHUNK_MS)
                    self._telemetry_acquire_ms_total += (time.perf_counter() - acq_t0) * 1000.0
                    
                    # Check for no data or invalid data
                    if data is None or not hasattr(data, 'shape') or len(data.shape) != 2 or data.shape[1] < 1:
                        self.empty_chunk_count += 1
                        self._telemetry_empty_chunks += 1
                        self.last_chunk_signature = None
                        self.repeated_chunk_count = 0
                        
                        # Check if we should emit "waiting" state
                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD:
                            if self.state_manager.get_current_state() == AppState.STREAMING:
                                self.acquisition_state_signal.emit("waiting")
                        
                        # Check for data loss timeout
                        if self._last_fresh_data_monotonic.elapsed() >= int(NO_DATA_LOSS_TIMEOUT_SEC * 1000):
                            connection_lost = True
                            break
                        
                        self.msleep(PLOT_INTERVAL_MS)
                        continue
                    
                    # Validate data
                    arr = np.asarray(data)
                    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
                        self._telemetry_invalid_chunks += 1
                        self.msleep(PLOT_INTERVAL_MS)
                        self._emit_telemetry_if_due()
                        continue
                    signature = (
                        arr.shape[0],
                        arr.shape[1],
                        float(np.round(arr[0, 0], 6)),
                        float(np.round(arr[0, -1], 6)),
                        float(np.round(np.mean(arr), 6)),
                    )
                    
                    # Check for repeated chunks (stale data)
                    if signature == self.last_chunk_signature:
                        self.repeated_chunk_count += 1
                        self._telemetry_repeated_chunks += 1
                        if self.repeated_chunk_count > 5:
                            self.empty_chunk_count += 1
                            
                            # Check if we should emit "waiting" state
                            if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD:
                                if self.state_manager.get_current_state() == AppState.STREAMING:
                                    self.acquisition_state_signal.emit("waiting")
                            
                            # Check for data loss timeout
                            if self._last_fresh_data_monotonic.elapsed() >= int(NO_DATA_LOSS_TIMEOUT_SEC * 1000):
                                connection_lost = True
                                break
                        
                        self.msleep(PLOT_INTERVAL_MS)
                        continue
                    
                    # Fresh data arrived
                    self.last_chunk_signature = signature
                    self.repeated_chunk_count = 0
                    self.empty_chunk_count = 0
                    self._last_fresh_data_monotonic.restart()
                    
                    # If we were in waiting state, transition back to streaming
                    if self.state_manager.get_current_state() == AppState.WAITING_FOR_DATA:
                        self.acquisition_state_signal.emit("running")
                    
                    # Process the data
                    csv_t0 = time.perf_counter()
                    self._append_chunk_to_csv(arr)
                    self._telemetry_csv_ms_total += (time.perf_counter() - csv_t0) * 1000.0

                    emit_t0 = time.perf_counter()
                    self.data_received_signal.emit(arr)
                    self._telemetry_emit_ms_total += (time.perf_counter() - emit_t0) * 1000.0

                    self._telemetry_chunk_count += 1
                    self._telemetry_sample_count += int(arr.shape[1])
                    self._emit_telemetry_if_due()
                    
                    self.msleep(PLOT_INTERVAL_MS)
                
                except Exception as e:
                    print(f"Error reading data: {e}")
                    if self.device is not None and not self.device.connected:
                        connection_lost = True
                    break
            
            # Check if connection was lost
            if self.device is not None and not self.device.connected and not self.isInterruptionRequested():
                connection_lost = True
            
            if connection_lost:
                self.acquisition_state_signal.emit("connection_lost")
        
        finally:
            # Cleanup
            if self.device is not None:
                try:
                    if is_currently_streaming:
                        self.device.stop_streaming()
                except Exception:
                    pass
            
            self._close_csv_writer()
            
            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
            
            self.acquisition_state_signal.emit("stopped")