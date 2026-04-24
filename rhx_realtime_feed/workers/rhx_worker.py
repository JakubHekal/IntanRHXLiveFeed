import time
from datetime import datetime

from PyQt5 import QtCore
import numpy as np

from rhx_realtime_feed.state_manager import StateManager, AppState
from rhx_realtime_feed.telemetry_logger import append_telemetry_line
from rhx_realtime_feed.workers.chunk_writer import ChunkWriter
from rhx_realtime_feed.workers.marker_manager import MarkerManager
from rhx_realtime_feed.device import IntanRHXDevice


CHANNELS_PORT = "B"
CHANNELS_TO_PLOT = [0]

PLOT_UPDATE_HZ = 20
PLOT_INTERVAL_MS = int(1000 / PLOT_UPDATE_HZ)
ACQUIRE_CHUNK_MS = max(20, PLOT_INTERVAL_MS)
NO_DATA_SLEEP_MS = max(1, PLOT_INTERVAL_MS // 4)
READ_WAIT_SLICE_MS = 2
WAITING_CHUNK_THRESHOLD = 10
NO_DATA_LOSS_TIMEOUT_SEC = 3.0
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

    def __init__(self, host, command_port, data_port, sample_rate, project_name, project_path=None, port="B", channel=0):
        super().__init__()
        # Device connection config
        self.host = host
        self.command_port = command_port
        self.data_port = data_port
        self.sample_rate = sample_rate
        self.port = port
        self.channel = channel
        self.device = None
        
        # Collaborators for I/O and data management
        self.chunk_writer = ChunkWriter(
            sample_rate=sample_rate,
            channel=channel,
            chunk_max_sec=RAW_CHUNK_SEC,
            buffer_bytes=CSV_FILE_BUFFER_BYTES,
            flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
        )
        self.marker_manager = MarkerManager()
        
        # Project context
        self.project_name = project_name
        self.project_path = project_path
        self.project_run_dir = None
        
        # Acquisition state
        self.empty_chunk_count = 0
        self.repeated_chunk_count = 0
        self._last_chunk_data = None
        self._last_window_cursor = None
        self._seen_nonzero_chunk = False
        self._stream_started_t = 0.0
        self._last_fresh_data_monotonic = 0.0
        
        # Telemetry
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
        
        # State machine integration
        self.state_manager = StateManager.get_instance()
        self._last_emitted_state = None

    def __del__(self):
        self.stop()

    def stop(self, timeout_ms: int = 3000) -> bool:
        if not self.isRunning():
            return True
        self.requestInterruption()
        try:
            if self.device is not None:
                try:
                    self.device.stop_streaming()
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
        """Pause receiving (state machine transition handled by caller)."""
        pass

    def resume_receiving(self):
        """Resume receiving (state machine transition handled by caller)."""
        pass

    def request_marker(self, marker_name: str = ""):
        """Add a marker and emit signals."""
        marker_id, marker_ts, markers_snapshot = self.marker_manager.add_marker(
            sample_index=self.chunk_writer.csv_sample_counter,
            sample_rate=self.sample_rate,
            marker_name=marker_name,
        )
        self.marker_added_signal.emit(marker_ts)
        self.marker_catalog_signal.emit(markers_snapshot)

    def get_project_paths(self):
        """Get current project paths."""
        return {
            "run_dir": str(self.project_run_dir) if self.project_run_dir else "",
            "raw_chunks_dir": str(self.chunk_writer.raw_chunks_dir) if self.chunk_writer.raw_chunks_dir else "",
            "markers_csv": str(self.marker_manager.markers_csv_path) if self.marker_manager.markers_csv_path else "",
        }

    def get_markers(self):
        """Get snapshot of all markers."""
        return self.marker_manager.get_markers()

    def request_rename_marker(self, marker_id: int, new_name: str):
        """Queue marker rename operation."""
        return self.marker_manager.request_rename(marker_id, new_name)

    def request_delete_marker(self, marker_id: int):
        """Queue marker delete operation."""
        return self.marker_manager.request_delete(marker_id)
    
    def _emit_telemetry_if_due(self):
        """Emit telemetry metrics if the emission interval has elapsed."""
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
            self.device = IntanRHXDevice(
                host=self.host, 
                command_port=self.command_port, 
                data_port=self.data_port, 
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            try:
                device_fs = float(self.device.get_sample_rate())
                if device_fs > 0:
                    self.sample_rate = device_fs
                    self._chunk_max_samples = max(1, int(round(float(self.sample_rate) * RAW_CHUNK_SEC)))
                    if hasattr(self.device, "sample_rate"):
                        self.device.sample_rate = device_fs
                    if hasattr(self.device, "effective_fs"):
                        self.device.effective_fs = device_fs
            except Exception:
                pass

            self.device.enable_wide_channel([self.channel], port=self.port)
            self.device.set_blocks_per_write(1)
            self.device.start_streaming()
            
            # Initialize session with collaborators
            self.project_run_dir, raw_chunks_dir = self.chunk_writer.start_session(
                self.project_name, self.project_path
            )
            self.marker_manager.initialize(self.project_run_dir)
        except Exception as e:
            self.connection_request_result_signal.emit(False, str(e))
            return

        self.connection_request_result_signal.emit(self.device.connected, None)
        if self.device.connected:
            self.acquisition_state_signal.emit("running")
            self._last_fresh_data_monotonic = QtCore.QElapsedTimer()
            self._last_fresh_data_monotonic.start()
            self._stream_started_t = time.perf_counter()
            self._seen_nonzero_chunk = False

        try:
            connection_lost = False
            is_currently_streaming = True  # Started above at device init
            read_period_s = float(ACQUIRE_CHUNK_MS) / 1000.0
            next_read_t = time.perf_counter()
            
            while not self.isInterruptionRequested():
                # Process any pending marker commands
                marker_catalog, marker_csv_path = self.marker_manager.process_commands()
                if marker_catalog is not None:
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
                    self.empty_chunk_count = 0
                    self.repeated_chunk_count = 0
                    self.last_chunk_signature = None
                    self._last_chunk_data = None
                    self._last_window_cursor = None
                    self._seen_nonzero_chunk = False
                    self._stream_started_t = time.perf_counter()
                    if isinstance(self._last_fresh_data_monotonic, QtCore.QElapsedTimer):
                        self._last_fresh_data_monotonic.restart()
                    self.acquisition_state_signal.emit("paused")
                
                # Start streaming if state machine says we should stream
                if should_stream and not is_currently_streaming:
                    try:
                        self.device.start_streaming()
                        is_currently_streaming = True
                        self.empty_chunk_count = 0
                        self.repeated_chunk_count = 0
                        self.last_chunk_signature = None
                        self._last_chunk_data = None
                        self._last_window_cursor = None
                        self._seen_nonzero_chunk = False
                        self._stream_started_t = time.perf_counter()
                        if isinstance(self._last_fresh_data_monotonic, QtCore.QElapsedTimer):
                            self._last_fresh_data_monotonic.restart()
                        self.acquisition_state_signal.emit("running")
                    except Exception as e:
                        print(f"[WORKER] Error starting stream on resume: {e}")
                        is_currently_streaming = False
                        # Don't break - allow retry on next iteration
                        self.msleep(PLOT_INTERVAL_MS)
                        continue
                
                # If not streaming, just sleep
                if not is_currently_streaming:
                    next_read_t = time.perf_counter()
                    self.msleep(PLOT_INTERVAL_MS)
                    continue

                now_t = time.perf_counter()
                if now_t < next_read_t:
                    wait_ms = int((next_read_t - now_t) * 1000.0)
                    self.msleep(max(1, min(READ_WAIT_SLICE_MS, wait_ms)))
                    continue
                
                # Try to get data
                try:
                    acq_t0 = time.perf_counter()
                    window_cursor = None
                    if hasattr(self.device, "get_latest_window_with_cursor"):
                        data, window_cursor = self.device.get_latest_window_with_cursor(duration_ms=ACQUIRE_CHUNK_MS)
                    else:
                        data = self.device.get_latest_window(duration_ms=ACQUIRE_CHUNK_MS)
                    self._telemetry_acquire_ms_total += (time.perf_counter() - acq_t0) * 1000.0
                    
                    # Check for no data or invalid data
                    if data is None or not hasattr(data, 'shape') or len(data.shape) != 2 or data.shape[1] < 1:
                        self.empty_chunk_count += 1
                        self._telemetry_empty_chunks += 1
                        self.last_chunk_signature = None
                        self._last_chunk_data = None
                        self._last_window_cursor = None
                        self.repeated_chunk_count = 0
                        next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)
                        
                        # Check if we should emit "waiting" state
                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD:
                            if self.state_manager.get_current_state() == AppState.STREAMING:
                                self.acquisition_state_signal.emit("waiting")
                        
                        # Check for data loss timeout
                        if self._last_fresh_data_monotonic.elapsed() >= int(NO_DATA_LOSS_TIMEOUT_SEC * 1000):
                            connection_lost = True
                            break
                        
                        self.msleep(NO_DATA_SLEEP_MS)
                        continue
                    
                    # Validate data
                    arr = np.asarray(data)
                    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
                        self._telemetry_invalid_chunks += 1
                        next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)
                        self.msleep(NO_DATA_SLEEP_MS)
                        self._emit_telemetry_if_due()
                        continue

                    # Suppress warm-up zero chunks briefly after stream start/resume.
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
                    if window_cursor is not None and self._last_window_cursor is not None:
                        is_repeated_chunk = int(window_cursor) == int(self._last_window_cursor)
                    else:
                        is_repeated_chunk = (
                            self._last_chunk_data is not None
                            and self._last_chunk_data.shape == arr.shape
                            and np.array_equal(self._last_chunk_data, arr)
                        )
                    
                    # Check for repeated chunks (stale data)
                    if is_repeated_chunk:
                        self.repeated_chunk_count += 1
                        self._telemetry_repeated_chunks += 1
                        self.empty_chunk_count += 1
                        next_read_t = time.perf_counter() + (NO_DATA_SLEEP_MS / 1000.0)

                        # Check if we should emit "waiting" state
                        if self.empty_chunk_count >= WAITING_CHUNK_THRESHOLD:
                            if self.state_manager.get_current_state() == AppState.STREAMING:
                                self.acquisition_state_signal.emit("waiting")

                        # Check for data loss timeout
                        if self._last_fresh_data_monotonic.elapsed() >= int(NO_DATA_LOSS_TIMEOUT_SEC * 1000):
                            connection_lost = True
                            break
                        
                        self.msleep(NO_DATA_SLEEP_MS)
                        continue
                    
                    # Fresh data arrived
                    self._last_window_cursor = int(window_cursor) if window_cursor is not None else None
                    if self._last_window_cursor is None:
                        self._last_chunk_data = np.array(arr, copy=True)
                    else:
                        self._last_chunk_data = None
                    self.repeated_chunk_count = 0
                    self.empty_chunk_count = 0
                    self._last_fresh_data_monotonic.restart()
                    next_read_t = max(next_read_t + read_period_s, time.perf_counter())
                    
                    # If we were in waiting state, transition back to streaming
                    if self.state_manager.get_current_state() == AppState.WAITING_FOR_DATA:
                        self.acquisition_state_signal.emit("running")
                    
                    # Get pending markers and append data with markers
                    pending_markers = self.marker_manager.get_pending_markers()
                    marker_dict = {int(m.get("sample_index", -1)): m for m in pending_markers}
                    
                    # Write data to CSV
                    csv_t0 = time.perf_counter()
                    self.chunk_writer.append_data(arr, marker_info=marker_dict if marker_dict else None)
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
                        connection_lost = True
                    break
            
            # Check if connection was lost
            if self.device is not None and not self.device.connected and not self.isInterruptionRequested():
                connection_lost = True
            
            if connection_lost:
                self.acquisition_state_signal.emit("connection_lost")
        
        finally:
            # Cleanup: finalize collaborators and close device
            markers_snapshot, markers_csv = self.marker_manager.finalize_session()
            self.chunk_writer.close()
            
            if self.device is not None:
                try:
                    if is_currently_streaming:
                        self.device.stop_streaming()
                except Exception:
                    pass
            
            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
            
            self.acquisition_state_signal.emit("stopped")