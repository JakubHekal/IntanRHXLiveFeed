import time
from pathlib import Path

from PyQt5 import QtCore
import numpy as np

from rhx_realtime_feed.workers.device_worker import RAW_CHUNK_SEC, CSV_FILE_BUFFER_BYTES, CSV_FLUSH_INTERVAL_SEC
from rhx_realtime_feed.workers.chunk_writer import ChunkWriter
from rhx_realtime_feed.telemetry_logger import append_telemetry_line


class ExperimentRunner(QtCore.QObject):
    step_started = QtCore.pyqtSignal(int, str, str, float)  # step_index, device_name, action, duration
    step_completed = QtCore.pyqtSignal(int, str, str)
    experiment_finished = QtCore.pyqtSignal(bool, str)  # success, message
    error_occurred = QtCore.pyqtSignal(str, str)  # device_name, error_message
    data_received = QtCore.pyqtSignal(str, object)  # device_name, chunk
    device_configured = QtCore.pyqtSignal(str, int, list, float)  # device_name, num_channels, labels, sample_rate
    user_input_requested = QtCore.pyqtSignal(str)  # message

    def __init__(self, devices: list, sequence: list, run_path: str, parent=None):
        """
        Args:
            devices: list of (name, blocks, device_type, config_dict, device_instance)
            sequence: list of SequenceStep (or dict with action, parameters, device_name)
            run_path: path to the run directory for storing outputs
        """
        super().__init__(parent)
        self._devices = devices
        self._sequence = sequence
        self._run_path = Path(run_path)
        self._thread = None
        self._abort = False

    def start(self):
        if self._thread is not None and self._thread.isRunning():
            return
        self._abort = False
        self._thread = _RunnerThread(self._devices, self._sequence, self._run_path, self)
        self._thread.step_started.connect(self.step_started)
        self._thread.step_completed.connect(self.step_completed)
        self._thread.experiment_finished.connect(self.experiment_finished)
        self._thread.error_occurred.connect(self.error_occurred)
        self._thread.data_received.connect(self.data_received)
        self._thread.device_configured.connect(self.device_configured)
        self._thread.user_input_requested.connect(self.user_input_requested)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def stop(self):
        self._abort = True
        if self._thread is not None:
            self._thread.abort()
            if not self._thread.wait(5000):
                print("[Runner] Thread did not stop within 5s")

    def pause(self):
        if self._thread is not None:
            self._thread._paused = True

    def resume(self):
        if self._thread is not None:
            self._thread._paused = False

    def _on_thread_finished(self):
        self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.isRunning()


class _RunnerThread(QtCore.QThread):
    step_started = QtCore.pyqtSignal(int, str, str, float)
    step_completed = QtCore.pyqtSignal(int, str, str)
    experiment_finished = QtCore.pyqtSignal(bool, str)
    error_occurred = QtCore.pyqtSignal(str, str)
    data_received = QtCore.pyqtSignal(str, object)
    user_input_requested = QtCore.pyqtSignal(str)
    device_configured = QtCore.pyqtSignal(str, int, list, float)  # device_name, num_channels, labels, sample_rate

    def __init__(self, devices, sequence, run_path, parent=None):
        super().__init__(parent)
        self._devices = devices
        self._sequence = sequence
        self._run_path = Path(run_path)
        self._abort = False
        self._paused = False

    def abort(self):
        self._abort = True
        self.requestInterruption()

    def _find_device(self, device_name: str):
        for d in self._devices:
            name = d[0] if isinstance(d, (list, tuple)) else ""
            if name == device_name:
                inst = d[4] if len(d) >= 5 else None
                return inst
        return None

    def _find_device_type(self, device_name: str):
        for d in self._devices:
            name = d[0] if isinstance(d, (list, tuple)) else ""
            if name == device_name:
                return d[2] if len(d) >= 3 else ""
        return ""

    def run(self):
        try:
            device_map = {}
            for d in self._devices:
                if isinstance(d, (list, tuple)) and len(d) >= 5:
                    inst = d[4]
                    if inst is not None:
                        device_map[d[0]] = inst

            for step_idx, step in enumerate(self._sequence):
                if self._abort or self.isInterruptionRequested():
                    break

                action = step.action if hasattr(step, 'action') else step.get('action', '')
                device_name = step.device_name if hasattr(step, 'device_name') else step.get('device_name', '')
                params = dict(step.parameters) if hasattr(step, 'parameters') else dict(step.get('parameters', {}))

                device = device_map.get(device_name)
                duration = params.get('duration_s', 2.0)

                self.step_started.emit(step_idx, device_name, action, duration)
                print(f"[Runner] Step {step_idx + 1}/{len(self._sequence)}: {device_name} → {action} ({duration}s)")
                append_telemetry_line(
                    f"step_start | {step_idx} | {device_name} | {action} | duration={duration}"
                )

                if device is None and action not in ("wait_input", "log_event", "start_recording", "stop_recording", "pause"):
                    self.error_occurred.emit(device_name or "__system__", f"Device '{device_name}' not found or not connected")
                    self.msleep(500)
                    continue

                if self._abort:
                    break

                try:
                    if action == "Stream":
                        if not getattr(device, 'connected', False):
                            self.error_occurred.emit(device_name, "Device not connected")
                            self.msleep(500)
                            continue

                        # configure first so device.channels reflects the right channel count
                        device.configure(**{k: v for k, v in params.items() if k not in ('duration_s', 'block_label')})

                        sr = device.sample_rate if device.sample_rate else 20000.0
                        ch_labels = [c.name for c in getattr(device, 'channels', [])]
                        num_ch = len(ch_labels) or 1
                        self.device_configured.emit(device_name, num_ch, ch_labels, sr)

                        # raw/{device_name}/ per run
                        dev_raw_dir = self._run_path / "raw" / (device_name.replace(" ", "_"))
                        dev_raw_dir.mkdir(parents=True, exist_ok=True)

                        sink = ChunkWriter(
                            sample_rate=sr,
                            num_channels=num_ch,
                            chunk_max_sec=RAW_CHUNK_SEC,
                            buffer_bytes=CSV_FILE_BUFFER_BYTES,
                            flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
                        )
                        label = params.get("block_label", "Stream")
                        sink.raw_chunks_dir = dev_raw_dir
                        sink.filename_prefix = label
                        sink._open_new_chunk_locked()

                        device.start_acquisition()
                        append_telemetry_line(
                            f"acq_start | {step_idx} | {device_name} | Stream | "
                            f"ch={num_ch} sr={sr} label={label}"
                        )
                        first_chunk = True

                        deadline = time.perf_counter() + duration
                        while time.perf_counter() < deadline and not self._abort and not self.isInterruptionRequested() and getattr(device, 'connected', True):
                            # pause extends deadline so pause doesn't eat into stream time
                            if self._paused:
                                pause_start = time.perf_counter()
                                while self._paused and not self._abort and not self.isInterruptionRequested():
                                    self.msleep(100)
                                deadline += time.perf_counter() - pause_start
                            try:
                                data = device.read_data()
                            except Exception:
                                data = None
                            if data is not None:
                                arr = np.asarray(data)
                                if arr.ndim == 2 and arr.shape[1] > 0:
                                    marker_info = (
                                        {0: {"id": step_idx + 1, "name": label}}
                                        if first_chunk
                                        else None
                                    )
                                    sink.append_data(arr, marker_info=marker_info)
                                    self.data_received.emit(device_name, arr)
                                    first_chunk = False
                            remaining = deadline - time.perf_counter()
                            self.msleep(int(min(100, max(0, remaining * 1000))) if remaining > 0 else 10)

                        sink.close()
                        append_telemetry_line(
                            f"acq_end | {step_idx} | {device_name} | Stream | samples_in_chunk={sink._chunk_samples_written}"
                        )
                        try:
                            device.stop_acquisition()
                        except Exception:
                            pass

                    elif action in ("Configure",):
                        device.configure(**params)
                        self.msleep(100)

                    elif action == "Write":
                        ch = params.get("channel", 1) - 1
                        val = params.get("value", 0.0)
                        device.write_output(ch, val)
                        self.msleep(100)

                    elif action == "Trigger":
                        ch = params.get("channel", 1) - 1
                        device.trigger_action(ch)
                        self.msleep(100)

                    elif action in ("Stimulus", "force_voltage"):
                        if hasattr(device, 'configure'):
                            device.configure(mode="FVMI")
                        if hasattr(device, 'write_output'):
                            ch = params.get("channel", 1) - 1
                            val = params.get("voltage", params.get("amplitude", 5.0))
                            device.write_output(max(0, ch), val)
                        dur = params.get("duration_s", 1.0)
                        self.msleep(int(dur * 1000))
                        try:
                            device.write_output(max(0, ch), 0.0)
                        except Exception:
                            pass

                    elif action == "Measure":
                        if hasattr(device, 'read_data'):
                            data = device.read_data()
                            print(f"[Runner] {device_name} Measure: got {data.shape if data is not None else 'None'}")
                        self.msleep(100)

                    elif action in ("force_current",):
                        ch = params.get("channel", 1) - 1
                        current_nA = float(params.get("current", params.get("current_nA", -100000.0)))
                        dur = float(params.get("duration_s", 1200.0))
                        label = params.get("block_label", "ForceCurrent")

                        if hasattr(device, 'configure'):
                            device.configure(mode="FIMV")

                        sr = getattr(device, 'sample_rate', 1000.0) or 1000.0
                        num_ch = len(getattr(device, 'channels', [])) or 2

                        dev_raw_dir = self._run_path / "raw" / (device_name.replace(" ", "_"))
                        dev_raw_dir.mkdir(parents=True, exist_ok=True)

                        sink = ChunkWriter(
                            sample_rate=sr,
                            num_channels=num_ch,
                            chunk_max_sec=RAW_CHUNK_SEC,
                            buffer_bytes=CSV_FILE_BUFFER_BYTES,
                            flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
                        )
                        sink.raw_chunks_dir = dev_raw_dir
                        sink.filename_prefix = label
                        sink._open_new_chunk_locked()

                        device.start_acquisition()
                        append_telemetry_line(
                            f"acq_start | {step_idx} | {device_name} | force_current | "
                            f"current_nA={current_nA} ch={ch} dur={dur} label={label}"
                        )
                        first_chunk = True
                        device.write_output(ch, current_nA / 1e9)

                        deadline = time.perf_counter() + dur
                        while time.perf_counter() < deadline and not self._abort and not self.isInterruptionRequested():
                            if self._paused:
                                pause_start = time.perf_counter()
                                while self._paused and not self._abort and not self.isInterruptionRequested():
                                    self.msleep(100)
                                deadline += time.perf_counter() - pause_start
                            try:
                                data = device.read_data()
                            except Exception:
                                data = None
                            if data is not None:
                                arr = np.asarray(data)
                                if arr.ndim == 2 and arr.shape[1] > 0:
                                    marker_info = (
                                        {0: {"id": step_idx + 1, "name": label}}
                                        if first_chunk
                                        else None
                                    )
                                    sink.append_data(arr, marker_info=marker_info)
                                    self.data_received.emit(device_name, arr)
                                    first_chunk = False
                            remaining = deadline - time.perf_counter()
                            self.msleep(int(min(100, max(0, remaining * 1000))) if remaining > 0 else 10)

                        sink.close()
                        append_telemetry_line(
                            f"acq_end | {step_idx} | {device_name} | force_current | samples_in_chunk={sink._chunk_samples_written}"
                        )
                        try:
                            device.write_output(ch, 0.0)
                        except Exception:
                            pass
                        try:
                            device.stop_acquisition()
                        except Exception:
                            pass

                    elif action == "wait_input":
                        msg = params.get("message", "Click OK to continue")
                        self._input_result = None
                        self.user_input_requested.emit(msg)
                        while self._input_result is None and not self._abort and not self.isInterruptionRequested():
                            self.msleep(100)
                        print(f"[Runner] Wait input result: {self._input_result}")

                    elif action == "log_event":
                        append_telemetry_line(f"log | {step_idx} | {device_name} | {params}")
                        print(f"[Runner] Log event: {params}")

                    elif action == "pause":
                        self.msleep(int(duration * 1000))

                    elif action in ("start_recording", "stop_recording"):
                        # ponytail: recording handled by Stream block's ChunkWriter lifecycle
                        print(f"[Runner] Recording op {action} (handled by Stream)")

                    else:
                        print(f"[Runner] Unknown action '{action}', skipping")

                except Exception as e:
                    self.error_occurred.emit(device_name, str(e))
                    append_telemetry_line(
                        f"step_error | {step_idx} | {device_name} | {action} | {e}"
                    )
                    print(f"[Runner] Error in step {step_idx}: {e}")

                self.step_completed.emit(step_idx, device_name, action)
                append_telemetry_line(f"step_end | {step_idx} | {device_name} | {action} | ok")

                # ponytail: busy-wait, replace with QWaitCondition if throughput matters
                while self._paused and not self._abort and not self.isInterruptionRequested():
                    self.msleep(100)

            success = not self._abort
            msg = "Completed" if success else "Aborted"
            self.experiment_finished.emit(success, msg)
            print(f"[Runner] Experiment {msg}")

        except Exception as e:
            self.experiment_finished.emit(False, str(e))
            print(f"[Runner] Fatal error: {e}")

        finally:
            for d in self._devices:
                if isinstance(d, (list, tuple)) and len(d) >= 5:
                    inst = d[4]
                    if inst is not None and hasattr(inst, 'close'):
                        try:
                            inst.close()
                        except Exception as e:
                            print(f"[Runner] Error closing device {d[0]}: {e}")
