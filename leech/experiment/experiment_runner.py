import threading
import time
from pathlib import Path

from PyQt5 import QtCore
import numpy as np

from leech.workers.device_worker import RAW_CHUNK_SEC, CSV_FILE_BUFFER_BYTES, CSV_FLUSH_INTERVAL_SEC
from leech.workers.chunk_writer import ChunkWriter
from leech.telemetry_logger import append_telemetry_line

_SYSTEM_ACTIONS = frozenset(("wait_input", "log_event", "pause", "start_recording", "stop_recording"))


class ExperimentRunner(QtCore.QObject):
    step_started = QtCore.pyqtSignal(int, str, str, float, str)  # step_index, device_name, action, duration, block_label
    step_completed = QtCore.pyqtSignal(int, str, str)
    experiment_finished = QtCore.pyqtSignal(bool, str)  # success, message
    error_occurred = QtCore.pyqtSignal(str, str)  # device_name, error_message
    data_received = QtCore.pyqtSignal(str, object)  # device_name, chunk
    device_configured = QtCore.pyqtSignal(str, int, list, float)  # device_name, num_channels, labels, sample_rate
    user_input_requested = QtCore.pyqtSignal(str)  # message

    def __init__(self, devices: list, sequence: list, run_path: str, parent=None):
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

    def wait_input_response(self, result):
        if self._thread is not None:
            self._thread._input_result = result

    def _on_thread_finished(self):
        self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.isRunning()


class _RunnerThread(QtCore.QThread):
    step_started = QtCore.pyqtSignal(int, str, str, float, str)
    step_completed = QtCore.pyqtSignal(int, str, str)
    experiment_finished = QtCore.pyqtSignal(bool, str)
    error_occurred = QtCore.pyqtSignal(str, str)
    data_received = QtCore.pyqtSignal(str, object)
    user_input_requested = QtCore.pyqtSignal(str)
    device_configured = QtCore.pyqtSignal(str, int, list, float)

    def __init__(self, devices, sequence, run_path, parent=None):
        super().__init__(parent)
        self._devices = devices
        self._sequence = sequence
        self._run_path = Path(run_path)
        self._abort = False
        self._paused = False
        self._input_result = None
        self._device_map = {}

    def abort(self):
        self._abort = True
        self.requestInterruption()

    def _find_device(self, device_name: str):
        return self._device_map.get(device_name)

    def _find_device_type(self, device_name: str):
        for d in self._devices:
            name = d[0] if isinstance(d, (list, tuple)) else ""
            if name == device_name:
                return d[2] if len(d) >= 3 else ""
        return ""

    def run(self):
        try:
            self._device_map = {}
            for d in self._devices:
                if isinstance(d, (list, tuple)) and len(d) >= 5:
                    inst = d[4]
                    if inst is not None:
                        self._device_map[d[0]] = inst

            # assign device-only step indices matching timeline block order
            device_idx = 0
            for step in self._sequence:
                action = self._action_of(step)
                if action not in _SYSTEM_ACTIONS:
                    step._device_step_idx = device_idx
                    device_idx += 1
                else:
                    step._device_step_idx = -1

            groups = self._group_by_start()

            for group in groups:
                if self._abort:
                    break

                system_steps = [s for s in group if self._action_of(s) in _SYSTEM_ACTIONS]
                device_steps = [s for s in group if self._action_of(s) not in _SYSTEM_ACTIONS]

                for step in system_steps:
                    if self._abort:
                        break
                    self._execute_step(step, emit_signals=False)

                if self._abort:
                    break

                if len(device_steps) == 0:
                    pass
                elif len(device_steps) == 1:
                    self._execute_step(device_steps[0])
                else:
                    # emit step_started once for the group using the first device step
                    first = device_steps[0]
                    self._emit_group_start(first)
                    threads = []
                    for step in device_steps:
                        t = threading.Thread(target=self._execute_step, args=(step,), kwargs={"emit_signals": False})
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    self._emit_group_complete(first)

                while self._paused and not self._abort:
                    time.sleep(0.1)

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

    def _group_by_start(self):
        groups = []
        current_start = None
        for step in self._sequence:
            params = dict(step.parameters) if hasattr(step, 'parameters') else dict(step.get('parameters', {}))
            start = params.get('_start', 0.0)
            if start != current_start:
                groups.append([])
                current_start = start
            groups[-1].append(step)
        return groups

    @staticmethod
    def _action_of(step):
        return step.action if hasattr(step, 'action') else step.get('action', '')

    @staticmethod
    def _name_of(step):
        return step.device_name if hasattr(step, 'device_name') else step.get('device_name', '')

    @staticmethod
    def _params_of(step):
        return dict(step.parameters) if hasattr(step, 'parameters') else dict(step.get('parameters', {}))

    def _emit_group_start(self, step):
        action = self._action_of(step)
        device_name = self._name_of(step)
        params = self._params_of(step)
        step_idx = getattr(step, '_device_step_idx', -1)
        duration = params.get('duration_s', 2.0)
        block_label = params.get('block_label', '')
        self.step_started.emit(step_idx, device_name, action, duration, block_label)
        print(f"[Runner] Group start: {device_name} → {action} ({duration}s) label={block_label}")

    def _emit_group_complete(self, step):
        action = self._action_of(step)
        device_name = self._name_of(step)
        step_idx = getattr(step, '_device_step_idx', -1)
        self.step_completed.emit(step_idx, device_name, action)

    def _execute_step(self, step, emit_signals=True):
        action = self._action_of(step)
        device_name = self._name_of(step)
        params = self._params_of(step)
        step_idx = getattr(step, '_device_step_idx', -1)
        is_system = step_idx < 0

        device = self._device_map.get(device_name)
        duration = params.get('duration_s', 2.0)

        if emit_signals and not is_system:
            block_label = params.get('block_label', '')
            self.step_started.emit(step_idx, device_name, action, duration, block_label)
        print(f"[Runner] Step {step_idx + 1 if not is_system else '?'}: {device_name} → {action} ({duration}s)")
        append_telemetry_line(f"step_start | {step_idx} | {device_name} | {action} | duration={duration}")

        if device is None and not is_system:
            self.error_occurred.emit(device_name or "__system__", f"Device '{device_name}' not found or not connected")
            time.sleep(0.5)
            if emit_signals:
                self.step_completed.emit(step_idx, device_name, action)
            return

        try:
            if action == "Stream":
                self._run_stream(step_idx, device, device_name, params, duration)
            elif action in ("force_current",):
                self._run_force_current(step_idx, device, device_name, params)
            elif action == "Configure":
                device.configure(**params)
                time.sleep(0.1)
            elif action == "Write":
                ch = params.get("channel", 1) - 1
                val = params.get("value", 0.0)
                device.write_output(ch, val)
                time.sleep(0.1)
            elif action == "Trigger":
                ch = params.get("channel", 1) - 1
                device.trigger_action(ch)
                time.sleep(0.1)
            elif action in ("Stimulus", "force_voltage"):
                if hasattr(device, 'configure'):
                    device.configure(mode="FVMI")
                if hasattr(device, 'write_output'):
                    ch = params.get("channel", 1) - 1
                    val = params.get("voltage", params.get("amplitude", 5.0))
                    device.write_output(max(0, ch), val)
                dur = params.get("duration_s", 1.0)
                time.sleep(dur)
                try:
                    device.write_output(max(0, ch), 0.0)
                except Exception:
                    pass
            elif action == "Measure":
                if hasattr(device, 'read_data'):
                    data = device.read_data()
                    print(f"[Runner] {device_name} Measure: got {data.shape if data is not None else 'None'}")
                time.sleep(0.1)
            elif action == "wait_input":
                msg = params.get("message", "Click OK to continue")
                self._input_result = None
                self.user_input_requested.emit(msg)
                while self._input_result is None and not self._abort:
                    time.sleep(0.1)
                print(f"[Runner] Wait input result: {self._input_result}")
            elif action == "log_event":
                append_telemetry_line(f"log | {step_idx} | {device_name} | {params}")
                print(f"[Runner] Log event: {params}")
            elif action == "pause":
                time.sleep(duration)
            elif action in ("start_recording", "stop_recording"):
                print(f"[Runner] Recording op {action} (handled by Stream)")
            else:
                print(f"[Runner] Unknown action '{action}', skipping")

        except Exception as e:
            self.error_occurred.emit(device_name, str(e))
            append_telemetry_line(f"step_error | {step_idx} | {device_name} | {action} | {e}")
            print(f"[Runner] Error in step {step_idx}: {e}")

        if emit_signals and not is_system:
            self.step_completed.emit(step_idx, device_name, action)
        append_telemetry_line(f"step_end | {step_idx} | {device_name} | {action} | ok")

    def _run_stream(self, step_idx, device, device_name, params, duration):
        if not getattr(device, 'connected', False):
            self.error_occurred.emit(device_name, "Device not connected")
            return

        device.configure(**{k: v for k, v in params.items() if k not in ('duration_s', 'block_label')})

        sr = device.sample_rate if device.sample_rate else 20000.0
        ch_labels = [c.name for c in getattr(device, 'channels', [])]
        num_ch = len(ch_labels) or 1
        self.device_configured.emit(device_name, num_ch, ch_labels, sr)

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
        while time.perf_counter() < deadline and not self._abort and getattr(device, 'connected', True):
            if self._paused:
                pause_start = time.perf_counter()
                while self._paused and not self._abort:
                    time.sleep(0.1)
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
            time.sleep(min(0.1, max(0, remaining)) if remaining > 0 else 0.01)

        sink.close()
        append_telemetry_line(
            f"acq_end | {step_idx} | {device_name} | Stream | samples_in_chunk={sink._chunk_samples_written}"
        )
        try:
            device.stop_acquisition()
        except Exception:
            pass

    def _run_force_current(self, step_idx, device, device_name, params):
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
        while time.perf_counter() < deadline and not self._abort:
            if self._paused:
                pause_start = time.perf_counter()
                while self._paused and not self._abort:
                    time.sleep(0.1)
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
            time.sleep(min(0.1, max(0, remaining)) if remaining > 0 else 0.01)

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
