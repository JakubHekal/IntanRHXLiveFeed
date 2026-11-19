import re
import threading
import time
from pathlib import Path

from PyQt5 import QtCore
import numpy as np

RAW_CHUNK_SEC = 300
CSV_FILE_BUFFER_BYTES = 1024 * 1024
CSV_FLUSH_INTERVAL_SEC = 1.0
from leech.workers.storage_worker import StorageWorker
from leech.telemetry_logger import append_telemetry_line
from leech.experiment.migrations import SYSTEM_DEVICE_ID, SYSTEM_DEVICE_TYPE
_SYSTEM_ACTIONS = frozenset(("wait_input", "log_event", "pause", "start_recording", "stop_recording"))


def sanitize_raw_key(key):
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._")
    if not key:
        raise ValueError("Device raw-data key is empty")
    return key


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
        self._device_key_by_instance = {}
        self._failed = False

    @staticmethod
    def _instance_of(device):
        instance = getattr(device, "instance", None)
        if instance is not None:
            return instance
        return device[4] if isinstance(device, (list, tuple)) and len(device) >= 5 else None

    @staticmethod
    def _device_id_of(device):
        return getattr(device, "device_id", "") or ""

    @staticmethod
    def _device_id_of_step(step):
        if hasattr(step, "device_id"):
            return step.device_id or ""
        return step.get("device_id", "")

    def _device_for_step(self, step):
        device_id = self._device_id_of_step(step)
        if device_id and device_id != SYSTEM_DEVICE_ID:
            return self._device_map.get(device_id)
        return self._device_map.get(self._name_of(step))

    def _raw_device_key(self, device, fallback):
        return sanitize_raw_key(
            str(self._device_key_by_instance.get(id(device), fallback)))

    def _report_error(self, device_name, message):
        self._failed = True
        self.error_occurred.emit(device_name, message)

    def abort(self):
        self._abort = True
        self.requestInterruption()

    def run(self):
        try:
            self._device_map = {}
            self._device_key_by_instance = {}
            name_counts = {}
            for d in self._devices:
                if d[0]:
                    name_counts[d[0]] = name_counts.get(d[0], 0) + 1
            for d in self._devices:
                inst = self._instance_of(d)
                if inst is None:
                    continue
                device_id = self._device_id_of(d)
                if device_id and device_id != SYSTEM_DEVICE_ID:
                    self._device_map[device_id] = inst
                if name_counts.get(d[0]) == 1:
                    self._device_map[d[0]] = inst
                if device_id:
                    self._device_key_by_instance[id(inst)] = f"{d[0]}_{device_id}"
                else:
                    self._device_key_by_instance[id(inst)] = d[0].replace(" ", "_")

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
                else:
                    # Prepare-then-run: program stim (and any other config) while
                    # the board is stopped. Uploading stim parameters while the
                    # board runs wedges the Intan server's run loop (the 45 s
                    # data stall), so it must happen before streaming starts.
                    for step in device_steps:
                        if self._abort:
                            break
                        self._prepare_step(step)
                    if self._abort:
                        break
                    if len(device_steps) == 1:
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

            success = not self._abort and not self._failed
            msg = "Completed" if success else ("Aborted" if self._abort else "Failed")
            self.experiment_finished.emit(success, msg)
            print(f"[Runner] Experiment {msg}")

        except Exception as e:
            self.experiment_finished.emit(False, str(e))
            print(f"[Runner] Fatal error: {e}")

        finally:
            for d in self._devices:
                inst = self._instance_of(d)
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
        if hasattr(step, "operation_id"):
            return step.operation_id or step.action
        return step.get("operation_id") or step.get("action", "")

    @staticmethod
    def _name_of(step):
        return step.device_name if hasattr(step, 'device_name') else step.get('device_name', '')

    @staticmethod
    def _params_of(step):
        return dict(step.parameters) if hasattr(step, 'parameters') else dict(step.get('parameters', {}))

    @staticmethod
    def _operation_for_device(device, action):
        if device is None:
            from leech.device.registry import _SYSTEM_OPERATIONS
            operations = _SYSTEM_OPERATIONS
        else:
            operations = device.get_operations()
            action = device.canonical_operation_id(action)
        for operation in operations:
            if action in (operation.operation_id, operation.name):
                return operation
        return None

    def _emit_group_start(self, step):
        action = self._action_of(step)
        device_name = self._name_of(step)
        params = self._params_of(step)
        step_idx = getattr(step, '_device_step_idx', -1)
        duration = params.get('duration_s', 2.0)
        block_label = params.get('block_label', '')
        self.step_started.emit(step_idx, device_name, action, duration, block_label)
        print(f"[Runner] Group start: {device_name} -> {action} ({duration}s) label={block_label}")

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

        device = self._device_for_step(step)
        duration = params.get('duration_s', 2.0)

        if emit_signals and not is_system:
            block_label = params.get('block_label', '')
            self.step_started.emit(step_idx, device_name, action, duration, block_label)
        print(f"[Runner] Step {step_idx + 1 if not is_system else '?'}: {device_name} -> {action} ({duration}s)")
        append_telemetry_line(f"step_start | {step_idx} | {device_name} | {action} | duration={duration}")

        if device is None and not is_system:
            self._report_error(
                device_name or SYSTEM_DEVICE_TYPE,
                f"Device ID {self._device_id_of_step(step)!r} not found or not connected",
            )
            time.sleep(0.5)
            if emit_signals:
                self.step_completed.emit(step_idx, device_name, action)
            return

        try:
            operation = self._operation_for_device(device, action)
            handler = operation.handler if operation is not None else None
            if handler is not None:
                handler(self, step_idx, device, device_name, params, duration)
            elif action == "Stream":
                self._run_stream(step_idx, device, device_name, params, duration)
            elif action in ("Configure", "configure"):
                device.configure(**{k: v for k, v in params.items()
                                    if k not in ('duration_s', 'block_label', '_start')})
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
            else:
                self._report_error(device_name or SYSTEM_DEVICE_TYPE, f"Unknown action {action!r}")

        except Exception as e:
            self._report_error(device_name or SYSTEM_DEVICE_TYPE, str(e))
            append_telemetry_line(f"step_error | {step_idx} | {device_name} | {action} | {e}")
            print(f"[Runner] Error in step {step_idx}: {e}")

        if emit_signals and not is_system:
            self.step_completed.emit(step_idx, device_name, action)
        append_telemetry_line(
            f"step_end | {step_idx} | {device_name} | {action} | "
            f"{'failed' if self._failed else 'ok'}"
        )

    def _run_stream(self, step_idx, device, device_name, params, duration):
        if not getattr(device, 'connected', False):
            self._report_error(device_name, "Device not connected")
            return

        device.configure(**{k: v for k, v in params.items() if k not in ('duration_s', 'block_label')})

        sr = device.sample_rate if device.sample_rate else 20000.0
        ch_labels = [c.name for c in getattr(device, 'channels', [])]
        num_ch = len(ch_labels) or 1
        self.device_configured.emit(device_name, num_ch, ch_labels, sr)

        dev_raw_dir = self._run_path / "raw" / self._raw_device_key(device, device_name)
        dev_raw_dir.mkdir(parents=True, exist_ok=True)

        label = str(params.get("block_label", "Stream") or "Stream")
        file_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or "Stream"
        file_prefix = f"{step_idx + 1:06d}_{file_label}"

        rhs_folder = None
        sink = device.create_output_sink(
            sample_rate=sr,
            channel_names=ch_labels,
            raw_dir=dev_raw_dir,
            file_prefix=file_prefix,
            flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
        )
        if sink is None:
            # Device records natively (e.g. RHX .rhs), one folder per stage.
            rhs_folder = device.start_recording_session(dev_raw_dir, file_prefix)

        storage = StorageWorker(sink) if sink is not None else None
        device.start_acquisition()
        append_telemetry_line(
            f"acq_start | {step_idx} | {device_name} | Stream | "
            f"ch={num_ch} sr={sr} label={label}"
            + (f" rhs={rhs_folder}" if rhs_folder else "")
        )
        first_chunk = True
        stalled = False
        last_data_time = time.perf_counter()

        deadline = time.perf_counter() + duration
        while (time.perf_counter() < deadline and not self._abort
               and getattr(device, 'connected', True) and not stalled):
            if self._paused:
                pause_start = time.perf_counter()
                while self._paused and not self._abort:
                    time.sleep(0.1)
                deadline += time.perf_counter() - pause_start
                last_data_time = time.perf_counter()
            try:
                data = device.read_data()
            except Exception:
                data = None
            if data is not None:
                arr = np.asarray(data)
                if arr.ndim == 2 and arr.shape[1] > 0:
                    if storage is not None:
                        marker_info = (
                            {0: {"id": step_idx + 1, "name": label}}
                            if first_chunk
                            else None
                        )
                        if not storage.submit(arr, marker_info=marker_info):
                            self._report_error(device_name, f"Storage writer failed: {storage.error}")
                            stalled = True
                        first_chunk = False
                    self.data_received.emit(device_name, arr)
                last_data_time = time.perf_counter()
            # ponytail: watchdog — if the Intan TCP output stalls (buffer
            # overflow / controller wedged) and the connection silently stops
            # delivering, fail the step now instead of streaming zeros for the
            # rest of the duration.
            if time.perf_counter() - last_data_time > 5.0 and deadline - time.perf_counter() > 5.0:
                stalled = True
                msg = f"Data stall: no data from {device_name} for 5 s while streaming (device data output stalled)"
                self._report_error(device_name, msg)
                append_telemetry_line(f"acq_stall | {step_idx} | {device_name} | Stream | no data for 5s")
                print(f"[Runner] {msg}")
            remaining = deadline - time.perf_counter()
            time.sleep(min(0.01, max(0, remaining)) if remaining > 0 else 0.001)

        if not self._abort and not stalled and not getattr(device, "connected", True):
            self._report_error(device_name, "Device disconnected during stream")
        if storage is not None:
            storage_error = storage.close()
            if storage_error is not None and not self._abort:
                self._report_error(device_name, f"Storage writer failed: {storage_error}")
            append_telemetry_line(
                f"acq_end | {step_idx} | {device_name} | Stream | "
                f"samples_in_chunk={sink._chunk_samples_written} "
                f"backpressure={storage.backpressure_count}"
            )
        else:
            append_telemetry_line(
                f"acq_end | {step_idx} | {device_name} | Stream | rhs={rhs_folder}"
            )
        try:
            device.stop_acquisition()
        except Exception:
            pass

    def _prepare_step(self, step):
        """Let the device configure itself while its board is stopped, before
        the group runs. Uploads are rejected (or wedge the server) while running."""
        action = self._action_of(step)
        device_name = self._name_of(step)
        params = self._params_of(step)
        step_idx = getattr(step, '_device_step_idx', -1)
        device = self._device_for_step(step)
        if device is None:
            return
        operation = self._operation_for_device(device, action)
        prepare = operation.prepare if operation is not None else None
        if prepare is None:
            return
        prepare(self, step_idx, device, device_name, params, params.get('duration_s', 2.0))
