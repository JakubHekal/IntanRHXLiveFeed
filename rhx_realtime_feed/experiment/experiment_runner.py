import time
from pathlib import Path

from PyQt5 import QtCore
import numpy as np

from rhx_realtime_feed.device import Device
from rhx_realtime_feed.workers.device_worker import DeviceWorker, RAW_CHUNK_SEC, CSV_FILE_BUFFER_BYTES, CSV_FLUSH_INTERVAL_SEC
from rhx_realtime_feed.workers.chunk_writer import ChunkWriter


class ExperimentRunner(QtCore.QObject):
    step_started = QtCore.pyqtSignal(int, str, str, float)  # step_index, device_name, action, duration
    step_completed = QtCore.pyqtSignal(int, str, str)
    experiment_finished = QtCore.pyqtSignal(bool, str)  # success, message
    error_occurred = QtCore.pyqtSignal(str, str)  # device_name, error_message

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

            raw_chunks_base = self._run_path / "raw_chunks"
            raw_chunks_base.mkdir(parents=True, exist_ok=True)

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

                        sr = device.sample_rate if device.sample_rate else 20000.0
                        num_ch = len(getattr(device, 'channels', [])) or 1
                        step_dir = raw_chunks_base / f"{device_name}_step_{step_idx + 1:02d}"
                        step_dir.mkdir(parents=True, exist_ok=True)

                        sink = ChunkWriter(
                            sample_rate=sr,
                            num_channels=num_ch,
                            chunk_max_sec=RAW_CHUNK_SEC,
                            buffer_bytes=CSV_FILE_BUFFER_BYTES,
                            flush_interval_sec=CSV_FLUSH_INTERVAL_SEC,
                        )

                        worker = DeviceWorker(device, sink, project_name=device_name, project_path=str(step_dir), close_device_on_stop=False)
                        stream_ok = [True]  # assume success, signal overrides on failure

                        def _on_connection_result(ok, msg):
                            stream_ok[0] = ok
                            print(f"[Runner] {device_name} stream: ok={ok}, msg={msg}")

                        worker.connection_request_result_signal.connect(_on_connection_result)

                        device.configure(**{k: v for k, v in params.items() if k != 'duration_s'})
                        worker.start()

                        # give the worker a moment to connect, then check
                        self.msleep(500)
                        if not stream_ok[0]:
                            self.error_occurred.emit(device_name, "Stream connection failed")
                            worker.stop(timeout_ms=1000)
                            worker.deleteLater()
                            continue

                        deadline = time.perf_counter() + duration
                        while time.perf_counter() < deadline and not self._abort and not self.isInterruptionRequested() and getattr(device, 'connected', True):
                            remaining = deadline - time.perf_counter()
                            self.msleep(int(min(100, remaining * 1000)) if remaining > 0 else 10)

                        worker.stop(timeout_ms=3000)
                        worker.deleteLater()

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

                    elif action == "Stimulus":
                        if hasattr(device, 'write_output'):
                            ch = params.get("channel", 1) - 1
                            val = params.get("voltage", params.get("amplitude", 5.0))
                            device.write_output(max(0, ch), val)
                        dur = params.get("duration_s", 1.0)
                        self.msleep(int(dur * 1000))

                    elif action == "Measure":
                        if hasattr(device, 'read_data'):
                            data = device.read_data()
                            print(f"[Runner] {device_name} Measure: got {data.shape if data is not None else 'None'}")
                        self.msleep(100)

                    elif action == "wait_input":
                        # ponytail: skip wait_input for now, add when experiment explicitly needs it
                        print(f"[Runner] Wait for user input (skipped)")

                    elif action == "log_event":
                        # ponytail: log_event is a no-op without a defined logging interface
                        print(f"[Runner] Log event: {params}")

                    elif action in ("start_recording", "stop_recording", "pause"):
                        # ponytail: system ops are no-ops in basic sequential execution
                        print(f"[Runner] System op {action} (skipped)")

                    else:
                        print(f"[Runner] Unknown action '{action}', skipping")

                except Exception as e:
                    self.error_occurred.emit(device_name, str(e))
                    print(f"[Runner] Error in step {step_idx}: {e}")

                self.step_completed.emit(step_idx, device_name, action)

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
