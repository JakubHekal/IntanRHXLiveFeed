from PyQt5 import QtCore
import numpy as np
import time


class ExpensiveTaskWorker(QtCore.QThread):
    """Background worker for cancellable long-running recompute jobs.

    Jobs are keyed by task_type. Enqueuing a new job of the same task_type replaces
    the pending one so UI-driven configuration changes do not build long backlogs.
    """

    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QtCore.QMutex()
        self._condition = QtCore.QWaitCondition()
        self._running = True
        self._pending_by_type = {}

    def stop(self, timeout_ms: int = 3000) -> bool:
        if not self.isRunning():
            return True
        with QtCore.QMutexLocker(self._mutex):
            self._running = False
            self._pending_by_type.clear()
            self._condition.wakeAll()
        if self.wait(max(0, int(timeout_ms))):
            return True
        print(f"[WORKER] ExpensiveTaskWorker stop timeout after {int(timeout_ms)} ms")
        return False

    def schedule(self, job: dict):
        task_type = str(job.get("task_type", ""))
        if not task_type:
            return
        with QtCore.QMutexLocker(self._mutex):
            self._pending_by_type[task_type] = dict(job)
            self._condition.wakeOne()

    def run(self):
        while True:
            with QtCore.QMutexLocker(self._mutex):
                while not self._pending_by_type and self._running:
                    self._condition.wait(self._mutex)
                if not self._running:
                    return

                # Execute the oldest queued task_type first for fairness.
                task_type = next(iter(self._pending_by_type.keys()))
                job = self._pending_by_type.pop(task_type)

            started = time.perf_counter()
            result = {
                "task_type": task_type,
                "task_id": int(job.get("task_id", 0)),
                "session_id": int(job.get("session_id", 0)),
                "status": "ok",
                "error": "",
                "duration_ms": 0.0,
                "data": None,
            }

            try:
                if task_type == "spike_rebin":
                    result["data"] = self._run_spike_rebin(job)
                else:
                    raise ValueError(f"Unsupported task_type: {task_type}")
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
            finally:
                result["duration_ms"] = (time.perf_counter() - started) * 1000.0

            self.result_ready.emit(result)

    def _run_spike_rebin(self, job: dict) -> dict:
        spike_times = np.asarray(job.get("spike_times", []), dtype=np.float64)
        bin_sec = float(job.get("bin_sec", 1.0))
        last_time_s = float(job.get("last_time_s", 0.0))

        if bin_sec <= 0:
            raise ValueError("bin_sec must be > 0")

        total_bins = max(1, int(np.floor(last_time_s / bin_sec)) + 1)
        counts = np.zeros(total_bins, dtype=np.int64)

        if spike_times.size:
            bins = np.floor(spike_times / bin_sec).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < total_bins)]
            if bins.size:
                counts += np.bincount(bins, minlength=total_bins).astype(np.int64)

        minute_idx = (np.arange(total_bins, dtype=np.float64) * bin_sec) / 60.0
        return {
            "minute_idx": minute_idx,
            "counts": counts,
            "bin_sec": bin_sec,
            "last_time_s": last_time_s,
        }
