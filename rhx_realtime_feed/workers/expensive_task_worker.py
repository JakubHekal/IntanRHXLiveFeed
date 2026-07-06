from PyQt5 import QtCore
import numpy as np

class ExpensiveTaskWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QtCore.QMutex()
        self._condition = QtCore.QWaitCondition()
        self._running = True
        self._pending = None

    def stop(self, timeout_ms: int = 3000) -> bool:
        if not self.isRunning():
            return True
        with QtCore.QMutexLocker(self._mutex):
            self._running = False
            self._condition.wakeAll()
        if self.wait(max(0, int(timeout_ms))):
            return True
        print(f"[WORKER] ExpensiveTaskWorker stop timeout after {int(timeout_ms)} ms")
        return False

    def schedule(self, job: dict):
        with QtCore.QMutexLocker(self._mutex):
            self._pending = dict(job)
            self._condition.wakeOne()

    def run(self):
        while True:
            with QtCore.QMutexLocker(self._mutex):
                while self._pending is None and self._running:
                    self._condition.wait(self._mutex)
                if not self._running:
                    return
                job = self._pending
                self._pending = None

            try:
                data = self._run_spike_rebin(job)
                self.result_ready.emit({
                    "task_type": "spike_rebin",
                    "task_id": job.get("task_id", 0),
                    "session_id": job.get("session_id", 0),
                    "status": "ok", "data": data, "error": "",
                })
            except Exception as exc:
                self.result_ready.emit({
                    "task_type": "spike_rebin",
                    "task_id": job.get("task_id", 0),
                    "session_id": job.get("session_id", 0),
                    "status": "error", "data": None, "error": str(exc),
                })

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
