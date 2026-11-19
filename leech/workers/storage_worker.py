import queue
import threading
import time

import numpy as np


class StorageWorker:
    """Owns an output sink on its own thread; acquisition never touches disk."""

    def __init__(self, sink, max_queue=512):
        self._sink = sink
        self._queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="storage-writer", daemon=True)
        self._closed = False
        self._lock = threading.Lock()
        self.error = None
        self.backpressure_count = 0
        self.samples_written = 0
        self._thread.start()

    def submit(self, arr: np.ndarray, marker_info=None):
        if self._closed:
            return False
        item = ("data", np.asarray(arr), marker_info)
        while True:
            try:
                if self.error is not None:
                    return False
                self._queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                with self._lock:
                    self.backpressure_count += 1

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item[0] == "stop":
                    return
                _, arr, marker_info = item
                self._sink.append_data(arr, marker_info=marker_info)
                self.samples_written += int(arr.shape[1]) if arr.ndim == 2 else int(arr.size)
            except Exception as exc:  # surface writer failure to the acquisition loop
                with self._lock:
                    self.error = exc
            finally:
                self._queue.task_done()

    def close(self):
        if self._closed:
            return self.error
        self._closed = True
        deadline = time.perf_counter() + 30.0
        while self._queue.unfinished_tasks and time.perf_counter() < deadline:
            time.sleep(0.01)
        try:
            self._queue.put(("stop", None, None), timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        try:
            self._sink.close()
        except Exception as exc:
            with self._lock:
                if self.error is None:
                    self.error = exc
        return self.error
