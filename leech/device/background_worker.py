from PyQt5 import QtCore


class BackgroundWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QtCore.QMutex()
        self._condition = QtCore.QWaitCondition()
        self._running = True
        self._pending = None

    def schedule(self, fn, *args, **kwargs):
        with QtCore.QMutexLocker(self._mutex):
            self._pending = (fn, args, kwargs)
            self._condition.wakeOne()

    def stop(self, timeout_ms: int = 3000) -> bool:
        if not self.isRunning():
            return True
        with QtCore.QMutexLocker(self._mutex):
            self._running = False
            self._pending = None
            self._condition.wakeAll()
        if self.wait(max(0, int(timeout_ms))):
            return True
        return False

    def run(self):
        while True:
            with QtCore.QMutexLocker(self._mutex):
                while self._pending is None and self._running:
                    self._condition.wait(self._mutex)
                if not self._running:
                    return
                fn, args, kwargs = self._pending
                self._pending = None
            try:
                result = fn(*args, **kwargs)
                self.result_ready.emit(result)
            except Exception as exc:
                self.result_ready.emit(exc)
