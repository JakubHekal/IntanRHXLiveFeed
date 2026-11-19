import unittest

import numpy as np

from leech.workers.storage_worker import StorageWorker


class _Sink:
    def __init__(self, fail_at=None):
        self.items = []
        self.fail_at = fail_at
        self.closed = False

    def append_data(self, arr, marker_info=None):
        if self.fail_at is not None and len(self.items) == self.fail_at:
            raise OSError("disk full")
        self.items.append((arr.copy(), marker_info))

    def close(self):
        self.closed = True


class StorageWorkerTest(unittest.TestCase):
    def test_submits_everything_before_close(self):
        sink = _Sink()
        worker = StorageWorker(sink, max_queue=2)
        for i in range(5):
            self.assertTrue(worker.submit(np.full((2, 3), i), marker_info={"i": i}))
        self.assertIsNone(worker.close())
        self.assertTrue(sink.closed)
        self.assertEqual([int(arr[0, 0]) for arr, _ in sink.items], [0, 1, 2, 3, 4])
        self.assertEqual(worker.samples_written, 15)

    def test_sink_error_is_returned_once(self):
        sink = _Sink(fail_at=1)
        worker = StorageWorker(sink)
        worker.submit(np.zeros((1, 4)))
        worker.submit(np.ones((1, 4)))
        error = worker.close()
        self.assertIsInstance(error, OSError)
        self.assertTrue(sink.closed)
        self.assertFalse(worker.submit(np.zeros((1, 4))))


if __name__ == "__main__":
    unittest.main()
