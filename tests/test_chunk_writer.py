import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from leech.workers.chunk_writer import ChunkWriter


class ChunkWriterTest(unittest.TestCase):
    def _writer(self, tmp, **kwargs):
        w = ChunkWriter(sample_rate=1000.0, num_channels=3,
                        channel_names=["A", "B", "C"], **kwargs)
        w.raw_chunks_dir = Path(tmp)
        w.filename_prefix = "Step_000001"
        w._open_new_chunk_locked()
        return w

    def test_columns_and_values_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.append_data(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            w.close()
            path = sorted(Path(tmp).glob("chunk_*.csv"))[0]
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "time_s,A_uV,B_uV,C_uV,marker_id,marker_name")
            self.assertEqual(lines[1], "0.000000,1.0000,3.0000,5.0000,0,")
            self.assertEqual(lines[2], "0.001000,2.0000,4.0000,6.0000,0,")

    def test_marker_lands_on_first_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.append_data(np.ones((3, 2)), marker_info={0: {"id": 7, "name": "Stream"}})
            w.close()
            path = sorted(Path(tmp).glob("chunk_*.csv"))[0]
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(lines[1].endswith(",7,Stream"))

    def test_rotates_on_measured_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp, chunk_target_bytes=100)
            w.append_data(np.ones((3, 4)))
            w._chunk_start_t = time.perf_counter() - 2.0
            w.append_data(np.ones((3, 4)))
            w.close()
            self.assertGreaterEqual(len(list(Path(tmp).glob("chunk_*.csv"))), 2)

    def test_does_not_rotate_before_one_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp, chunk_target_bytes=1)
            w.append_data(np.ones((3, 2)))
            w.append_data(np.ones((3, 2)))
            w.close()
            self.assertEqual(len(list(Path(tmp).glob("chunk_*.csv"))), 1)

    def test_rotates_at_sample_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp, chunk_max_sec=1.0)
            w.append_data(np.ones((3, 1500)))
            w.close()
            self.assertEqual(len(list(Path(tmp).glob("chunk_*.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
