import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leech.workers.hdf5_chunk_writer import Hdf5ChunkWriter
from leech.device.csv_source import CsvSource
from leech.device.mcs_mea.replay import McsHdf5Source
from leech.device.replay_paths import chunk_files
from leech.workers.replay_source import make_source
from leech.workers.replay_session import ReplaySession

ADC_STEP_UV = 0.059605


def _write_hdf5(tmp, raw):
    channels = raw.shape[0]
    writer = Hdf5ChunkWriter(
        sample_rate=20000,
        num_channels=channels,
        channel_names=[f"CH{i:03d}" for i in range(channels)],
        chunk_max_sec=0.001,
        flush_interval_sec=0,
        adc_step_uv=ADC_STEP_UV,
    )
    writer.raw_chunks_dir = Path(tmp)
    microvolts = (raw.astype(np.float64) * ADC_STEP_UV).astype(np.float32)
    for offset in range(0, raw.shape[1], 4):
        writer.append_data(
            microvolts[:, offset:offset + 4],
            marker_info={4: {"id": 3, "name": "stim"}} if offset == 4 else None,
            force_flush=True,
        )
    writer.close()
    return microvolts


class ReplaySourceTest(unittest.TestCase):
    def test_hdf5_source_round_trip_and_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = (np.arange(36, dtype=np.int32).reshape(3, 12) - 10)
            expected = _write_hdf5(tmp, raw)
            source = McsHdf5Source(Path(tmp))
            self.assertEqual(source.num_channels, 3)
            self.assertEqual(source.sample_rate, 20000)
            self.assertEqual(source.total_samples, 12)
            self.assertEqual(source.labels, ["CH000", "CH001", "CH002"])

            pieces = []
            markers = []
            while not source.exhausted:
                chunk, found = source.read(5)
                if chunk is None:
                    break
                pieces.append(chunk)
                markers.extend(found)
            joined = np.concatenate(pieces, axis=1)
            source.close()
            np.testing.assert_allclose(joined, expected, rtol=0, atol=1e-6)
            self.assertEqual(markers, [(4, 3, "stim")])

    def test_replay_sources_live_with_their_devices(self):
        self.assertEqual(CsvSource.__module__, "leech.device.csv_source")
        self.assertEqual(
            McsHdf5Source.__module__, "leech.device.mcs_mea.replay"
        )

    def test_factory_dispatches_to_device_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = (np.arange(36, dtype=np.int32).reshape(3, 12) - 10)
            _write_hdf5(tmp, raw)
            source = make_source(Path(tmp), "hdf5")
            self.assertIsInstance(source, McsHdf5Source)
            source.close()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "chunk_A_000001.csv").write_text(
                "time_s,A_uV,marker_id,marker_name\n0,1,0,\n",
                encoding="utf-8",
            )
            source = make_source(root, "csv", sample_rate=10.0)
            self.assertIsInstance(source, CsvSource)
            source.close()

    def test_legacy_chunks_follow_recorded_block_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "chunk_First_000001.csv"
            second = root / "chunk_Second_000001.csv"
            for path in (first, second):
                path.write_text("time_s,A_uV,marker_id,marker_name\n0,1,0,\n", encoding="utf-8")
            ordered = chunk_files(root, "csv", {"second": 0, "first": 1})
            self.assertEqual([path.name for path in ordered], [second.name, first.name])

    def test_csv_source_streams_blocks_and_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "chunk_First_000001.csv").write_text(
                "time_s,A_uV,B_uV,marker_id,marker_name\n"
                "0.0,1,3,0,\n"
                "0.1,2,4,7,alpha\n",
                encoding="utf-8",
            )
            (root / "chunk_First_000002.csv").write_text(
                "time_s,A_uV,B_uV,marker_id,marker_name\n"
                "0.2,5,6,0,\n",
                encoding="utf-8",
            )
            source = CsvSource(root, sample_rate=10.0)
            self.assertEqual(source.labels, ["A", "B"])
            first, markers = source.read(10)
            np.testing.assert_array_equal(first, np.array([[1, 2, 5], [3, 4, 6]], dtype=np.float32))
            self.assertEqual(markers, [(1, 7, "alpha")])
            self.assertTrue(source.exhausted)
            self.assertIsNone(source.read(2)[0])
            source.close()


class ReplaySessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_start_refuses_second_session_and_stop_clears(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            raw = run / "raw" / "A"
            raw.mkdir(parents=True)
            (raw / "chunk_A_000001.csv").write_text(
                "time_s,A_uV,marker_id,marker_name\n0,1,0,\n0.1,2,0,\n",
                encoding="utf-8",
            )
            (run / "run.json").write_text(
                json.dumps({
                    "schema_version": 3,
                    "devices": [{
                        "device_id": "a",
                        "name": "A",
                        "device_type": "simulated",
                        "config": {"sample_rate": 10.0},
                        "storage": {"format": "csv", "data_dir": "raw/A"},
                    }],
                }),
                encoding="utf-8",
            )
            configured = []
            first = ReplaySession(run)
            first.device_configured.connect(lambda *args: configured.append(args))
            self.assertTrue(first.start(1.0))
            self.assertEqual(len(configured), 1)

            second = ReplaySession(run)
            errors = []
            second.error.connect(errors.append)
            self.assertFalse(second.start(1.0))
            self.assertIn("already running", errors[0])

            first.stop()
            self.assertIsNone(ReplaySession.active())
            self.assertTrue(second.start(1.0))
            second.stop()

    def test_one_tick_drives_every_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            devices = []
            for name in ("A", "B"):
                raw = run / "raw" / name
                raw.mkdir(parents=True)
                rows = "\n".join(
                    f"{i / 1000.0},{i},{5 if i == 50 else 0},"
                    + ("mid" if i == 50 else "")
                    for i in range(100)
                )
                (raw / f"chunk_{name}_000001.csv").write_text(
                    f"time_s,{name}_uV,marker_id,marker_name\n{rows}\n",
                    encoding="utf-8",
                )
                devices.append({
                    "device_id": name,
                    "name": name,
                    "device_type": "simulated",
                    "config": {"sample_rate": 1000.0},
                    "storage": {"format": "csv", "data_dir": f"raw/{name}"},
                })
            (run / "run.json").write_text(
                json.dumps({"schema_version": 3, "devices": devices}),
                encoding="utf-8",
            )
            received = []
            markers = []
            done = []
            session = ReplaySession(run)
            session.data_received.connect(lambda n, c: received.append((n, c.shape[1])))
            session.markers_received.connect(lambda n, m: markers.extend((n, item) for item in m))
            session.finished.connect(lambda: done.append(True))
            self.assertTrue(session.start("instant"))
            for _ in range(500):
                self.app.processEvents()
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(sum(n for name, n in received if name == "A"), 100)
            self.assertEqual(sum(n for name, n in received if name == "B"), 100)
            self.assertEqual(len(markers), 2)
            for _name, marker in markers:
                self.assertAlmostEqual(marker["timestamp_s"], 0.05)
            self.assertIsNone(ReplaySession.active())


if __name__ == "__main__":
    unittest.main()
