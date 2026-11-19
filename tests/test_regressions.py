import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from leech.device.base import DeviceOperation
from leech.device.csv_source import CsvSource
from leech.device.replay_paths import chunk_files
from leech.experiment.experiment import ExperimentManager, SequenceStep
from leech.experiment.experiment_dialog import RunExperimentDialog
from leech.experiment.experiment_runner import _RunnerThread
from leech.screens.timeline import DeviceRow
from leech.workers.replay_session import ReplaySession


class RegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_runner_uses_explicit_id_and_latches_failure(self):
        class Device:
            connected = True

            def close(self):
                pass

        first = Device()
        second = Device()
        row_a = DeviceRow("Same", [], "future", device_id="a")
        row_b = DeviceRow("Same", [], "future", device_id="b")
        row_a.instance = first
        row_b.instance = second
        step = SequenceStep(
            action="legacy_name",
            operation_id="missing_operation",
            device_name="Same",
            device_id="missing",
        )
        runner = _RunnerThread([row_a, row_b], [step], Path("run"))
        results = []
        runner.experiment_finished.connect(lambda success, message: results.append((success, message)))
        with patch("leech.experiment.experiment_runner.append_telemetry_line"):
            runner.run()
        self.assertIsNone(
            runner._device_for_step(SequenceStep(device_name="Same", device_id="missing"))
        )
        runner._device_map = {"a": first, "b": second}
        self.assertIs(
            runner._device_for_step(SequenceStep(device_name="Same", device_id="a")),
            first,
        )
        self.assertEqual(results, [(False, "Failed")])

    def test_runner_dispatches_custom_operation_handler(self):
        calls = []

        class CustomDevice:
            connected = True

            def get_operations(self):
                return [DeviceOperation(
                    "Zap", "Zap",
                    handler=lambda runner, idx, dev, name, params, dur:
                        calls.append((name, params.get("x"))),
                )]

            def canonical_operation_id(self, operation_id):
                return operation_id

            def close(self):
                pass

        row = DeviceRow("Custom", [], "future", device_id="c1")
        row.instance = CustomDevice()
        step = SequenceStep(
            action="Zap",
            operation_id="Zap",
            device_name="Custom",
            device_id="c1",
            parameters={"x": 7, "_start": 0},
        )
        runner = _RunnerThread([row], [step], Path("run"))
        with patch("leech.experiment.experiment_runner.append_telemetry_line"):
            runner.run()
        self.assertEqual(calls, [("Custom", 7)])

    def test_replay_reads_channels_and_uses_run_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp)
            raw_dir = run_path / "raw" / "device-a"
            raw_dir.mkdir(parents=True)
            (run_path / "run.json").write_text(json.dumps({
                "devices": [{
                    "device_id": "device-a",
                    "name": "A",
                    "blocks": [{"label": "second"}, {"label": "first"}],
                }]
            }), encoding="utf-8")
            first = raw_dir / "chunk_First_000001.csv"
            second = raw_dir / "chunk_Second_000001.csv"
            first.write_text(
                "time_s,A_uV,B_uV,marker_id,marker_name\n"
                "0,1,3,0,alpha\n"
                "0.1,2,4,0,\n",
                encoding="utf-8",
            )
            second.write_text(
                "time_s,A_uV,B_uV,marker_id,marker_name\n"
                "0,5,6,0,beta\n",
                encoding="utf-8",
            )
            ordered = chunk_files(raw_dir, "csv", {"second": 0, "first": 1})
            self.assertEqual(ordered, [second, first])
            source = CsvSource(
                raw_dir, sample_rate=0.1, label_order={"second": 0, "first": 1}
            )
            chunk, _ = source.read(1)
            np.testing.assert_array_equal(
                chunk,
                np.array([[5.0], [6.0]], dtype=np.float32),
            )
            source.close()

    def test_dialog_transfers_or_closes_devices_once(self):
        class Device:
            connected = True

            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        group = {
            "device_id": "device-a",
            "name": "A",
            "device_type": "future",
            "device_class": None,
            "param_defs": [],
            "current_config": {},
        }
        dialog = RunExperimentDialog("Test", str(Path("experiment")), [group])
        device = Device()
        dialog._device_sections[0]._device_instance = device
        instances, instance_map = dialog.take_devices()
        dialog._close_devices()
        self.assertEqual(instances, [device])
        self.assertEqual(instance_map, {"device-a": device})
        self.assertEqual(device.close_count, 0)
        device.close()

        cancelled_device = Device()
        dialog._device_sections[0]._device_instance = cancelled_device
        dialog.reject()
        self.assertEqual(cancelled_device.close_count, 1)

    def test_run_init_and_status_update_keep_schema_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            device_a = "00000000-0000-0000-0000-000000000001"
            device_b = "00000000-0000-0000-0000-000000000002"
            devices = [
                {"device_id": device_a, "name": "A", "device_type": "rhx", "config": {}},
                {"device_id": device_b, "name": "B", "device_type": "rhx", "config": {}},
            ]
            sequence = [{
                "step_id": 1,
                "action": "Stream",
                "operation_id": "Stream",
                "device_id": device_b,
                "device_name": "B",
                "parameters": {},
            }]
            ExperimentManager.init_run(
                run_path,
                {"metadata": {"experiment_name": "Test"}},
                devices,
                sequence,
            )
            loaded = ExperimentManager.load_run(run_path)
            self.assertEqual(loaded["schema_version"], 3)
            self.assertEqual(loaded["storage"], {"format": "rhs", "data_dir": "raw"})
            self.assertEqual(loaded["sequence"][0]["device_id"], device_b)
            ExperimentManager.update_run(run_path, "failed", error_count=2)
            updated = ExperimentManager.load_run(run_path)
            self.assertEqual(updated["run"]["status"], "failed")
            self.assertEqual(updated["run"]["error_count"], 2)

    def test_missing_hdf5_chunks_error_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp)
            (run_path / "run.json").write_text(json.dumps({
                "schema_version": 3,
                "devices": [{
                    "device_id": "a",
                    "name": "A",
                    "device_type": "mcs_mea2100",
                    "config": {},
                    "storage": {"format": "hdf5", "data_dir": "raw"},
                }],
                "sequence": [],
                "storage": {"format": "hdf5", "data_dir": "raw"},
            }), encoding="utf-8")
            errors = []
            session = ReplaySession(run_path)
            session.error.connect(errors.append)
            self.assertFalse(session.start())
            self.assertEqual(len(errors), 1)
            self.assertIn("No HDF5 chunks", errors[0])

    def test_rerun_into_non_empty_dir_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = Path(tmp) / "exp"
            run_path = exp_path / "runs" / "run1"
            run_path.mkdir(parents=True)
            (run_path / "keep.txt").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ExperimentManager.start_run(exp_path, "run1")

    def test_rename_corrupt_metadata_leaves_original_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = Path(tmp)
            src = exp_path / "old"
            src.mkdir()
            (src / "run.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperimentManager.rename_run(src, "new")
            self.assertTrue(src.exists())
            self.assertFalse((exp_path / "new").exists())


if __name__ == "__main__":
    unittest.main()
