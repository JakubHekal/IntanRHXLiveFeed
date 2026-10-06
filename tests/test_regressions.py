import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from leech.experiment.experiment import ExperimentManager, SequenceStep
from leech.experiment.experiment_dialog import RunExperimentDialog
from leech.experiment.experiment_runner import _RunnerThread
from leech.screens.timeline import DeviceRow
from leech.workers.replay_worker import ReplayWorker


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
            worker = ReplayWorker(
                run_path,
                "Replay",
                device_dir=raw_dir,
                device_id="device-a",
                source_device_name="A",
            )
            ordered = sorted([first, second], key=worker._chunk_sort_key)
            self.assertEqual(ordered, [second, first])
            np.testing.assert_array_equal(
                worker._read_chunk(first),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )

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

    def test_run_init_and_status_update_keep_schema_v2(self):
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
            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(loaded["sequence"][0]["device_id"], device_b)
            ExperimentManager.update_run(run_path, "failed", error_count=2)
            updated = ExperimentManager.load_run(run_path)
            self.assertEqual(updated["run"]["status"], "failed")
            self.assertEqual(updated["run"]["error_count"], 2)


if __name__ == "__main__":
    unittest.main()
