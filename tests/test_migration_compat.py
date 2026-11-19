"""Old projects keep loading now that Intan storage is native-only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from leech.device.intan_rhx.device import IntanRHXDevice
from leech.experiment.experiment import ExperimentManager, resolve_storage
from leech.experiment.migrations import CONFIG_SCHEMA_VERSION, RUN_SCHEMA_VERSION

DEVICE_ID = "00000000-0000-0000-0000-000000000001"


def _old_config(schema_version):
    return {
        "schema_version": schema_version,
        "metadata": {"experiment_name": "Old"},
        "execution_control": {"required_devices": []},
        "sequence": [],
        "post_processing": [],
        "devices": [{
            "device_id": DEVICE_ID,
            "name": "Intan",
            "device_type": "rhx",
            "config": {"storage_mode": "csv"},
            "config_version": 1,
        }],
    }


def _old_run(schema_version, storage=None, devices=None):
    data = {
        "schema_version": schema_version,
        "devices": devices if devices is not None else [{
            "device_id": DEVICE_ID,
            "name": "Intan",
            "device_type": "rhx",
            "config": {"storage_mode": "csv"},
            "config_version": 1,
        }],
        "sequence": [],
    }
    if storage is not None:
        data["storage"] = storage
    return data


class MigrationCompatTest(unittest.TestCase):
    def test_schema_versions_stay_3(self):
        self.assertEqual(RUN_SCHEMA_VERSION, 3)
        self.assertEqual(CONFIG_SCHEMA_VERSION, 3)

    def test_old_config_csv_intan_loads_and_device_coerces(self):
        for version in (2, 3):
            with tempfile.TemporaryDirectory() as tmp:
                exp = Path(tmp)
                (exp / "config.json").write_text(
                    json.dumps(_old_config(version)), encoding="utf-8")
                config = ExperimentManager.load(exp)
                self.assertEqual(config.devices[0]["config"]["storage_mode"], "csv")
                with patch("builtins.print") as printed:
                    device = IntanRHXDevice(
                        autolaunch=False, **config.devices[0]["config"])
                self.assertEqual(device.storage_mode, "rhs")
                warnings = [
                    call for call in printed.call_args_list
                    if call.args and "storage_mode" in call.args[0]
                ]
                self.assertEqual(len(warnings), 1)

    def test_old_run_with_native_rhd_resolves_rhs(self):
        for version in (2, 3):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "raw" / "Intan_1").mkdir(parents=True)
                (root / "raw" / "Intan_1" / "rest_260101_120000").mkdir()
                (root / "raw" / "Intan_1" / "rest_260101_120000" / "rest.rhd").write_bytes(b"")
                (root / "run.json").write_text(
                    json.dumps(_old_run(version)), encoding="utf-8")
                loaded = ExperimentManager.load_run(root)
                fmt, data_dir = resolve_storage(loaded["devices"][0], root)
                self.assertEqual(fmt, "rhs")
                self.assertTrue(data_dir.is_dir())

    def test_old_run_with_native_rhs_and_no_storage_resolves_rhs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "Intan_1").mkdir(parents=True)
            (root / "raw" / "Intan_1" / "rest.rhs").write_bytes(b"")
            (root / "run.json").write_text(
                json.dumps(_old_run(3)), encoding="utf-8")
            loaded = ExperimentManager.load_run(root)
            self.assertNotIn("storage", loaded["devices"][0])
            self.assertEqual(resolve_storage(loaded["devices"][0], root)[0], "rhs")

    def test_old_intan_csv_only_run_resolves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "Intan_1").mkdir(parents=True)
            (root / "raw" / "Intan_1" / "chunk_000001_rest.csv").write_text(
                "x", encoding="utf-8")
            (root / "run.json").write_text(
                json.dumps(_old_run(3, storage={"format": "csv", "data_dir": "raw"})),
                encoding="utf-8")
            loaded = ExperimentManager.load_run(root)
            self.assertEqual(resolve_storage(loaded["devices"][0], root)[0], "csv")

    def test_init_run_writes_schema_3_and_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            ExperimentManager.init_run(
                run_path,
                {"metadata": {"experiment_name": "Old"}},
                [{
                    "device_id": DEVICE_ID,
                    "name": "Intan",
                    "device_type": "rhx",
                    "config": {"storage_mode": "csv"},
                }],
                [],
            )
            loaded = ExperimentManager.load_run(run_path)
            self.assertEqual(loaded["schema_version"], 3)
            self.assertEqual(loaded["storage"], {"format": "rhs", "data_dir": "raw"})
            self.assertEqual(
                loaded["devices"][0]["storage"],
                {"format": "rhs", "data_dir": f"raw/Intan_{DEVICE_ID}"},
            )


if __name__ == "__main__":
    unittest.main()
