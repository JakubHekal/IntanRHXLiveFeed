import json
import tempfile
import unittest
from pathlib import Path

from leech.experiment.experiment import ExperimentManager, resolve_storage


class StorageResolveTest(unittest.TestCase):
    def _write_metadata(self, root, devices, storage=None):
        (root / "run.json").write_text(json.dumps({
            "schema_version": 3,
            "devices": devices,
            "sequence": [],
            "storage": storage or {"format": "rhs", "data_dir": "raw"},
        }), encoding="utf-8")

    def test_mixed_run_resolves_per_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "Intan_1").mkdir(parents=True)
            (root / "raw" / "MCS_2").mkdir(parents=True)
            (root / "raw" / "Intan_1" / "chunk_a_000001.csv").write_text("x", encoding="utf-8")
            (root / "raw" / "MCS_2" / "chunk_b_000001.h5").write_bytes(b"")
            devices = [
                {"name": "Intan", "device_id": "1", "device_type": "rhx",
                 "config": {"storage_mode": "csv"},
                 "storage": {"format": "csv", "data_dir": "raw/Intan_1"}},
                {"name": "MCS", "device_id": "2", "device_type": "mcs_mea",
                 "config": {"storage_mode": "hdf5"},
                 "storage": {"format": "hdf5", "data_dir": "raw/MCS_2"}},
            ]
            self._write_metadata(root, devices)
            fmt_a, dir_a = resolve_storage(devices[0], root)
            fmt_b, dir_b = resolve_storage(devices[1], root)
            self.assertEqual((fmt_a, dir_a.name), ("csv", "Intan_1"))
            self.assertEqual((fmt_b, dir_b.name), ("hdf5", "MCS_2"))

    def test_falls_back_to_device_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "chunk_a_000001.csv").write_text("x", encoding="utf-8")
            devices = [{"name": "A", "device_type": "rhx", "config": {"storage_mode": "csv"}}]
            self._write_metadata(root, devices)
            self.assertEqual(resolve_storage(devices[0], root)[0], "csv")

    def test_disk_probe_overrides_wrong_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "chunk_a_000001.h5").write_bytes(b"")
            devices = [{"name": "A", "device_type": "mcs_mea",
                        "config": {"storage_mode": "csv"}}]
            self._write_metadata(root, devices)
            self.assertEqual(resolve_storage(devices[0], root)[0], "hdf5")

    def test_new_run_metadata_is_additive(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            ExperimentManager.init_run(
                run_path,
                {"metadata": {"experiment_name": "Mixed"}},
                [
                    {"name": "Intan",
                     "device_id": "00000000-0000-0000-0000-000000000001",
                     "device_type": "rhx", "config": {"storage_mode": "csv"}},
                    {"name": "MCS",
                     "device_id": "00000000-0000-0000-0000-000000000002",
                     "device_type": "mcs_mea", "config": {"storage_mode": "csv"}},
                ],
                [],
            )
            loaded = ExperimentManager.load_run(run_path)
            self.assertEqual(loaded["schema_version"], 3)
            self.assertEqual(loaded["storage"], {"format": "csv", "data_dir": "raw"})
            self.assertEqual(loaded["devices"][0]["storage"]["format"], "csv")
            self.assertEqual(loaded["devices"][1]["storage"]["format"], "hdf5")


if __name__ == "__main__":
    unittest.main()
