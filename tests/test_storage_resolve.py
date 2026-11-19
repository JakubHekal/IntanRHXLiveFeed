import json
import tempfile
import unittest
from pathlib import Path

from leech.experiment.experiment import ExperimentManager, resolve_storage


class StorageResolveTest(unittest.TestCase):
    def _write_metadata(self, root, devices, storage=None):
        data = {
            "schema_version": 3,
            "devices": devices,
            "sequence": [],
        }
        if storage is not None:
            data["storage"] = storage
        (root / "run.json").write_text(json.dumps(data), encoding="utf-8")

    def test_mixed_run_resolves_per_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "Intan_1" / "rest_260101_120000").mkdir(parents=True)
            (root / "raw" / "MCS_2").mkdir(parents=True)
            (root / "raw" / "Intan_1" / "rest_260101_120000" / "rest.rhs").write_bytes(b"")
            (root / "raw" / "MCS_2" / "chunk_b_000001.h5").write_bytes(b"")
            devices = [
                {"name": "Intan", "device_id": "1", "device_type": "rhx",
                 "config": {"storage_mode": "csv"},
                 "storage": {"format": "rhs", "data_dir": "raw/Intan_1"}},
                {"name": "MCS", "device_id": "2", "device_type": "mcs_mea",
                 "config": {"storage_mode": "hdf5"},
                 "storage": {"format": "hdf5", "data_dir": "raw/MCS_2"}},
            ]
            self._write_metadata(root, devices)
            fmt_a, dir_a = resolve_storage(devices[0], root)
            fmt_b, dir_b = resolve_storage(devices[1], root)
            self.assertEqual((fmt_a, dir_a.name), ("rhs", "Intan_1"))
            self.assertEqual((fmt_b, dir_b.name), ("hdf5", "MCS_2"))

    def test_old_per_device_csv_resolves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "Intan_1").mkdir(parents=True)
            (root / "raw" / "Intan_1" / "chunk_a_000001.csv").write_text("x", encoding="utf-8")
            devices = [{"name": "Intan", "device_id": "1", "device_type": "rhx",
                        "config": {"storage_mode": "csv"},
                        "storage": {"format": "csv", "data_dir": "raw/Intan_1"}}]
            self._write_metadata(root, devices)
            self.assertEqual(resolve_storage(devices[0], root)[0], "csv")

    def test_native_disk_files_override_csv_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "session_260101_120000").mkdir()
            (root / "raw" / "session_260101_120000" / "session.rhd").write_bytes(b"")
            devices = [{"name": "A", "device_type": "rhx", "config": {"storage_mode": "csv"}}]
            self._write_metadata(root, devices, storage={"format": "csv", "data_dir": "raw"})
            self.assertEqual(resolve_storage(devices[0], root)[0], "rhs")

    def test_old_run_without_storage_metadata_resolves_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "A").mkdir(parents=True)
            (root / "raw" / "A" / "rest_260101_120000").mkdir()
            (root / "raw" / "A" / "rest_260101_120000" / "rest.rhs").write_bytes(b"")
            devices = [{"name": "A", "device_id": "42", "device_type": "rhx",
                        "config": {"storage_mode": "csv"}}]
            self._write_metadata(root, devices)
            self.assertEqual(resolve_storage(devices[0], root)[0], "rhs")

    def test_old_intan_csv_only_run_resolves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "A").mkdir(parents=True)
            (root / "raw" / "A" / "chunk_000001_rest.csv").write_text("x", encoding="utf-8")
            devices = [{"name": "A", "device_id": "42", "device_type": "rhx",
                        "config": {"storage_mode": "csv"}}]
            self._write_metadata(root, devices)
            self.assertEqual(resolve_storage(devices[0], root)[0], "csv")

    def test_disk_probe_overrides_wrong_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "chunk_a_000001.h5").write_bytes(b"")
            devices = [{"name": "A", "device_type": "mcs_mea",
                        "config": {"storage_mode": "csv"}}]
            self._write_metadata(root, devices, storage={"format": "csv", "data_dir": "raw"})
            self.assertEqual(resolve_storage(devices[0], root)[0], "hdf5")

    def test_mixed_files_prefer_device_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw" / "A").mkdir(parents=True)
            (root / "raw" / "A" / "rest.rhs").write_bytes(b"")
            (root / "raw" / "A" / "chunk_000001.csv").write_text("x", encoding="utf-8")
            (root / "raw" / "M").mkdir(parents=True)
            (root / "raw" / "M" / "chunk_000001.h5").write_bytes(b"")
            (root / "raw" / "M" / "chunk_000001.csv").write_text("x", encoding="utf-8")
            intan = {"name": "A", "device_id": "1", "device_type": "rhx",
                     "config": {}, "storage": {"format": "csv", "data_dir": "raw/A"}}
            mcs = {"name": "M", "device_id": "2", "device_type": "mcs_mea",
                   "config": {}, "storage": {"format": "csv", "data_dir": "raw/M"}}
            self._write_metadata(root, [intan, mcs])
            self.assertEqual(resolve_storage(intan, root)[0], "rhs")
            self.assertEqual(resolve_storage(mcs, root)[0], "hdf5")

    def test_missing_files_and_format_raises_naming_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            devices = [{"name": "A", "device_id": "42", "device_type": "rhx", "config": {}}]
            self._write_metadata(root, devices)
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_storage({}, root)
            message = str(ctx.exception)
            self.assertIn("raw", message)
            self.assertIn(str(root), message)

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
            self.assertEqual(loaded["storage"], {"format": "rhs", "data_dir": "raw"})
            self.assertEqual(loaded["devices"][0]["storage"]["format"], "rhs")
            self.assertEqual(loaded["devices"][1]["storage"]["format"], "hdf5")


if __name__ == "__main__":
    unittest.main()
