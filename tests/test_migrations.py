import json
import tempfile
import unittest
from pathlib import Path

from leech.experiment.experiment import ExperimentManager
from leech.experiment.migrations import (
    UnsupportedSchemaError,
    migrate_config_data,
    migrate_device_config,
    migrate_run_data,
)


class MigrationTest(unittest.TestCase):
    def test_legacy_config_gets_ids_and_preserves_device_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "metadata": {"experiment_name": "Legacy"},
                "devices": [{
                    "name": "Rig",
                    "device_type": "rhx",
                    "config": {"host": "board.local", "custom": 7},
                }],
                "sequence": [
                    {
                        "step_id": 1,
                        "action": "Stream",
                        "parameters": {},
                        "device_name": "Rig",
                    },
                    {
                        "step_id": 2,
                        "action": "wait_input",
                        "parameters": {},
                        "device_name": "__System__",
                    },
                ],
            }), encoding="utf-8")

            config = ExperimentManager.load(root)
            self.assertTrue(config.migration_changed)
            self.assertEqual(config.sequence[1].device_id, "system")
            self.assertEqual(config.sequence[1].device_name, "")
            self.assertTrue(config.devices[0]["device_id"])

            ExperimentManager.save(root, config, backup=True)
            self.assertTrue((root / "config.json.bak").exists())
            saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["devices"][0]["config"]["custom"], 7)
            self.assertEqual(saved["sequence"][0]["device_id"], config.devices[0]["device_id"])
            self.assertEqual(saved["sequence"][1]["device_id"], "system")

            reloaded = ExperimentManager.load(root)
            self.assertFalse(reloaded.migration_changed)
            self.assertEqual(reloaded.devices[0]["device_id"], config.devices[0]["device_id"])

    def test_migration_is_idempotent_and_rejects_newer_schema(self):
        legacy = {
            "devices": [{"name": "A", "device_type": "rhx"}],
            "sequence": [],
        }
        first, first_report = migrate_config_data(legacy)
        second, second_report = migrate_config_data(first)
        self.assertEqual(first, second)
        self.assertTrue(first_report.changed)
        self.assertFalse(second_report.changed)

        with self.assertRaises(UnsupportedSchemaError):
            migrate_config_data({"schema_version": 99})

    def test_same_type_devices_resolve_by_id(self):
        data = {
            "devices": [
                {"name": "A", "device_type": "rhx"},
                {"name": "B", "device_type": "rhx"},
            ],
            "sequence": [
                {"step_id": 1, "action": "Stream", "parameters": {}, "device_name": "B"},
            ],
        }
        migrated, report = migrate_config_data(data)
        self.assertFalse(report.warnings)
        self.assertNotEqual(
            migrated["devices"][0]["device_id"],
            migrated["devices"][1]["device_id"],
        )
        self.assertEqual(migrated["sequence"][0]["device_id"], migrated["devices"][1]["device_id"])

    def test_current_schema_normalizes_reference_case(self):
        device_id = "00000000-0000-0000-0000-00000000000a"
        migrated, report = migrate_config_data({
            "schema_version": 2,
            "execution_control": {"required_devices": []},
            "devices": [{
                "device_id": device_id,
                "name": "A",
                "device_type": "rhx",
                "config": {},
                "config_version": 1,
            }],
            "sequence": [{
                "step_id": 1,
                "action": "Stream",
                "operation_id": "Stream",
                "device_id": device_id.upper(),
                "device_name": "A",
                "parameters": {},
            }],
        })
        self.assertTrue(report.changed)
        self.assertEqual(migrated["sequence"][0]["device_id"], device_id)

    def test_unknown_plugin_is_preserved(self):
        data = {
            "devices": [{
                "name": "Future",
                "device_type": "future_device",
                "config": {"future": True},
            }],
            "sequence": [],
        }
        migrated, _ = migrate_config_data(data)
        self.assertEqual(migrated["devices"][0]["device_type"], "future_device")
        self.assertEqual(migrated["devices"][0]["config"], {"future": True})

    def test_plugin_config_migration_uses_declared_version(self):
        class Plugin:
            config_version = 2

            @classmethod
            def migrate_config(cls, config, from_version):
                result = dict(config)
                result["migrated_from"] = from_version
                return result

        config, version, changed = migrate_device_config(Plugin, {"host": "board"}, 1)
        self.assertEqual(config, {"host": "board", "migrated_from": 1})
        self.assertEqual(version, 2)
        self.assertTrue(changed)

    def test_run_migration_adds_ids_without_rewriting_source_shape(self):
        data = {
            "devices": [
                {"name": "A", "device_type": "rhx", "config": {"sample_rate": 1000}},
                {"name": "B", "device_type": "rhx", "config": {}},
            ],
            "sequence": [
                {"step_id": 1, "action": "Stream", "parameters": {}, "device_name": "A"},
                {"step_id": 2, "action": "Stream", "parameters": {}, "device_name": "B"},
            ],
        }
        migrated, _ = migrate_run_data(data)
        self.assertEqual(migrated["schema_version"], 2)
        self.assertNotEqual(
            migrated["devices"][0]["device_id"],
            migrated["devices"][1]["device_id"],
        )
        self.assertEqual(migrated["sequence"][0]["device_id"], migrated["devices"][0]["device_id"])
        self.assertEqual(migrated["sequence"][1]["device_id"], migrated["devices"][1]["device_id"])


if __name__ == "__main__":
    unittest.main()
