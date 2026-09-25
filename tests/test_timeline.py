import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QContextMenuEvent
from PyQt5.QtWidgets import QApplication, QMenu, QMessageBox

from leech.experiment.experiment import ExperimentManager
from leech.screens.timeline import ExperimentTimeline
from leech.ui import MainWindow


class TimelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_migration_and_name_only_device_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "metadata": {"experiment_name": "Legacy"},
                "execution_control": {"required_devices": []},
                "devices": [{"name": "Dev", "device_type": "unknown"}],
                "sequence": [{
                    "step_id": 1,
                    "action": "wait_input",
                    "parameters": {"duration_s": 0},
                    "device_name": "__System__",
                }],
            }), encoding="utf-8")

            class Fake:
                _current_experiment_path = str(root)
                main_stage = SimpleNamespace(
                    timeline=ExperimentTimeline(),
                    plot_screen=SimpleNamespace(clear_all=lambda: None),
                )

                def _set_edit_mode(self, enabled):
                    self.main_stage.timeline.set_edit_mode(enabled)

            fake = Fake()
            config = ExperimentManager.load(root)
            MainWindow._populate_timeline_from_config(fake, config)
            system_rows = [row for row in fake.main_stage.timeline._devices if row[2] == "__system__"]
            self.assertEqual(len(system_rows), 1)
            self.assertEqual(len(system_rows[0][1]), 1)
            saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["sequence"][0]["device_name"], "")

        menus = []
        select_remove = [False]

        def fake_exec(menu, _pos):
            texts = [action.text() for action in menu.actions()]
            menus.append(texts)
            if select_remove[0]:
                return next((action for action in menu.actions()
                             if action.text().startswith("Remove Device")), None)
            return None

        def event_at(x, y):
            return QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(x, y), QPoint(x, y))

        with patch.object(QMenu, "exec_", new=fake_exec):
            timeline = ExperimentTimeline()
            timeline.resize(900, 400)
            timeline.add_device("Dev", "unknown")
            origin = timeline._row_origins()[0]
            name_y = origin + timeline.FLAG_HEIGHT + 10

            timeline.contextMenuEvent(event_at(10, name_y))
            self.assertTrue(any(text.startswith("Remove Device") for text in menus[-1]))

            menus.clear()
            timeline.contextMenuEvent(event_at(timeline.LABEL_WIDTH + 20, name_y))
            self.assertFalse(any(text.startswith("Remove Device") for text in menus[-1]))

            timeline.add_block(0, "New Block", start=0, duration=30)
            menus.clear()
            timeline.contextMenuEvent(event_at(timeline._x_from_time(0) + 20, name_y))
            self.assertFalse(any(text.startswith("Remove Device") for text in menus[-1]))

            select_remove[0] = True
            with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
                timeline.contextMenuEvent(event_at(10, name_y))
            self.assertTrue(any(row[0] == "Dev" for row in timeline._devices))
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                timeline.contextMenuEvent(event_at(10, name_y))
            self.assertFalse(any(row[0] == "Dev" for row in timeline._devices))

            system = ExperimentTimeline()
            system._devices.append(["System actions", [], "__system__", {}])
            system._update_total_time()
            system._update_height()
            system_origin = system._row_origins()[0]
            system.contextMenuEvent(event_at(10, system_origin + system.FLAG_HEIGHT + 10))
            self.assertFalse(any(text.startswith("Remove Device") for text in menus[-1]))


if __name__ == "__main__":
    unittest.main()
