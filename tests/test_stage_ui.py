import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QSplitter

from leech import __version__
from leech.plot_settings import _SETTINGS, load_list, save_list
from leech.screens.stage import FluentExpander, MainStage
from leech.gui.tabs.host import TabHost
from leech.screens.timeline import ExperimentTimeline


def _spin(seconds=0.4):
    app = QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


class StageLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_version_bumped(self):
        self.assertEqual(__version__, "1.3.0")

    def test_main_stage_uses_splitter(self):
        stage = MainStage()
        self.assertEqual(len(stage.findChildren(QSplitter)), 2)
        h = stage._h_splitter
        self.assertEqual(h.orientation(), Qt.Horizontal)
        self.assertEqual(h.count(), 3)
        for i in range(3):
            self.assertFalse(h.isCollapsible(i))

    def test_vertical_splitter_between_plot_and_timeline(self):
        stage = MainStage()
        v = stage._v_splitter
        self.assertEqual(v.count(), 2)
        self.assertIsInstance(v.widget(0), TabHost)
        self.assertIsInstance(v.widget(1), ExperimentTimeline)
        for i in range(2):
            self.assertFalse(v.isCollapsible(i))

    def test_splitter_sizes_roundtrip(self):
        key = "layout/test_sizes"
        old = _SETTINGS.value(key)
        try:
            save_list(key, [111, 222, 333])
            self.assertEqual(load_list(key, [1, 2]), [111, 222, 333])
            _SETTINGS.setValue(key, "garbage")
            self.assertEqual(load_list(key, [9, 9]), [9, 9])
            _SETTINGS.remove(key)
            self.assertEqual(load_list(key, [7, 7]), [7, 7])
        finally:
            if old is None:
                _SETTINGS.remove(key)
            else:
                _SETTINGS.setValue(key, old)

    def test_expander_release_max_height(self):
        e = FluentExpander("t", expanded=True)
        e.setContentWidget(QLabel("hi"))
        e.resize(200, 100)
        e.show()
        e._toggle()
        _spin()
        self.assertFalse(e._content_frame.isVisible())
        e._toggle()
        _spin()
        self.assertTrue(e._content_frame.isVisible())
        self.assertEqual(e._content_frame.maximumHeight(), 16777215)


if __name__ == "__main__":
    unittest.main()
