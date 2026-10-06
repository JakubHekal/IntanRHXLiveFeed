"""Checks for RHX-native recording (storage_mode='rhs') + autolaunch plumbing."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from leech.device.intan_rhx.device import IntanRHXDevice
from leech.experiment import experiment_runner as er
from leech.experiment.experiment_runner import _RunnerThread


def _fake_connected_device(**over):
    dev = IntanRHXDevice(autolaunch=False, **over)
    dev._connected = True
    dev._sample_rate = 1000.0
    return dev


class StartRecordingSessionTest(unittest.TestCase):
    def test_commands_and_folder_resolution(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        dev = _fake_connected_device()
        calls = []
        with patch.object(dev, "set_parameter", side_effect=lambda p, v: calls.append((p, v))), \
                patch.object(dev, "get_run_mode", return_value="record"):
            prefix = "000001_rest"
            folder = root / f"{prefix}_260101_120000"
            folder.mkdir()  # RHX-created folder already present
            out = dev.start_recording_session(root, prefix)
        self.assertEqual(out, str(folder))
        self.assertEqual(calls, [
            ("createnewdirectory", "true"),
            ("fileformat", "Traditional"),
            ("filename.path", str(root)),
            ("filename.basefilename", prefix),
            ("runmode", "record"),
        ])

    def test_refused_record_raises(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dev = _fake_connected_device()
        with patch.object(dev, "set_parameter"), \
                patch.object(dev, "get_run_mode", return_value="stop"):
            with self.assertRaisesRegex(RuntimeError, "refused runmode=record"):
                dev.start_recording_session(tmp.name, "000001_x")


class TemplateRenderTest(unittest.TestCase):
    def test_render_defaults(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dev = _fake_connected_device(command_port=6000, data_port=6001,
                                     rhx_device_serial="", rhx_sample_rate_hz=0)
        with patch("tempfile.gettempdir", return_value=tmp.name):
            ini = dev._render_rhx_templates()
        self.assertTrue(ini.is_file())
        xml = (Path(tmp.name) / "leech_rhx" / "rhx_settings.xml").read_text(encoding="utf-8")
        self.assertIn('TCPCommandSocket.Port="6000"', xml)
        self.assertIn('TCPWaveformDataSocket.Port="6001"', xml)
        self.assertIn('TCPCommandSocket.Status="Pending"', xml)
        # Host/Port must precede Status (RHX parses in document order).
        self.assertIn("TCPCommandSocket.Host", xml.split("TCPCommandSocket.Status")[0])
        ini_text = ini.read_text(encoding="utf-8")
        self.assertIn("default_settings_file=", ini_text)
        self.assertIn("rhx_settings.xml", ini_text)
        self.assertNotIn("device_serial", ini_text)  # empty -> omitted

    def test_render_optional_keys(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dev = _fake_connected_device(rhx_device_serial="224800130O", rhx_sample_rate_hz=30000)
        with patch("tempfile.gettempdir", return_value=tmp.name):
            ini = dev._render_rhx_templates().read_text(encoding="utf-8")
        self.assertIn("device_serial=224800130O", ini)
        self.assertIn("sample_rate_hz=30000", ini)

    def test_find_rhx_exe_explicit(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        exe = Path(tmp.name) / "IntanRHX.exe"
        exe.write_bytes(b"")
        self.assertEqual(IntanRHXDevice._find_rhx_exe(str(exe)), str(exe))
        self.assertIsNone(IntanRHXDevice._find_rhx_exe(str(Path(tmp.name) / "missing.exe")))


class RunModeTest(unittest.TestCase):
    def test_wait_for_run_mode_accepts_record(self):
        dev = _fake_connected_device()
        with patch.object(dev, "get_run_mode", return_value="record"):
            self.assertTrue(dev.wait_for_run_mode(timeout=0.2))


class FailFastLaunchTest(unittest.TestCase):
    def test_already_running_not_listening_raises_without_launch(self):
        dev = IntanRHXDevice()
        with patch.object(IntanRHXDevice, "_connect_command_socket",
                          side_effect=OSError), \
                patch.object(IntanRHXDevice, "_rhx_process_running",
                             return_value=True), \
                patch.object(IntanRHXDevice, "_find_rhx_exe") as find_exe:
            with self.assertRaisesRegex(ConnectionError, "already running"):
                dev._ensure_rhx_running()
        find_exe.assert_not_called()

    def test_exited_process_raises_immediately(self):
        dev = IntanRHXDevice()
        proc = unittest.mock.Mock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with patch.object(IntanRHXDevice, "_connect_command_socket",
                          side_effect=OSError), \
                patch.object(IntanRHXDevice, "_rhx_process_running",
                             return_value=False), \
                patch.object(IntanRHXDevice, "_find_rhx_exe",
                             return_value="C:/Intan/IntanRHX.exe"), \
                patch.object(IntanRHXDevice, "_render_rhx_templates",
                             return_value=Path("startup.ini")), \
                patch("leech.device.intan_rhx.device.subprocess.Popen",
                      return_value=proc) as popen:
            with self.assertRaisesRegex(ConnectionError, r"exited \(code 1\)"):
                dev._ensure_rhx_running()
        popen.assert_called_once()


class RawDeviceKeyTest(unittest.TestCase):
    def test_folder_key_contains_name_and_id(self):
        class _Dev:
            device_id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
            instance = None

            def __getitem__(self, i):
                return "Intan A"

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dev = _Dev()
        inst = object()
        dev.instance = inst
        t = _RunnerThread(devices=[dev], sequence=[], run_path=tmp.name)
        t.run()
        key = t._device_key_by_instance[id(inst)]
        self.assertEqual(key, f"Intan A_{dev.device_id}")
        self.assertEqual(
            t._raw_device_key(inst, "fallback"),
            f"Intan_A_{dev.device_id}")


class _FakeRhsDevice:
    storage_mode = "rhs"
    connected = True
    sample_rate = 1000.0
    channels = []

    def __init__(self):
        self.session = None
        self.acq_started = False
        self.acq_stopped = False

    def configure(self, **kw):
        pass

    def start_recording_session(self, path, prefix):
        self.session = (str(path), prefix)
        return str(Path(path) / f"{prefix}_260101_120000")

    def start_acquisition(self):
        self.acq_started = True

    def stop_acquisition(self):
        self.acq_stopped = True

    def read_data(self):
        return np.zeros((4, 16), dtype=np.float32)


class RunStreamStorageModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def _thread_for(dev, root):
        t = _RunnerThread(devices=[], sequence=[], run_path=str(root))
        t._device_key_by_instance = {id(dev): "intan_a"}
        t._device_map = {"intan_a": dev, "IntanA": dev}
        return t

    def test_rhs_mode_no_csv(self):
        dev = _FakeRhsDevice()
        t = self._thread_for(dev, self.root)
        telemetry = []
        with patch.object(er, "append_telemetry_line", telemetry.append):
            t._run_stream(0, dev, "IntanA", {"block_label": "rest", "duration_s": 0.3}, 0.3)
        self.assertEqual(dev.session, (str(self.root / "raw" / "intan_a"), "000001_rest"))
        self.assertTrue(dev.acq_started)
        self.assertTrue(dev.acq_stopped)
        self.assertEqual(list(self.root.rglob("*.csv")), [])
        self.assertTrue(any("rhs=" in line for line in telemetry if line.startswith("acq_start")))
        self.assertTrue(any("rhs=" in line for line in telemetry if line.startswith("acq_end")))

    def test_csv_mode_still_writes_chunks(self):
        dev = _FakeRhsDevice()
        dev.storage_mode = "csv"
        t = self._thread_for(dev, self.root)
        with patch.object(er, "append_telemetry_line", lambda line: None):
            t._run_stream(0, dev, "IntanA", {"block_label": "rest", "duration_s": 0.3}, 0.3)
        self.assertIsNone(dev.session)
        csvs = list(self.root.rglob("*.csv"))
        self.assertTrue(csvs, "csv mode must still write a chunk file")
        self.assertIn("000001_rest", csvs[0].name)


if __name__ == "__main__":
    unittest.main()
