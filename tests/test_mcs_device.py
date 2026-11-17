"""Checks for MCS USBNET DLL discovery."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from leech.device.mcs_mea import device as mcs_device


class ResolveDllTest(unittest.TestCase):
    def test_prefers_explicit_then_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "explicit.dll"
            env_dll = Path(tmp) / "env.dll"
            explicit.touch()
            env_dll.touch()
            with patch.dict(os.environ, {"MCS_USBNET_DLL": str(env_dll)}), \
                    patch.object(mcs_device, "_ASSET_DIR", Path(tmp) / "assets"):
                self.assertEqual(mcs_device._resolve_dll(str(explicit)), explicit)
                self.assertEqual(mcs_device._resolve_dll(), env_dll)

    def test_error_lists_searched_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "assets"
            with patch.dict(os.environ, {"MCS_USBNET_DLL": ""}), \
                    patch.object(mcs_device, "_ASSET_DIR", asset_dir), \
                    patch.object(mcs_device.Path, "cwd", return_value=Path(tmp)):
                with self.assertRaises(FileNotFoundError) as raised:
                    mcs_device._resolve_dll()
            message = str(raised.exception)
            self.assertIn(str(asset_dir / "McsUsbNet.dll"), message)
            self.assertIn(str(Path(tmp) / "McsUsbNet.dll"), message)


if __name__ == "__main__":
    unittest.main()
