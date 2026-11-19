"""MCS MEA2100 device driver for LEECH.

The vendor ``McsUsbNet.dll`` is loaded lazily. Importing this module never
requires pythonnet, the DLL, or connected hardware.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from leech.device.base import ChannelInfo, Device, DeviceOperation, ParamDef

from .layout import (
    channel_names as layout_channel_names,
    mea_layout as layout_mea_layout,
    mea_name as layout_mea_name,
)


_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "mcs" / "x64"


class _Api:
    """Lazy pythonnet bindings for McsUsbNet."""

    def __init__(self, dll_path: Path) -> None:
        import clr

        dll_dir = str(dll_path.parent)
        if dll_dir not in sys.path:
            sys.path.insert(0, dll_dir)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except OSError:
                pass

        clr.AddReference(dll_path.stem)
        from Mcs.Usb import (  # type: ignore
            CMcsUsbListNet,
            CMcsUsbNet,
            CMeaDeviceNet,
            DacqGroupChannelEnumNet,
            DataModeEnumNet,
            DeviceEnumNet,
            McsBusTypeEnumNet,
            SampleDstSizeNet,
            SampleSizeNet,
        )
        from System import Boolean, Int32, UInt32
        from System.Runtime.InteropServices import GCHandle, GCHandleType

        self.clr = clr
        self.CMcsUsbListNet = CMcsUsbListNet
        self.CMcsUsbNet = CMcsUsbNet
        self.CMeaDeviceNet = CMeaDeviceNet
        self.DacqGroupChannelEnumNet = DacqGroupChannelEnumNet
        self.DataModeEnumNet = DataModeEnumNet
        self.DeviceEnumNet = DeviceEnumNet
        self.McsBusTypeEnumNet = McsBusTypeEnumNet
        self.SampleDstSizeNet = SampleDstSizeNet
        self.SampleSizeNet = SampleSizeNet
        self.Boolean = Boolean
        self.Int32 = Int32
        self.UInt32 = UInt32
        self.GCHandle = GCHandle
        self.GCHandleType = GCHandleType


def _resolve_dll(explicit: str = "") -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("MCS_USBNET_DLL", "")
    if env:
        candidates.append(Path(env))
    candidates.append(_ASSET_DIR / "McsUsbNet.dll")
    candidates.append(Path.cwd() / "McsUsbNet.dll")
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        "McsUsbNet.dll not found. Set dll_path, MCS_USBNET_DLL, or place it in "
        f"assets/mcs/x64/.\nSearched:\n{searched}"
    )


def _status_text(api: _Api, device: Any, status: int) -> str:
    try:
        text = api.CMcsUsbNet.GetErrorText(int(status))
        if text:
            return str(text)
    except Exception:
        pass
    try:
        text = device.GetErrorText(int(status))
        if text:
            return str(text)
    except Exception:
        pass
    return f"status {int(status)}"


def _net_int_array_to_numpy(api: _Api, values: Any) -> np.ndarray:
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=np.int32)
    handle = api.GCHandle.Alloc(values, api.GCHandleType.Pinned)
    try:
        pointer = handle.AddrOfPinnedObject()
        buffer = (ctypes.c_int32 * count).from_address(int(pointer))
        return np.frombuffer(buffer, dtype=np.int32).copy()
    finally:
        handle.Free()


class McsMEA2100Device(Device):
    name = "MCS MEA2100"
    device_type = "mcs_mea2100"

    @staticmethod
    def _coerce_storage_mode(mode: Any) -> str:
        mode = str(mode or "hdf5")
        if mode != "hdf5":
            print(f"[MCS] storage_mode {mode!r} is no longer supported; using hdf5")
        return "hdf5"

    def __init__(
        self,
        sample_rate: int = 20000,
        num_channels: int = 60,
        buffer_duration_sec: float = 5.0,
        storage_mode: str = "hdf5",
        dll_path: str = "",
        **kwargs: Any,
    ) -> None:
        self._sample_rate = int(sample_rate)
        self._num_channels = int(num_channels)
        self._buffer_duration_sec = float(buffer_duration_sec)
        self.storage_mode = self._coerce_storage_mode(storage_mode)
        self._dll_path = str(dll_path or "")
        self._connected = False
        self._last_connect_error = ""
        self._thread_error = ""
        self._api: Optional[_Api] = None
        self._dev: Any = None
        self._entry: Any = None
        self._serial = ""

        self._voltage_step_uv = 1.0
        self._channels_in_block = 0
        self._total_channels_32 = 0
        self._samples_per_frame = 1
        self._analog_channels = 0

        self._chunks: deque = deque()
        self._buffer_lock = threading.RLock()
        self._buffer_capacity = max(1, int(self._sample_rate * self._buffer_duration_sec))
        self._base_pos = 0
        self._total_written = 0
        self._read_pos = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Device contract ────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> float:
        return float(self._sample_rate)

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        self._sample_rate = int(value)

    @property
    def channels(self) -> List[ChannelInfo]:
        try:
            names = layout_channel_names(self._num_channels)
        except ValueError:
            names = [f"CH{i + 1:03d}" for i in range(self._num_channels)]
        return [
            ChannelInfo(i, label, "input", "uV", -5000.0, 5000.0)
            for i, label in enumerate(names)
        ]

    @property
    def voltage_step_uv(self) -> float:
        """ADC step in microvolts, as reported by the connected hardware."""
        return float(self._voltage_step_uv)

    @property
    def adc_bits(self) -> int:
        return 24

    @property
    def mea_name(self) -> str:
        return layout_mea_name(self._num_channels)

    @property
    def mea_layout(self) -> str:
        return layout_mea_layout(self._num_channels)

    def configure(self, **kwargs: Any) -> None:
        new_rate = int(kwargs.get("sample_rate", self._sample_rate))
        new_count = int(kwargs.get("num_channels", self._num_channels))
        new_buffer = float(kwargs.get("buffer_duration_sec", self._buffer_duration_sec))
        if "storage_mode" in kwargs:
            self.storage_mode = self._coerce_storage_mode(kwargs["storage_mode"])
        if "dll_path" in kwargs:
            self._dll_path = str(kwargs["dll_path"] or "")

        changed = (new_rate != self._sample_rate or new_count != self._num_channels)
        self._sample_rate = new_rate
        self._num_channels = new_count
        self._buffer_duration_sec = max(1.0, new_buffer)
        self._buffer_capacity = max(1, int(self._sample_rate * self._buffer_duration_sec))
        if changed and self._connected:
            self.stop_acquisition()
            self._configure_hardware()
            self._reset_buffer()

    def connect(self) -> None:
        self._last_connect_error = ""
        try:
            dll = _resolve_dll(self._dll_path)
            self._api = _Api(dll)
            api = self._api
            device_list = api.CMcsUsbListNet(api.DeviceEnumNet.MCS_MEA_DEVICE)
            try:
                count = int(device_list.Count)
            except Exception:
                count = int(device_list.GetNumberOfDevices())
            if count <= 0:
                raise RuntimeError("No MCS MEA device found on USB")

            entry = device_list.GetUsbListEntry(0)
            dev = api.CMeaDeviceNet(api.McsBusTypeEnumNet.MCS_USB_BUS)
            status = int(dev.Connect(entry))
            if status != 0:
                raise RuntimeError(f"Connect failed: {_status_text(api, dev, status)}")

            self._entry = entry
            self._dev = dev
            self._serial = str(getattr(entry, "SerialNumber", "") or "")
            if self._serial:
                self.name = f"MCS MEA2100 ({self._serial})"
            self._configure_hardware()
            self._reset_buffer()
            self._connected = True
        except Exception as exc:
            self._last_connect_error = str(exc)
            self._connected = False
            self._cleanup_device()
            raise

    def close(self) -> None:
        self.stop_acquisition()
        self._cleanup_device()
        self._connected = False

    def start_acquisition(self) -> None:
        if not self._connected or self._dev is None:
            raise RuntimeError("MCS device is not connected")
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread_error = ""
        self._stop_event.clear()
        self._dev.StartDacq()
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="McsMeaAcquisition",
            daemon=True,
        )
        self._thread.start()

    def stop_acquisition(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        if self._dev is not None:
            try:
                self._dev.SendStopDacq()
            except Exception:
                pass
            try:
                self._dev.StopDacq(0)
            except Exception:
                pass

    def read_data(self) -> Optional[np.ndarray]:
        if self._thread_error:
            error = self._thread_error
            self._thread_error = ""
            raise RuntimeError(error)
        with self._buffer_lock:
            start = max(self._read_pos, self._base_pos)
            end = self._total_written
            if end <= start:
                return None
            parts = []
            pos = self._base_pos
            for chunk in self._chunks:
                chunk_end = pos + chunk.shape[1]
                lo = max(start, pos)
                hi = min(end, chunk_end)
                if lo < hi:
                    parts.append(chunk[:, lo - pos:hi - pos])
                pos = chunk_end
            self._read_pos = end
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0].copy()
            return np.concatenate(parts, axis=1)

    def write_output(self, *args: Any, **kwargs: Any) -> None:
        return None

    def trigger_action(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ── Configuration ──────────────────────────────────────────────────
    @classmethod
    def get_config_params(cls) -> List[ParamDef]:
        return [
            ParamDef("sample_rate", "Sample rate (Hz)", "int", default=20000, min_val=1000, max_val=100000),
            ParamDef("num_channels", "Electrode channels", "choice", default="60", choices=["60", "120"]),
            ParamDef("buffer_duration_sec", "Device buffer (s)", "float", default=5.0, min_val=1.0, max_val=60.0),
            ParamDef("storage_mode", "Storage format", "choice", default="hdf5", choices=["hdf5"]),
            ParamDef("dll_path", "McsUsbNet.dll override", "str", default=""),
        ]

    @classmethod
    def get_operations(cls) -> List[DeviceOperation]:
        return [
            DeviceOperation(
                "Configure",
                "Reconfigure",
                instantaneous=True,
                default_duration=0,
                params=list(cls.get_config_params()),
            ),
            DeviceOperation(
                "Stream",
                "Record",
            ),
        ]

    @classmethod
    def get_tab_class(cls):
        from .tab import McsMeaTab

        return McsMeaTab

    # ── Hardware setup ─────────────────────────────────────────────────
    def _configure_hardware(self) -> None:
        if self._dev is None or self._api is None:
            raise RuntimeError("MCS device is not connected")
        api = self._api
        dev = self._dev
        if self._num_channels not in (60, 120):
            raise ValueError("MCS MEA supports 60 or 120 channels")

        try:
            dev.StopDacq(0)
        except Exception:
            pass

        dev.SetSamplerate(int(self._sample_rate), 1, 0)
        dev.SetDataMode(api.DataModeEnumNet.Signed_32bit, 0)
        try:
            dev.SetVoltageRangeByIndex(0, 0)
        except Exception:
            pass

        hs2 = 60 if self._num_channels == 120 else 0
        dev.SetNumberOfAnalogChannels(60, hs2, 0, 8, 0)

        try:
            dev.EnableDigitalIn(api.Boolean(False), api.UInt32(0))
        except Exception:
            pass
        try:
            dev.EnableChecksum(api.Boolean(True), api.UInt32(0))
        except Exception:
            pass

        res = api.clr.Reference[api.Int32]()
        res_unit = api.clr.Reference[api.Int32]()
        dev.GetResolutionPerDigit(
            0,
            api.DacqGroupChannelEnumNet.HeadstageElectrodeGroup,
            res,
            res_unit,
        )
        step = float(res.Value) * (10.0 ** (-float(res_unit.Value))) * 1e6
        if not np.isfinite(step) or step <= 0:
            raise RuntimeError(f"Invalid MCS voltage resolution: {step}")
        self._voltage_step_uv = step

        analog = api.clr.Reference[api.Int32]()
        digital = api.clr.Reference[api.Int32]()
        checksum = api.clr.Reference[api.Int32]()
        timestamp = api.clr.Reference[api.Int32]()
        block = api.clr.Reference[api.Int32]()
        try:
            dev.GetChannelLayout(analog, digital, checksum, timestamp, block, 0)
            self._channels_in_block = int(block.Value)
            self._analog_channels = int(analog.Value)
        except Exception:
            self._channels_in_block = int(dev.GetChannelsInBlock(0))
            self._analog_channels = self._channels_in_block // 2

        self._total_channels_32 = self._channels_in_block // 2
        if self._total_channels_32 < self._num_channels:
            raise RuntimeError(
                f"Device reports {self._total_channels_32} 32-bit channels, "
                f"but {self._num_channels} electrode channels were requested"
            )

        actual_rate = int(dev.GetSamplerate(0))
        if actual_rate > 0:
            self._sample_rate = actual_rate
        ratio = int(dev.GetSampleratePerFramerate(0))
        self._samples_per_frame = max(1, ratio)

        threshold = max(1, self._sample_rate // 10)
        queue_size = max(threshold * 10, self._sample_rate)
        dev.ChannelBlock.SetSelectedData(
            self._total_channels_32,
            queue_size,
            threshold,
            api.SampleSizeNet.SampleSize32Signed,
            api.SampleDstSizeNet.SampleDstSize32,
            self._channels_in_block,
        )
        try:
            dev.ChannelBlock.SetCommonThreshold(threshold)
        except Exception:
            pass
        try:
            dev.ChannelBlock.SetCheckChecksum(checksum.Value, timestamp.Value)
        except Exception:
            pass

    def _reset_buffer(self) -> None:
        with self._buffer_lock:
            self._chunks.clear()
            self._buffer_capacity = max(1, int(self._sample_rate * self._buffer_duration_sec))
            self._base_pos = 0
            self._total_written = 0
            self._read_pos = 0

    # ── Acquisition ────────────────────────────────────────────────────
    def _acquisition_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._read_available()
                if data is None:
                    time.sleep(0.005)
                    continue
                self._append_buffer(data)
            except Exception as exc:
                self._thread_error = f"MCS acquisition failed: {exc}"
                break

    def _read_available(self) -> Optional[np.ndarray]:
        if self._api is None or self._dev is None:
            return None
        api = self._api
        block = self._dev.ChannelBlock
        available = int(block.AvailFrames(0, 0))
        if available <= 0:
            return None
        read_count = min(available, max(1, self._sample_rate // 5))
        frames_ret = api.clr.Reference[api.Int32]()
        raw_array = block.ReadFramesI32(0, 0, read_count, frames_ret)
        frames = int(frames_ret.Value)
        if frames <= 0:
            return None

        flat = _net_int_array_to_numpy(api, raw_array)
        total = self._total_channels_32
        ratio = self._samples_per_frame
        if ratio > 1 and flat.size >= frames * total * ratio:
            matrix = flat[: frames * total * ratio].reshape(frames, total, ratio)
            matrix = matrix.transpose(1, 0, 2).reshape(total, frames * ratio)
        elif flat.size >= frames * total:
            matrix = flat[: frames * total].reshape(frames, total).T
        else:
            raise RuntimeError(
                f"MCS frame shape mismatch: got {flat.size} values, "
                f"expected {frames * total} or {frames * total * ratio}"
            )

        data = matrix[: self._num_channels].astype(np.float32, copy=False)
        data *= np.float32(self._voltage_step_uv)
        return np.ascontiguousarray(data)

    def _append_buffer(self, data: np.ndarray) -> None:
        with self._buffer_lock:
            self._chunks.append(data)
            self._total_written += data.shape[1]
            while self._total_written - self._base_pos > self._buffer_capacity:
                old = self._chunks.popleft()
                self._base_pos += old.shape[1]
            if self._read_pos < self._base_pos:
                self._read_pos = self._base_pos

    def _cleanup_device(self) -> None:
        dev = self._dev
        self._dev = None
        if dev is not None:
            try:
                dev.Disconnect()
            except Exception:
                pass
            try:
                dev.Dispose()
            except Exception:
                pass
        entry = self._entry
        self._entry = None
        if entry is not None:
            try:
                entry.Dispose()
            except Exception:
                pass
