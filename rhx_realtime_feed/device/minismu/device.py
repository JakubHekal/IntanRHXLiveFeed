import numpy as np
from typing import Optional, List, Union

from ..base import Device, ChannelInfo
from .tab import SmuDeviceTab
from rhx_realtime_feed.telemetry_logger import append_telemetry_line

from minismu_py import SMU as MiniSMU, ConnectionType, SMUException


def _smu_tel(event: str, *details):
    line = f"minismu | {event}" + (" | " + " | ".join(str(d) for d in details) if details else "")
    append_telemetry_line(line)


class MiniSMUDevice(Device):
    name = "miniSMU MS01"
    device_type = "smu"

    def __init__(self, connection_type="usb", port="COM3",
                 host="192.168.1.1", tcp_port=3333, mode="FVMI"):

        ct_map = {
            "usb": ConnectionType.USB,
            "network": ConnectionType.NETWORK,
        }
        ct = ct_map.get(connection_type, ConnectionType.USB)

        self._smu = MiniSMU(ct, port=port, host=host, tcp_port=tcp_port)
        self._mode = mode.upper()
        self._connected = False
        self._acquisition_running = False
        self._sample_rate = 10

        self._channels = [
            ChannelInfo(0, "CH1", "bidirectional", "V/A", -12.0, 12.0),
            ChannelInfo(1, "CH2", "bidirectional", "V/A", -12.0, 12.0),
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate

    @property
    def channels(self) -> List[ChannelInfo]:
        return self._channels

    def connect(self) -> bool:
        try:
            idn = self._smu.get_identity()
            self._connected = bool(idn)
            _smu_tel("connect", "ok", idn)
            return self._connected
        except SMUException as e:
            _smu_tel("connect", "fail", e)
            self._connected = False
            return False

    def close(self) -> None:
        if not self._connected:
            return
        _smu_tel("close")
        self.stop_acquisition()
        try:
            self._smu.close()
        except SMUException as e:
            _smu_tel("close_error", e)
        self._connected = False

    def start_acquisition(self) -> None:
        _smu_tel("start_acquisition", "mode=" + self._mode)
        try:
            self._smu.set_mode(1, self._mode)
            self._smu.enable_channel(1)
        except SMUException as e:
            _smu_tel("start_acquisition_error", e)
        self._acquisition_running = True

    def stop_acquisition(self) -> None:
        if not self._acquisition_running:
            return
        _smu_tel("stop_acquisition")
        self._acquisition_running = False
        try:
            self._smu.disable_channel(1)
        except SMUException as e:
            _smu_tel("stop_acquisition_error", e)

    def read_data(self) -> Optional[np.ndarray]:
        try:
            v, i = self._smu.measure_voltage_and_current(1)
        except SMUException as e:
            _smu_tel("read_data_error", e)
            return None
        _smu_tel("read_data", f"v={v:.6f}", f"i={i:.9f}")
        return np.array([[v], [i]], dtype=np.float32)

    def write_output(self, channel_index: int, value: Union[float, bool]) -> None:
        ch = channel_index + 1
        _smu_tel("write_output", f"ch={ch}", f"mode={self._mode}", f"value={value}")
        try:
            if self._mode == "FVMI":
                self._smu.set_voltage(ch, float(value))
            else:
                self._smu.set_current(ch, float(value))
            self._smu.enable_channel(ch)
        except SMUException as e:
            _smu_tel("write_output_error", e)

    def trigger_action(self, channel_index: int) -> None:
        ch = channel_index + 1
        _smu_tel("trigger_action", f"ch={ch}")
        try:
            self._smu.enable_channel(ch)
        except SMUException as e:
            _smu_tel("trigger_action_error", e)

    def configure(self, **kwargs) -> None:
        _smu_tel("configure", *[f"{k}={v}" for k, v in kwargs.items()])
        if "mode" in kwargs:
            self._mode = kwargs["mode"].upper()
            try:
                self._smu.set_mode(1, self._mode)
            except SMUException as e:
                _smu_tel("configure_mode_error", e)
        if "current_protection" in kwargs:
            v = float(kwargs["current_protection"])
            try:
                self._smu.set_current_protection(1, v)
            except SMUException as e:
                _smu_tel("configure_current_protection_error", e)
        if "voltage_protection" in kwargs:
            v = float(kwargs["voltage_protection"])
            try:
                self._smu.set_voltage_protection(1, v)
            except SMUException as e:
                _smu_tel("configure_voltage_protection_error", e)
        if "oversampling" in kwargs:
            osr = int(kwargs["oversampling"])
            try:
                self._smu.set_oversampling_ratio(1, osr)
            except SMUException as e:
                _smu_tel("configure_oversampling_error", e)
        if "sample_rate" in kwargs:
            v = int(kwargs["sample_rate"])
            self._sample_rate = v
            try:
                self._smu.set_sample_rate(1, v)
            except SMUException as e:
                _smu_tel("configure_sample_rate_error", e)

    @classmethod
    def get_operations(cls):
        from ..base import DeviceOperation, ParamDef
        return [
            DeviceOperation("configure", "Configure", instantaneous=True, default_duration=0, color="#E74856", params=[
                ParamDef("current_protection", "Current Protection (A)", "float", default=0.1, min_val=0.001, max_val=1.0),
                ParamDef("voltage_protection", "Voltage Protection (V)", "float", default=12.0, min_val=0.1, max_val=24.0),
                ParamDef("oversampling", "Oversampling", "int", default=16, min_val=1, max_val=256),
                ParamDef("sample_rate", "Sample Rate (Hz)", "int", default=10, min_val=1, max_val=1000)
            ]),
            DeviceOperation("force_voltage", "Force Voltage", default_duration=5.0, color="#F1707A", params=[
                ParamDef("channel", "Channel", "int", default=1, min_val=1, max_val=2),
                ParamDef("voltage", "Voltage (V)", "float", default=5.0, min_val=-12.0, max_val=12.0),
                ParamDef("current_limit", "Current Limit (A)", "float", default=0.1, min_val=0.001, max_val=1.0),
            ]),
            DeviceOperation("measure", "Measure", instantaneous=True, default_duration=0, color="#E74856"),
            DeviceOperation("force_current", "Force Current", default_duration=60.0, color="#D13438", params=[
                ParamDef("channel", "Channel", "int", default=1, min_val=1, max_val=2),
                ParamDef("current", "Current (nA)", "float", default=-100000.0, min_val=-1000000.0, max_val=1000000.0),
                ParamDef("voltage_limit", "Voltage Limit (V)", "float", default=12.0, min_val=0.1, max_val=24.0),
            ]),
        ]

    @classmethod
    def get_config_params(cls):
        from ..base import ParamDef
        return [
            ParamDef("connection_type", "Connection Type", "choice", default="usb", choices=["usb", "network"]),
            ParamDef("port", "Port", "str", default="COM3"),
            ParamDef("host", "Host", "str", default="192.168.1.1"),
            ParamDef("tcp_port", "TCP Port", "int", default=3333, min_val=1024, max_val=65535),
        ]

    @classmethod
    def get_tab_class(cls):
        return SmuDeviceTab
