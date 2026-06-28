import time
import threading
import numpy as np
from typing import Optional, List, Union

from .device import Device, ChannelInfo

from minismu_py import SMU as MiniSMU, ConnectionType, SMUException

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
        self._thread = None

        self._buffer = []
        self._buffer_lock = threading.Lock()

        self._channels = [
            ChannelInfo(0, "CH1", "bidirectional", "V/A", -12.0, 12.0),
            ChannelInfo(1, "CH2", "bidirectional", "V/A", -12.0, 12.0),
        ]

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> Optional[float]:
        return 1000.0

    @property
    def channels(self) -> List[ChannelInfo]:
        return self._channels

    def connect(self) -> bool:
        try:
            idn = self._smu.get_identity()
            self._connected = bool(idn)
            return self._connected
        except SMUException:
            self._connected = False
            return False

    def close(self) -> None:
        self.stop_acquisition()
        try:
            self._smu.close()
        except SMUException:
            pass
        self._connected = False

    def start_acquisition(self) -> None:
        if self._acquisition_running:
            return
        try:
            self._smu.set_mode(1, self._mode)
            self._smu.set_mode(2, self._mode)
            self._smu.set_sample_rate(1, 1000)
            self._smu.set_sample_rate(2, 1000)
            self._smu.start_streaming(1)
            self._smu.start_streaming(2)
        except SMUException:
            pass
        self._acquisition_running = True
        self._thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._thread.start()

    def stop_acquisition(self) -> None:
        self._acquisition_running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._smu.stop_streaming(1)
            self._smu.stop_streaming(2)
        except SMUException:
            pass

    def _stream_worker(self):
        while self._acquisition_running:
            try:
                ch, ts, v, i = self._smu.read_streaming_data()
                with self._buffer_lock:
                    self._buffer.append((ch - 1, ts, v, i))
            except SMUException:
                time.sleep(0.001)

    def read_data(self) -> Optional[np.ndarray]:
        with self._buffer_lock:
            if not self._buffer:
                return None
            buf = list(self._buffer)
            self._buffer.clear()

        if self._mode == "FVMI":
            ch1 = np.array([b[2] for b in buf if b[0] == 0], dtype=np.float32)  # voltage
            ch2 = np.array([b[2] for b in buf if b[0] == 1], dtype=np.float32)
            arr = np.stack([ch1, ch2]) if len(ch1) and len(ch2) else None
        else:
            ch1 = np.array([b[3] for b in buf if b[0] == 0], dtype=np.float32)  # current
            ch2 = np.array([b[3] for b in buf if b[0] == 1], dtype=np.float32)
            arr = np.stack([ch1, ch2]) if len(ch1) and len(ch2) else None
        return arr

    def write_output(self, channel_index: int, value: Union[float, bool]) -> None:
        ch = channel_index + 1
        try:
            if self._mode == "FVMI":
                self._smu.set_voltage(ch, float(value))
            else:
                self._smu.set_current(ch, float(value))
        except SMUException:
            pass

    def trigger_action(self, channel_index: int) -> None:
        ch = channel_index + 1
        try:
            self._smu.enable_channel(ch)
        except SMUException:
            pass

    def configure(self, **kwargs) -> None:
        if "mode" in kwargs:
            self._mode = kwargs["mode"].upper()
            try:
                self._smu.set_mode(1, self._mode)
                self._smu.set_mode(2, self._mode)
            except SMUException:
                pass
        if "current_protection" in kwargs:
            v = float(kwargs["current_protection"])
            try:
                self._smu.set_current_protection(1, v)
                self._smu.set_current_protection(2, v)
            except SMUException:
                pass
        if "voltage_protection" in kwargs:
            v = float(kwargs["voltage_protection"])
            try:
                self._smu.set_voltage_protection(1, v)
                self._smu.set_voltage_protection(2, v)
            except SMUException:
                pass
        if "oversampling" in kwargs:
            osr = int(kwargs["oversampling"])
            try:
                self._smu.set_oversampling_ratio(1, osr)
                self._smu.set_oversampling_ratio(2, osr)
            except SMUException:
                pass

    @classmethod
    def get_operations(cls):
        from .device import DeviceOperation, ParamDef
        return [
            DeviceOperation("Configure", "Configure", instantaneous=True, default_duration=0, color="#E74856", params=[
                ParamDef("current_protection", "Current Protection (A)", "float", default=0.1, min_val=0.001, max_val=1.0),
                ParamDef("voltage_protection", "Voltage Protection (V)", "float", default=12.0, min_val=0.1, max_val=24.0),
                ParamDef("oversampling", "Oversampling", "int", default=16, min_val=1, max_val=256),
            ]),
            DeviceOperation("Stimulus", "Stimulus", default_duration=5.0, color="#F1707A", params=[
                ParamDef("voltage", "Voltage (V)", "float", default=5.0, min_val=-12.0, max_val=12.0),
                ParamDef("current_limit", "Current Limit (A)", "float", default=0.1, min_val=0.001, max_val=1.0),
                ParamDef("duration_s", "Duration (s)", "float", default=1.0, min_val=0.1, max_val=60.0),
            ]),
            DeviceOperation("Measure", "Measure", instantaneous=True, default_duration=0, color="#E74856"),
        ]

    @classmethod
    def get_config_params(cls):
        from .device import ParamDef
        return [
            ParamDef("connection_type", "Connection Type", "choice", default="usb", choices=["usb", "network"]),
            ParamDef("port", "Port", "str", default="COM3"),
            ParamDef("host", "Host", "str", default="192.168.1.1"),
            ParamDef("tcp_port", "TCP Port", "int", default=3333, min_val=1024, max_val=65535),
            ParamDef("mode", "Mode", "choice", default="FVMI", choices=["FVMI", "FIMV"]),
        ]
