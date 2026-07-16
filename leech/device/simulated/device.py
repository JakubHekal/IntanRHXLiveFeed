import time
import numpy as np
import threading
from typing import Optional, List, Union

from ..base import Device, ChannelInfo
from .tab import SimpleDeviceTab


class SimulatedRecordingDevice(Device):
    name = "Simulated Recorder"
    device_type = "simulated_recorder"

    def __init__(self, sample_rate=20000, num_channels=1, amplitude=500.0,
                 frequency=10.0, noise_level=10.0):
        self._sample_rate = float(sample_rate)
        self._num_channels = int(num_channels)
        self._amplitude = float(amplitude)
        self._frequency = float(frequency)
        self._noise_level = float(noise_level)
        self._connected = False
        self._streaming = False
        self._phase = 0.0
        self._buffer = np.zeros((self._num_channels, 0), dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._thread = None
        self._t0 = 0.0

        self._channels = [
            ChannelInfo(i, f"CH{i + 1}", "input", "uV", -self._amplitude * 1.2, self._amplitude * 1.2)
            for i in range(self._num_channels)
        ]

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
        self._connected = True
        return True

    def close(self) -> None:
        self.stop_acquisition()
        self._connected = False

    def start_acquisition(self) -> None:
        if self._streaming:
            return
        self._streaming = True
        self._t0 = time.perf_counter()
        self._buffer = np.zeros((self._num_channels, 0), dtype=np.float32)
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def stop_acquisition(self) -> None:
        self._streaming = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _generate(self):
        while self._streaming:
            elapsed = time.perf_counter() - self._t0
            n = int(self._sample_rate * 0.05)
            t = np.arange(n, dtype=np.float32) / self._sample_rate
            phase = 2.0 * np.pi * self._frequency * (elapsed + t)
            data = self._amplitude * np.sin(phase[np.newaxis, :] + np.arange(self._num_channels)[:, np.newaxis])
            data += np.random.normal(0, self._noise_level, data.shape)
            with self._buffer_lock:
                self._buffer = np.hstack([self._buffer, data.astype(np.float32)])
                max_len = int(self._sample_rate * 5)
                if self._buffer.shape[1] > max_len:
                    self._buffer = self._buffer[:, -max_len:]
            time.sleep(0.05)

    def read_data(self) -> Optional[np.ndarray]:
        with self._buffer_lock:
            if self._buffer.shape[1] < 1:
                return None
            data = self._buffer.copy()
            self._buffer = np.zeros((self._num_channels, 0), dtype=np.float32)
        return data

    def write_output(self, channel_index: int, value: Union[float, bool]) -> None:
        pass

    def trigger_action(self, channel_index: int) -> None:
        pass

    @classmethod
    def get_operations(cls):
        from ..base import DeviceOperation
        return [
            DeviceOperation("Configure", "Configure", instantaneous=True, default_duration=0, color="#2B88D8"),
            DeviceOperation("Stream", "Stream", default_duration=12.0, color="#4BA3E3"),
        ]

    @classmethod
    def get_config_params(cls):
        from ..base import ParamDef
        return [
            ParamDef("sample_rate", "Sample Rate (Hz)", "int", default=20000, min_val=1000, max_val=50000),
            ParamDef("num_channels", "Num Channels", "int", default=1, min_val=1, max_val=256),
            ParamDef("amplitude", "Amplitude (µV)", "float", default=500.0, min_val=0.0, max_val=5000.0),
            ParamDef("frequency", "Frequency (Hz)", "float", default=10.0, min_val=0.1, max_val=1000.0),
            ParamDef("noise_level", "Noise Level (µV)", "float", default=10.0, min_val=0.0, max_val=500.0),
        ]

    @classmethod
    def get_tab_class(cls):
        return SimpleDeviceTab


class SimulatedActorDevice(Device):
    name = "Simulated Actor"
    device_type = "simulated_actor"

    def __init__(self, num_outputs=2):
        self._num_outputs = int(num_outputs)
        self._connected = False
        self.command_log = []

        self._channels = [
            ChannelInfo(i, f"OUT{i + 1}", "output", "V", -12.0, 12.0)
            for i in range(self._num_outputs)
        ]

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> Optional[float]:
        return None

    @property
    def channels(self) -> List[ChannelInfo]:
        return self._channels

    def connect(self) -> bool:
        self._connected = True
        return True

    def close(self) -> None:
        self._connected = False

    def start_acquisition(self) -> None:
        pass

    def stop_acquisition(self) -> None:
        pass

    def read_data(self) -> Optional[np.ndarray]:
        return None

    def write_output(self, channel_index: int, value: Union[float, bool]) -> None:
        self.command_log.append(("write", channel_index, value, time.perf_counter()))

    def trigger_action(self, channel_index: int) -> None:
        self.command_log.append(("trigger", channel_index, None, time.perf_counter()))

    def configure(self, **kwargs) -> None:
        self.command_log.append(("configure", kwargs, time.perf_counter()))

    @classmethod
    def get_operations(cls):
        from ..base import DeviceOperation, ParamDef
        return [
            DeviceOperation("Configure", "Configure", instantaneous=True, default_duration=0, color="#B146C2"),
            DeviceOperation("Write", "Write", default_duration=2.0, color="#C239B3", params=[
                ParamDef("channel", "Channel", "int", default=1, min_val=1, max_val=64),
                ParamDef("value", "Value (V)", "float", default=5.0, min_val=-12.0, max_val=12.0),
            ]),
            DeviceOperation("Trigger", "Trigger", default_duration=0.5, color="#8764B8", params=[
                ParamDef("channel", "Channel", "int", default=1, min_val=1, max_val=64),
            ]),
        ]

    @classmethod
    def get_config_params(cls):
        from ..base import ParamDef
        return [
            ParamDef("num_outputs", "Num Outputs", "int", default=2, min_val=1, max_val=64),
        ]

    @classmethod
    def get_tab_class(cls):
        return SimpleDeviceTab


class SimulatedCombinedDevice(Device):
    name = "Simulated Combined"
    device_type = "simulated_combined"

    def __init__(self, sample_rate=10000, num_inputs=2, num_outputs=2):
        self._recorder = SimulatedRecordingDevice(
            sample_rate=sample_rate, num_channels=num_inputs,
        )
        self._actor = SimulatedActorDevice(num_outputs=num_outputs)
        self._connected = False

        combined = []
        for c in self._recorder.channels:
            combined.append(ChannelInfo(c.index, c.name, "bidirectional", c.unit, c.range_min, c.range_max))
        base = len(self._recorder.channels)
        for c in self._actor.channels:
            combined.append(ChannelInfo(base + c.index, c.name, "bidirectional", c.unit, c.range_min, c.range_max))
        self._channels = combined

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> Optional[float]:
        return self._recorder.sample_rate

    @property
    def channels(self) -> List[ChannelInfo]:
        return self._channels

    def connect(self) -> bool:
        self._connected = self._recorder.connect() and self._actor.connect()
        return self._connected

    def close(self) -> None:
        self._recorder.close()
        self._actor.close()
        self._connected = False

    def start_acquisition(self) -> None:
        self._recorder.start_acquisition()

    def stop_acquisition(self) -> None:
        self._recorder.stop_acquisition()

    def read_data(self) -> Optional[np.ndarray]:
        return self._recorder.read_data()

    def write_output(self, channel_index: int, value: Union[float, bool]) -> None:
        self._actor.write_output(channel_index, value)

    def trigger_action(self, channel_index: int) -> None:
        self._actor.trigger_action(channel_index)

    def configure(self, **kwargs) -> None:
        self._recorder.configure(**{k: v for k, v in kwargs.items() if k in ("frequency", "amplitude", "noise_level")})
        self._actor.configure(**{k: v for k, v in kwargs.items() if k not in ("frequency", "amplitude", "noise_level")})

    @classmethod
    def get_operations(cls):
        from ..base import DeviceOperation, ParamDef
        return [
            DeviceOperation("Configure", "Configure", instantaneous=True, default_duration=0, color="#2B88D8"),
            DeviceOperation("Stream", "Stream", default_duration=10.0, color="#4BA3E3"),
            DeviceOperation("Write", "Write", default_duration=2.0, color="#C239B3", params=[
                ParamDef("channel", "Channel", "int", default=1, min_val=1, max_val=64),
                ParamDef("value", "Value (V)", "float", default=5.0, min_val=-12.0, max_val=12.0),
            ]),
            DeviceOperation("Trigger", "Trigger", default_duration=0.5, color="#8764B8", params=[
                ParamDef("channel", "Channel", "int", default=1, min_val=1, max_val=64),
            ]),
        ]

    @classmethod
    def get_config_params(cls):
        from ..base import ParamDef
        return [
            ParamDef("sample_rate", "Sample Rate (Hz)", "int", default=10000, min_val=1000, max_val=50000),
            ParamDef("num_inputs", "Num Inputs", "int", default=2, min_val=1, max_val=256),
            ParamDef("num_outputs", "Num Outputs", "int", default=2, min_val=1, max_val=64),
        ]

    @classmethod
    def get_tab_class(cls):
        return SimpleDeviceTab
