"""rhx_realtime_feed.device - Hardware device abstraction layer."""

from .device import Device, ChannelInfo, OutputSink
from ._rhx_config import RHXConfig, GetSampleRateFailure
from ._rhx_device import IntanRHXDevice
from ._simulated_devices import (
    SimulatedRecordingDevice,
    SimulatedActorDevice,
    SimulatedCombinedDevice,
)

try:
    from ._minismu_device import MiniSMUDevice
    _has_minismu = True
except ImportError:
    MiniSMUDevice = None
    _has_minismu = False


__all__ = [
    "Device", "ChannelInfo", "OutputSink",
    "RHXConfig", "GetSampleRateFailure",
    "IntanRHXDevice",
    "SimulatedRecordingDevice",
    "SimulatedActorDevice",
    "SimulatedCombinedDevice",
]

if _has_minismu:
    __all__.append("MiniSMUDevice")