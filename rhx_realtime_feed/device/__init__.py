"""rhx_realtime_feed.device - Hardware device abstraction layer."""

from .device import Device, ChannelInfo, OutputSink, DeviceOperation, ParamDef
from .intan_rhx import IntanRHXDevice, GetSampleRateFailure
from .simulated import SimulatedRecordingDevice, SimulatedActorDevice, SimulatedCombinedDevice

try:
    from .minismu import MiniSMUDevice
    _has_minismu = True
except ImportError:
    MiniSMUDevice = None
    _has_minismu = False

# Plugin registry: device_type -> Device subclass
_PLUGIN_REGISTRY = {
    IntanRHXDevice.device_type: IntanRHXDevice,
    SimulatedRecordingDevice.device_type: SimulatedRecordingDevice,
    SimulatedActorDevice.device_type: SimulatedActorDevice,
    SimulatedCombinedDevice.device_type: SimulatedCombinedDevice,
}
if MiniSMUDevice:
    _PLUGIN_REGISTRY[MiniSMUDevice.device_type] = MiniSMUDevice

__all__ = [
    "Device", "ChannelInfo", "OutputSink", "DeviceOperation", "ParamDef",
    "GetSampleRateFailure",
    "IntanRHXDevice",
    "SimulatedRecordingDevice",
    "SimulatedActorDevice",
    "SimulatedCombinedDevice",
]

if _has_minismu:
    __all__.append("MiniSMUDevice")