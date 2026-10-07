"""Device plugin registry + system (pseudo-device) operations.

This is device knowledge: which device implementations exist and what they
can do. Screens and UI import from here, never the other way round.
"""

from leech.device.base import DeviceOperation, ParamDef
from leech.device.intan_rhx import IntanRHXDevice
from leech.device.simulated import SimulatedRecordingDevice, SimulatedActorDevice, SimulatedCombinedDevice

try:
    from leech.device.minismu import MiniSMUDevice
except ImportError:
    MiniSMUDevice = None

# device_type -> Device subclass
_PLUGIN_REGISTRY = {
    IntanRHXDevice.device_type: IntanRHXDevice,
    SimulatedRecordingDevice.device_type: SimulatedRecordingDevice,
    SimulatedActorDevice.device_type: SimulatedActorDevice,
    SimulatedCombinedDevice.device_type: SimulatedCombinedDevice,
}
if MiniSMUDevice:
    _PLUGIN_REGISTRY[MiniSMUDevice.device_type] = MiniSMUDevice

_DEVICE_CLASSES = dict(_PLUGIN_REGISTRY)

_SYSTEM_OPERATIONS = [
    DeviceOperation("wait_input", "Wait for User Input", instantaneous=True, default_duration=0, color="#FFD700", params=[
        ParamDef("message", "Message", "str", default="Click OK to continue"),
    ]),
    DeviceOperation("log_event", "Log Event", instantaneous=True, default_duration=0, color="#FFA500"),
    DeviceOperation("start_recording", "Start Recording", instantaneous=True, default_duration=0, color="#00CC66"),
    DeviceOperation("stop_recording", "Stop Recording", instantaneous=True, default_duration=0, color="#CC3333"),
    DeviceOperation("pause", "Pause", default_duration=1.0, color="#FF8C00"),
]
