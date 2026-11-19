"""Device plugin registry + system (pseudo-device) operations.

This is device knowledge: which device implementations exist and what they
can do. Screens and UI import from here, never the other way round.
"""

from leech.system_ids import SYSTEM_DEVICE_ID, SYSTEM_DEVICE_TYPE
from leech.system_ops import _SYSTEM_OPERATIONS
from leech.device.intan_rhx import IntanRHXDevice
from leech.device.mcs_mea import McsMEA2100Device
from leech.device.simulated import SimulatedRecordingDevice, SimulatedActorDevice, SimulatedCombinedDevice

try:
    from leech.device.minismu import MiniSMUDevice
except ImportError:
    MiniSMUDevice = None

# device_type -> Device subclass
DEVICE_CLASSES = {}


def register(device_class):
    """Register a Device subclass; returns it so it can be used as a decorator."""
    DEVICE_CLASSES[device_class.device_type] = device_class
    return device_class


for _device_class in (
    IntanRHXDevice,
    McsMEA2100Device,
    SimulatedRecordingDevice,
    SimulatedActorDevice,
    SimulatedCombinedDevice,
):
    register(_device_class)
if MiniSMUDevice:
    register(MiniSMUDevice)

# replay format -> device class that owns the reader
REPLAY_SOURCE_CLASSES = {}
for _device_class in DEVICE_CLASSES.values():
    for _fmt in getattr(_device_class, "replay_source_formats", ()):
        REPLAY_SOURCE_CLASSES.setdefault(_fmt, _device_class)
