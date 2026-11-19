"""Stable identifiers for the system (pseudo-device) rows in experiment JSON.

Leaf module: no imports, so migrations and the device registry can both use it.
"""

SYSTEM_DEVICE_ID = "system"
SYSTEM_DEVICE_TYPE = "__system__"
_SYSTEM_DEVICE_NAMES = frozenset(("__System__", "System actions", SYSTEM_DEVICE_TYPE))
_SYSTEM_OPERATION_IDS = frozenset(("wait_input", "log_event", "pause", "start_recording", "stop_recording"))
