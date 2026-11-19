"""System (pseudo-device) identity and operations.

These steps belong to the experiment timeline, not to a hardware device.
Values are stored in experiment/run JSON, so they must stay stable.
"""

import time

from leech.telemetry_logger import append_telemetry_line
from leech.system_ids import SYSTEM_DEVICE_ID, SYSTEM_DEVICE_TYPE
from leech.device.base import DeviceOperation, ParamDef


def _wait_input(runner, step_idx, device, device_name, params, duration):
    runner._input_result = None
    runner.user_input_requested.emit(params.get("message", "Click OK to continue"))
    while runner._input_result is None and not runner._abort:
        time.sleep(0.1)
    print(f"[Runner] Wait input result: {runner._input_result}")


def _log_event(runner, step_idx, device, device_name, params, duration):
    append_telemetry_line(f"log | {step_idx} | {device_name} | {params}")


def _pause(runner, step_idx, device, device_name, params, duration):
    time.sleep(duration)


def _recording(runner, step_idx, device, device_name, params, duration):
    print(f"[Runner] Recording op {params.get('operation_id', '')} (handled by Stream)")


_SYSTEM_OPERATIONS = [
    DeviceOperation("wait_input", "Wait for User Input", instantaneous=True, default_duration=0, color="#FFD700", params=[
        ParamDef("message", "Message", "str", default="Click OK to continue"),
    ], handler=_wait_input),
    DeviceOperation("log_event", "Log Event", instantaneous=True, default_duration=0, color="#FFA500", handler=_log_event),
    DeviceOperation("start_recording", "Start Recording", instantaneous=True, default_duration=0, color="#00CC66", handler=_recording),
    DeviceOperation("stop_recording", "Stop Recording", instantaneous=True, default_duration=0, color="#CC3333", handler=_recording),
    DeviceOperation("pause", "Pause", default_duration=1.0, color="#FF8C00", handler=_pause),
]
